# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Singleton HaMeR loader and per-crop forward helper.

Wraps the upstream API:
    - hamer.models.load_hamer                          → (model, cfg)
    - hamer.datasets.utils.generate_image_patch_cv2    → 256×256 crop

`cam_crop_to_full` is reimplemented locally — HaMeR ships it inside
``utils_detectron2.py`` which top-level imports ``detectron2``; we don't need
detectron2 for inference, so we keep that dep out of the image.
"""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import torch

_MODEL = None
_CFG   = None


def get_model(weights_dir: str) -> Tuple[object, object]:
    """Return a singleton (HaMeR model, cfg).

    HaMeR's model_config.yaml carries relative paths (e.g.
    ``./_DATA/data/mano_mean_params.npz``) that resolve against the *current
    working directory*. We chdir into ``weights_dir`` for the duration of
    model construction so those paths land on the user's mounted weights tree.
    """
    global _MODEL, _CFG
    if _MODEL is None:
        # Keep single-crop and batched kernels numerically consistent.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
        from hamer.models import load_hamer

        ckpt = os.path.join(weights_dir, "_DATA", "hamer_ckpts", "checkpoints", "hamer.ckpt")
        if not os.path.exists(ckpt):
            raise FileNotFoundError(
                f"HaMeR checkpoint not found at {ckpt}. "
                "Run `python -m v2d.hamer.lib.download_weights --weights_dir <dir>` first."
            )
        prev_cwd = os.getcwd()
        try:
            os.chdir(weights_dir)
            _MODEL, _CFG = load_hamer(ckpt)
        finally:
            os.chdir(prev_cwd)
        _MODEL = _MODEL.to("cuda").eval()
    return _MODEL, _CFG


def crop_for_hamer(
    image: np.ndarray, cx: float, cy: float, size: float, cfg, flip: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Square crop centred at (cx, cy) of side `size`, resized to HaMeR input.

    HaMeR is trained as a right-hand model and processes left hands by
    horizontally flipping the input patch. The caller passes ``flip=True``
    for left hands so the model receives a right-hand-oriented patch. The
    returned ``box_center`` is in original-image pixels (matching HaMeR's
    demo convention; ``cam_crop_to_full`` lifts to the original frame).

    Returns:
        patch          (3, H, W) float32, normalized.
        box_center     (2,)   the crop center in original-image pixels.
        box_size       scalar — the crop side in original-image pixels.
    """
    from hamer.datasets.utils import generate_image_patch_cv2

    patch_w = patch_h = cfg.MODEL.IMAGE_SIZE
    bgr = image[..., ::-1].copy()  # generate_image_patch_cv2 expects BGR
    patch_bgr, _ = generate_image_patch_cv2(
        bgr, cx, cy, size, size, patch_w, patch_h,
        flip, 1.0, 0,    # flip, scale, rot
        # default border_mode = cv2.BORDER_CONSTANT
    )
    patch_rgb = patch_bgr[..., ::-1].astype(np.float32) / 255.0
    mean = np.array(cfg.MODEL.IMAGE_MEAN, dtype=np.float32)
    std  = np.array(cfg.MODEL.IMAGE_STD,  dtype=np.float32)
    patch = (patch_rgb - mean) / std
    patch = patch.transpose(2, 0, 1).astype(np.float32)   # (3, H, W)
    return patch, np.array([cx, cy], dtype=np.float64), float(size)


def _cam_crop_to_full(
    cam_bbox: np.ndarray,    # (3,) — (s, tx, ty) per-crop weak-perspective
    box_center: np.ndarray,  # (2,) in original-image pixels
    box_size: float,         # crop side in original-image pixels
    img_size: np.ndarray,    # (2,) — (W, H)
    focal_length: float,
) -> np.ndarray:
    """HaMeR's weak-perspective → full-image perspective conversion.

    Re-implemented to avoid ``detectron2`` (which HaMeR's bundled util pulls in
    only because the file shares a module with detector glue). Output is the
    3-vec camera translation under a virtual pinhole at focal=focal_length,
    principal point at the image center.
    """
    img_w, img_h = float(img_size[0]), float(img_size[1])
    cx, cy = float(box_center[0]), float(box_center[1])
    b = float(box_size)
    s, tx_c, ty_c = (float(v) for v in cam_bbox)
    bs = b * s + 1e-9
    tz = 2.0 * focal_length / bs
    tx = (2.0 * (cx - img_w / 2.0) / bs) + tx_c
    ty = (2.0 * (cy - img_h / 2.0) / bs) + ty_c
    return np.array([tx, ty, tz], dtype=np.float64)


def run_hamer(
    model, cfg, image: np.ndarray, cx: float, cy: float, size: float, is_right: bool
) -> dict:
    """Run HaMeR on a single hand crop. Returns a dict of CPU numpy arrays.

    Output keys:
      betas              (10,)
      hand_pose          (15, 3, 3)   rotation matrices (raw HaMeR output)
      global_orient      (1, 3, 3)    rotation matrix
      pred_cam           (3,)         per-crop weak-perspective (s, tx, ty)
      pred_cam_t         (3,)         per-crop camera translation
      pred_cam_t_full    (3,)         translation lifted to full-image pinhole
      scaled_focal_length scalar      focal length for the full image
      pred_vertices      (778, 3)     MANO vertices in cam space
      pred_keypoints_3d  (21, 3)
    """
    return run_hamer_batch(model, cfg, [(image, cx, cy, size, is_right)])[0]


def run_hamer_batch(model, cfg, crops) -> list[dict]:
    """Infer independent hand crops in one batch, preserving crop order."""
    if not crops:
        return []
    prepared = [crop_for_hamer(image, cx, cy, size, cfg, flip=not right)
                for image, cx, cy, size, right in crops]
    patch_t = torch.from_numpy(np.stack([p[0] for p in prepared])).to("cuda")
    right_t = torch.tensor([int(c[-1]) for c in crops], dtype=torch.float32, device="cuda")
    with torch.no_grad():
        out = model({"img": patch_t, "right": right_t})
    arrays = {key: out[key].detach().cpu().numpy() for key in
              ("pred_cam", "pred_cam_t", "pred_vertices", "pred_keypoints_3d")}
    mano = {key: value.detach().cpu().numpy() for key, value in out["pred_mano_params"].items()}
    results = []
    for index, ((image, cx, cy, size, is_right), (_, box_center, box_size)) in enumerate(zip(crops, prepared)):
        H, W = image.shape[:2]
        pred_cam = arrays["pred_cam"][index].copy()
        pred_cam_t = arrays["pred_cam_t"][index].copy()
        betas = mano["betas"][index].copy()
        hand_pose = mano["hand_pose"][index].copy()
        global_orient = mano["global_orient"][index].copy()
        vertices = arrays["pred_vertices"][index].copy()
        joints3d = arrays["pred_keypoints_3d"][index].copy()
        # HaMeR demo: pred_cam[1] (= tx in crop frame) gets sign-flipped for left
        # hand BEFORE cam_crop_to_full. This compensates for the patch flip so
        # cam_t lands in the original-image frame, with box_center kept unflipped.
        multiplier = 1 if is_right else -1
        pred_cam[1] *= multiplier

        scaled_focal_length = float(
            cfg.EXTRA.FOCAL_LENGTH / cfg.MODEL.IMAGE_SIZE * max(W, H)
        )
        pred_cam_t_full = _cam_crop_to_full(
            pred_cam, box_center, box_size, np.array([W, H], dtype=np.float64),
            scaled_focal_length,
        )

        # Vertex/joint x-flip for left hand. cam_t stays as-is — the pred_cam
        # correction above already put it in the original-image frame.
        if not is_right:
            vertices[:, 0] *= -1
            joints3d[:, 0] *= -1

        results.append({
            "betas":               betas,
            "hand_pose":           hand_pose,
            "global_orient":       global_orient,
            "pred_cam":            pred_cam,
            "pred_cam_t":          pred_cam_t,
            "pred_cam_t_full":     pred_cam_t_full,
            "scaled_focal_length": scaled_focal_length,
            "pred_vertices":       vertices,
            "pred_keypoints_3d":   joints3d,
        })
    return results


def rotmat_to_axis_angle(R: np.ndarray) -> np.ndarray:
    """Batched rotation-matrix → axis-angle. Accepts (..., 3, 3); returns (..., 3).

    Uses a stable SO(3) projection and log map, including rotations near 180°.
    """
    from scipy.spatial.transform import Rotation

    R = np.asarray(R, dtype=np.float64)
    if R.shape[-2:] != (3, 3):
        raise ValueError("Expected rotation matrices with shape (..., 3, 3)")
    if R.size == 0:
        return np.empty(R.shape[:-2] + (3,), dtype=np.float64)
    # Project small network/float32 orthogonality errors onto SO(3). The
    # trace/sin(theta) formula amplifies those errors near 180 degrees.
    return Rotation.from_matrix(R.reshape(-1, 3, 3)).as_rotvec().reshape(R.shape[:-2] + (3,))
