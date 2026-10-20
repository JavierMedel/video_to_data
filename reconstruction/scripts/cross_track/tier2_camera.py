"""Static camera fit from own-tier correspondences with held-out review."""
import argparse
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from .common import load, new_run, save, sha, complete


def project(points, parameters):
    fx, fy = np.exp(parameters[:2]); cx, cy = parameters[2:4]
    rotation = Rotation.from_rotvec(parameters[4:7]).as_matrix()
    camera = np.asarray(points)@rotation.T+parameters[7:10]
    depth = camera[..., 2]
    xy = camera[..., :2]/np.maximum(depth[..., None], 1e-6)
    return xy*np.array([fx, fy])+[cx, cy], depth


def calibrate(points, pixels, visible, frames, image_shape):
    import cv2
    points, pixels, visible, frames = np.asarray(points, float), np.asarray(pixels, float), np.asarray(visible), np.asarray(frames)
    if (points.ndim != 3 or points.shape[-1] != 3 or pixels.shape != (*points.shape[:2], 2)
            or visible.shape != points.shape[:2] or visible.dtype.kind != 'b'
            or frames.shape != (len(points),) or len(np.unique(frames)) != len(frames)
            or frames.dtype.kind not in 'iu' or not np.isfinite(points).all() or not np.isfinite(pixels).all()):
        raise ValueError('Invalid finite camera correspondence grid')
    train = visible & (frames[:, None] % 2 == 0)
    held = visible & (frames[:, None] % 2 != 0)
    if min(train.sum(), held.sum()) < 20 or min(np.count_nonzero(train.any(1)), np.count_nonzero(held.any(1))) < 2:
        raise ValueError('Insufficient independent training/held-out frame observations')
    world, xy = points[train], pixels[train]
    if np.linalg.matrix_rank(world-world.mean(0), tol=1e-6) < 3: raise ValueError('Camera self-calibration is spatially degenerate')
    height, width = map(int, image_shape)
    if min(height, width) < 16: raise ValueError('Invalid image size')
    if np.any(xy < 0) or np.any(xy[:, 0] >= width) or np.any(xy[:, 1] >= height): raise ValueError('Training keypoints outside declared image')
    candidates = []
    for scale in (.6, 1., 1.6, 2.5):
        focal = max(height, width)*scale; k = np.array([[focal, 0, width/2], [0, focal, height/2], [0, 0, 1.]])
        cv2.setRNGSeed(42)
        ok, r, t, _ = cv2.solvePnPRansac(world, xy, k, None, iterationsCount=100, reprojectionError=8.)
        if not ok: continue
        initial = np.r_[np.log([focal, focal]), [width/2, height/2], r.ravel(), t.ravel()]
        def residual(parameters):
            predicted, depth = project(world, parameters)
            return np.r_[(predicted-xy).ravel(), np.minimum(depth-.001, 0)*1000]
        lower = np.r_[np.log([.2*width, .2*height]), [0, 0], [-np.inf]*6]
        upper = np.r_[np.log([5*width, 5*height]), [width, height], [np.inf]*6]
        result = least_squares(residual, initial, bounds=(lower, upper), loss='huber', f_scale=3., max_nfev=500)
        if result.success: candidates.append(result)
    if not candidates: raise ValueError('Static camera optimization failed')
    fit = min(candidates, key=lambda v: v.cost)
    predicted, depth = project(points, fit.x)
    train_error = np.linalg.norm(predicted[train]-pixels[train], axis=1)
    held_error = np.linalg.norm(predicted[held]-pixels[held], axis=1)
    k = np.array([[np.exp(fit.x[0]), 0, fit.x[2]], [0, np.exp(fit.x[1]), fit.x[3]], [0, 0, 1.]])
    w2c = np.eye(4); w2c[:3, :3] = Rotation.from_rotvec(fit.x[4:7]).as_matrix(); w2c[:3, 3] = fit.x[7:10]
    singular = np.linalg.svd(fit.jac, compute_uv=False)
    full_rank = singular[-1] > 1e-8*singular[0]
    passed = bool(full_rank and np.all(depth[visible] > .001) and np.median(held_error) <= 3. and np.quantile(held_error, .95) <= 10.)
    return {'intrinsics': k.tolist(), 'world_to_camera': w2c.tolist(), 'held_out_passed': passed,
            'train_median_error_px': float(np.median(train_error)), 'held_out_median_error_px': float(np.median(held_error)),
            'held_out_p95_error_px': float(np.quantile(held_error, .95)), 'jacobian_full_rank': bool(full_rank),
            'training_frames': frames[train.any(1)].tolist(), 'held_out_frames': frames[held.any(1)].tolist(),
            'scope': 'Own-tier static pinhole model; no distortion model or hidden-reference pose claim'}


def run(source, evidence, output):
    source, output = Path(source), Path(output)
    manifest = load(source/'source_manifest.json')
    if manifest.get('tier') != 2: raise ValueError('Self-calibration requires own Tier 2 observations')
    with np.load(evidence, allow_pickle=False) as raw: e = dict(raw)
    if str(e['source_manifest_sha256'].item()) != sha(source/'source_manifest.json'):
        raise ValueError('Camera evidence belongs to another source')
    if str(e['coordinate_frame'].item()) != 'released_track2_opencv_world': raise ValueError('Camera evidence world frame mismatch')
    new_run(output, [source/'source_manifest.json', evidence], {'method': 'tier2_static_pinhole_v1'})
    result = calibrate(e['points_world'], e['points_xy'], e['visible'], e['frames'], e['image_shape'])
    save(output/'camera.json', result)
    # FoundationPose expects calibrated K; downstream review must still verify
    # silhouette agreement, depth/mask resolution, and the body/world transform.
    k = result['intrinsics']; height, width = map(int, e['image_shape'])
    save(output/'intrinsics.json', {'fx': k[0][0], 'fy': k[1][1], 'cx': k[0][2], 'cy': k[1][2], 'width': width, 'height': height})
    complete(output, 'CAMERA_REQUIRES_SILHOUETTE_REVIEW' if result['held_out_passed'] else 'CAMERA_REJECTED', **result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'evidence', 'output'): p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args(); run(a.source, a.evidence, a.output)
