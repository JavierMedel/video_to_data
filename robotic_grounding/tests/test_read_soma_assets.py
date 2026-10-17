# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the reproducible SOMA body-model asset contract."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from robotic_grounding.retarget.read_soma import (
    SOMA,
    _SOMA_NEUTRAL_RIG_KEYS,
    _resolve_data_root,
)


def _write_minimal_asset_bundle(root: Path) -> None:
    root.mkdir(parents=True)
    np.savez(
        root / "SOMA_neutral.npz",
        **{key: np.empty(0) for key in _SOMA_NEUTRAL_RIG_KEYS},
    )
    (root / "correctives_model.pt").touch()
    for relative in (
        "MHR/mhr_model_lod1.pt",
        "MHR/base_body_lod1.obj",
        "MHR/SOMA_wrap_lod1.obj",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


def test_resolve_data_root_accepts_complete_explicit_bundle(tmp_path: Path) -> None:
    root = tmp_path / "soma"
    _write_minimal_asset_bundle(root)

    assert _resolve_data_root(root, "mhr") == root


def test_resolve_data_root_rejects_mutable_upstream_fallback(tmp_path: Path) -> None:
    root = tmp_path / "missing-soma"

    with pytest.raises(FileNotFoundError, match="setup_soma_assets.py") as exc_info:
        _resolve_data_root(root, "mhr")

    assert "mutable SOMA-X default snapshot is intentionally not used" in str(
        exc_info.value
    )


def test_reconstruction_forwards_time_varying_bone_lengths():
    import torch

    model = SOMA.__new__(SOMA)
    model.device = torch.device("cpu")
    received = {}

    def layer(poses, identity, **kwargs):
        received.update(kwargs)
        return {"vertices": torch.zeros((2, 3, 3)), "joints": torch.zeros((2, 1, 3))}

    model.layer = layer
    model._compute_global_rotations = lambda *args, **kwargs: torch.eye(3).expand(2, 1, 3, 3)
    flex = np.arange(12, dtype=np.float32).reshape(2, 6)
    model._reconstruct(poses_rotvec=np.zeros((2, 1, 3), dtype=np.float32),
        transl=np.zeros((2, 3), dtype=np.float32), identity=np.zeros(45, dtype=np.float32),
        scale=np.zeros(68, dtype=np.float32), joint_orient=np.eye(3)[None],
        keep_root=False, bone_length_flexibles=flex)
    np.testing.assert_array_equal(received["kwargs"]["bone_length_flexibles"].numpy(), flex)


def test_reconstruction_rejects_wrong_bone_length_timeline():
    import torch

    model = SOMA.__new__(SOMA)
    model.device = torch.device("cpu")
    with pytest.raises(ValueError, match="shape"):
        model._reconstruct(poses_rotvec=np.zeros((2, 1, 3), dtype=np.float32),
            transl=np.zeros((2, 3), dtype=np.float32), identity=np.zeros(45, dtype=np.float32),
            scale=np.zeros(68, dtype=np.float32), joint_orient=np.eye(3)[None],
            keep_root=False, bone_length_flexibles=np.zeros((1, 6)))
