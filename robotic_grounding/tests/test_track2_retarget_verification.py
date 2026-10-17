"""Check that preparation gates reject changes to object motion across frames."""
import importlib
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


@pytest.fixture
def verifier(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module("retarget_track2_tier1")


def fixture_motion(tmp_path, *, position_drift=False, rotation_drift=False):
    poses = np.broadcast_to(np.eye(4), (3, 4, 4)).copy()
    poses[:, 0, 3] = [0, 1, 2]
    np.save(tmp_path / "poses.npy", poses)
    positions = np.array([[3., 4., 5.], [3., 5., 5.], [3., 6., 5.]])
    quaternions = np.tile([np.sqrt(.5), 0, 0, np.sqrt(.5)], (3, 1))
    if position_drift:
        positions[2, 0] += .001
    if rotation_drift:
        quaternions[2] = [1, 0, 0, 0]
    path = tmp_path / "motion.parquet"
    pq.write_table(pa.Table.from_pylist([{
        "object_body_position": positions[:, None].tolist(),
        "object_body_wxyz": quaternions[:, None].tolist(),
        "robot_joint_positions": np.zeros((3, 43)).tolist(),
    }]), path)
    return path


def test_accepts_one_shared_rigid_frame_transform(tmp_path, verifier):
    report = verifier.verify_motion(fixture_motion(tmp_path), tmp_path)
    assert report["frames"] == 3
    assert report["object_translation_se3_max_error_m"] < 1e-10


@pytest.mark.parametrize("drift", ["position_drift", "rotation_drift"])
def test_rejects_framewise_object_motion_changes(tmp_path, verifier, drift):
    path = fixture_motion(tmp_path, **{drift: True})
    with pytest.raises(ValueError, match="rigid transform"):
        verifier.verify_motion(path, tmp_path)
