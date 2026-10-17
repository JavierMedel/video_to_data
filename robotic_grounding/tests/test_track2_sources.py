"""Frame-conversion and missing-observation checks for challenge source preparation."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "track2_sources", Path(__file__).parents[1] / "scripts/prepare_track2_sources.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_wxyz_rotation_and_world_translation():
    poses = np.array([[1., 2., 3., np.sqrt(.5), 0., 0., np.sqrt(.5)]])
    matrix = module.poses_to_matrices(poses)[0]
    np.testing.assert_allclose(matrix @ [1, 0, 0, 1], [1, 3, 3, 1], atol=1e-7)
    np.testing.assert_allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-7)


def test_missing_pose_interpolation_preserves_visible_observations():
    poses = np.array([[0, 0, 0, 1, 0, 0, 0], [0]*7, [2, 0, 0, 0, 0, 0, 1.]])
    result = module.continuous_object_poses(poses, [True, False, True])
    np.testing.assert_array_equal(result[[0, 2]], poses[[0, 2]])
    np.testing.assert_allclose(result[1], [1, 0, 0, np.sqrt(.5), 0, 0, np.sqrt(.5)])
    np.testing.assert_array_equal(poses[1], np.zeros(7))


def test_antipodal_and_endpoint_missing_poses():
    poses = np.array([[0]*7, [0, 0, 0, 1, 0, 0, 0], [0]*7,
                      [2, 0, 0, -1, 0, 0, 0], [0]*7], dtype=float)
    result = module.continuous_object_poses(poses, [False, True, False, True, False])
    np.testing.assert_array_equal(result[0], poses[1])
    np.testing.assert_array_equal(result[-1], poses[3])
    np.testing.assert_allclose(result[2], [1, 0, 0, 1, 0, 0, 0])


def test_invalid_visible_quaternion_is_rejected():
    with pytest.raises(ValueError, match="unit length"):
        module.continuous_object_poses(np.zeros((1, 7)), [True])
