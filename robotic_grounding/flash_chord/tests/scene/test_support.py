# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for support-surface spawning."""

import pytest

from flash_chord.assets import ASSETS_DIR
from flash_chord.assets.registry import support_usda_path

pytestmark = pytest.mark.gpu

_HOT3D = (
    ASSETS_DIR
    / "human_motion_data"
    / "hot3d"
    / "hot3d_processed"
    / "sequence_id=P0002_59a84a3a_seg025"
    / "robot_name=sharpa_wave"
)


@pytest.mark.sequence_data
def test_support_surfaces_added():
    import warp as wp

    import newton
    from flash_chord.scene.support import add_support_surfaces

    support = support_usda_path("hot3d", "P0002_59a84a3a_seg025")
    assert support.exists()
    with wp.ScopedDevice("cuda:0"):
        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        binding = add_support_surfaces(builder, support)
        builder.finalize()
    assert binding.body_id == 0
    assert binding.free_q_ids == tuple(range(7))
    assert binding.free_dof_ids == tuple(range(6))
    assert tuple(binding.shapes.ids()) == (0, 1)  # hot3d support = two cylinder pads


def test_explicit_missing_support_is_rejected(tmp_path):
    import newton
    from flash_chord.scene.support import add_support_surfaces

    with pytest.raises(FileNotFoundError, match="support surface does not exist"):
        add_support_surfaces(newton.ModelBuilder(), tmp_path / "missing.usda")


def test_empty_support_is_rejected_before_mutating_builder(tmp_path):
    import newton
    from flash_chord.scene.support import add_support_surfaces

    support = tmp_path / "empty.usda"
    support.write_text('#usda 1.0\n\ndef Xform "World" {}\n')
    builder = newton.ModelBuilder()

    with pytest.raises(ValueError, match="contains no Cylinder or Cube"):
        add_support_surfaces(builder, support)

    assert len(builder.body_label) == 0
    assert len(builder.shape_body) == 0


def test_cube_support_applies_authored_nonuniform_scale(tmp_path):
    import newton
    from flash_chord.scene.support import add_support_surfaces

    support = tmp_path / "scaled_cube.usda"
    support.write_text(
        """#usda 1.0

def Cube "tabletop"
{
    double size = 1
    double3 xformOp:scale = (0.84, 0.282, 0.01)
    uniform token[] xformOpOrder = ["xformOp:scale"]
}
"""
    )
    builder = newton.ModelBuilder()

    binding = add_support_surfaces(builder, support)

    assert binding is not None
    assert tuple(float(value) for value in builder.shape_scale[0]) == pytest.approx((0.42, 0.141, 0.005))


@pytest.mark.sequence_data
def test_build_scene_includes_support():
    import warp as wp

    from flash_chord.data.mano_sharpa import load_mano_sharpa
    from flash_chord.embodiments.sharpa_hands import SharpaHands
    from flash_chord.scene.builder import build_scene

    ref = load_mano_sharpa(str(_HOT3D))
    support = support_usda_path("hot3d", "P0002_59a84a3a_seg025")
    with wp.ScopedDevice("cuda:0"):
        scene = build_scene(SharpaHands(), ref, world_count=1, support_usda=support)
    assert scene.support_shape_count == 2
    assert scene.support is not None
    assert scene.support.shapes == scene.collision_layout.support


@pytest.mark.parametrize('allow_empty', [False, True])
def test_declared_empty_support_requires_explicit_ground_context(tmp_path, allow_empty):
    import newton
    from flash_chord.scene.support import add_support_surfaces
    path = tmp_path / 'released_empty.usda'
    path.write_text('#usda 1.0\n(defaultPrim = "support_surfaces")\ndef Xform "support_surfaces" {}\n')
    builder = newton.ModelBuilder()
    if allow_empty:
        assert add_support_surfaces(builder, path, allow_empty=True) is None
    else:
        with pytest.raises(ValueError, match='contains no Cylinder or Cube'):
            add_support_surfaces(builder, path)
    assert builder.body_count == 0 and builder.shape_count == 0


@pytest.mark.parametrize('content', [
    '#usda 1.0\n(defaultPrim = "World")\ndef Xform "World" {}\n',
    '#usda 1.0\n(defaultPrim = "support_surfaces")\ndef Xform "support_surfaces" {\n def Mesh "unhandled" {}\n}\n',
])
def test_allow_empty_never_drops_unknown_support_geometry(tmp_path, content):
    import newton
    from flash_chord.scene.support import add_support_surfaces
    path = tmp_path / 'invalid.usda'; path.write_text(content)
    with pytest.raises(ValueError, match='contains no Cylinder or Cube'):
        add_support_surfaces(newton.ModelBuilder(), path, allow_empty=True)
