from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("omegaconf")
pytest.importorskip("kornia")
pytest.importorskip("pytorch3d")
pytest.importorskip("nvdiffrast")

from v2d.foundation_pose.lib.trt_predictors import (
    REFINER_MAX_BATCH,
    SCORER_MAX_BATCH,
    _model_inputs,
    _crop_inputs_in_chunks,
    _run_refiner_chunks,
)


@pytest.mark.parametrize('batch', [1, 43, 252])
def test_crop_workspace_is_bounded_and_keeps_every_hypothesis(batch):
    seen = []
    poses = torch.arange(batch, dtype=torch.float32)

    def crop(render_size, chunk):
        seen.append(len(chunk))
        values = chunk[:, None, None, None].expand(-1, 3, 2, 2)
        return SimpleNamespace(rgbAs=values, xyz_mapAs=values + 1000,
                               rgbBs=values + 2000, xyz_mapBs=values + 3000)

    first, second = _crop_inputs_in_chunks(crop, poses)
    assert max(seen) <= REFINER_MAX_BATCH
    assert sum(seen) == batch
    assert torch.equal(first[:, 0, 0, 0], poses)
    assert torch.equal(first[:, 3, 0, 0], poses + 1000)
    assert torch.equal(second[:, 0, 0, 0], poses + 2000)


@pytest.mark.parametrize("batch", [1, 20, 42, 43, 252])
def test_refiner_chunking_preserves_hypothesis_order(batch):
    hypothesis_ids = torch.arange(batch, dtype=torch.float32).reshape(batch, 1, 1, 1)
    input1 = hypothesis_ids.expand(-1, 6, 160, 160).contiguous()
    input2 = (hypothesis_ids + 1000).expand(-1, 6, 160, 160).contiguous()
    seen_batches = []

    def runner(first, second):
        seen_batches.append(len(first))
        assert torch.equal(second[:, 0, 0, 0], first[:, 0, 0, 0] + 1000)
        ids = first[:, 0, 0, 0].reshape(-1, 1)
        return {"trans": ids.expand(-1, 3), "rot": -ids.expand(-1, 3)}

    outputs = _run_refiner_chunks(runner, input1, input2)
    expected = torch.arange(batch, dtype=torch.float32)
    assert torch.equal(outputs["trans"][:, 0], expected)
    assert torch.equal(outputs["rot"][:, 0], -expected)
    assert max(seen_batches) <= REFINER_MAX_BATCH
    assert sum(seen_batches) == batch


def test_model_inputs_keep_nchw_rgb_xyz_as_fp32_nchw():
    batch = 2
    pose_data = SimpleNamespace(
        rgbAs=torch.ones(batch, 3, 160, 160, dtype=torch.float64),
        xyz_mapAs=torch.full((batch, 3, 160, 160), 2.0),
        rgbBs=torch.full((batch, 3, 160, 160), 3.0),
        xyz_mapBs=torch.full((batch, 3, 160, 160), 4.0),
    )
    input1, input2 = _model_inputs(pose_data)
    assert input1.shape == (batch, 6, 160, 160)
    assert input2.shape == (batch, 6, 160, 160)
    assert input1.dtype == input2.dtype == torch.float32
    assert input1.is_contiguous() and input2.is_contiguous()
    assert torch.equal(input1[0, :, 0, 0], torch.tensor([1, 1, 1, 2, 2, 2.0]))
    assert torch.equal(input2[0, :, 0, 0], torch.tensor([3, 3, 3, 4, 4, 4.0]))


def test_contract_batch_limits():
    assert REFINER_MAX_BATCH == 42
    assert SCORER_MAX_BATCH == 252
