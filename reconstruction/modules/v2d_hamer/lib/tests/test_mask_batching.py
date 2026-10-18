import json
from pathlib import Path

import numpy as np
from PIL import Image

from v2d.hamer.lib import masks_to_hands as module


def test_batches_preserve_frame_hand_mapping_tail_and_resume(tmp_path, monkeypatch):
    frames = tmp_path / 'frames'
    masks = tmp_path / 'masks'
    frames.mkdir()
    tracks = tmp_path / 'tracks.json'
    tracks.write_text(json.dumps({'tracks': [
        {'object_id': 100, 'is_right': False, 'role': 'hand'},
        {'object_id': 101, 'is_right': True, 'role': 'hand'}]}))
    for index in range(3):
        Image.fromarray(np.full((20, 20, 3), index * 50, dtype=np.uint8)).save(frames / f'{index:06d}.png')
    for oid in (100, 101):
        folder = masks / str(oid)
        folder.mkdir(parents=True)
        for index in range(3):
            Image.fromarray(np.full((20, 20), 255, dtype=np.uint8)).save(folder / f'{index:06d}.png')
    monkeypatch.setattr(module, 'get_model', lambda _: (None, None))
    calls = []

    def predict(model, cfg, crops):
        calls.append(len(crops))
        return [dict(betas=np.full(10, int(image[0, 0, 0])),
                     hand_pose=np.broadcast_to(np.eye(3), (15, 3, 3)),
                     global_orient=np.eye(3)[None],
                     pred_cam_t_full=np.array([cx, cy, int(right)]),
                     scaled_focal_length=100.)
                for image, cx, cy, size, right in crops]

    monkeypatch.setattr(module, 'run_hamer_batch', predict)
    baseline, batched = tmp_path / 'single', tmp_path / 'batched'
    args = (str(frames), str(masks), str(tracks))
    module.masks_to_hands(*args, str(baseline), 'weights', batch_size=1)
    calls.clear()
    module.masks_to_hands(*args, str(batched), 'weights', batch_size=2)
    assert calls == [2, 1, 2, 1]
    for path in baseline.glob('*/*.json'):
        relative = path.relative_to(baseline)
        expected = json.loads(path.read_text())
        actual = json.loads((batched / relative).read_text())
        assert actual == expected
        assert actual['mano']['betas'] == [actual['frame_idx'] * 50] * 10
        assert actual['camera']['cam_t'][2] == int(actual['is_right'])
    calls.clear()
    module.masks_to_hands(*args, str(batched), 'weights', batch_size=2)
    assert not calls


def test_empty_batch_needs_no_model():
    from v2d.hamer.lib._hamer import run_hamer_batch
    assert run_hamer_batch(None, None, []) == []


def test_rotation_export_is_stable_near_pi_with_float32_roundoff():
    from scipy.spatial.transform import Rotation
    from v2d.hamer.lib._hamer import rotmat_to_axis_angle
    axis = np.array([1., 2., -3.]) / np.sqrt(14.)
    vectors = np.array([axis * (np.pi - gap) for gap in (0., 1e-5, .001, .01)])
    matrices = Rotation.from_rotvec(vectors).as_matrix().astype(np.float32)
    matrices[:, 0, 0] += 1e-7
    result = rotmat_to_axis_angle(matrices)
    error = (Rotation.from_rotvec(result) * Rotation.from_rotvec(vectors).inv()).magnitude()
    assert np.max(error) < 1e-6
    assert np.all(np.linalg.norm(result, axis=-1) <= np.pi + 1e-12)
