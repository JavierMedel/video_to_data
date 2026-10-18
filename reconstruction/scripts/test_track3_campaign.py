"""CPU checks for the Track 3 pilot's expensive-stage resume contract."""
import json
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

import run_track3_object_pilot as pilot
import run_track3_pose_pilot as tracking
import run_track3_hand_pilot as hands
import run_track3_depth_batch as depth_batch
import run_track3_camera_batch as camera_batch


def test_resume_rejects_modified_inputs_and_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(pilot, 'ROOT', tmp_path)
    source = tmp_path / 'source'; source.write_text('source v1')
    output = tmp_path / 'output'
    reports = tmp_path / 'reports'; reports.mkdir()
    calls = []
    monkeypatch.setattr(pilot.subprocess, 'check_output', lambda *a, **k: 'sha256:image-v1')

    def run(*args, **kwargs):
        calls.append(args)
        output.write_text('generated')
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(pilot.subprocess, 'run', run)
    command = ['docker', 'run', 'image:tag', '-m', 'producer']
    invoke = lambda: pilot.run_stage('stage', command, [output], [source], reports)
    invoke(); invoke()
    assert len(calls) == 1
    output.write_text('corrupted')
    invoke()
    assert len(calls) == 2
    source.write_text('source v2')
    invoke()
    assert len(calls) == 3


def test_failed_stage_is_not_reused(tmp_path, monkeypatch):
    monkeypatch.setattr(pilot, 'ROOT', tmp_path)
    source = tmp_path / 'source'; source.write_text('source')
    reports = tmp_path / 'reports'; reports.mkdir()
    monkeypatch.setattr(pilot.subprocess, 'check_output', lambda *a, **k: 'sha256:image')
    monkeypatch.setattr(pilot.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=7))
    with pytest.raises(RuntimeError, match='stage failed'):
        pilot.run_stage('stage', ['docker', 'run', 'image', '-m', 'producer'],
                        [tmp_path / 'missing'], [source], reports)
    assert json.loads((reports / 'stage.json').read_text())['state'] == 'FAILED'


@pytest.mark.parametrize('memory', [18000, 19001, 20999, 21000, 21001])
@pytest.mark.parametrize('module,target', [(tracking, 'v2d-track3-pilot-tracking'),
                                          (hands, 'v2d-track3-hand-stage'),
                                          (depth_batch, 'v2d-track3-depth-stage'),
                                          (camera_batch, 'v2d-track3-camera-stage')])
def test_memory_guard_stops_only_its_container(tmp_path, monkeypatch, memory, module, target):
    threshold = (depth_batch.GPU_MEMORY_GUARD_MIB if module is depth_batch
                 else camera_batch.GPU_LIMIT_MIB if module is camera_batch else 19000)
    should_stop = memory >= threshold
    calls = []

    class Stop:
        samples = 0
        def wait(self, seconds):
            self.samples += 1
            return self.samples > 1

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout=str(memory))

    monkeypatch.setattr(module.subprocess, 'run', run)
    report = tmp_path / 'guard.json'
    watchdog = tracking.memory_watchdog if module is tracking else module.guard
    watchdog(Stop(), report)
    kills = [c for c in calls if c[:2] == ['docker', 'kill']]
    assert kills == ([['docker', 'kill', target]] if should_stop else [])
    assert report.exists() == should_stop


def test_gap_fill_retains_empty_track_without_inventing_hand_motion(tmp_path):
    source = Path(__file__).resolve().parents[1] / 'modules/v2d_wilor/lib/tracks_interpolate.py'
    spec = importlib.util.spec_from_file_location('track3_interpolation', source)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    aligned, masks, output = tmp_path / 'aligned', tmp_path / 'masks', tmp_path / 'output'
    (aligned / '2').mkdir(parents=True); (aligned / '3').mkdir()
    masks.mkdir()
    record = {'track_id': 3, 'frame_idx': 0, 'is_right': True,
              'mano': {'betas': [0] * 10, 'global_orient': [0] * 3, 'hand_pose': [0] * 45},
              'camera': {'cam_t': [.1, .1, 1], 'intrinsics': {'fx': 100, 'fy': 100, 'cx': 50, 'cy': 50}},
              'image_size': [100, 100], 'diagnostics': {}}
    (aligned / '3/000000.json').write_text(json.dumps(record))
    module.tracks_interpolate(str(aligned), str(masks), str(output), max_gap_frames=15, extrapolate=False)
    assert not list((output / '2').glob('*.json'))
    assert json.loads((output / '3/000000.json').read_text())['interpolated'] is False
    assert (output / 'handedness.json').exists()


def test_budget_resume_preserves_evidence_and_rejects_changed_data(tmp_path, monkeypatch):
    monkeypatch.setattr(depth_batch, 'ROOT', tmp_path)
    scripts = tmp_path / 'scripts'; scripts.mkdir()
    runner = scripts / 'run_track3_object_pilot.py'; runner.write_text('runner')
    source = tmp_path / 'model'; source.write_text('weights')
    output = tmp_path / 'depth'; output.mkdir()
    frame = output / '000000.png'; frame.write_bytes(b'depth')
    old_cmd = ['docker', 'run', '-e', 'V2D_TORCH_MEMORY_FRACTION=0.55', 'image', '-m', 'producer', '--batch_size', '4']
    new_cmd = [x.replace('=0.55', '=0.70') for x in old_cmd]
    new_cmd[-1] = '8'
    record = tmp_path / 'stage.json'
    record.write_text(json.dumps({'state': 'COMPLETE', 'command': old_cmd,
        'image_id': 'image-id', 'runner_sha256': depth_batch.sha256(runner),
        'input_sha256': {'model': depth_batch.sha256(source)},
        'output_sha256': {'depth/000000.png': depth_batch.sha256(frame)}}))
    original = record.read_bytes()
    monkeypatch.setattr(depth_batch.subprocess, 'check_output', lambda *a, **k: 'image-id')
    reuse = lambda: depth_batch.reuse_with_budget_change(record, new_cmd, [output], [source])
    assert reuse()
    assert record.read_bytes() == original
    assert not depth_batch.reuse_with_budget_change(record, new_cmd + ['--changed-inference'], [output], [source])
    source.write_text('different weights')
    assert not reuse()
    source.write_text('weights')
    frame.write_bytes(b'corrupted')
    assert not reuse()


def test_camera_pose_validation_rejects_missing_frames_and_bad_quaternions(tmp_path):
    pose = {'rotation': [1, 0, 0, 0], 'translation': [0, 0, 0], 'scale': [1, 1, 1]}
    (tmp_path / '000000.json').write_text(json.dumps(pose))
    with pytest.raises(ValueError, match='every source frame'):
        camera_batch.validate_poses(tmp_path, 2)
    pose['translation'] = [0.01, 0, 0]
    (tmp_path / '000001.json').write_text(json.dumps(pose))
    assert camera_batch.validate_poses(tmp_path, 2)['largest_translation_step_m'] == .01
    pose['rotation'] = [0, 0, 0, 0]
    (tmp_path / '000001.json').write_text(json.dumps(pose))
    with pytest.raises(ValueError, match='rotation/scale'):
        camera_batch.validate_poses(tmp_path, 2)


def test_multi_object_merge_preserves_shared_world_and_contact_body_ids():
    import copy
    import run_track3_full_campaign as full
    row = {'fps': 20., 'object_body_position': [[[0., 0., 1.]], [[0., 0., 1.]]],
           'object_body_wxyz': [[[1., 0., 0., 0.]], [[1., 0., 0., 0.]]],
           'object_body_names': ['first'], 'object_mesh_paths': ['first.obj'],
           'object_urdf_paths': ['first.urdf'], 'object_mesh_radius': [.1]}
    for side in ('left', 'right'):
        row[f'mano_{side}_joints'] = [[[0., 0., 1.]], [[0., 0., 1.]]]
        row[f'mano_{side}_object_contact_positions'] = [[[0., 0., 1.]], [[0., 0., 1.]]]
        row[f'mano_{side}_object_contact_normals'] = [[[0., 0., 1.]], [[0., 0., 1.]]]
        row[f'mano_{side}_link_contact_positions'] = [[[0., 0., .9]], [[0., 0., .9]]]
        row[f'mano_{side}_link_contact_normals'] = [[[0., 0., -1.]], [[0., 0., -1.]]]
        row[f'mano_{side}_object_contact_part_ids'] = [[1], [0]]
        row[f'mano_{side}_tips_distance'] = [[.1], [.2]]
    second = copy.deepcopy(row)
    second.update(object_body_names=['second'], object_mesh_paths=['second.obj'], object_urdf_paths=['second.urdf'])
    second['object_body_position'] = [[[.25, 0., 1.]], [[.25, 0., 1.]]]
    for side in ('left', 'right'):
        second[f'mano_{side}_object_contact_positions'] = [[[0., 0., .91]], [[0., 0., .91]]]
        second[f'mano_{side}_object_contact_part_ids'] = [[1], [0]]
    out = full.merge_rows([row, second], {'episode_index': 10, 'frames': 2})
    assert out['object_body_position'][0] == [[0., 0., 1.], [.25, 0., 1.]]
    assert out['mano_left_object_contact_part_ids'] == [[2], [0]]
    assert out['mano_left_object_contact_positions'][0] == [[0., 0., .91]]
    assert out['mano_left_link_contact_positions'][0] == [[0., 0., .9]]
    assert out['mano_left_object_contact_positions'][1] == [[0., 0., 0.]]
    second['mano_left_joints'][0][0][0] = .1
    with pytest.raises(ValueError, match='one hand/world'):
        full.merge_rows([row, second], {'episode_index': 10, 'frames': 2})


def test_reconstruction_detection_does_not_assign_same_box_to_two_objects():
    from run_track3_reconstruction_campaign import choose_detection
    primary = {'confidence': .9, 'box': {'x0': 0, 'y0': 0, 'x1': 100, 'y1': 100}}
    other = {'confidence': .7, 'box': {'x0': 110, 'y0': 0, 'x1': 200, 'y1': 100}}
    assert choose_detection([primary, other], [primary]) == other
    with pytest.raises(ValueError, match='distinct confident'):
        choose_detection([primary], [primary])
