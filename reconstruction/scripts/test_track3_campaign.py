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


@pytest.mark.parametrize('memory,should_stop', [(18000, False), (19001, True)])
@pytest.mark.parametrize('module,target', [(tracking, 'v2d-track3-pilot-tracking'),
                                          (hands, 'v2d-track3-hand-stage'),
                                          (depth_batch, 'v2d-track3-depth-stage')])
def test_memory_guard_stops_only_its_container(tmp_path, monkeypatch, memory, should_stop, module, target):
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
    old_cmd = ['docker', 'run', '-e', 'V2D_TORCH_MEMORY_FRACTION=0.55', 'image', '-m', 'producer']
    new_cmd = [x.replace('=0.55', '=0.65') for x in old_cmd]
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
    source.write_text('different weights')
    assert not reuse()
    source.write_text('weights')
    frame.write_bytes(b'corrupted')
    assert not reuse()
