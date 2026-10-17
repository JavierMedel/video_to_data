"""CPU checks for the Track 3 pilot's expensive-stage resume contract."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import run_track3_object_pilot as pilot
import run_track3_pose_pilot as tracking


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
def test_memory_guard_stops_only_the_tracking_container(tmp_path, monkeypatch, memory, should_stop):
    calls = []

    class Stop:
        samples = 0
        def wait(self, seconds):
            self.samples += 1
            return self.samples > 1

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout=str(memory))

    monkeypatch.setattr(tracking.subprocess, 'run', run)
    report = tmp_path / 'guard.json'
    tracking.memory_watchdog(Stop(), report)
    kills = [c for c in calls if c[:2] == ['docker', 'kill']]
    assert kills == ([['docker', 'kill', 'v2d-track3-pilot-tracking']] if should_stop else [])
    assert report.exists() == should_stop
