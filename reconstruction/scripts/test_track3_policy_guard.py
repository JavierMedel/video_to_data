import json
import subprocess
import sys
from types import SimpleNamespace

import pytest
import run_track3_policy_stage as policy


def exercise(monkeypatch, tmp_path, readings, cap_mib=None):
    queries = iter(readings)
    calls = []

    class Stop:
        remaining = len(readings)
        def wait(self, seconds):
            self.remaining -= 1
            return self.remaining < 0

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0)

    def sample():
        value = next(queries)
        if isinstance(value, Exception):
            raise value
        value = float(value)
        if not policy.math.isfinite(value) or value < 0:
            raise ValueError('Invalid reading')
        return value, f'timestamp, 99, {value}, 24463, 95'

    monkeypatch.setattr(policy.subprocess, 'run', run)
    monkeypatch.setattr(policy, 'read_gpu_sample', sample)
    policy.guard(Stop(), tmp_path, cap_mib=cap_mib)
    return [command for command in calls if command[:2] == ['docker', 'kill']]


def timeout():
    return subprocess.TimeoutExpired('nvidia-smi', 5)


def test_monitor_recovers_from_two_timeouts_without_killing(monkeypatch, tmp_path):
    assert exercise(monkeypatch, tmp_path, [10296, timeout(), timeout(), 10296]) == []
    assert not (tmp_path / 'memory_guard.json').exists()
    events = [json.loads(line) for line in (tmp_path / 'monitor_events.jsonl').read_text().splitlines()]
    assert events[-1]['event'] == 'MONITOR_RECOVERED'
    assert events[-1]['prior_failures'] == 2


def test_sustained_monitor_loss_stops_only_owned_container(monkeypatch, tmp_path):
    assert exercise(monkeypatch, tmp_path, [timeout(), timeout(), timeout()]) == [['docker', 'kill', policy.NAME]]
    assert '3 consecutive failures' in json.loads((tmp_path / 'memory_guard.json').read_text())['reason']


@pytest.mark.parametrize('memory', [21000, 22000])
def test_confirmed_memory_limit_stops_immediately(monkeypatch, tmp_path, memory):
    assert exercise(monkeypatch, tmp_path, [memory], cap_mib=21000) == [['docker', 'kill', policy.NAME]]


@pytest.mark.parametrize('memory', [21000, 22000, 24400])
def test_default_has_no_vram_cutoff(monkeypatch, tmp_path, memory):
    assert exercise(monkeypatch, tmp_path, [memory]) == []
    assert not (tmp_path / 'memory_guard.json').exists()
    assert str(float(memory)) in (tmp_path / 'gpu.csv').read_text()


def test_valid_reading_resets_failure_count(monkeypatch, tmp_path):
    assert exercise(monkeypatch, tmp_path, [timeout(), timeout(), 10000, timeout(), timeout(), 10000]) == []


def test_invalid_readings_never_count_as_safe_memory(monkeypatch, tmp_path):
    assert exercise(monkeypatch, tmp_path, ['nan', '', -1]) == [['docker', 'kill', policy.NAME]]


def test_query_timeout_never_waits_for_an_uninterruptible_child(monkeypatch):
    from io import StringIO
    class Process:
        stdout = StringIO()
        stderr = StringIO()
        killed = False
        def communicate(self, timeout):
            raise subprocess.TimeoutExpired('nvidia-smi', timeout)
        def kill(self):
            self.killed = True
        def wait(self, *args, **kwargs):
            pytest.fail('Must not wait for a child blocked in the WSL driver')
    process = Process()
    monkeypatch.setattr(policy.subprocess, 'Popen', lambda *args, **kwargs: process)
    with pytest.raises(subprocess.TimeoutExpired):
        policy.read_gpu_sample()
    assert process.killed and process.stdout.closed and process.stderr.closed


@pytest.mark.parametrize('age,run_id,memory,valid', [(0, 'current', 10296, True),
    (7, 'current', 10296, False), (0, 'old-run', 10296, False), (0, 'current', float('nan'), False)])
def test_native_heartbeat_rejects_stale_wrong_run_and_bad_memory(tmp_path, monkeypatch, age, run_id, memory, valid):
    monkeypatch.setattr(policy.time, 'time', lambda: 1000.)
    (tmp_path / 'gpu_heartbeat.json').write_text(json.dumps(dict(run_id=run_id, unix_time=1000.-age,
        memory_used_mib=memory, memory_total_mib=24463.)))
    assert (policy.read_gpu_heartbeat(tmp_path, 'current') is not None) == valid


@pytest.mark.parametrize('memory,cap,kills', [(10296, None, False), (21000, None, False), (24400, None, False),
                                          (10296, 21000, False), (21000, 21000, True)])
def test_fresh_native_memory_is_guarded_without_external_query(tmp_path, monkeypatch, memory, cap, kills):
    monkeypatch.setattr(policy.time, 'time', lambda: 1000.)
    (tmp_path / 'gpu_heartbeat.json').write_text(json.dumps(dict(run_id='current', unix_time=1000.,
        memory_used_mib=memory, memory_total_mib=24463.)))
    class Stop:
        calls = 0
        def wait(self, seconds):
            self.calls += 1
            return self.calls > 1
    def no_external_query():
        pytest.fail('Fresh native memory must bypass the blocked external query')
    monkeypatch.setattr(policy, 'read_gpu_sample', no_external_query)
    commands = []
    monkeypatch.setattr(policy.subprocess, 'run', lambda command, **kw: commands.append(command))
    policy.guard(Stop(), tmp_path, 'current', cap_mib=cap)
    assert commands == ([['docker', 'kill', policy.NAME]] if kills else [])


def test_new_training_doubles_worlds_and_pins_resume(tmp_path):
    args = ['scene.world_count=512', 'training.total_environment_steps=5000192']
    tuned = policy.tune_training_arguments('scripts.train_flash_sac', args, tmp_path)
    assert 'scene.world_count=1024' in tuned
    resumed = policy.tune_training_arguments('scripts.train_flash_sac', args + ['training.resume=true'], tmp_path)
    assert 'scene.world_count=1024' in resumed
    assert json.loads((tmp_path / 'gpu_training_profile.json').read_text())['memory_cap_mib'] is None


def test_explicit_memory_cap_is_recorded_without_changing_training_worlds(tmp_path):
    args = ['scene.world_count=1024']
    assert policy.tune_training_arguments('scripts.train_flash_sac', args, tmp_path, cap_mib=23000) == args
    assert json.loads((tmp_path / 'gpu_training_profile.json').read_text())['memory_cap_mib'] == 23000


def test_existing_checkpoint_keeps_worlds_and_evaluation_is_unchanged(tmp_path):
    args = ['scene.world_count=512', 'training.resume=true']
    assert policy.tune_training_arguments('scripts.train_flash_sac', args, tmp_path) == args
    assert policy.tune_training_arguments('scripts.evaluate_policy', ['evaluation.world_count=1'], tmp_path) == ['evaluation.world_count=1']


@pytest.mark.parametrize('cap', [None, 23000])
def test_policy_container_can_allocate_full_vram_without_preallocation(tmp_path, monkeypatch, cap):
    root = tmp_path / 'reconstruction'
    (root / 'data/track3/pilot/episode_000038').mkdir(parents=True)
    report = tmp_path / 'robotic_grounding/reports/evaluation'
    monkeypatch.setattr(policy, 'ROOT', root)
    monkeypatch.setattr(policy, 'NAME', policy.NAME)
    monkeypatch.setattr(policy, 'guard', lambda *args: None)
    extra = [] if cap is None else ['--gpu-memory-limit-mib', str(cap)]
    monkeypatch.setattr(sys, 'argv', ['policy', '--report', str(report), '--track', 'track_2_tier2',
                                    *extra, 'scripts.evaluate_policy', 'evaluation.world_count=1'])
    commands = []
    def process(command, **kw):
        commands.append(command)
        return SimpleNamespace(stdout=iter([]), wait=lambda: 0)
    monkeypatch.setattr(policy.subprocess, 'Popen', process)
    policy.main()
    assert len(commands) == 1
    assert 'XLA_PYTHON_CLIENT_MEM_FRACTION=1.0' in commands[0]
    assert 'XLA_PYTHON_CLIENT_PREALLOCATE=false' in commands[0]
    assert commands[0][commands[0].index('--gpus') + 1] == 'all'
    receipt = json.loads((report / 'command.json').read_text())
    assert receipt['gpu_memory_limit_mib'] == cap
