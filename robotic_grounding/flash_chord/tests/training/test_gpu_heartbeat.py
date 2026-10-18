import json

from flash_chord.training import gpu_heartbeat as heartbeat


def test_heartbeat_is_atomic_rate_limited_and_reports_global_memory(tmp_path, monkeypatch):
    path = tmp_path / 'memory.json'
    monkeypatch.setenv('V2D_GPU_HEARTBEAT_PATH', str(path))
    monkeypatch.setenv('V2D_GPU_HEARTBEAT_ID', 'run-1')
    monkeypatch.setattr(heartbeat, '_last_write', float('-inf'))
    clock = [1000.]
    monkeypatch.setattr(heartbeat.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(heartbeat.time, 'time', lambda: clock[0])
    monkeypatch.setattr(heartbeat, '_query_memory', lambda: (10296., 24463.))
    heartbeat.maybe_write_gpu_heartbeat(512)
    data = json.loads(path.read_text())
    assert data['run_id'] == 'run-1' and data['memory_used_mib'] == 10296.
    assert data['memory_total_mib'] == 24463. and data['environment_steps'] == 512
    clock[0] += 1
    heartbeat.maybe_write_gpu_heartbeat(1024)
    assert json.loads(path.read_text())['environment_steps'] == 512
    clock[0] += 2
    heartbeat.maybe_write_gpu_heartbeat(1536)
    assert json.loads(path.read_text())['environment_steps'] == 1536
    assert not path.with_suffix('.json.tmp').exists()


def test_bad_native_memory_sample_never_produces_safe_memory(tmp_path, monkeypatch):
    path = tmp_path / 'memory.json'
    monkeypatch.setenv('V2D_GPU_HEARTBEAT_PATH', str(path))
    monkeypatch.setattr(heartbeat, '_last_write', float('-inf'))
    monkeypatch.setattr(heartbeat, '_query_memory', lambda: (float('nan'), 24463.))
    heartbeat.maybe_write_gpu_heartbeat(1)
    data = json.loads(path.read_text())
    assert 'error' in data and 'memory_used_mib' not in data


def test_disabled_heartbeat_never_queries_cuda(monkeypatch):
    monkeypatch.delenv('V2D_GPU_HEARTBEAT_PATH', raising=False)
    def fail():
        raise AssertionError('Disabled progress telemetry must not initialize CUDA')
    monkeypatch.setattr(heartbeat, '_query_memory', fail)
    heartbeat.maybe_write_gpu_heartbeat(1)
