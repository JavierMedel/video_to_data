# SPDX-License-Identifier: Apache-2.0
"""Optional device-wide CUDA memory heartbeat for an external safety monitor."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import time
import threading

_last_write = float('-inf')
_steps = 0
_lock = threading.RLock()
_worker = None


def start_gpu_heartbeat() -> None:
    """Keep sampling real CUDA memory during checkpoint I/O and finalization."""
    global _worker
    if not os.environ.get('V2D_GPU_HEARTBEAT_PATH') or _worker is not None:
        return
    def sample_forever():
        while True:
            time.sleep(2)
            maybe_write_gpu_heartbeat(_steps)
    _worker = threading.Thread(target=sample_forever, name='cuda-memory-heartbeat', daemon=True)
    _worker.start()


def _query_memory():
    import warp as wp
    device = wp.get_device('cuda:0')
    total, free = device.total_memory, device.free_memory
    return (total - free) / 2**20, total / 2**20


def maybe_write_gpu_heartbeat(environment_steps: int) -> None:
    global _steps
    if not os.environ.get('V2D_GPU_HEARTBEAT_PATH'):
        return
    with _lock:
        _steps = max(_steps, environment_steps)
        _write_gpu_heartbeat(_steps)


def _write_gpu_heartbeat(environment_steps: int) -> None:
    global _last_write
    target = os.environ.get('V2D_GPU_HEARTBEAT_PATH')
    if not target or time.monotonic() - _last_write < 2:
        return
    _last_write = time.monotonic()
    data = {'run_id': os.environ.get('V2D_GPU_HEARTBEAT_ID'), 'unix_time': time.time(),
            'environment_steps': environment_steps, 'source': 'CUDA_device_memory_info'}
    try:
        used, total = _query_memory()
        if not math.isfinite(used) or not math.isfinite(total) or not 0 <= used <= total or total <= 0:
            raise ValueError('Invalid device-wide CUDA memory sample')
        data.update(memory_used_mib=used, memory_total_mib=total)
    except Exception as exc:
        # An error is not a safe sample; the external guard will use its
        # fallback query or stop on sustained loss of monitoring.
        data['error'] = str(exc)
    try:
        path = Path(target)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + '.tmp')
        temp.write_text(json.dumps(data) + '\n')
        temp.replace(path)
    except OSError as exc:
        print(f'GPU heartbeat write failed: {exc}', flush=True)
