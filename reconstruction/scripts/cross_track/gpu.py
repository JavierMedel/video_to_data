"""GPU scope for Torch diagnostics, sharing the native policy queue's lock."""
import fcntl
import os
from pathlib import Path
import signal
import threading
import time
from .common import REPO, save


def start_guard(report, cap_mib=21000):
    """CLI-only scope: hold the GPU lock for this process and monitor global VRAM."""
    import torch
    lock_path = REPO/'reconstruction/data/track3/pilot/episode_000038/pilot.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = lock_path.open('a')
    fcntl.flock(lock, fcntl.LOCK_EX)
    report = Path(report); report.mkdir(parents=True, exist_ok=True)
    _, total = torch.cuda.mem_get_info()
    torch.cuda.set_per_process_memory_fraction(min(1., cap_mib*1024**2/total))
    def monitor():
        failures = 0
        while True:
            try:
                free, total = torch.cuda.mem_get_info()
                used = (total-free)/1024**2
                save(report/'gpu_heartbeat.json', {'unix_time': time.time(), 'memory_used_mib': used,
                                                 'memory_total_mib': total/1024**2, 'cap_mib': cap_mib})
                failures = 0
                if used > cap_mib:
                    save(report/'memory_guard.json', {'reason': 'Global device VRAM exceeded cap', 'used_mib': used})
                    os.kill(os.getpid(), signal.SIGTERM); return
            except Exception as exc:
                failures += 1
                if failures >= 3:
                    save(report/'memory_guard.json', {'reason': 'GPU monitor unavailable', 'error': str(exc)})
                    os.kill(os.getpid(), signal.SIGTERM); return
            time.sleep(2)
    threading.Thread(target=monitor, daemon=True).start()
    return lock
