#!/usr/bin/env python3
"""Run a native FlashCHORD command under the shared Track 3 GPU lock/guard."""
import argparse
import fcntl
from pathlib import Path
import subprocess
import threading

from prepare_track3_campaign import ROOT
from run_track3_object_pilot import save, telemetry

NAME = 'v2d-track3-policy-stage'


def guard(stop, report):
    while not stop.wait(.5):
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                                    check=True, capture_output=True, text=True, timeout=5)
            memory = float(result.stdout.strip().splitlines()[0])
            if memory < 21000: continue
            reason = f'Total GPU memory {memory} MiB >= 21000'
        except (subprocess.SubprocessError, ValueError, IndexError) as exc:
            reason = f'GPU monitor unavailable: {exc}'
        save(report / 'memory_guard.json', {'state': 'STOPPED_BY_GPU_GUARD', 'reason': reason})
        subprocess.run(['docker', 'kill', NAME], capture_output=True, timeout=10)
        return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('module')
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    args.report.mkdir(parents=True, exist_ok=True)
    monorepo = ROOT.parent
    cache = monorepo / 'robotic_grounding/data/track3/runtime_cache'; cache.mkdir(parents=True, exist_ok=True)
    lock = (ROOT / 'data/track3/pilot/episode_000038/pilot.lock').open('a')
    print('Waiting for shared Track 3 GPU lock', flush=True)
    fcntl.flock(lock, fcntl.LOCK_EX)
    command = ['docker', 'run', '--rm', '--name', NAME, '--gpus', 'all',
               '--memory', '20g', '--memory-swap', '22g', '--cpus', '8', '--user', '1000:1000',
               '-e', 'HOME=/tmp', '-e', 'LOGNAME=alien51', '-e', 'OMP_NUM_THREADS=8',
               '-e', 'PYTHONUNBUFFERED=1', '-e', 'XLA_PYTHON_CLIENT_PREALLOCATE=false',
               '-e', 'XDG_CACHE_HOME=/cache', '-e', 'MUJOCO_GL=egl', '-e', 'PYOPENGL_PLATFORM=egl',
               '-e', 'PYTHONPATH=/workspace:/workspace/src',
               '-v', f'{monorepo}:/repo', '-v', f'{monorepo}/robotic_grounding/flash_chord:/workspace',
               '-v', f'{cache}:/cache', '-w', '/workspace',
               '--entrypoint', 'python', 'flash_chord:track2-baseline',
               '-m', args.module, *args.arguments]
    save(args.report / 'command.json', {'state': 'RUNNING', 'command': command, 'submission_ready': False})
    stop = threading.Event()
    threads = [threading.Thread(target=guard, args=(stop, args.report), daemon=True),
               threading.Thread(target=telemetry, args=(stop, args.report / 'gpu.csv'), daemon=True)]
    for thread in threads: thread.start()
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
        physics_error = None
        for line in process.stdout:
            print(line, end='', flush=True)
            if 'nefc overflow' in line or 'ncon overflow' in line:
                physics_error = line.strip()
                save(args.report / 'physics_buffer_failure.json', {'state': 'FAILED', 'reason': physics_error})
                subprocess.run(['docker', 'kill', NAME], capture_output=True, timeout=10)
        returncode = process.wait()
        if physics_error: returncode = returncode or 3
        save(args.report / 'exit.json', {'state': 'COMPLETE' if returncode == 0 else 'FAILED', 'exit_code': returncode})
        if returncode: raise SystemExit(returncode)
    finally:
        stop.set()
        for thread in threads: thread.join(timeout=10)


if __name__ == '__main__': main()
