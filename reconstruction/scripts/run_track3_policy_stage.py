#!/usr/bin/env python3
"""Run a native FlashCHORD command under the shared Track 3 GPU lock/guard."""
import argparse
import csv
import fcntl
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import threading
import time
import uuid

from prepare_track3_campaign import ROOT
from run_track3_object_pilot import save

NAME = 'v2d-track3-policy-stage'
MAX_MONITOR_FAILURES = 3
WINDOWS_SMI = Path('/mnt/c/Windows/System32/nvidia-smi.exe')
GPU_SMI = os.environ.get('V2D_GPU_MONITOR_BIN') or (str(WINDOWS_SMI) if WINDOWS_SMI.is_file() else 'nvidia-smi')


def read_gpu_heartbeat(report, run_id):
    if not run_id:
        return None
    try:
        data = json.loads((report / 'gpu_heartbeat.json').read_text())
        age = time.time() - float(data['unix_time'])
        memory, total = float(data['memory_used_mib']), float(data['memory_total_mib'])
        if data.get('run_id') != run_id or not -2 <= age <= 6:
            return None
        if not math.isfinite(memory) or not math.isfinite(total) or not 0 <= memory <= total or total <= 0:
            return None
        stamp = datetime.now().strftime('%Y/%m/%d %H:%M:%S.%f')
        return memory, f'{stamp}, N/A, {memory}, {total}, N/A'
    except (OSError, KeyError, ValueError, TypeError):
        return None


def read_gpu_sample():
    command = [GPU_SMI, '--query-gpu=timestamp,utilization.gpu,memory.used,memory.total,power.draw',
               '--format=csv,noheader,nounits']
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        output, error = process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        # WSL driver calls can leave nvidia-smi in uninterruptible sleep.
        # Sending SIGKILL is safe; waiting for that process can hang too.
        process.kill()
        process.stdout.close()
        process.stderr.close()
        raise
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, command, output, error)
    line = output.strip().splitlines()[0]
    memory = float(next(csv.reader([line]))[2])
    if not math.isfinite(memory) or memory < 0:
        raise ValueError(f'Invalid GPU memory reading: {memory}')
    return memory, line


def guard(stop, report, run_id=None):
    consecutive_failures = 0
    last_memory = None
    def event(kind, **fields):
        with (report / 'monitor_events.jsonl').open('a') as output:
            output.write(json.dumps({'utc': datetime.now(timezone.utc).isoformat(),
                                     'event': kind, **fields}) + '\n')
    while not stop.wait(1):
        try:
            sample = read_gpu_heartbeat(report, run_id)
            if sample is None:
                try:
                    sample = read_gpu_sample()
                except (subprocess.SubprocessError, OSError, ValueError, IndexError, csv.Error):
                    sample = read_gpu_heartbeat(report, run_id)
                    if sample is None:
                        raise
            memory, line = sample
            with (report / 'gpu.csv').open('a') as output:
                output.write(line + '\n')
            if consecutive_failures:
                event('MONITOR_RECOVERED', prior_failures=consecutive_failures, memory_mib=memory)
            consecutive_failures = 0
            last_memory = memory
            if memory < 21000: continue
            reason = f'Total GPU memory {memory} MiB >= 21000'
        except (subprocess.SubprocessError, OSError, ValueError, IndexError, csv.Error) as exc:
            consecutive_failures += 1
            event('MONITOR_FAILURE', consecutive_failures=consecutive_failures,
                  last_memory_mib=last_memory, reason=str(exc))
            print(f'GPU monitor warning ({consecutive_failures}/{MAX_MONITOR_FAILURES}): {exc}', flush=True)
            if consecutive_failures < MAX_MONITOR_FAILURES:
                continue
            reason = f'GPU monitor unavailable after {consecutive_failures} consecutive failures: {exc}'
        save(report / 'memory_guard.json', {'state': 'STOPPED_BY_GPU_GUARD', 'reason': reason})
        subprocess.run(['docker', 'kill', NAME], capture_output=True, timeout=10)
        return


def tune_training_arguments(module, arguments, report):
    """Double new-job simulation concurrency; pin it for checkpoint resumes."""
    if module != 'scripts.train_flash_sac':
        return arguments
    profile = report / 'gpu_training_profile.json'
    if profile.exists():
        worlds = int(json.loads(profile.read_text())['world_count'])
    elif 'training.resume=true' in arguments:
        # Existing checkpoints were collected with 512 worlds.
        worlds = int(next(a.split('=', 1)[1] for a in arguments if a.startswith('scene.world_count=')))
    else:
        worlds = 1024
    if worlds not in (512, 1024):
        raise ValueError('Unexpected saved Track 3 world count')
    save(profile, {'world_count': worlds, 'memory_cap_mib': 21000,
                   'note': 'Concurrency tuning; measure throughput and memory before claiming improvement'})
    return [f'scene.world_count={worlds}' if a.startswith('scene.world_count=') else a for a in arguments]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('module')
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    args.report.mkdir(parents=True, exist_ok=True)
    args.arguments = tune_training_arguments(args.module, args.arguments, args.report)
    monorepo = ROOT.parent
    cache = monorepo / 'robotic_grounding/data/track3/runtime_cache'; cache.mkdir(parents=True, exist_ok=True)
    lock = (ROOT / 'data/track3/pilot/episode_000038/pilot.lock').open('a')
    print('Waiting for shared Track 3 GPU lock', flush=True)
    fcntl.flock(lock, fcntl.LOCK_EX)
    run_id = uuid.uuid4().hex if args.module == 'scripts.train_flash_sac' else None
    heartbeat_env = []
    if run_id:
        heartbeat_path = '/repo/' + str((args.report.resolve() / 'gpu_heartbeat.json').relative_to(monorepo))
        heartbeat_env = ['-e', f'V2D_GPU_HEARTBEAT_PATH={heartbeat_path}', '-e', f'V2D_GPU_HEARTBEAT_ID={run_id}']
    command = ['docker', 'run', '--rm', '--init', '--name', NAME, '--gpus', 'all',
               '--memory', '20g', '--memory-swap', '22g', '--cpus', '8', '--user', '1000:1000',
               '-e', 'HOME=/tmp', '-e', 'LOGNAME=alien51', '-e', 'OMP_NUM_THREADS=8',
               '-e', 'PYTHONUNBUFFERED=1', '-e', 'XLA_PYTHON_CLIENT_PREALLOCATE=false',
               '-e', 'XDG_CACHE_HOME=/cache', '-e', 'MUJOCO_GL=egl', '-e', 'PYOPENGL_PLATFORM=egl',
               '-e', 'PYTHONPATH=/workspace:/workspace/src',
               *heartbeat_env,
               '-v', f'{monorepo}:/repo', '-v', f'{monorepo}/robotic_grounding/flash_chord:/workspace',
               '-v', f'{cache}:/cache', '-w', '/workspace',
               '--entrypoint', 'python', 'flash_chord:track2-baseline',
               '-m', args.module, *args.arguments]
    save(args.report / 'command.json', {'state': 'RUNNING', 'command': command, 'gpu_monitor_executable': GPU_SMI,
                                        'gpu_heartbeat_id': run_id,
                                        'submission_ready': False})
    stop = threading.Event()
    threads = [threading.Thread(target=guard, args=(stop, args.report, run_id), daemon=True)]
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
