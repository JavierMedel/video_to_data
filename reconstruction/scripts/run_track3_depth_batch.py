#!/usr/bin/env python3
"""Prepare bounded MoGe depth/intrinsics for the remaining scored Track 3 videos.

Uses the pilot's raw-RGB baseline; camera calibration/metric validation remain
separate stages. Each episode has independent hash-verified resume evidence.
"""
from datetime import datetime, timezone
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import threading

from prepare_track3_campaign import ROOT, sha256
from run_track3_object_pilot import cpath, run_stage, save, telemetry
from run_track3_hand_pilot import command

NAME = 'v2d-track3-depth-stage'
MEMORY_FRACTION = .70
BATCH_SIZE = 8


def guard(stop, path):
    while not stop.wait(.5):
        try:
            sample = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                                    capture_output=True, text=True, timeout=5)
            if sample.returncode: raise RuntimeError(sample.stderr)
            used = float(sample.stdout.strip().splitlines()[0])
            if used < 19000: continue
            reason = f'GPU memory reached {used} MiB (limit 19000)'
        except (subprocess.TimeoutExpired, RuntimeError, ValueError, IndexError) as exc:
            reason = f'GPU monitor unavailable: {exc}'
        save(path, {'state': 'STOPPED_BY_GPU_GUARD', 'reason': reason,
                    'observed_utc': datetime.now(timezone.utc).isoformat()})
        subprocess.run(['docker', 'kill', NAME], capture_output=True, timeout=10)
        return


def reuse_with_budget_change(record_path, cmd, outputs, inputs):
    """Preserve original evidence when only resource settings/queue code changed."""
    if not record_path.exists():
        return False
    old = json.loads(record_path.read_text())
    if old.get('state') != 'COMPLETE':
        return False
    def normalize(command):
        normalized = []
        skip_batch_value = False
        for item in command:
            if skip_batch_value:
                if not item.isdigit() or int(item) < 1:
                    raise ValueError('Invalid batch size in resource-resume command')
                skip_batch_value = False
                continue
            if item == '--batch_size':
                skip_batch_value = True
                normalized.append(item)
            elif not item.startswith('V2D_TORCH_MEMORY_FRACTION='):
                normalized.append(item)
        if skip_batch_value:
            raise ValueError('Missing batch size in resource-resume command')
        return normalized
    if normalize(old['command']) != normalize(cmd):
        return False
    image = cmd[cmd.index('-m') - 1]
    image_id = subprocess.check_output(['docker', 'image', 'inspect', image, '--format', '{{.Id}}'], text=True).strip()
    if image_id != old['image_id'] or old['runner_sha256'] != sha256(ROOT / 'scripts/run_track3_object_pilot.py'):
        return False
    for path in inputs:
        if path == Path(__file__):
            continue  # Explicit opt-in permits this queue's resource/resume edit only.
        if old['input_sha256'].get(str(path.relative_to(ROOT))) != sha256(path):
            return False
    actual = {str(p.relative_to(ROOT)): sha256(p) for output in outputs
              for p in output.rglob('*') if p.is_file()}
    return bool(actual) and actual == old.get('output_sha256')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume-budget-change', action='store_true',
                        help='Hash-verify completed episodes across this resource-only update; retain original provenance')
    args = parser.parse_args()
    report = ROOT / 'incident_reports/track3_baseline/depth_batch'
    report.mkdir(parents=True, exist_ok=True)
    batch_lock = (report / 'batch.lock').open('w')
    fcntl.flock(batch_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    inventory_path = ROOT / 'incident_reports/track3_baseline/inputs.json'
    inventory = json.loads(inventory_path.read_text())
    episodes = sorted((e for e in inventory['episodes'] if e['episode_index'] != 38), key=lambda e: e['frames'])
    state = {'state': 'QUEUED_FOR_GPU', 'track': 'track_3', 'episode_count': len(episodes),
             'completed_episode_indices': [], 'existing_completed_pilot_episode': 38,
             'dataset_revision': inventory['dataset_revision'], 'batch_size': BATCH_SIZE,
             'pytorch_memory_fraction': MEMORY_FRACTION, 'preserved_episode_indices': [], 'gpu_memory_guard_mib': 19000,
             'container_ram_limit_gib': 16, 'policy_trained': False, 'submission_ready': False, 'uploaded': False}
    save(report / 'execution.json', state)
    print('QUEUED: remaining 19 depth episodes, shortest first; one GPU container at a time', flush=True)
    # Share the pilot lock so any hand/refinement stage finishes first and a
    # future GPU job can be scheduled between depth episodes.
    lock = (ROOT / 'data/track3/pilot/episode_000038/pilot.lock').open('w')
    try:
        for episode in episodes:
            index = episode['episode_index']
            fcntl.flock(lock, fcntl.LOCK_EX)
            state.update(state='RUNNING', active_episode_index=index,
                         observed_utc=datetime.now(timezone.utc).isoformat())
            save(report / 'execution.json', state)
            source = episode['videos']['ego_cam_a']
            video = ROOT / source['path']
            if sha256(video) != source['sha256']:
                raise ValueError(f'Source video changed for episode {index}')
            work = ROOT / f'data/track3/episodes/episode_{index:06d}'
            work.mkdir(parents=True, exist_ok=True)
            depth, intrinsics = work / 'depth', work / 'intrinsics'
            cmd = command('v2d_cari4d:blackwell', 'v2d.moge.lib.video_to_depth',
                          ['--video_path', cpath(video), '--depth_folder', cpath(depth),
                           '--intrinsics_folder', cpath(intrinsics), '--weights_path', '/repo/data/weights/moge',
                           '--batch_size', str(BATCH_SIZE)])
            cmd[cmd.index('--name') + 1] = NAME
            cmd = [f'V2D_TORCH_MEMORY_FRACTION={MEMORY_FRACTION}' if x.startswith('V2D_TORCH_MEMORY_FRACTION=') else x for x in cmd]
            stop = threading.Event()
            threads = [threading.Thread(target=telemetry, args=(stop, report / 'gpu.csv'), daemon=True),
                       threading.Thread(target=guard, args=(stop, report / 'memory_guard.json'), daemon=True)]
            for thread in threads: thread.start()
            try:
                stage_inputs = [Path(__file__), ROOT / 'scripts/run_track3_hand_pilot.py',
                                ROOT / 'scripts/track3_gpu_python.py', inventory_path, video,
                                ROOT / 'data/weights/moge/model.pt', ROOT / 'modules/v2d_moge/lib/video_to_depth.py']
                record_path = report / f'episode_{index:06d}.json'
                if args.resume_budget_change and reuse_with_budget_change(record_path, cmd, [depth, intrinsics], stage_inputs):
                    state['preserved_episode_indices'].append(index)
                    print('PRESERVED_WITH_ORIGINAL_PROVENANCE', index, flush=True)
                else:
                    run_stage(f'episode_{index:06d}', cmd, [depth, intrinsics], stage_inputs, report)
                expected = {f'{i:06d}' for i in range(episode['frames'])}
                if {p.stem for p in depth.glob('*.png')} != expected or {p.stem for p in intrinsics.glob('*.json')} != expected:
                    raise ValueError(f'Incomplete source-frame coverage for episode {index}')
                save(work / 'depth_manifest.json', {'episode_index': index, 'source_sha256': source['sha256'],
                     'frames': episode['frames'], 'fps': source['fps'], 'source_camera': 'ego_cam_a',
                     'depth_encoding': 'v2d uint16 inverse depth, 65535/(depth_metres+1)',
                     'input_rectified': False, 'metric_scale_independently_verified': False,
                     'provenance_report': str((report / f'episode_{index:06d}.json').relative_to(ROOT))})
                state['completed_episode_indices'].append(index)
                print('DEPTH_COMPLETE', index, episode['frames'], 'frames', flush=True)
            finally:
                stop.set()
                for thread in threads: thread.join(timeout=10)
                fcntl.flock(lock, fcntl.LOCK_UN)
        state.update(state='COMPLETE', active_episode_index=None)
    except Exception as exc:
        state.update(state='FAILED', error=str(exc))
        raise
    finally:
        state['observed_utc'] = datetime.now(timezone.utc).isoformat()
        save(report / 'execution.json', state)


if __name__ == '__main__':
    main()
