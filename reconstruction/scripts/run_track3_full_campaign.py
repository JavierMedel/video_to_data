#!/usr/bin/env python3
"""Prepare complete shared-world episodes, then train/evaluate genuine policies.

Runs alongside the reconstruction queue: CPU preparation overlaps inference;
policy training starts after reconstruction ends and shares its GPU lock.
Optional --upload validates entry and the packed file before sending six metrics.
Incomplete/failed cohorts cannot be labelled submission-ready.
"""
from datetime import datetime, timezone
import argparse
import fcntl
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import trimesh
from prepare_track3_campaign import ROOT, KIT, sha256
from run_track3_object_pilot import save

REPO = ROOT.parent
DATA = REPO / 'robotic_grounding/data/track3'
REPORT = ROOT / 'incident_reports/track3_baseline/full_campaign'
RECON = ROOT / 'incident_reports/track3_baseline/reconstruction_campaign/execution.json'
PYTHON = str(ROOT / '.venv/bin/python')
FULL_STEPS = 250003456
TOTAL_STEPS = FULL_STEPS
PROFILE = 'full'
POLICY_SUBDIR = 'policies'
RESULTS_SUBDIR = 'results'
SUBMISSION_SUBDIR = 'submission'
BASE_THRESHOLDS = [19660800, 34406400, 49152000, 63897600, 78643200, 93388800,
                   108134400, 122880000, 137625600, 200007680, FULL_STEPS]


def scaled_thresholds(total_steps, world_count=512):
    if total_steps < world_count * len(BASE_THRESHOLDS) or total_steps % world_count:
        raise ValueError('Training budget must fit every curriculum stage and be divisible by world count')
    result = [(value * total_steps + FULL_STEPS * world_count - 1) // (FULL_STEPS * world_count) * world_count
              for value in BASE_THRESHOLDS]
    if any(b <= a for a, b in zip([0] + result[:-1], result)):
        raise ValueError('Compressed curriculum thresholds must be strictly increasing')
    return result


def cpath(path): return '/repo/' + str(path.relative_to(REPO))


def run(command, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('a') as output:
        output.write(json.dumps({'command': command, 'utc': datetime.now(timezone.utc).isoformat()}) + '\n')
        output.flush()
        result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT)
    if result.returncode: raise RuntimeError(f'Exit {result.returncode}: {log}')


def cpu_command(image, script, args, module=False):
    return ['docker', 'run', '--rm', '--memory', '12g', '--memory-swap', '14g', '--cpus', '6',
            '--user', f'{os.getuid()}:{os.getgid()}', '--entrypoint', 'python',
            '-e', 'HOME=/tmp', '-e', 'LOGNAME=alien51', '-e', 'OMP_NUM_THREADS=4',
            '-e', 'PYTHONUNBUFFERED=1', '-e', 'PYTHONPATH=/repo/robotic_grounding/source/robotic_grounding',
            '-e', f'HUMAN_MOTION_DATA_DIR={cpath(DATA / "assets/human_motion_data")}',
            '-e', f'ROBOTIC_GROUNDING_INTERMEDIATE_DIR={cpath(DATA / "intermediate")}',
            '-v', f'{REPO}:/repo', image, *(['-m'] if module else []), script, *args]


def matrices(position, quaternion):
    result = np.stack([trimesh.transformations.quaternion_matrix(q) for q in np.asarray(quaternion)])
    result[:, :3, 3] = position
    return result


def shared_alignment(bundles, output):
    vertices = []
    for bundle in bundles:
        with np.load(bundle / 'result.npz', allow_pickle=False) as raw:
            transform = raw['object_to_world_transform'][0]
        mesh = trimesh.load(bundle / 'mesh.obj', force='scene').to_geometry()
        vertices.append(mesh.vertices @ transform[:3, :3].T + transform[:3, 3])
    centered = vertices[0] - vertices[0].mean(axis=0)
    _, axes = np.linalg.eigh(centered.T @ centered)
    extents = np.ptp(centered @ axes, axis=0)
    up = axes[:, int(np.argmin(extents))]
    if up @ vertices[0].mean(axis=0) > 0: up = -up
    rotation = trimesh.geometry.align_vectors(up, [0, 0, 1])[:3, :3]
    world = np.eye(4); world[:3, :3] = rotation
    rotated = [v @ rotation.T for v in vertices]
    world[:2, 3] = -rotated[0][:, :2].mean(axis=0)
    world[2, 3] = -min(float(v[:, 2].min()) for v in rotated)
    save(output, {'world_alignment': world.tolist(), 'method': 'primary reconstructed-object thin PCA axis; union initial object bottom at z=0',
                  'gravity_independently_verified': False, 'shared_by_all_objects': True})
    return world


def merge_rows(rows, episode):
    base = dict(rows[0])
    frames = episode['frames']
    for row in rows:
        if float(row['fps']) != 20 or len(row['object_body_position']) != frames:
            raise ValueError('Source cadence/frame mapping changed in loader')
        for side in ('left', 'right'):
            if not np.allclose(row[f'mano_{side}_joints'], base[f'mano_{side}_joints'], atol=1e-4, rtol=0):
                raise ValueError('Object bundles do not share one hand/world reference')
    for key in ('object_body_position', 'object_body_wxyz'):
        base[key] = np.concatenate([np.asarray(r[key]) for r in rows], axis=1).tolist()
    for key in ('object_body_names', 'object_mesh_paths', 'object_urdf_paths', 'object_mesh_radius', 'safe_object_body_names'):
        if key not in base: continue
        base[key] = [x for r in rows for x in r[key]]
    for side in ('left', 'right'):
        prefix = f'mano_{side}_object_contact_'
        # Native ManoSharpa has one contact per MANO link (16), regardless
        # of object count. Select the nearest active body contact per link.
        active = np.stack([np.asarray(r[prefix + 'part_ids']) > 0 for r in rows])
        distances = np.stack([np.sum((np.asarray(r[prefix + 'positions']) -
            np.asarray(r[f'mano_{side}_link_contact_positions'])) ** 2, axis=-1) for r in rows])
        choice = np.argmin(np.where(active, distances, np.inf), axis=0)
        valid = active.any(axis=0)
        for kind in ('object', 'link'):
            for suffix in ('positions', 'normals'):
                key = f'mano_{side}_{kind}_contact_{suffix}'
                values = np.stack([np.asarray(r[key]) for r in rows])
                selected = np.take_along_axis(values, choice[None, ..., None], axis=0)[0]
                base[key] = np.where(valid[..., None], selected, 0).tolist()
        base[prefix + 'part_ids'] = np.where(valid, choice + 1, 0).tolist()
        base[f'mano_{side}_tips_distance'] = np.min([np.asarray(r[f'mano_{side}_tips_distance']) for r in rows], axis=0).tolist()
    base.update(sequence_id=f'episode_{episode["episode_index"]:06d}', robot_name='sharpa_wave',
                object_name=f'episode_{episode["episode_index"]:06d}',
                safe_object_name=f'episode_{episode["episode_index"]:06d}', source_dataset='ego_recon')
    return base


def prepare(episode):
    index = episode['episode_index']; report = REPORT / f'episode_{index:06d}'; report.mkdir(parents=True, exist_ok=True)
    work = ROOT / ('data/track3/pilot/episode_000038' if index == 38 else f'data/track3/episodes/episode_{index:06d}')
    manifest = json.loads((work / 'reconstruction_manifest.json').read_text())
    bundles = [ROOT / p for p in manifest['bundles']]
    if len(bundles) != len(episode['objects_in_scored_order']): raise ValueError('Missing episode objects')
    target = DATA / f'references/episode_{index:06d}.parquet'
    receipt = report / 'preparation.json'
    source_hashes = {str(b.relative_to(ROOT)): {'npz': sha256(b / 'result.npz'), 'mesh': sha256(b / 'mesh.obj')} for b in bundles}
    if PROFILE == 'today' and not receipt.exists() and target.exists():
        original = ROOT / f'incident_reports/track3_baseline/full_campaign/episode_{index:06d}/preparation.json'
        if original.exists():
            old = json.loads(original.read_text())
            if old.get('source_hashes') == source_hashes and old.get('reference_sha256') == sha256(target):
                save(receipt, {**old, 'reused_from': str(original.relative_to(ROOT))})
                return target
    if receipt.exists() and target.exists():
        old = json.loads(receipt.read_text())
        if old.get('source_hashes') == source_hashes and old.get('reference_sha256') == sha256(target): return target
        raise ValueError('Existing prepared reference changed; preserve for investigation')
    # Preserve the already validated pilot retarget used by the training smoke.
    if index == 38 and target.exists():
        row = pq.ParquetFile(target).read().to_pylist()[0]
        with np.load(bundles[0] / 'result.npz', allow_pickle=False) as raw:
            achieved = matrices(np.asarray(row['object_body_position'])[:, 0], np.asarray(row['object_body_wxyz'])[:, 0])
            world = achieved[0] @ np.linalg.inv(raw['object_to_world_transform'][0])
            if not np.allclose(achieved, world[None] @ raw['object_to_world_transform'], atol=1e-5, rtol=0):
                raise ValueError('Pilot retarget changed object motion beyond a rigid world transform')
        save(report / 'world_alignment.json', {'world_alignment': world.tolist(), 'method': 'recovered verified native pilot rigid alignment'})
    else:
        alignment_path = report / 'world_alignment.json'
        world = shared_alignment(bundles, alignment_path)
        rows = []
        for slot, bundle in enumerate(bundles):
            sequence = f'episode_{index:06d}_object_{slot:02d}'
            run(cpu_command('v2d_track3_loader:blackwell', 'v2d.task_library_loader.lib.ego_recon_loader',
                ['--dataset_root', cpath(bundle.parent.parent), '--mano_model_dir', cpath(ROOT / 'data/weights/track3_mano'),
                 '--object_name', sequence, '--sequence_name', sequence, '--scene_alignment_path', cpath(alignment_path),
                 '--fps', '20', '--device', 'cpu', '--save'], module=True), report / f'load_object_{slot}.log')
            files = list((DATA / f'intermediate/ego_recon/loaded/sequence_id={sequence}').rglob('*.parquet'))
            if len(files) != 1: raise ValueError('Ambiguous loaded output')
            row = pq.ParquetFile(files[0]).read().to_pylist()[0]
            row.update(sequence_id=sequence, robot_name='sharpa_wave')
            with np.load(bundle / 'result.npz', allow_pickle=False) as raw:
                achieved = matrices(np.asarray(row['object_body_position'])[:, 0], np.asarray(row['object_body_wxyz'])[:, 0])
                if not np.allclose(achieved, world[None] @ raw['object_to_world_transform'], atol=1e-5, rtol=0):
                    raise ValueError('Loader changed object trajectories beyond common rigid alignment')
            rows.append(row)
        merged = merge_rows(rows, episode)
        loaded = DATA / f'merged_loaded/sequence_id=episode_{index:06d}/robot_name=sharpa_wave/data.parquet'
        loaded.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist([merged]), loaded)
        run(cpu_command('v2d_track2_retarget:baseline', '/repo/robotic_grounding/scripts/retarget/ego_recon_to_sharpa.py',
            ['--device', 'cpu', '--save', '--smooth_reference', '--input_dir', cpath(DATA / 'merged_loaded'),
             '--output_dir', cpath(DATA / 'retargeted'), '--sequence_pattern', f'episode_{index:06d}']), report / 'retarget.log')
        retargeted = list((DATA / f'retargeted/sequence_id=episode_{index:06d}').rglob('*.parquet'))
        if len(retargeted) != 1: raise ValueError('Ambiguous retarget output')
        row = pq.ParquetFile(retargeted[0]).read().to_pylist()[0]
        row.update(sequence_id=f'episode_{index:06d}', robot_name='sharpa_wave', source_dataset='ego_recon')
        target.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist([row]), target)
    run(cpu_command('v2d_track2_retarget:baseline', '/repo/robotic_grounding/scripts/generate_rigid_urdfs.py',
                    ['--dataset', 'ego_recon']), report / 'urdf.log')
    row = pq.ParquetFile(target).read().to_pylist()[0]
    for name in ['robot_left_finger_joints', 'robot_right_finger_joints', 'object_body_position', 'object_body_wxyz']:
        values = np.asarray(row[name])
        if len(values) != episode['frames'] or not np.isfinite(values).all(): raise ValueError('Invalid prepared reference')
    mesh_export = DATA / f'submission_meshes/episode_{index:06d}'; mesh_export.mkdir(parents=True, exist_ok=True)
    for name, path in zip(episode['objects_in_scored_order'], row['object_mesh_paths'], strict=True):
        original = REPO / path.removeprefix('/repo/')
        # URDF visual STL conversion preserves the metric body origin/axes.
        trimesh.load(original, force='scene').to_geometry().export(mesh_export / f'{name}.glb')
    save(receipt, {'state': 'PREPARED', 'source_hashes': source_hashes, 'reference_sha256': sha256(target),
                   'frames': episode['frames'], 'objects': episode['objects_in_scored_order'], 'fps': 20,
                   'world_alignment': world.tolist(), 'gravity_independently_verified': False})
    return target


def policy_command(report, module, overrides):
    return [PYTHON, str(ROOT / 'scripts/run_track3_policy_stage.py'), '--report', str(report), module, *overrides]


def upload_packed(packed, update, track='track_3'):
    """Validate before upload; recover scores from exact IDs without reuploading."""
    expected_metrics = 4 if track.startswith('track_2') else 6
    robot = 'G1+Dex3' if track.startswith('track_2') else 'Sharpa'
    uploader = KIT / 'submit/v2d_submit.py'
    receipt, manifest = REPORT / 'kaggle_receipt.json', REPORT / 'upload_manifest.json'
    digest = sha256(packed)
    if receipt.exists():
        if not manifest.exists() or json.loads(manifest.read_text()).get('packet_sha256') != digest:
            update(state='UPLOAD_REQUIRES_REVIEW', upload_error='Existing receipt belongs to a different or unverified packet')
            return
        command = [PYTHON, str(uploader), '--track', track, '--poll-receipt', str(receipt)]
    else:
        geometry = ', reconstructed meshes' if track == 'track_3' else ''
        message = f'{track} {PROFILE} baseline: per-sequence {robot} policies, target {TOTAL_STEPS} steps{geometry}, unassisted rollouts'
        base = [PYTHON, str(uploader), str(packed), '--track', track, '-m', message]
        try:
            run(base + ['--dry-run'], REPORT / 'upload_dry_run.log')
        except Exception as exc:
            update(state='PACKED_UPLOAD_BLOCKED', upload_error=str(exc))
            return
        save(manifest, {'packet_sha256': digest, 'receipt': str(receipt), 'profile': PROFILE})
        command = base + ['--wait', '--receipt', str(receipt)]
    while True:
        try:
            run(command, REPORT / 'upload.log')
        except Exception as exc:
            update(upload_error=str(exc))
        if not receipt.exists():
            update(state='UPLOAD_REQUIRES_REVIEW', upload_error='Uploader produced no receipt; inspect logs before retrying')
            return
        entries = json.loads(receipt.read_text()).get('submissions', {})
        accepted = any(value.get('ref') for value in entries.values())
        if len(entries) == expected_metrics and all(value.get('state') == 'COMPLETE' for value in entries.values()):
            update(state='SCORED', uploaded=True, kaggle_receipt=str(receipt),
                   kaggle_results=entries, upload_error=None)
            return
        terminal_failure = len(entries) != expected_metrics or any(not value.get('ref') or value.get('state') in
            ('ERROR', 'UPLOAD_FAILED', 'UPLOAD_UNKNOWN', 'UNREADABLE') for value in entries.values())
        if terminal_failure:
            update(state='SUBMISSION_REQUIRES_REVIEW', uploaded=accepted, kaggle_receipt=str(receipt), kaggle_results=entries)
            return
        update(state='SUBMITTED_PROCESSING', uploaded=accepted, kaggle_receipt=str(receipt), kaggle_results=entries)
        command = [PYTHON, str(uploader), '--track', track, '--poll-receipt', str(receipt)]
        time.sleep(20)


def train_and_evaluate(episode, reference):
    index = episode['episode_index']; report = REPORT / f'episode_{index:06d}'
    output = DATA / f'{POLICY_SUBDIR}/episode_{index:06d}'
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = sorted(output.glob('policy_*.safetensors'), key=lambda p: int(p.stem.split('_')[-1]))
    if PROFILE == 'today':
        # Existing longer-trained policies are legitimate baseline candidates;
        # retain their actual step count and never relabel them as full-budget.
        existing = [p for p in (DATA / f'policies/episode_{index:06d}').glob('policy_*.safetensors')
                    if int(p.stem.split('_')[-1]) >= TOTAL_STEPS]
        checkpoints += existing
        checkpoints.sort(key=lambda p: int(p.stem.split('_')[-1]))
    if not checkpoints or int(checkpoints[-1].stem.split('_')[-1]) < TOTAL_STEPS - 4096:
        overrides = ['experiment=sharpa_flash_sac', f'task.parquet={cpath(reference)}',
                     'task.motion_speed=1.0', 'task.source_frame_playback=false', 'sim.fps=20',
                     'sim.njmax=2048', 'sim.nconmax=512',
                     'scene.world_count=512', 'scene.support=false', 'training.replay.capacity=1000000',
                     'training.replay.minimum_size=51200', 'training.replay.batch_size=1024',
                     f'training.total_environment_steps={TOTAL_STEPS}', 'logging.mode=disabled',
                     f'output_dir={cpath(output)}', 'reset.seed=42']
        if PROFILE == 'today':
            overrides += ['curriculum.thresholds=' + json.dumps(scaled_thresholds(TOTAL_STEPS), separators=(',', ':')),
                          'training.checkpoint.save_interval_environment_steps=1000448']
        states = sorted(output.glob('state_*.safetensors'), key=lambda p: int(p.stem.split('_')[-1]))
        if states: overrides += ['training.resume=true', f'training.checkpoint_path={cpath(states[-1])}']
        run(policy_command(report / 'training', 'scripts.train_flash_sac', overrides), report / 'training.log')
        checkpoints = sorted(output.glob('policy_*.safetensors'), key=lambda p: int(p.stem.split('_')[-1]))
    if not checkpoints: raise ValueError('Training produced no real policy checkpoint')
    checkpoint = checkpoints[-1]
    results = DATA / RESULTS_SUBDIR; results.mkdir(parents=True, exist_ok=True)
    trajectory, metrics = results / f'episode_{index:06d}.parquet', results / f'episode_{index:06d}.json'
    receipt = report / 'policy.json'
    hashes = {'checkpoint_sha256': sha256(checkpoint), 'reference_sha256': sha256(reference)}
    reused = False
    if receipt.exists() and trajectory.exists() and metrics.exists():
        old = json.loads(receipt.read_text())
        reused = old.get('state') == 'EVALUATED' and all(old.get(k) == v for k, v in hashes.items())
        reused = reused and old.get('trajectory_sha256') == sha256(trajectory) and old.get('metrics_sha256') == sha256(metrics)
    if not reused:
        run(policy_command(report / 'evaluation', 'scripts.evaluate_policy',
            [f'evaluation.checkpoint={cpath(checkpoint)}', 'evaluation.world_count=1', 'evaluation.reset_mode=explicit',
             f'evaluation.metrics_output={cpath(metrics)}', f'evaluation.object_trajectories_output={cpath(trajectory)}',
             'logging.mode=disabled']), report / 'evaluation.log')
    table = pq.ParquetFile(trajectory).read().to_pandas()
    expected = {(f, o) for f in range(episode['frames']) for o in range(len(episode['objects_in_scored_order']))}
    actual = set(zip(table['reference_frame'], table['object_body_id']))
    if actual != expected or len(table) != len(expected) or not table.environment_valid.all():
        raise ValueError('Policy rollout is missing frames/objects or contains invalid worlds')
    save(report / 'policy.json', {'state': 'EVALUATED', 'checkpoint_sha256': sha256(checkpoint),
         'reference_sha256': sha256(reference), 'trajectory_sha256': sha256(trajectory),
         'metrics_sha256': sha256(metrics), 'trained_environment_steps': int(checkpoint.stem.split('_')[-1]),
         'profile': PROFILE, 'requested_training_steps': TOTAL_STEPS,
         'checkpoint_path': str(checkpoint), 'reference_path': str(reference),
         'full_budget_policy': int(checkpoint.stem.split('_')[-1]) >= FULL_STEPS - 4096,
         'learned_policy': True, 'official_held_out_score': False})


def submission_rollout_names(table, episode):
    """Map recorded body IDs to scored names without changing any pose values."""
    index = episode['episode_index']
    roster = episode['objects_in_scored_order']
    slots = table['object_body_id'].to_pylist()
    labels = table['object_body_name'].to_pylist()
    canonical = []
    for slot, label in zip(slots, labels, strict=True):
        if slot is None or not 0 <= slot < len(roster):
            raise ValueError('Invalid rollout body ID for scored roster')
        expected = roster[slot]
        if label not in (expected, f'episode_{index:06d}_object_{slot:02d}'):
            raise ValueError(f'Unexpected rollout body name {label!r} in slot {slot}')
        canonical.append(expected)
    column = table.schema.get_field_index('object_body_name')
    values = pa.array(canonical).dictionary_encode().cast(table.schema.field(column).type)
    return table.set_column(column, table.schema.field(column), values)


def stage_submission_rollouts(episodes):
    """Keep raw evaluator evidence; export only verified semantic body labels."""
    source = DATA / RESULTS_SUBDIR
    target = DATA / (RESULTS_SUBDIR + '_submission'); target.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for episode in episodes:
        stem = f'episode_{episode["episode_index"]:06d}'
        raw = source / (stem + '.parquet')
        original = pq.ParquetFile(raw).read()
        mapped = submission_rollout_names(original, episode)
        for name in original.column_names:
            if name != 'object_body_name' and not original[name].equals(mapped[name]):
                raise ValueError(f'Export changed rollout values: {name}')
        destination = target / raw.name
        pq.write_table(mapped, destination)
        shutil.copyfile(source / (stem + '.json'), target / (stem + '.json'))
        manifest[stem] = {'raw_sha256': sha256(raw), 'export_sha256': sha256(destination),
                          'object_names_in_body_id_order': episode['objects_in_scored_order'],
                          'changed_column': 'object_body_name', 'pose_values_unchanged': True}
    save(REPORT / 'submission_name_mapping.json', manifest)
    return target


def train_with_monitor_recovery(episode, reference):
    """Retry monitoring-only interruptions, never memory-cap/physics failures."""
    report = REPORT / f'episode_{episode["episode_index"]:06d}'
    for attempt in range(4):
        try:
            return train_and_evaluate(episode, reference)
        except Exception:
            guards = list(report.glob('*/memory_guard.json'))
            if attempt == 3 or not guards:
                raise
            if any(not json.loads(p.read_text()).get('reason', '').startswith('GPU monitor unavailable') for p in guards):
                raise
            # Require Docker to be responsive and the previous container gone.
            running = subprocess.check_output(['docker', 'ps', '--filter',
                'name=^/v2d-track3-policy-stage$', '--format', '{{.Names}}'], text=True, timeout=15)
            if running.strip():
                raise RuntimeError('Previous GPU container still exists; automatic recovery stopped')
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            for guard in guards:
                archive = guard.parent / 'previous_attempts' / stamp
                archive.mkdir(parents=True, exist_ok=True)
                guard.rename(archive / guard.name)
            print(f'MONITOR_RECOVERY {episode["episode_index"]} retry {attempt + 1}/3; resuming saved checkpoint', flush=True)
            time.sleep(10)


def main():
    global REPORT, TOTAL_STEPS, PROFILE, POLICY_SUBDIR, RESULTS_SUBDIR, SUBMISSION_SUBDIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=('full', 'today'), default='full')
    parser.add_argument('--training-steps', type=int)
    parser.add_argument('--upload', action='store_true', help='Validate and upload all six metrics after a complete pack')
    args = parser.parse_args()
    PROFILE = args.profile
    if PROFILE == 'today':
        REPORT = ROOT / 'incident_reports/track3_baseline/today_campaign'
        POLICY_SUBDIR, RESULTS_SUBDIR, SUBMISSION_SUBDIR = 'policies_today', 'results_today', 'submission_today'
    TOTAL_STEPS = args.training_steps if args.training_steps is not None else (5000192 if PROFILE == 'today' else FULL_STEPS)
    scaled_thresholds(TOTAL_STEPS)
    REPORT.mkdir(parents=True, exist_ok=True)
    lock = (REPORT / 'campaign.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    inventory = json.loads((ROOT / 'incident_reports/track3_baseline/inputs.json').read_text())
    episodes = sorted(inventory['episodes'], key=lambda e: (e['episode_index'] != 38, e['frames']))
    state = {'state': 'WAITING_FOR_RECONSTRUCTION', 'prepared': [], 'evaluated': [], 'failed': {},
             'submission_ready': False, 'uploaded': False, 'training_steps_per_episode': TOTAL_STEPS,
             'profile': PROFILE, 'full_budget_steps': FULL_STEPS}
    def update(**kw):
        state.update(**kw, observed_utc=datetime.now(timezone.utc).isoformat()); save(REPORT / 'execution.json', state)
    prepared = {}
    while True:
        recon = json.loads(RECON.read_text()) if RECON.exists() else {}
        for episode in episodes:
            index = episode['episode_index']
            if index not in recon.get('completed_episode_indices', []) or index in prepared or str(index) in state['failed']: continue
            update(state='CPU_PREPARATION', active_episode_index=index)
            try:
                prepared[index] = prepare(episode)
                state['prepared'].append(index)
                print('PREPARED', index, flush=True)
            except Exception as exc:
                state['failed'][str(index)] = f'preparation: {exc}'
                print('PREPARATION_FAILED', index, str(exc), flush=True)
            update(state='WAITING_FOR_RECONSTRUCTION')
        if recon.get('state') in ('COMPLETE', 'PARTIAL', 'STOPPED_BY_GPU_GUARD'): break
        time.sleep(15)
    if recon.get('state') == 'STOPPED_BY_GPU_GUARD':
        update(state='STOPPED_BY_GPU_GUARD'); return
    for episode in episodes:
        index = episode['episode_index']
        if index not in prepared: continue
        update(state='TRAINING_AND_EVALUATING', active_episode_index=index)
        try:
            train_with_monitor_recovery(episode, prepared[index]); state['evaluated'].append(index)
        except Exception as exc:
            state['failed'][str(index)] = f'policy: {exc}'
            print('POLICY_FAILED', index, str(exc), flush=True)
            if list((REPORT / f'episode_{index:06d}').glob('*/memory_guard.json')):
                update(state='STOPPED_BY_GPU_GUARD'); return
        update()
    expected_ids = {e['episode_index'] for e in episodes}
    if set(state['evaluated']) == expected_ids:
        kit = KIT
        submission_results = stage_submission_rollouts(episodes)
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
        packed = DATA / f'{SUBMISSION_SUBDIR}/track3_submission.parquet'; packed.parent.mkdir(parents=True, exist_ok=True)
        run([PYTHON, str(kit / 'tools/pack_submission.py'), '--track', 'track_3', '--results', str(submission_results),
             '--meshes', str(DATA / 'submission_meshes'), '--out', str(packed),
             '--eval-script', str(REPO / 'robotic_grounding/flash_chord/scripts/evaluate_policy.py'),
             '--code-commit', f'https://github.com/JavierMedel/video_to_data/commit/{commit}'], REPORT / 'packing.log')
        update(state='PACKED_FOR_REVIEW', submission_ready=True, submission_path=str(packed))
        save(packed.parent / 'episode_checkpoints.json', {
            str(e['episode_index']): json.loads((REPORT / f'episode_{e["episode_index"]:06d}/policy.json').read_text())['checkpoint_path']
            for e in episodes})
        if args.upload:
            upload_packed(packed, update)
    else:
        update(state='INCOMPLETE', active_episode_index=None, missing_episode_indices=sorted(expected_ids - set(state['evaluated'])))


if __name__ == '__main__': main()
