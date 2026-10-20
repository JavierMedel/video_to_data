#!/usr/bin/env python3
"""Train genuine per-sequence Tier 1 policies, evaluate, pack and optionally submit.

Raw evaluator evidence stays immutable. One 350M pilot may be reused; remaining
policies use the explicitly selected budget with all native curriculum phases.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(REPO / 'reconstruction/scripts'))
import run_track3_full_campaign as common
from retarget_track2_tier1 import verify_motion

PYTHON = str(REPO / 'reconstruction/.venv/bin/python')
DATA = ROOT / 'data/track2/tier1'
REPORT = ROOT / 'incident_reports/track2_baseline/short_campaign'
FULL_STEPS = 349999104
STEPS = 5000192
TIER = 1
POLICY_SUBDIR, RESULTS_SUBDIR, EXPORT_SUBDIR = 'policies_short', 'results_short', 'results_short_submission'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def scaled_thresholds(steps):
    config = yaml.safe_load((ROOT / 'flash_chord/src/flash_chord/configs/curriculum/recon_body_flash_sac.yaml').read_text())
    if steps % 1024 or steps < 1024 * len(config['thresholds']):
        raise ValueError('Budget must be a multiple of 1024 and cover all curriculum stages')
    result = [(v * steps + FULL_STEPS * 1024 - 1) // (FULL_STEPS * 1024) * 1024 for v in config['thresholds']]
    if result[-1] != steps or any(a >= b for a, b in zip(result, result[1:])):
        raise ValueError('Invalid scaled curriculum')
    return result


def run(command, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('a') as output:
        output.write(json.dumps({'utc': datetime.now(timezone.utc).isoformat(), 'command': command}) + '\n'); output.flush()
        subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, check=True)


def native(stage, module, arguments):
    return [PYTHON, str(REPO / 'reconstruction/scripts/run_track3_policy_stage.py'), '--track', f'track_2_tier{TIER}', '--report', str(stage), module, *arguments]


def cpath(path):
    return '/repo/' + str(path.resolve().relative_to(ROOT))


def verify_input(episode):
    i = episode['episode_index']; reference = Path(episode['reference'])
    if sha(reference) != episode['reference_sha256']:
        raise ValueError(f'Retargeted reference changed for episode {i}')
    source = DATA / 'source_episodes' / episode['sequence_id']
    manifest = json.loads((source / 'source_manifest.json').read_text())
    if manifest.get('tier', 1) != TIER:
        raise ValueError('Source manifest belongs to another Track 2 tier')
    if 'source_manifest_sha256' in episode and sha(source / 'source_manifest.json') != episode['source_manifest_sha256']:
        raise ValueError('Inventory source lineage changed')
    if TIER != 1:
        if sha(episode['frame_reference']) != episode['frame_reference_sha256']:
            raise ValueError('Inventory packer frame reference changed')
    for filename, digest in manifest['output_sha256'].items():
        if sha(source / filename) != digest:
            raise ValueError(f'Source artifact changed: {source / filename}')
    verify_motion(reference, source)
    return reference, manifest


def validate_rollout(table, episode):
    actual = list(zip(table['environment_id'].to_pylist(), table['reference_frame'].to_pylist(), table['object_body_id'].to_pylist()))
    expected = {(0, frame, 0) for frame in range(episode['frames'])}
    if len(actual) != len(expected) or set(actual) != expected or not all(table['environment_valid'].to_pylist()):
        raise ValueError('Rollout must contain exactly every source frame in one valid world')
    for name in table.column_names:
        if name.startswith(('achieved_', 'reference_')) and not np.isfinite(table[name].to_numpy()).all():
            raise ValueError(f'Non-finite rollout poses: {name}')


def train_evaluate(episode):
    i = episode['episode_index']; stem = f'episode_{i:06d}'; report = REPORT / stem
    reference, manifest = verify_input(episode)
    output = DATA / POLICY_SUBDIR / stem; output.mkdir(parents=True, exist_ok=True)
    checkpoints = sorted(output.glob('policy_*.safetensors'), key=lambda p: int(p.stem.split('_')[-1]))
    if TIER == 1:
        checkpoints += [p for p in (DATA / 'baseline' / stem).glob('policy_*.safetensors') if int(p.stem.split('_')[-1]) >= STEPS]
    checkpoints.sort(key=lambda p: int(p.stem.split('_')[-1]))
    if not checkpoints or int(checkpoints[-1].stem.split('_')[-1]) < STEPS:
        arguments = ['experiment=g1_recon_body_flash_sac', f'task.parquet={json.dumps(cpath(reference))}', 'scene.world_count=1024', 'sim.njmax=4096', 'sim.nconmax=1024',
                     'training.replay.capacity=1000000', 'training.replay.minimum_size=102400', 'training.replay.batch_size=1024',
                     f'training.total_environment_steps={STEPS}', 'training.checkpoint.save_interval_environment_steps=1000448',
                     'curriculum.thresholds=' + json.dumps(scaled_thresholds(STEPS), separators=(',', ':')),
                     'reset.seed=42', 'logging.mode=disabled', f'output_dir={cpath(output)}']
        states = sorted(output.glob('state_*.safetensors'), key=lambda p: int(p.stem.split('_')[-1]))
        if states: arguments += ['training.resume=true', f'training.checkpoint_path={cpath(states[-1])}']
        run(native(report / 'training', 'scripts.train_flash_sac', arguments), report / 'training.log')
        checkpoints = sorted(output.glob('policy_*.safetensors'), key=lambda p: int(p.stem.split('_')[-1]))
    checkpoint = checkpoints[-1]
    results = DATA / RESULTS_SUBDIR; results.mkdir(parents=True, exist_ok=True)
    trajectory, metrics = results / (stem + '.parquet'), results / (stem + '.json')
    receipt = report / 'policy.json'; hashes = {'checkpoint_sha256': sha(checkpoint), 'reference_sha256': sha(reference)}
    cached = json.loads(receipt.read_text()) if receipt.exists() else {}
    reuse = cached.get('state') == 'EVALUATED' and all(cached.get(k) == v for k, v in hashes.items())
    reuse = reuse and trajectory.exists() and metrics.exists() and cached.get('trajectory_sha256') == sha(trajectory) and cached.get('metrics_sha256') == sha(metrics)
    if not reuse:
        pilot_trajectory, pilot_metrics = DATA / 'results' / trajectory.name, DATA / 'results' / metrics.name
        if TIER == 1 and i == 16 and pilot_trajectory.exists() and pilot_metrics.exists() and json.loads(pilot_metrics.read_text()).get('checkpoint_sha256') == sha(checkpoint):
            # Preserve the full-budget pilot's genuine native rollout and its actual hashes.
            shutil.copyfile(pilot_trajectory, trajectory); shutil.copyfile(pilot_metrics, metrics)
        else:
            run(native(report / 'evaluation', 'scripts.evaluate_policy', [f'evaluation.checkpoint={cpath(checkpoint)}',
                'evaluation.world_count=1', 'evaluation.reset_mode=explicit', f'evaluation.metrics_output={cpath(metrics)}',
                f'evaluation.object_trajectories_output={cpath(trajectory)}', 'logging.mode=disabled']), report / 'evaluation.log')
    table = pq.ParquetFile(trajectory).read(); validate_rollout(table, episode)
    metadata = table.schema.metadata or {}
    if metadata.get(b'flash_chord.checkpoint_sha256', b'').decode() != sha(checkpoint):
        raise ValueError('Rollout checkpoint provenance mismatch')
    # Native single-object tasks call the body "object"; roster identity comes from the verified source manifest.
    object_name = manifest['object']; labels = table['object_body_name'].to_pylist()
    if any(name not in ('object', object_name) for name in labels):
        raise ValueError('Unexpected object assignment in native rollout')
    field = table.schema.field('object_body_name'); values = pa.array([object_name] * len(table)).dictionary_encode().cast(field.type)
    staged = table.set_column(table.schema.get_field_index('object_body_name'), field, values)
    for name in table.column_names:
        if name != 'object_body_name' and not table[name].equals(staged[name]):
            raise ValueError(f'Export changed rollout values: {name}')
    export = DATA / EXPORT_SUBDIR; export.mkdir(parents=True, exist_ok=True)
    pq.write_table(staged, export / trajectory.name); shutil.copyfile(metrics, export / metrics.name)
    record = {'state': 'EVALUATED', **hashes, 'checkpoint_path': str(checkpoint), 'reference_path': str(reference),
              'trajectory_sha256': sha(trajectory), 'metrics_sha256': sha(metrics), 'export_sha256': sha(export / trajectory.name),
              'trained_environment_steps': int(checkpoint.stem.split('_')[-1]), 'requested_training_steps': STEPS,
              'full_budget_policy': int(checkpoint.stem.split('_')[-1]) >= FULL_STEPS, 'object_name': object_name,
              'rows': len(table), 'raw_poses_preserved': True}
    save(receipt, record)
    return record


def recoverable_train(episode):
    report = REPORT / f'episode_{episode["episode_index"]:06d}'
    for attempt in range(4):
        try:
            return train_evaluate(episode)
        except subprocess.CalledProcessError:
            guards = list(report.glob('*/memory_guard.json'))
            if attempt == 3 or not guards or any(not json.loads(p.read_text()).get('reason', '').startswith('GPU monitor unavailable') for p in guards):
                raise
            active = subprocess.check_output(['docker', 'ps', '--filter', 'name=^/v2d-track2-policy-stage$', '--format', '{{.Names}}'], text=True, timeout=15)
            if active.strip(): raise RuntimeError('Prior policy container still exists')
            for guard in guards:
                archive = guard.parent / 'previous_attempts' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
                archive.mkdir(parents=True, exist_ok=True); guard.rename(archive / guard.name)
            print(f'MONITOR_RECOVERY episode {episode["episode_index"]} retry {attempt + 1}/3', flush=True)
            time.sleep(10)


def main():
    global STEPS, TIER, DATA, REPORT, POLICY_SUBDIR, RESULTS_SUBDIR, EXPORT_SUBDIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-steps', type=int, default=5000192)
    parser.add_argument('--upload', action='store_true')
    parser.add_argument('--tier', type=int, choices=(1, 2, 3), default=1)
    parser.add_argument('--version', default='short')
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--inventory', type=Path)
    parser.add_argument('--parent-packet', type=Path)
    parser.add_argument('--rules', type=Path)
    parser.add_argument('--snapshots', type=Path)
    args = parser.parse_args(); STEPS = args.training_steps; scaled_thresholds(STEPS)
    if not args.version.replace('_', '').replace('-', '').isalnum():
        raise ValueError('Version must be one simple directory name')
    TIER = args.tier
    DATA = (args.data_root or ROOT / f'data/track2/tier{TIER}').resolve()
    DATA.relative_to(ROOT)
    legacy = TIER == 1 and args.version == 'short' and args.data_root is None
    if not legacy:
        if args.inventory is None: raise ValueError('New tier/version requires a verified inventory')
        REPORT = ROOT / f'incident_reports/track2_tier{TIER}/{args.version}_campaign'
        POLICY_SUBDIR, RESULTS_SUBDIR, EXPORT_SUBDIR = f'policies_{args.version}', f'results_{args.version}', f'results_{args.version}_submission'
        if args.upload and (args.rules is None or args.snapshots is None):
            raise ValueError('New uploads require documented weekly rules and fresh snapshots')
    REPORT.mkdir(parents=True, exist_ok=True)
    lock = (REPORT / 'campaign.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    inventory_path = args.inventory or ROOT / 'incident_reports/track2_baseline/submission_readiness_20261019.json'
    inventory = json.loads(inventory_path.read_text())
    if inventory.get('tier', 1) != TIER: raise ValueError('Inventory tier mismatch')
    signature = {'tier': TIER, 'version': args.version, 'inventory_sha256': sha(inventory_path), 'training_steps': STEPS,
                 'parent_packet_sha256': sha(args.parent_packet) if args.parent_packet else None}
    contract = REPORT / 'campaign_inputs.json'
    if contract.exists() and json.loads(contract.read_text()) != signature:
        raise ValueError('Campaign inputs changed; choose a new version')
    save(contract, signature)
    episodes = sorted(inventory['episodes'], key=lambda e: (e['episode_index'] != 16, e['frames']))
    if {e['episode_index'] for e in episodes} != set(range(30)): raise ValueError('Incomplete scored roster')
    state = {'state': 'TRAINING_AND_EVALUATING', 'track': f'track_2_tier{TIER}', 'evaluated': [], 'failed': {}, 'inputs': signature,
             'training_steps_per_new_episode': STEPS, 'full_budget_steps': FULL_STEPS, 'submission_ready': False, 'uploaded': False}
    def update(**fields):
        state.update(**fields, observed_utc=datetime.now(timezone.utc).isoformat()); save(REPORT / 'execution.json', state)
    update()
    for episode in episodes:
        i = episode['episode_index']; update(active_episode_index=i)
        print(f'START episode {i}; {len(state["evaluated"])}/30 evaluated', flush=True)
        try:
            recoverable_train(episode); state['evaluated'].append(i); update()
            print(f'EVALUATED episode {i}; {len(state["evaluated"])}/30 complete', flush=True)
        except Exception as exc:
            state['failed'][str(i)] = str(exc); update(state='STOPPED_FOR_REVIEW'); raise
    common.REPORT = REPORT; common.PROFILE = 'short'; common.TOTAL_STEPS = STEPS
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    packet = DATA / f'submission_{args.version}/track2_tier{TIER}.csv'; packet.parent.mkdir(parents=True, exist_ok=True)
    save(packet.parent / 'episode_checkpoints.json', {str(i): json.loads((REPORT / f'episode_{i:06d}/policy.json').read_text())['checkpoint_path'] for i in state['evaluated']})
    frame_args = [] if TIER == 1 else ['--frame-references', str(DATA / 'frame_references')]
    run([PYTHON, str(common.KIT / 'tools/pack_submission.py'), '--track', f'track_2_tier{TIER}', '--results', str(DATA / EXPORT_SUBDIR), *frame_args,
         '--dataset-root', str(REPO / 'reconstruction/data/v2d_challenge'), '--eval-script', str(ROOT / 'flash_chord/scripts/evaluate_policy.py'),
         '--code-commit', f'https://github.com/JavierMedel/video_to_data/commit/{commit}', '-o', str(packet)], REPORT / 'packing.log')
    update(state='PACKED_FOR_REVIEW', active_episode_index=None, submission_ready=True, submission_path=str(packet))
    if args.upload:
        if legacy:
            common.upload_packed(packet, update, track='track_2_tier1')
        else:
            from cross_track.submission import submit
            receipt = submit(packet, f'track_2_tier{TIER}', args.rules, args.snapshots, REPORT, execute=True)
            entries = receipt.get('submissions', {})
            update(state='SCORED' if len(entries) == 4 and all(r.get('state') == 'COMPLETE' for r in entries.values()) else 'SUBMITTED_REQUIRES_REVIEW',
                   uploaded=any(r.get('ref') for r in entries.values()), kaggle_receipt=str(REPORT / 'kaggle_receipt.json'))


if __name__ == '__main__':
    main()
