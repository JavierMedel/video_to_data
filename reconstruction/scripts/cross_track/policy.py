"""Deterministic policy evaluation, exact Tier 1 selection and budget planning."""
import argparse
import json
from pathlib import Path
import struct
import sys
import numpy as np
import pyarrow.parquet as pq
from .common import KIT, REPO, kit_imports, metric, sha, save, load, checked_command, new_run, complete

kit_imports()
from v2dlb import object_pose as op
from v2dlb.trajectory import read_episode
from v2dlb.vertices import sample_vertices, mesh_path_for


def checkpoint_metadata(path):
    with Path(path).open('rb') as stream:
        length = struct.unpack('<Q', stream.read(8))[0]
        if length > 64*1024*1024: raise ValueError('Unreasonable safetensors header')
        raw = json.loads(stream.read(length)).get('__metadata__', {})
    return {k.removeprefix('user.'): v for k, v in raw.items()}


def read_rollout(path, checkpoint=None):
    table = pq.ParquetFile(path).read()
    d = table.to_pandas().sort_values(['reference_frame', 'environment_id', 'object_body_id'])
    metadata = table.schema.metadata or {}
    if checkpoint is not None and metadata.get(b'flash_chord.checkpoint_sha256', b'').decode() != sha(checkpoint):
        raise ValueError('Trajectory checkpoint digest mismatch')
    if not d.environment_valid.all() or set(d.environment_id) != {0}:
        raise ValueError('Require one deterministic valid evaluation world')
    if d.duplicated(['reference_frame', 'environment_id', 'object_body_id']).any():
        raise ValueError('Duplicate world/frame/body rollout rows')
    frames, bodies = np.sort(d.reference_frame.unique()), np.sort(d.object_body_id.unique())
    if len(d) != len(frames)*len(bodies): raise ValueError('Incomplete rectangular rollout')
    if not np.array_equal(bodies, np.arange(len(bodies))): raise ValueError('Body slots must be contiguous')
    def poses(prefix):
        columns = [prefix+'_position_'+axis for axis in 'xyz']+[prefix+'_quaternion_'+axis for axis in 'xyzw']
        values = d[columns].to_numpy(dtype=np.float64).reshape(len(frames), 1, len(bodies), 7)
        if not np.isfinite(values).all(): raise ValueError('Nonfinite rollout')
        return op._normalise_quaternions(values, prefix, ValueError)
    names = [str(d[d.object_body_id == b].object_body_name.iloc[0]) for b in bodies]
    return frames, bodies, names, poses('achieved'), poses('reference')


def diagnostics(path, track, episode, *, reference_root=None, checkpoint=None):
    config = load(KIT/'config/tracks.json')['tracks'][track]
    frames, bodies, names, achieved, reference = read_rollout(path, checkpoint)
    vertices = None
    scope = 'reconstructed_reference_proxy' if track == 'track_3' else 'tier_internal_proxy'
    reference_lineage = {}
    if track.startswith('track_2'):
        if reference_root is None: raise ValueError('Track 2 requires its own tier input root')
        root = Path(reference_root).resolve()
        tier = track.rsplit('tier', 1)[1]
        # Require the exact tier path from configuration; no silent clean-data selection.
        permitted = config.get('input_path')
        if tier == '3':
            lineage = load(root/'reconstruction_manifest.json')
            if lineage.get('input_source') != 'reconstruction' or lineage.get('tier') != 3:
                raise ValueError('Tier 3 requires a reconstruction lineage manifest')
        elif not permitted or not root.as_posix().endswith(permitted.rstrip('/')):
            raise ValueError('Reference root does not match the configured tier input path')
        released = read_episode(root/f'data/chunk-000/episode_{episode:06d}.parquet', episode)
        reference_file = root/f'data/chunk-000/episode_{episode:06d}.parquet'
        reference_lineage = {'reference_parquet': str(reference_file), 'reference_parquet_sha256': sha(reference_file),
                             'vertex_mesh_sha256': {str(mesh_path_for(root/'mesh', name)): sha(mesh_path_for(root/'mesh', name)) for name in released.object_names}}
        if tier != '1':
            # Conditioned tier-owned targets need not be one rigid transform of
            # the raw noisy input. Compare to the actual native training target,
            # never to clean Tier 1 data or an inferred host reference.
            vertices = np.stack([sample_vertices(mesh_path_for(root/'mesh', name)) for name in released.object_names])
            position, orientation = op._per_world_pose_error(achieved, reference)
            values = {name: float(op.episode_metric(name, achieved, reference, bodies, vertices, align_initial_pose=False))
                      for name in config['metrics']}
            return {'episode_index': episode, 'track': track, 'scope': 'tier_internal_proxy', 'metrics': values,
                    'trajectory_sha256': sha(path), 'mean_position_error_m': float(position.mean()),
                    'mean_orientation_error_rad': float(orientation.mean()),
                    'closeness': float(min(max(position.mean()/.1, orientation.mean()/.5), 3)),
                    'pos_margin_m': float(position.mean()-.1), 'ori_margin_rad': float(orientation.mean()-.5),
                    'scored_frames': len(frames), 'body_count': len(bodies), 'align_initial_pose': False}
        index = {int(f): i for i, f in enumerate(frames)}
        if any(int(f) not in index for f in released.frame_index): raise ValueError('Missing scored source frames')
        chosen = np.array([index[int(f)] for f in released.frame_index])
        achieved, reference = achieved[chosen], reference[chosen]
        if len(bodies) != len(released.object_names): raise ValueError('Object roster differs')
        import importlib.util
        spec = importlib.util.spec_from_file_location('cross_track_packer', KIT/'tools/pack_submission.py')
        packer = importlib.util.module_from_spec(spec); spec.loader.exec_module(packer)
        for body in bodies:
            achieved[:, 0, body], *_ = packer.convert_track2_frame(achieved[:, 0, body], reference[:, 0, body], released.pose_xyzw[:, body])
        reference = released.pose_xyzw[:, None]
        vertices = np.stack([sample_vertices(mesh_path_for(root/'mesh', name)) for name in released.object_names])
        frames = released.frame_index
        scope = 'exact_released_tier1' if tier == '1' else 'tier_internal_proxy'
    # The official Track 2 packer writes %.6f CSV. Score those exact serialized
    # values, then normalize quaternions just as the Kaggle scorer does.
    if track == 'track_2_tier1':
        achieved = np.array([float(format(v, '.6f')) for v in achieved.ravel()]).reshape(achieved.shape)
    achieved = op._normalise_quaternions(achieved, 'scored prediction', ValueError)
    reference = op._normalise_quaternions(reference, 'scored reference', ValueError)
    align = bool(config.get('align_initial_pose', False))
    aligned = op._warp_to_reference_start(achieved, reference) if align else achieved
    position, orientation = op._per_world_pose_error(aligned, reference)
    values = {name: float(op.episode_metric(name, achieved, reference, bodies, vertices, align_initial_pose=align))
              for name in config['metrics'] if name != 'add_auc' or vertices is not None}
    # Track 3 native reconstructed-reference metrics do not include confidential mesh registration/symmetries.
    return {'episode_index': episode, 'track': track, 'scope': scope, 'trajectory_sha256': sha(path), **reference_lineage,
            'metrics': values, 'mean_position_error_m': float(position.mean()),
            'mean_orientation_error_rad': float(orientation.mean()),
            'pos_margin_m': float(position.mean()-.1), 'ori_margin_rad': float(orientation.mean()-.5),
            'closeness': float(min(max(position.mean()/.1, orientation.mean()/.5), 3)),
            'scored_frames': len(frames), 'body_count': len(bodies), 'align_initial_pose': align,
            'matches_track2_csv_precision': track == 'track_2_tier1'}


def rank(record):
    m = record['metrics']
    return (m['spider_sr'], m['maniptrans_sr'], m['add_auc'], -m['mppe_cm'])


def select(records, expected=range(30)):
    if any(r['scope'] != 'exact_released_tier1' for r in records):
        raise ValueError('Exact checkpoint selection is restricted to released Tier 1 scoring')
    selected = {}
    for r in records:
        key = str(r['episode_index'])
        if key not in selected or rank(r) > rank(selected[key]): selected[key] = r
    if set(selected) != set(map(str, expected)): raise ValueError('Checkpoint candidates do not cover the complete roster')
    return {'state': 'LOCAL_CHECKPOINT_SELECTION_COMPLETE', 'episodes': selected,
            'mean_metrics': {k: float(np.mean([r['metrics'][k] for r in selected.values()])) for k in selected[next(iter(selected))]['metrics']}}


def halving(records, count):
    eligible = [r for r in records if not (r['metrics']['spider_sr'] and r['metrics']['maniptrans_sr']
                    and r['metrics'].get('add_auc', 0) >= .6)]
    return [r['episode_index'] for r in sorted(eligible, key=lambda r: (r['closeness'], r['episode_index']))[:count]]


def curriculum(config_path, steps, worlds=1024):
    import yaml
    thresholds = yaml.safe_load(Path(config_path).read_text())['thresholds']
    if steps % worlds or steps < len(thresholds)*worlds: raise ValueError('Budget must cover every stage at full world batches')
    result = [(v*steps+thresholds[-1]*worlds-1)//(thresholds[-1]*worlds)*worlds for v in thresholds]
    if any(a >= b for a, b in zip(result, result[1:])): raise ValueError('Scaled stages overlap')
    return result


def training_job(track, episode, reference, output, steps, arm='fresh', checkpoint=None, rules=None, reward_profile='baseline'):
    """Build native commands without running them. Resume retains the old curriculum."""
    robot = track.startswith('track_2')
    if track not in ('track_3', 'track_2_tier1', 'track_2_tier2', 'track_2_tier3'):
        raise ValueError('Unknown policy track')
    mount = REPO/'robotic_grounding' if robot else REPO
    def container(path): return '/repo/'+str(Path(path).resolve().relative_to(mount))
    if arm not in ('fresh', 'resume', 'warm-start'): raise ValueError('Unknown training arm')
    reference, output = Path(reference), Path(output)
    if checkpoint and output.resolve() == Path(checkpoint).resolve().parent:
        raise ValueError('New arms must preserve parent checkpoint directory')
    config_root = REPO/'robotic_grounding/flash_chord/src/flash_chord/configs'
    experiment = 'g1_recon_body_flash_sac' if robot else 'sharpa_flash_sac'
    # Resolve the actual curriculum selected by the native experiment.
    import yaml
    experiment_config = yaml.safe_load((config_root/'experiment'/f'{experiment}.yaml').read_text())
    selected_curriculum = next(v for item in experiment_config['defaults'] if isinstance(item, dict) for k, v in item.items() if k.endswith('curriculum'))
    curriculum_path = config_root/'curriculum'/f'{selected_curriculum}.yaml'
    thresholds = curriculum(curriculum_path, steps)
    extra = []
    if arm != 'fresh':
        if checkpoint is None: raise ValueError('A parent checkpoint is required')
        saved = checkpoint_metadata(checkpoint)
        config = json.loads(saved['resolved_config_json'])
        if robot and f'/track2/tier{track[-1]}/' not in config['task']['parquet']:
            raise ValueError('Parent checkpoint cannot establish same-tier lineage')
        if arm == 'resume':
            if saved.get('checkpoint_kind') != 'training_state': raise ValueError('Resume requires native learner state')
            thresholds = config['curriculum']['thresholds']
            if int(saved['environment_steps']) >= steps: raise ValueError('Resume budget must exceed saved steps')
            extra = ['training.resume=true', f'training.checkpoint_path={json.dumps(container(checkpoint))}']
        else:
            if not rules or not load(rules).get('same_tier_cross_episode_warm_start_allowed'):
                raise ValueError('Warm start requires an explicit recorded rule decision')
            if saved.get('checkpoint_kind') != 'policy': raise ValueError('Warm start requires an actor-only policy')
            extra = [f'training.initial_policy_path={json.dumps(container(checkpoint))}']
    overrides = [f'experiment={experiment}', f'task.parquet={json.dumps(container(reference))}',
                 'scene.world_count=1024', 'sim.njmax=4096', 'sim.nconmax=1024',
                 'training.replay.capacity=1000000', 'training.replay.minimum_size=102400', 'training.replay.batch_size=1024',
                 f'training.total_environment_steps={steps}', 'training.checkpoint.save_interval_environment_steps=2500608',
                 'curriculum.thresholds='+json.dumps(thresholds, separators=(',', ':')), 'reset.seed=42', 'logging.mode=disabled',
                 f'output_dir={json.dumps(container(output))}', *extra]
    if not robot:
        overrides += ['task.motion_speed=1.0', 'task.source_frame_playback=false', 'sim.fps=20', 'scene.support=false']
    if reward_profile == 'object-focused' and not robot:
        stages = yaml.safe_load(curriculum_path.read_text())['stages']
        for index in range(len(stages)):
            overrides.extend([f'curriculum.stages.{index}.objective_weights.object_keypoints=40.0',
                              f'curriculum.stages.{index}.objective_weights.hand_keypoints=0.125',
                              f'curriculum.stages.{index}.objective_weights.hand_joint_pos=0.125'])
    elif reward_profile != 'baseline':
        raise ValueError('Unknown reward profile or incompatible embodiment')
    if arm == 'resume':
        # Saved world-shaped exploration state must retain its original world count.
        overrides = [f'scene.world_count={config["scene"]["world_count"]}' if v.startswith('scene.world_count=') else v for v in overrides]
    return {'track': track, 'episode_index': episode, 'arm': arm, 'reference': str(reference.resolve()), 'output_dir': str(output.resolve()),
            'reference_sha256': sha(reference), 'parent_checkpoint': str(checkpoint) if checkpoint else None,
            'parent_checkpoint_sha256': sha(checkpoint) if checkpoint else None, 'target_steps': steps,
            'curriculum_thresholds': thresholds, 'reward_profile': reward_profile, 'overrides': overrides}


def evaluate(checkpoint, track, report):
    """Deterministic evaluator; checkpoint's saved task/config owns the rollout."""
    checkpoint, report = Path(checkpoint).resolve(), Path(report).resolve()
    mount = REPO/'robotic_grounding' if track.startswith('track_2') else REPO
    def cp(path): return '/repo/'+str(path.relative_to(mount))
    signature = {'checkpoint_sha256': sha(checkpoint), 'track': track, 'world_count': 1, 'reset_mode': 'explicit'}
    new_run(report, [checkpoint], signature)
    trajectory, metrics = report/'trajectory.parquet', report/'metrics.json'
    if load(report/'execution.json')['state'] == 'EVALUATED':
        if sha(trajectory) != load(report/'execution.json')['trajectory_sha256']: raise ValueError('Cached rollout changed')
        if sha(metrics) != load(report/'execution.json')['metrics_sha256']: raise ValueError('Cached metrics changed')
        read_rollout(trajectory, checkpoint); return trajectory
    checked_command([sys.executable, REPO/'reconstruction/scripts/run_track3_policy_stage.py', '--track', track,
        '--report', report/'native', 'scripts.evaluate_policy', f'evaluation.checkpoint={json.dumps(cp(checkpoint))}',
        'evaluation.world_count=1', 'evaluation.reset_mode=explicit', f'evaluation.metrics_output={json.dumps(cp(metrics))}',
        f'evaluation.object_trajectories_output={json.dumps(cp(trajectory))}', 'logging.mode=disabled'], report/'evaluation.log')
    read_rollout(trajectory, checkpoint)
    complete(report, 'EVALUATED', trajectory_sha256=sha(trajectory), metrics_sha256=sha(metrics))
    return trajectory


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidates', type=Path, required=True, help='JSON list: episode_index, checkpoint, optional trajectory')
    p.add_argument('--reference-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--run-evaluations', action='store_true')
    a = p.parse_args(); candidates = load(a.candidates)
    new_run(a.output, [a.candidates], {'reference_root': str(a.reference_root.resolve()), 'track': 'track_2_tier1'})
    rows = []
    for item in candidates:
        trajectory = item.get('trajectory')
        if trajectory is None:
            if not a.run_evaluations: raise ValueError('Missing trajectory; enable explicit deterministic evaluation')
            trajectory = evaluate(item['checkpoint'], 'track_2_tier1', a.output/f'episode_{item["episode_index"]:06d}'/sha(item['checkpoint']))
        row = diagnostics(trajectory, 'track_2_tier1', item['episode_index'], reference_root=a.reference_root, checkpoint=item['checkpoint'])
        metadata = checkpoint_metadata(item['checkpoint'])
        row.update(checkpoint=str(Path(item['checkpoint']).resolve()), checkpoint_sha256=sha(item['checkpoint']),
                   trajectory=str(Path(trajectory).resolve()), trained_steps=int(metadata['environment_steps']),
                   curriculum_stage=metadata.get('curriculum_stage'))
        rows.append(row)
    save(a.output/'diagnostics.json', rows)
    selection = select(rows); save(a.output/'selection.json', selection)
    complete(a.output, 'LOCAL_SELECTION_COMPLETE_NOT_UPLOADED', selected_episodes=len(selection['episodes']))


if __name__ == '__main__': main()


def checkpoint_inventory(policy_roots, output, results=None):
    candidates, seen = [], set()
    for root in map(Path, policy_roots):
        for path in sorted(root.glob('episode_*/policy_*.safetensors')):
            episode = int(path.parent.name.removeprefix('episode_'))
            digest = sha(path)
            if (episode, digest) in seen: continue
            seen.add((episode, digest))
            item = {'episode_index': episode, 'checkpoint': str(path.resolve())}
            if results:
                trajectory = Path(results)/f'episode_{episode:06d}.parquet'
                if trajectory.exists():
                    metadata = pq.read_metadata(trajectory).metadata or {}
                    if metadata.get(b'flash_chord.checkpoint_sha256', b'').decode() == digest:
                        item['trajectory'] = str(trajectory.resolve())
            candidates.append(item)
    if {r['episode_index'] for r in candidates} != set(range(30)): raise ValueError('Checkpoint inventory must cover all 30 Tier 1 episodes')
    save(output, candidates)
    return candidates


def inventory_main():
    p = argparse.ArgumentParser(description='Inventory all saved Tier 1 actor checkpoints without evaluating them')
    p.add_argument('--policy-roots', type=Path, nargs='+', required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--existing-results', type=Path)
    a = p.parse_args(); checkpoint_inventory(a.policy_roots, a.output, a.existing_results)


def choose_arm(records):
    arms = {name: [r for r in records if r['arm'] == name] for name in sorted({r['arm'] for r in records})}
    roster = {r['episode_index'] for r in records}
    if len(roster) != 3 or any(len(v) != 3 or {r['episode_index'] for r in v} != roster for v in arms.values()):
        raise ValueError('Pilot arms require the same three episodes once each')
    if any(r['scope'] != 'exact_released_tier1' for r in records): raise ValueError('Tier 1 pilot needs exact local scoring')
    means = {name: {'auc': float(np.mean([r['metrics']['add_auc'] for r in group])),
                    'spider_count': int(sum(r['metrics']['spider_sr'] for r in group))} for name, group in arms.items()}
    return {'chosen_arm': max(means, key=lambda name: (means[name]['auc'], means[name]['spider_count'])), 'arms': means}


def collapse_decision(best, latest, learning_rate, retries=0):
    if latest['metrics']['add_auc'] >= best['metrics']['add_auc']-.1:
        return {'action': 'CONTINUE'}
    return {'action': 'REVERT_AND_RETRY' if retries == 0 else 'STOP_EPISODE',
            'checkpoint': best['checkpoint'], 'checkpoint_sha256': best['checkpoint_sha256'],
            'learning_rate': learning_rate*.5, 'retries': retries+1}
