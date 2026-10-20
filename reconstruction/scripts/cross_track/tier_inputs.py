"""Tier-owned source preparation and robust observation-only conditioning."""
import argparse
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation
from .common import KIT, REPO, load, sha, save, new_run, complete
from .temporal import smooth, smooth_rotations


def validate_root(dataset, tier):
    dataset = Path(dataset).resolve()
    config = load(KIT/'config/tracks.json')['tracks'][f'track_2_tier{tier}']
    if tier in (1, 2):
        if not dataset.as_posix().endswith(config['input_path']):
            raise ValueError('Input root is not the configured tier input path')
    elif tier == 3:
        provenance = load(dataset/'reconstruction_manifest.json')
        if provenance.get('tier') != 3 or provenance.get('input_source') != 'reconstruction':
            raise ValueError('Tier 3 has no supplied trajectory; provide reconstruction provenance')
        if not provenance.get('outputs_sha256') or not provenance.get('video_inputs_sha256'):
            raise ValueError('Tier 3 lineage requires reconstructed outputs and source video hashes')
        if provenance.get('coordinate_frame') != 'released_track2_opencv_world':
            raise ValueError('Tier 3 needs independently established released-world camera coordinates')
        for path, digest in provenance['outputs_sha256'].items():
            source = (dataset/path).resolve()
            if not source.is_relative_to(dataset) or sha(source) != digest:
                raise ValueError('Reconstruction output lineage mismatch')
        for path, digest in provenance['video_inputs_sha256'].items():
            if Path(path).suffix.lower() not in ('.mp4', '.mkv', '.mov') or sha(path) != digest:
                raise ValueError('Invalid Tier 3 source video lineage')
    else: raise ValueError('Unknown Track 2 tier')
    return dataset


def condition_positions(values, lam, *, noise_limit=True):
    values = np.asarray(values, float)
    frames = np.arange(len(values))
    candidate = smooth(values, frames, lam)
    residual = np.diff(values, n=2, axis=0)
    scale = float(np.median(np.linalg.norm(residual, axis=-1))/np.sqrt(6))
    if noise_limit:
        delta = candidate-values
        ratio = np.minimum(1., max(scale, 1e-6)/np.maximum(np.linalg.norm(delta, axis=1), 1e-12))
        candidate = values+ratio[:, None]*delta
    candidate[0] = values[0]
    return candidate, scale


def condition_source(source, output, lam=10.):
    """New source directory; source files and lineage are never overwritten."""
    import shutil
    source, output = Path(source), Path(output)
    manifest = load(source/'source_manifest.json')
    if manifest.get('tier') not in (2, 3): raise ValueError('Conditioning is only for explicitly tier-owned inputs')
    for name, digest in manifest['output_sha256'].items():
        if sha(source/name) != digest: raise ValueError('Source hash mismatch')
    new_run(output, [source/'source_manifest.json', *[source/p for p in manifest['output_sha256']]], {'lambda': lam})
    for name in manifest['output_sha256']:
        target = output/name; target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source/name, target)
    b = dict(np.load(source/'soma_params.npz', allow_pickle=False))
    b['transl'], noise = condition_positions(b['transl'], lam)
    frames = np.arange(len(b['poses']))
    for joint in range(b['poses'].shape[1]):
        original = b['poses'][:, joint].copy()
        matrices = Rotation.from_rotvec(original).as_matrix()
        filtered = smooth_rotations(matrices, frames, lam)
        # Limit angular changes to the observed high-frequency angular noise scale.
        velocity = Rotation.from_matrix(matrices[1:]@matrices[:-1].transpose(0, 2, 1)).as_rotvec()
        limit = max(1e-6, float(np.median(np.linalg.norm(np.diff(velocity, axis=0), axis=1))/np.sqrt(6)))
        delta = Rotation.from_matrix(filtered@matrices.transpose(0, 2, 1)).as_rotvec()
        delta *= np.minimum(1., limit/np.maximum(np.linalg.norm(delta, axis=1), 1e-12))[:, None]
        b['poses'][:, joint] = Rotation.from_matrix(Rotation.from_rotvec(delta).as_matrix()@matrices).as_rotvec()
        b['poses'][0, joint] = original[0]
    np.savez_compressed(output/'soma_params.npz', **b)
    poses = np.load(source/'poses.npy'); result = poses.copy()
    result[:, :3, 3], object_noise = condition_positions(poses[:, :3, 3], lam)
    filtered = smooth_rotations(poses[:, :3, :3], frames, lam)
    original_r = poses[:, :3, :3]
    velocity = Rotation.from_matrix(original_r[1:]@original_r[:-1].transpose(0, 2, 1)).as_rotvec()
    limit = max(1e-6, float(np.median(np.linalg.norm(np.diff(velocity, axis=0), axis=1))/np.sqrt(6)))
    delta = Rotation.from_matrix(filtered@original_r.transpose(0, 2, 1)).as_rotvec()
    delta *= np.minimum(1., limit/np.maximum(np.linalg.norm(delta, axis=1), 1e-12))[:, None]
    result[:, :3, :3] = Rotation.from_rotvec(delta).as_matrix()@original_r
    result[0] = poses[0]
    np.save(output/'poses.npy', result)
    record = {**manifest, 'conditioned_from_sha256': sha(source/'source_manifest.json'),
              'conditioning': {'lambda': lam, 'human_position_noise_m': noise, 'object_position_noise_m': object_noise,
                               'scope': 'Observation-only temporal fit; kinematic/contact feasibility must be checked after retargeting'},
              'output_sha256': {name: sha(output/name) for name in manifest['output_sha256']}}
    save(output/'source_manifest.json', record)
    complete(output, 'CONDITIONED_REQUIRES_RETARGET_VALIDATION')
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tier', type=int, choices=(1, 2, 3), required=True)
    p.add_argument('--dataset', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--episodes', type=int, nargs='+', default=list(range(30)))
    a = p.parse_args(); dataset = validate_root(a.dataset, a.tier)
    sys.path.insert(0, str(REPO/'robotic_grounding/scripts'))
    from prepare_track2_sources import prepare
    prepare(dataset, a.output, a.episodes, tier=a.tier)


if __name__ == '__main__': main()


def inventory(data, tier, output):
    """Hash every source/retarget pair; never infer Tier 3 from a clean-tier copy."""
    sys.path.insert(0, str(REPO/'robotic_grounding/scripts'))
    from retarget_track2_tier1 import verify_motion
    data = Path(data).resolve(); rows = []
    for path in sorted((data/'source_episodes').glob('*/source_manifest.json')):
        source = path.parent; manifest = load(path)
        if manifest.get('tier', 1) != tier: raise ValueError('Cross-tier source in inventory')
        if tier == 3 and manifest.get('input_source') != 'reconstruction': raise ValueError('Tier 3 source lacks reconstruction lineage')
        for name, digest in manifest['output_sha256'].items():
            if sha(source/name) != digest: raise ValueError('Source output digest mismatch')
        target = data/f'assets/human_motion_data/whole_body/challenge_tier{tier}/sequence_id={source.name}/robot_name=g1/data.parquet'
        verified = verify_motion(target, source)
        raw = np.load(source/'poses.npy', allow_pickle=False)
        frame_reference = data/'frame_references'/f'episode_{manifest["episode_index"]:06d}.npz'
        frame_reference.parent.mkdir(parents=True, exist_ok=True)
        poses = np.c_[raw[:, :3, 3], Rotation.from_matrix(raw[:, :3, :3]).as_quat()][:, None]
        if frame_reference.exists():
            stored = dict(np.load(frame_reference, allow_pickle=False))
            if not np.array_equal(stored['pose_xyzw'], poses): raise ValueError('Existing packer frame reference changed')
        else:
            np.savez_compressed(frame_reference, coordinate_frame='released_track2_world',
                                episode_index=manifest['episode_index'], frame_index=np.arange(len(raw)), pose_xyzw=poses)
        rows.append({'episode_index': manifest['episode_index'], 'sequence_id': manifest['sequence_id'],
                     'frames': manifest['frames'], 'reference': str(target), 'reference_sha256': sha(target),
                     'source_manifest_sha256': sha(path), 'input_source': manifest.get('input_source', 'released_tier_input'),
                     'frame_reference': str(frame_reference), 'frame_reference_sha256': sha(frame_reference),
                     'tier': tier, 'verification': verified})
    if len(rows) != 30 or {r['episode_index'] for r in rows} != set(range(30)):
        raise ValueError('The inventory must cover 30 unique Track 2 episodes')
    result = {'state': 'RETARGET_INVENTORY_VERIFIED', 'tier': tier, 'episodes': rows}
    save(output, result); return result


def inventory_main():
    p = argparse.ArgumentParser(description='Verify a complete tier-owned retarget inventory')
    p.add_argument('--data-root', type=Path, required=True); p.add_argument('--tier', type=int, choices=(1, 2, 3), required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); inventory(a.data_root, a.tier, a.output)
