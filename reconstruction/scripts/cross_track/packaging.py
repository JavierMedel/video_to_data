"""Assemble complete reviewed candidates; preserve every rejected parent episode."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
from .common import REPO, KIT, new_run, load, sha, save, checked_command, complete, input_digest


def assemble_track1(parent, parent_packet, reviews, scored_path, output, *, minimum_promoted=25, commit=None):
    parent, output = Path(parent), Path(output)
    decisions, scored = load(reviews), load(scored_path)
    if set(decisions) != set(map(str, range(30))): raise ValueError('Require a decision for all 30 episodes')
    inputs = [parent_packet, reviews, scored_path]+[parent/(f'episode_{i:06d}'+suffix) for i in range(30) for suffix in ('.npz', '_object.glb')]
    new_run(output, inputs, {'minimum_promoted': minimum_promoted, 'producing_commit': commit})
    destination = output/'submission_episodes'; destination.mkdir(exist_ok=True)
    promotions = []; artifacts = {}
    for episode in range(30):
        stem = f'episode_{episode:06d}'; decision = decisions[str(episode)]
        source = parent
        if decision.get('object'):
            candidate = Path(decision['object']); record = load(candidate/'execution.json')
            if record['state'] != 'CANDIDATE_REQUIRES_PACKED_VALIDATION': raise ValueError('Object candidate lacks a completed review')
            if record.get('evaluated_frames') != scored[str(episode)]: raise ValueError('Object evidence does not cover every scored frame')
            if record.get('lambda_policy') != 'gcv_capped' or record.get('lambda_cap') is None:
                raise ValueError('Promotion requires a held-out-calibrated lambda policy; max_passing is a diagnostic control')
            if not record.get('serialized_evidence_review', {}).get('passed'): raise ValueError('Serialized object candidate lacks passing evidence')
            lam = record.get('selected_lambda')
            if lam is None or not any(r.get('lambda') == lam and r.get('passed') for r in record['reviews']):
                raise ValueError('Rejected object candidate cannot be promoted')
            if sha(candidate/(stem+'.npz')) != record['npz_sha256'] or sha(candidate/(stem+'_object.glb')) != record['mesh_sha256']:
                raise ValueError('Candidate changed after evidence review')
            for path, digest in {**record['inputs_sha256'], **record['implementation_sha256']}.items():
                if sha(path) != digest: raise ValueError('Review input changed')
            expected_source = (Path(decision['human']) if decision.get('human') else parent) / (stem+'.npz')
            if input_digest(record, expected_source) != sha(expected_source):
                raise ValueError('Object fit does not descend from the selected parent/human')
            if input_digest(record, parent/(stem+'_object.glb')) != sha(parent/(stem+'_object.glb')):
                raise ValueError('Object fit used an unreviewed parent mesh')
            before = dict(np.load(parent/(stem+'.npz'), allow_pickle=False))
            after = dict(np.load(candidate/(stem+'.npz'), allow_pickle=False))
            if before.keys() != after.keys(): raise ValueError('Bundle fields changed')
            for key in before:
                if key not in ('pose', 'object_translation') and not np.array_equal(before[key], after[key]):
                    raise ValueError('Unreviewed bundle field changed: '+key)
            if before['pose'][scored[str(episode)][0]].tobytes() != after['pose'][scored[str(episode)][0]].tobytes():
                raise ValueError('Alignment frame changed')
            if any(not np.array_equal(before[k], after[k]) for k in ('shape', 'scales')): raise ValueError('Identity changed')
            if any(not np.isfinite(after[k]).all() for k in ('pose', 'scales', 'shape', 'object_rotation', 'object_translation', 'object_scale')):
                raise ValueError('Nonfinite candidate')
            if not np.array_equal(before['pose'], after['pose']):
                human = Path(decision['human']); human_review = load(human/'execution.json')
                if human_review['state'] != 'READY_FOR_COUPLED_OBJECT_REVIEW' or not human_review.get('passed'):
                    raise ValueError('Human change lacks geometry/reprojection gates')
                if human_review.get('evaluated_frames') != scored[str(episode)]: raise ValueError('Human evidence frame grid differs')
                if input_digest(human_review, parent/(stem+'.npz')) != sha(parent/(stem+'.npz')):
                    raise ValueError('Human fit does not descend from this parent')
                if sha(human/(stem+'.npz')) != human_review['npz_sha256']:
                    raise ValueError('Human candidate changed after review')
                human_bundle = dict(np.load(human/(stem+'.npz'), allow_pickle=False))
                if not np.array_equal(human_bundle['pose'], after['pose']): raise ValueError('Coupled fit changed reviewed human')
            source = candidate; promotions.append(episode)
        for suffix in ('.npz', '_object.glb'):
            name = stem+suffix; target = destination/name
            if target.exists() and sha(target) != sha(source/name): raise ValueError('Existing assembly artifact differs')
            shutil.copyfile(source/name, target); artifacts[name] = sha(target)
    if len(promotions) < minimum_promoted:
        complete(output, 'INSUFFICIENT_VALIDATED_EPISODES', promoted=promotions)
        raise ValueError(f'Only {len(promotions)} reviewed episodes; required {minimum_promoted}')
    result = {'state': 'COMPLETE_PAIRS_REQUIRES_OFFICIAL_PACK', 'promoted': promotions,
              'retained_parent': sorted(set(range(30))-set(promotions)), 'artifact_sha256': artifacts,
              'parent_packet_sha256': sha(parent_packet), 'uploaded': False}
    if commit:
        import re
        if not re.fullmatch('[0-9a-f]{40}', commit): raise ValueError('Require a full producing commit SHA')
        subprocess.run(['git', 'cat-file', '-e', commit+'^{commit}'], cwd=REPO, check=True)
        critical = sorted(str(p.relative_to(REPO)) for p in Path(__file__).parent.glob('*.py'))
        subprocess.run(['git', 'ls-files', '--error-unmatch', *critical], cwd=REPO, check=True, stdout=subprocess.DEVNULL)
        subprocess.run(['git', 'diff', '--quiet', commit, '--', *critical], cwd=REPO, check=True)
        packet = output/'track1_candidate.parquet'
        if packet.exists(): raise ValueError('Existing pack preserved; use a new version for another packing attempt')
        checked_command([sys.executable, KIT/'eval_reconstruction.py', '--episodes', destination,
                         '--sample', KIT/'data/track_1_sample_submission.parquet', '--commit',
                         'https://github.com/JavierMedel/video_to_data/commit/'+commit, '--out', packet], output/'packing.log')
        result.update(state='PACKED_REQUIRES_OFFICIAL_DRY_RUN', packet=str(packet), packet_sha256=sha(packet))
    save(output/'package_validation.json', result)
    complete(output, result['state'], **{k: v for k, v in result.items() if k != 'state'})
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('parent', 'parent-packet', 'reviews', 'scored', 'output'): p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--minimum-promoted', type=int, default=25); p.add_argument('--commit')
    a = p.parse_args(); assemble_track1(a.parent, a.parent_packet, a.reviews, a.scored, a.output, minimum_promoted=a.minimum_promoted, commit=a.commit)


if __name__ == '__main__': main()


def pack_selected_policy(selection_path, output, dataset_root, commit):
    """Stage only deterministic, hash-verified selected Tier 1 checkpoint rollouts."""
    import re
    import pyarrow as pa
    import pyarrow.parquet as pq
    from .policy import read_rollout
    from .common import kit_imports
    kit_imports()
    from v2dlb.trajectory import read_episode
    selection = load(selection_path); output = Path(output)
    if selection.get('state') != 'LOCAL_CHECKPOINT_SELECTION_COMPLETE' or set(selection['episodes']) != set(map(str, range(30))):
        raise ValueError('Require a complete deterministic Tier 1 selection')
    if not re.fullmatch('[0-9a-f]{40}', commit): raise ValueError('Require the full producing commit')
    subprocess.run(['git', 'cat-file', '-e', commit+'^{commit}'], cwd=REPO, check=True)
    new_run(output, [selection_path], {'commit': commit, 'track': 'track_2_tier1'})
    exports = output/'results'; exports.mkdir(exist_ok=True)
    for key, record in selection['episodes'].items():
        if record['scope'] != 'exact_released_tier1' or sha(record['checkpoint']) != record['checkpoint_sha256'] or sha(record['trajectory']) != record['trajectory_sha256']:
            raise ValueError('Selected checkpoint or rollout digest changed')
        if sha(record['reference_parquet']) != record['reference_parquet_sha256']:
            raise ValueError('Released reference changed after checkpoint selection')
        if any(sha(path) != digest for path, digest in record['vertex_mesh_sha256'].items()):
            raise ValueError('Canonical ADD mesh changed after selection')
        read_rollout(record['trajectory'], record['checkpoint'])
        reference = read_episode(Path(dataset_root)/f'track_2/tier_1_multiview_caption/data/chunk-000/episode_{int(key):06d}.parquet', int(key))
        table = pq.ParquetFile(record['trajectory']).read()
        if len(reference.object_names) != 1: raise ValueError('Unexpected Tier 1 object roster')
        field = table.schema.field('object_body_name')
        labels = pa.array([reference.object_names[0]]*len(table)).dictionary_encode().cast(field.type)
        staged = table.set_column(table.schema.get_field_index('object_body_name'), field, labels)
        for column in table.column_names:
            if column != 'object_body_name' and not table[column].equals(staged[column]): raise ValueError('Pose values changed during export')
        pq.write_table(staged, exports/f'episode_{int(key):06d}.parquet')
    packet = output/'track2_tier1.csv'
    if packet.exists(): raise ValueError('Preserve an existing packet; choose a new version')
    checked_command([sys.executable, KIT/'tools/pack_submission.py', '--track', 'track_2_tier1', '--results', exports,
                     '--dataset-root', dataset_root, '--code-commit', 'https://github.com/JavierMedel/video_to_data/commit/'+commit,
                     '--eval-script', REPO/'robotic_grounding/flash_chord/scripts/evaluate_policy.py', '--out', packet], output/'packing.log')
    complete(output, 'PACKED_REQUIRES_OFFICIAL_DRY_RUN', packet=str(packet), packet_sha256=sha(packet))
    return packet


def policy_main():
    p = argparse.ArgumentParser(description='Pack selected Tier 1 checkpoints without uploading')
    for name in ('selection', 'output', 'dataset-root'): p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--commit', required=True)
    a = p.parse_args(); pack_selected_policy(a.selection, a.output, a.dataset_root, a.commit)
