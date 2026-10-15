"""Review completed campaign outputs without refitting or uploading.

Add a conservative motion screen on consecutive scored triples. Preserve the
original all-video-frame screen and its results as separately labelled evidence.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh

from track1_contact_smoothing import scored_motion_diagnostics
from track1_episode7_ablation import verify_bundle


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--campaign', default='contact_smoothing_v8')
    a = p.parse_args(); root = a.root.resolve()
    assert Path(a.campaign).name == a.campaign and a.campaign not in ('.', '..')
    report = root/'incident_reports/track1_improvement'/a.campaign
    campaign = json.loads((report/'campaign.json').read_text())
    visual = json.loads((report/'visual_reviews.json').read_text()) if (report/'visual_reviews.json').exists() else {}
    scored = json.loads((root/'incident_reports/track1_scored_frames.json').read_text())
    controls = root/'data/v2d_challenge/track_1/cdo_v6_experiments/candidate_episodes'
    decisions = {}
    for episode in range(30):
        item = campaign['episodes'][str(episode)]
        if item['state'] != 'COMPLETE':
            decisions[str(episode)] = {'verdict': item['state']}; continue
        stem = f'episode_{episode:06d}'; folder = root/item['source']
        status = json.loads((root/item['status']).read_text())
        original = dict(np.load(controls/(stem+'.npz'), allow_pickle=False))
        candidate = dict(np.load(folder/(stem+'.npz'), allow_pickle=False)); verify_bundle(original, candidate)
        for suffix in ('.npz', '_object.glb'):
            assert sha(controls/(stem+suffix)) == campaign['control_sha256'][stem+suffix]
        assert sha(folder/(stem+'.npz')) == status['candidate_npz_sha256']
        assert sha(folder/(stem+'_object.glb')) == sha(controls/(stem+'_object.glb'))
        mesh = trimesh.load(controls/(stem+'_object.glb'), force='mesh', process=False)
        metrics = {label: scored_motion_diagnostics(bundle['object_rotation'], bundle['object_translation'], mesh.vertices*float(bundle['object_scale']), np.array(scored[str(episode)]))
                   for label, bundle in [('control', original), ('candidate', candidate)]}
        before, after = metrics['control'], metrics['candidate']; image = status['image_review']; depth = image['common_depth']
        motion_gain = before['sampled_vertex_acceleration_cm_proxy'] is not None and before['sampled_vertex_acceleration_cm_proxy'] > 0 and after['sampled_vertex_acceleration_cm_proxy'] <= before['sampled_vertex_acceleration_cm_proxy']*.95
        image_gain = image['summary']['candidate']['visible_iou_mean'] - image['summary']['control']['visible_iou_mean'] >= .005
        depth_gain = depth['supported_frames'] >= 4 and depth['before_cm']-depth['after_cm'] >= 1.
        scored_meaningful = bool(motion_gain or image_gain or depth_gain)
        guard = all(after[k] is None or after[k] <= max(.1, before[k]*1.1) for k in ('translation_acceleration_cm_proxy', 'sampled_vertex_acceleration_cm_proxy'))
        guard &= all(status['summary']['final_candidate'][k] <= max(.1, status['summary']['submitted_v6'][k]*1.1) for k in ('translation_acceleration_cm_proxy', 'sampled_vertex_acceleration_cm_proxy'))
        changed = not np.array_equal(original['object_translation'], candidate['object_translation'])
        numerical = bool(status['local_gates_pass'] and scored_meaningful and guard and changed)
        review = visual.get(str(episode), {}); reviewed = bool(review.get('review_complete'))
        if reviewed:
            for path, value in review.get('artifact_sha256', {}).items(): assert sha(root/path) == value
            if 'source_record' in review: assert sha(root/review['source_record']) == review['source_record_sha256']
        verdict = ('PROMOTE' if reviewed else 'VISUAL_REVIEW_PENDING') if numerical else 'RETAIN_V6'
        reasons = []
        if not status['local_gates_pass']: reasons.append('Original full-image/depth/hand/motion screen failed')
        if not scored_meaningful: reasons.append('No meaningful retained gain on consecutive scored triples, image IoU or cached depth')
        if not guard: reasons.append('Motion regression guard failed')
        if not changed: reasons.append('No object translation changed')
        if numerical and not reviewed: reasons.append('Sampled visual review pending')
        if verdict == 'PROMOTE': reasons.append(review['reason'])
        decisions[str(episode)] = {'verdict': verdict, 'gates': {'original_numerical_screen': bool(status['local_gates_pass']),
                'scored_meaningful_gain': scored_meaningful, 'motion_regression_guard': bool(guard), 'translation_changed': changed, 'visual_review_complete': reviewed},
                'source': item['source'], 'status': item['status'], 'status_sha256': sha(root/item['status']),
                'scored_motion': metrics, 'summary': status['summary'], 'reason': '; '.join(reasons),
                'candidate_npz_sha256': status['candidate_npz_sha256'], 'candidate_mesh_sha256': status['candidate_mesh_sha256']}
    result = {'state': 'COMPLETE' if campaign['state']=='COMPLETE' else 'PARTIAL', 'episodes': decisions,
              'promoted': [int(i) for i,d in decisions.items() if d['verdict']=='PROMOTE'],
              'pending_visual_review': [int(i) for i,d in decisions.items() if d['verdict']=='VISUAL_REVIEW_PENDING'],
              'retained_v6': [int(i) for i,d in decisions.items() if d['verdict']=='RETAIN_V6'],
              'scope': 'Original full-video screen retained, plus meaningful gain and regression guards on consecutive scored triples. Proxies are not official error scores.'}
    temporary = report/'selection_review.tmp.json'
    temporary.write_text(json.dumps(result, indent=2)+'\n'); temporary.replace(report/'selection_review.json')
    metadata = {row['episode_index']: row['object'] for row in map(json.loads,(root/'data/v2d_challenge/track_1/meta/episodes_metadata.jsonl').read_text().splitlines())}
    lines = ['# Contact-constrained smoothing decisions', '',
             f"State: {result['state']}. Promoted after scored-frame and visual review: {len(result['promoted'])}. Retain v6: {len(result['retained_v6'])}.", '',
             'Motion percentages below are reductions in local magnitude proxies on consecutive scored triples, not official errors. The original all-video-frame screen is also required.', '',
             '| Episode | Object | Translation gain | Surface gain | Decision |', '|---:|---|---:|---:|---|']
    for episode in range(30):
        item = decisions[str(episode)]; metrics = item.get('scored_motion')
        gains = ['pending', 'pending']
        if metrics:
            for index,key in enumerate(('translation_acceleration_cm_proxy','sampled_vertex_acceleration_cm_proxy')):
                before,after=metrics['control'][key],metrics['candidate'][key]
                gains[index] = f'{100*(1-after/before):+.2f}%' if before is not None and before>0 else 'unavailable'
        lines.append(f"| {episode} | {metadata[episode]} | {gains[0]} | {gains[1]} | {item['verdict']} |")
    lines += ['', 'Detailed gate results, rejection reasons and artifact fingerprints: `selection_review.json`. No leaderboard upload is performed by this review.']
    (report/'episode_decisions.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='episodes'}, indent=2))


if __name__ == '__main__': main()
