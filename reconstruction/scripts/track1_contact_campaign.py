"""Run isolated contact-constrained smoothing on all 30 v6 episodes.

Two GPU processes may run concurrently; no agent delegation, packaging or
uploading occurs. Resume verifies completed artifact and input hashes.
"""
import argparse
import concurrent.futures
import datetime
import fcntl
import hashlib
import json
import os
import subprocess
import threading
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--campaign', default='contact_smoothing_v8')
    p.add_argument('--workers', type=int, choices=[1, 2], default=2)
    p.add_argument('--status', action='store_true', help='Read current campaign progress without launching work')
    a = p.parse_args(); root = a.root.resolve()
    assert Path(a.campaign).name == a.campaign and a.campaign not in ('.', '..')
    data = root / 'data/v2d_challenge/track_1'
    report = root / 'incident_reports/track1_improvement' / a.campaign
    report.mkdir(parents=True, exist_ok=True)
    if a.status:
        state = json.loads((report/'campaign.json').read_text())
        running = []
        for episode, item in state['episodes'].items():
            if item['state'] != 'RUNNING': continue
            path = report/f'episode_{int(episode):06d}'/'status.json'
            detail = json.loads(path.read_text()) if path.exists() else {}
            running.append({'episode': int(episode), 'stage': detail.get('state', 'STARTING'),
                            'evaluated_frame_options': detail.get('completed_frame_options', 0),
                            'total_frame_options': item['frames']*2,
                            'evaluator': detail.get('evaluator_implementation', 'legacy eager')})
        print(json.dumps({'state': state['state'], 'completed': sum(d['state']=='COMPLETE' for d in state['episodes'].values()),
                          'local_pass': sorted(int(i) for i,d in state['episodes'].items() if d.get('verdict')=='LOCAL_PASS'),
                          'retained_v6': sorted(int(i) for i,d in state['episodes'].items() if d.get('verdict')=='RETAIN_V6'),
                          'running': running, 'failed': {i:d['error'] for i,d in state['episodes'].items() if d['state']=='FAILED'}}, indent=2))
        return
    lock = (report / 'runner.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    controls = data / 'cdo_v6_experiments/candidate_episodes'
    recorded = json.loads((root / 'incident_reports/track1_improvement/cdo_v6/candidate_validation.json').read_text())['artifact_sha256']
    actual = {f'episode_{i:06d}'+suffix: sha(controls/(f'episode_{i:06d}'+suffix)) for i in range(30) for suffix in ('.npz', '_object.glb')}
    assert actual == recorded, 'A submitted v6 control changed'
    frames = {i: len(np.load(controls/f'episode_{i:06d}.npz', allow_pickle=False)['pose']) for i in range(30)}
    state = {'state': 'RUNNING', 'scope': list(range(30)), 'control_version': 6,
             'workers': a.workers, 'control_sha256': actual, 'episodes': {}, 'uploaded': False,
             'hand_review_scope': 'Exact all-video-frame finite-option mesh checks reused after path selection; no independent rerender claim.'}
    mutex = threading.Lock()

    def save():
        state['updated_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        temporary = report / 'campaign.tmp.json'
        temporary.write_text(json.dumps(state, indent=2)+'\n')
        os.replace(temporary, report / 'campaign.json')

    def verified(episode, status, folder):
        d = json.loads(status.read_text()); stem = f'episode_{episode:06d}'
        assert d['episode'] == episode and d['state'] == 'LOCAL_VALIDATION_COMPLETE' and d['source_hashes_preserved']
        assert all(sha(root/path) == value for path, value in d['inputs'].items())
        assert d['candidate_npz_sha256'] == sha(folder/(stem+'.npz'))
        assert d['candidate_mesh_sha256'] == sha(folder/(stem+'_object.glb'))
        scored = json.loads((root/'incident_reports/track1_scored_frames.json').read_text())[str(episode)]
        assert d['image_review']['frames'] == scored and d['hand_review']['frames'] == scored
        assert (folder/'exact_candidate_comparison.jpg').exists()
        return {'state': 'COMPLETE', 'verdict': 'LOCAL_PASS' if d['local_gates_pass'] else 'RETAIN_V6',
                'source': str(folder.relative_to(root)), 'status': str(status.relative_to(root)),
                'frames': frames[episode], 'selected_smoothed_frames': d['selected_smoothed_frames'],
                'summary': d['summary'], 'candidate_npz_sha256': d['candidate_npz_sha256']}

    jobs = []
    for episode in range(30):
        stem = f'episode_{episode:06d}'
        status = report / stem / 'status.json'; folder = data/a.campaign/stem
        if episode == 7:
            status = root/'incident_reports/track1_improvement/episode7_contact_smoothing/status.json'
            folder = data/'episode7_contact_smoothing'/stem
        if status.exists() and json.loads(status.read_text()).get('state') == 'LOCAL_VALIDATION_COMPLETE' and (folder/'exact_candidate_comparison.jpg').exists():
            state['episodes'][str(episode)] = verified(episode, status, folder)
            state['episodes'][str(episode)]['reused'] = True
        else:
            state['episodes'][str(episode)] = {'state': 'PENDING', 'frames': frames[episode]}
            jobs.append(episode)
    save()

    def run(episode):
        stem = f'episode_{episode:06d}'; folder = report/stem; folder.mkdir(parents=True, exist_ok=True)
        log = folder/'run.log'
        with mutex:
            state['episodes'][str(episode)].update(state='RUNNING', log=str(log.relative_to(root))); save()
        print('CONTACT_CAMPAIGN_START', episode, frames[episode], flush=True)
        command = ['docker', 'run', '--rm', '--gpus', 'all', '--user', '1000:1000',
                   '--name', f'v2d-{a.campaign}-{episode:06d}', '--entrypoint', 'python',
                   '-e', 'PYTHONUNBUFFERED=1', '-e', 'XDG_CACHE_HOME=/tmp/contact-campaign-cache',
                   '-e', 'MPLCONFIGDIR=/tmp/contact-campaign-mpl', '-v', f'{root}:/repo',
                   'v2d_cari4d:blackwell', '/repo/scripts/track1_episode7_contact_smoothing.py',
                   '--episode', str(episode), '--campaign', a.campaign, '--validate', '--fit-hand-review', '--mask-sheet']
        try:
            with log.open('a') as handle:
                result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f'exit {result.returncode}: ' + '\n'.join(log.read_text(errors='replace').splitlines()[-12:]))
            record = verified(episode, folder/'status.json', data/a.campaign/stem)
        except Exception as error:
            record = {'state': 'FAILED', 'error': str(error), 'log': str(log.relative_to(root)), 'frames': frames[episode]}
        with mutex:
            state['episodes'][str(episode)] = record; save()
        print('CONTACT_CAMPAIGN_EPISODE', episode, record['state'], record.get('verdict', ''), flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        list(pool.map(run, sorted(jobs, key=lambda i: frames[i])))
    assert all(sha(controls/name) == value for name, value in actual.items())
    state['state'] = 'COMPLETE' if all(d['state'] == 'COMPLETE' for d in state['episodes'].values()) else 'FAILED'
    state['local_pass'] = sorted(int(i) for i, d in state['episodes'].items() if d.get('verdict') == 'LOCAL_PASS')
    state['retained_v6'] = sorted(int(i) for i, d in state['episodes'].items() if d.get('verdict') == 'RETAIN_V6')
    state['source_hashes_preserved'] = True; save()
    print('CONTACT_CAMPAIGN_COMPLETE', state['state'], 'local_pass', state['local_pass'], 'retained_v6', state['retained_v6'], flush=True)
    if state['state'] != 'COMPLETE': raise SystemExit(1)


if __name__ == '__main__':
    main()
