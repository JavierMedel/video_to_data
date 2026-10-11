"""Read-only artifact checks; write a durable recovery inventory."""
import json
import re
import sys
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import trimesh

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/v2d_challenge/track_1'
KIT = ROOT / 'v2d_submission_kit/v2d_submission_kit'
sys.path.insert(0, str(KIT))
from v2dlb.mhr_submission import _rotations, EPISODE_FRAME

def mesh_check(path):
    mesh = trimesh.load(path, force='mesh')
    assert len(mesh.vertices) and len(mesh.faces)
    assert np.isfinite(mesh.vertices).all() and mesh.area > 0
    return mesh.extents.tolist()

sample = pd.read_parquet(KIT / 'data/track_1_sample_submission.parquet', columns=['row_id'])
frames = {}
for row in sample.row_id:
    _, ep, frame, _, _ = row.split('_')
    if int(frame) != EPISODE_FRAME:
        frames[int(ep)] = max(frames.get(int(ep), 0), int(frame))
metadata = [json.loads(line) for line in (DATA / 'meta/episodes_metadata.jsonl').read_text().splitlines()]
markers = list(DATA.rglob('.stages/*.json'))
details = []
table = ['| Episode | Object | S01 | S02 | S03 | S04 | S05 | S06 | S07 | S08 | Mesh | NPZ | GLB | Next Action |', '|' + '---|' * 14]
full = valid = 0
for meta in metadata:
    i = meta['episode_index']; stem = f'episode_{i:06d}'; work = DATA / stem
    relevant = [p for p in markers if stem in str(p)]
    stages = ['MISSING'] * 8
    evidence = []
    for path in relevant:
        marker = json.loads(path.read_text()); stage = int(marker['name'][:2]) - 1
        run = path.parent.parent
        checks = []
        for out in marker['outputs']:
            container = out['path']; anchor = '/' + run.name + '/'
            positions = [m.start() for m in re.finditer('(?=' + re.escape(anchor) + ')', container)]
            candidates = [run / container[pos + len(anchor):] for pos in positions]
            local = next((p for p in candidates if p.is_file()), candidates[-1])
            ok = local.is_file() and local.stat().st_size == out['size'] and local.stat().st_mtime_ns == out['mtime_ns']
            if ok and local.suffix == '.npy':
                try:
                    arr = np.load(local, mmap_mode='r'); ok = arr.size > 0
                except Exception: ok = False
            checks.append(ok)
            evidence.append({'stage': marker['name'], 'path': str(local.relative_to(ROOT)), 'identity_match': ok})
        status = 'COMPLETE' if checks and all(checks) else 'PARTIAL'
        if stages[stage] != 'COMPLETE': stages[stage] = status
    if i == 16:
        legacy = ['moge', 'cari4d_export', 'cari4d_sam3d_body.pt', 'cari4d_aligned_depth.h5', 'cari4d_foundationpose.pt', 'cari4d_coconet.pth', 'cari4d_refined.pth', 'cari4d_reconstruction.mp4']
        stages = ['UNKNOWN' if (work / name).exists() else 'MISSING' for name in legacy]
    npz = DATA / 'submission_episodes' / (stem + '.npz')
    glb = npz.with_name(stem + '_object.glb')
    ns = gs = 'MISSING'; errors = []
    if npz.exists():
        try:
            d = dict(np.load(npz, allow_pickle=False)); t = len(d['pose'])
            assert d['pose'].shape == (t,136) and d['scales'].shape == (68,) and d['shape'].shape == (45,)
            assert d['object_rotation'].shape == (t,3,3) and d['object_translation'].shape == (t,3)
            assert all(np.isfinite(d[k]).all() for k in ['pose','scales','shape','object_rotation','object_translation','object_scale'])
            assert float(d['object_scale'].reshape(())) > 0 and t > frames[i]
            _rotations(d['object_rotation'])
            ns = 'COMPLETE'
        except Exception as exc: ns = 'FAILED'; errors.append('NPZ: ' + repr(exc))
    if glb.exists():
        try: mesh_check(glb); gs = 'COMPLETE'
        except Exception as exc: gs = 'FAILED'; errors.append('GLB: ' + repr(exc))
    mesh_paths = list(work.rglob('*.glb'))
    candidate = DATA / f'episode_{(i//3)*3:06d}_candidate_mesh'
    mesh_paths += list(candidate.glob('*.glb'))
    meshes = []
    for path in mesh_paths:
        try: meshes.append({'path': str(path.relative_to(ROOT)), 'extents_m_or_unverified_units': mesh_check(path)})
        except Exception as exc: errors.append(str(path) + ': ' + repr(exc))
    ms = 'COMPLETE' if meshes else ('FAILED' if mesh_paths else 'MISSING')
    if all(s == 'COMPLETE' for s in stages): full += 1
    if ns == gs == 'COMPLETE': valid += 1
    action = 'Preserve; pair ready' if ns == gs == 'COMPLETE' else ('Convert refined output' if stages[6] == 'COMPLETE' else ('Resume S07; preserve S01–S06' if i == 15 else 'Run missing stages after Docker restored'))
    if i == 16: action = 'Protected; never rerun without approval'
    table.append('| ' + ' | '.join([stem,meta['object'],*stages,ms,ns,gs,action]) + ' |')
    details.append({'episode':stem,'stages':stages,'evidence':evidence,'meshes':meshes,'mask_h5_present':(work/'cari4d_masks.h5').is_file(),'errors':errors,'next_action':action})
report = ROOT / 'incident_reports/track1_submission_progress.md'
text = f'# Track 1 submission recovery\n\nUpdated {datetime.now(timezone.utc).isoformat()}\n\nBranch: v2d-track1-blackwell. No inference or upload performed during recovery. Episode 16 untouched.\n\nVerified pairs: {valid}/30. Pipelines with all eight marker output identities verified: {full}/30.\n\n' + '\n'.join(table)
text += '\n\nCOMPLETE stages require matching output sizes and nanosecond timestamps from stage signatures; NPY headers were opened. This verifies preserved checkpoint identities, not full tensor contents. Episode 16 uses legacy artifacts without stage signatures and is UNKNOWN for stage completion. Mesh COMPLETE means loadable finite triangle geometry; metre scale still requires checking against source transform/calibration before inference. Candidate meshes are shared only within their same-object episode triplet.\n\n## Blockers and next execution\n\n- Docker CLI reports missing WSL integration both inside and outside sandbox. `/var/run/docker.sock`, Docker Desktop CLI mount, and Windows Docker executable are unavailable. GPU is visible outside sandbox: RTX 5090 Laptop, 24463 MiB. No host configuration changed.\n- Kaggle authenticated successfully outside sandbox; all five Track 1 competitions entered. No upload issued.\n- Host Python lacks torch and h5py. Existing refined PTH conversion cannot run here without the normal Docker environment or added dependencies.\n- Finish converting episodes 3–5, resume episode 15 at S07 with its existing stage signatures and original command settings, then process remaining missing episodes sequentially using Blackwell images. Preserve every existing stage.\n- Official format is one Parquet file for five metric competitions. No ZIP required. One authorized submission version only.\n- Producing-code GitHub commit must include actual conversion/pipeline changes and be accessible to organizers; current producing worktree is dirty. Do not invent a provenance URL.\n\n## Recorded failure\n\nDocker availability command: `docker ps --format ...`; CLI prints "The command docker could not be found in this WSL 2 distro". GPU command in sandbox exited 255; escalation succeeded. Kaggle entry check in sandbox exited 1 with DNS failure; escalated check exited 0 with all five competitions entered. No inference stage failed in this recovery turn.\n'
report.write_text(text)
(report.parent/'track1_recovery_inventory.json').write_text(json.dumps(details,indent=2))
print('\n'.join(table)); print(f'Valid pairs: {valid}/30; fully verified pipelines: {full}/30')
