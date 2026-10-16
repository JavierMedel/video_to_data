"""Isolated v8-parent temporal pilot; never stages a promotion or uploads."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from track1_whittaker import whittaker_smooth, scored_acceleration_magnitude

PARENT_SHA = 'bd73a931e9f9ec85b920884415c5dc2123d86e2d3d91395c7820ebf564fd3198'
PILOT = [1, 4, 7, 11, 13, 24, 28]
LAMBDAS = [3., 10., 30., 100., 300., 1000.]


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp.json'); tmp.write_text(json.dumps(data, indent=2)+'\n'); tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--episodes', type=int, nargs='+', default=PILOT)
    p.add_argument('--lambdas', type=float, nargs='+', default=LAMBDAS)
    p.add_argument('--stream', choices=['human','object'], required=True)
    p.add_argument('--robust', action='store_true')
    p.add_argument('--anchors', type=Path, help='JSON episode -> original frame indices for repair')
    p.add_argument('--variant', default='initial')
    a = p.parse_args(); root=a.root.resolve()
    assert Path(a.variant).name == a.variant and a.variant not in ('.','..')
    assert all(0 <= e < 30 for e in a.episodes)
    data=root/'data/v2d_challenge/track_1'; out=data/'v9_temporal'
    parent=data/'contact_smoothing_v8/submission_episodes'; parquet=data/'submission_packages/track1_contact_smoothing_v8.parquet'
    assert sha(parquet)==PARENT_SHA
    reference=json.loads((root/'incident_reports/track1_improvement/contact_smoothing_v8/package_validation.json').read_text())
    hashes={f.name:sha(f) for f in parent.iterdir() if f.suffix in ('.npz','.glb')}
    assert len(hashes)==60 and hashes==reference['artifact_sha256']
    write_json(out/'parent_manifest.json',{'parquet_sha256':PARENT_SHA,'artifact_sha256':hashes,'parent':str(parent)})
    scored=json.loads((root/'incident_reports/track1_scored_frames.json').read_text())
    periodic=json.loads((out/'parameter_probe.json').read_text())['periodic_pose_columns'] if a.stream=='human' else []
    anchors=json.loads(a.anchors.read_text()) if a.anchors else {}
    for episode in a.episodes:
        stem=f'episode_{episode:06d}'; b=dict(np.load(parent/(stem+'.npz'),allow_pickle=False)); f0=int(scored[str(episode)][0])
        key='pose' if a.stream=='human' else 'object_translation'; original=b[key]; y=original.astype('float64')
        if periodic:
            y[:,periodic]=np.unwrap(y[:,periodic],axis=0)
            y[:,periodic]+=original[f0,periodic]-y[f0,periodic]
        w=np.where(b.get('valid_input',np.ones(len(y),bool)),1.,.001)
        for frame in anchors.get(str(episode),[]):
            for offset in range(-3,4):
                index=frame+offset
                if 0<=index<len(w): w[index]=max(w[index],1e4 if offset==0 else 1e2)
        for lam in a.lambdas:
            folder=out/a.stream/('robust' if a.robust else 'plain')/a.variant/f'lam_{lam:g}'/stem
            folder.mkdir(parents=True,exist_ok=True)
            signature={'parent_npz':hashes[stem+'.npz'],'parent_glb':hashes[stem+'_object.glb'],
                       'driver':sha(__file__),'solver':sha(Path(__file__).with_name('track1_whittaker.py')),
                       'probe':sha(out/'parameter_probe.json') if a.stream=='human' else None,
                       'lambda':lam,'robust':a.robust,'variant':a.variant,'anchors_sha256':sha(a.anchors) if a.anchors else None,'stream':a.stream}
            done=folder/'done.json'
            if done.exists():
                record=json.loads(done.read_text()); assert record['inputs']==signature
                assert record['npz_sha256']==sha(folder/(stem+'.npz')) and record['mesh_sha256']==sha(folder/(stem+'_object.glb'))
                continue
            x,info=whittaker_smooth(y,w,lam,pin=f0 if a.stream=='human' else None,robust_iters=3 if a.robust else 0)
            x=x.astype(original.dtype)
            if a.stream=='human': x[f0]=original[f0]
            candidate=dict(b); candidate[key]=x
            assert all(np.array_equal(candidate[k],b[k]) for k in b if k!=key)
            if a.stream=='human': assert candidate[key][f0].tobytes()==original[f0].tobytes()
            destination=folder/(stem+'.npz')
            if destination.exists():
                saved=dict(np.load(destination,allow_pickle=False)); assert all(np.array_equal(saved[k],candidate[k]) for k in candidate)
            else: np.savez_compressed(destination,**candidate)
            mesh=folder/(stem+'_object.glb')
            if not mesh.exists(): shutil.copy2(parent/mesh.name,mesh)
            assert sha(mesh)==hashes[mesh.name]
            record={'inputs':signature,'episode':episode,'info':info,'npz_sha256':sha(destination),'mesh_sha256':sha(mesh),'state':'FIT_COMPLETE_NOT_PROMOTED','source_hashes_preserved':True}
            if a.stream=='object':
                delta=np.linalg.norm((x-original).astype('float64'),axis=1)*100
                record['diagnostics']={'mean_displacement_cm':float(delta.mean()),'max_displacement_cm':float(delta.max()),
                    'cd_o_bound_cm_per_unit_alignment_scale':float(2*delta[np.array(scored[str(episode)])].mean()),
                    'control_acceleration_cm_unaligned':scored_acceleration_magnitude(original,np.array(scored[str(episode)])),
                    'candidate_acceleration_cm_unaligned':scored_acceleration_magnitude(x,np.array(scored[str(episode)]))}
            write_json(done,record)
            print('FIT_COMPLETE',a.stream,episode,lam,flush=True)


if __name__=='__main__': main()
