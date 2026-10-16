"""Validate an explicitly selected complete package; optional official pack, never upload."""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
import numpy as np
from track1_v9_temporal import sha, write_json


def validate_pair(parent, candidate, first_scored, allowed_arrays):
    if set(parent)!=set(candidate): raise ValueError('Bundle fields changed')
    allowed_keys={k for index,k in [(0,'pose'),(1,'scales'),(2,'shape'),(3,'object_rotation'),(4,'object_translation'),(5,'object_scale')] if index in allowed_arrays}
    for key in parent:
        if key not in allowed_keys and not np.array_equal(parent[key],candidate[key]): raise ValueError('Unintended field changed: '+key)
    for key in ('pose','scales','shape','object_rotation','object_translation','object_scale'):
        if candidate[key].shape != parent[key].shape or not np.isfinite(candidate[key]).all(): raise ValueError('Invalid prediction: '+key)
    if candidate['pose'][first_scored].tobytes()!=parent['pose'][first_scored].tobytes(): raise ValueError('Alignment frame changed')
    if any(not np.array_equal(parent[k],candidate[k]) for k in ('scales','shape')): raise ValueError('Alignment identity changed')
    if float(candidate['object_scale'])<=0: raise ValueError('Nonpositive object scale')


def validate_packed_rows(base, new, allowed, first_frames):
    if len(base)!=len(new) or set(base.index)!=set(new.index): raise ValueError('Packed row set changed')
    new=new.loc[base.index]
    changed=np.any(base[['x','y','z']].to_numpy()!=new[['x','y','z']].to_numpy(),axis=1)
    parts=base.index.str.split('_',expand=True)
    indices=np.array(parts.get_level_values(3),dtype=int)
    episodes=np.array(parts.get_level_values(1),dtype=int)
    frames=np.array(parts.get_level_values(2),dtype=int)
    if changed[~np.isin(indices,list(allowed))].any(): raise ValueError('Disallowed packed array changed')
    pinned=np.array([f==first_frames[str(e)] for e,f in zip(episodes,frames)]) & (indices==0)
    if changed[pinned].any(): raise ValueError('Packed alignment frame changed')
    return changed, episodes


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--parent-episodes',type=Path,required=True);p.add_argument('--parent-parquet',type=Path,required=True)
    p.add_argument('--candidate-episodes',type=Path,required=True);p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--allowed-arrays',default='0,4');p.add_argument('--commit',help='Actual full SHA, or GitHub URL ending with that SHA')
    p.add_argument('--out',type=Path,required=True);p.add_argument('--report',type=Path,required=True)
    a=p.parse_args(); root=a.root.resolve();allowed={int(x) for x in a.allowed_arrays.split(',')}
    assert allowed <= {0,4},'This temporal packer deliberately supports only arrays0/4'
    decisions=json.loads(a.selection.read_text());assert decisions['state']=='LOCAL_VALIDATION_COMPLETE' and set(decisions['episodes'])==set(map(str,range(30)))
    frames=json.loads((root/'incident_reports/track1_scored_frames.json').read_text())
    kit=root/'v2d_submission_kit/v2d_submission_kit';sys.path.insert(0,str(kit))
    from v2dlb.mhr_submission import _rotations
    import pandas as pd
    import trimesh
    artifacts={}
    for e in range(30):
        stem=f'episode_{e:06d}'; parent=dict(np.load(a.parent_episodes/(stem+'.npz'),allow_pickle=False));candidate=dict(np.load(a.candidate_episodes/(stem+'.npz'),allow_pickle=False))
        validate_pair(parent,candidate,frames[str(e)][0],allowed);_rotations(candidate['object_rotation'])
        assert len(candidate['pose'])>max(frames[str(e)])
        for suffix in ('.npz','_object.glb'):
            name=stem+suffix;artifacts[name]=sha(a.candidate_episodes/name)
        assert artifacts[stem+'_object.glb']==sha(a.parent_episodes/(stem+'_object.glb'))
        mesh=trimesh.load(a.candidate_episodes/(stem+'_object.glb'),force='mesh',process=False)
        assert len(mesh.faces) and np.isfinite(mesh.vertices).all() and mesh.area>0
        changed=any(not np.array_equal(parent[k],candidate[k]) for k in ('pose','object_translation'))
        if changed: assert all(decisions['episodes'][str(e)]['gates'].values()),'Unreviewed changed episode'
    result={'state':'PAIRS_VALIDATED','parent_parquet_sha256':sha(a.parent_parquet),'artifact_sha256':artifacts,'selection_sha256':sha(a.selection),'uploaded':False}
    if a.commit:
        match=re.fullmatch(r'(?:https://github.com/JavierMedel/video_to_data/commit/)?([0-9a-f]{40})',a.commit);assert match
        commit=match[1];subprocess.run(['git','cat-file','-e',commit+'^{commit}'],cwd=root,check=True)
        critical=['scripts/'+n for n in ['track1_whittaker.py','track1_v9_temporal.py','track1_v9_motion_report.py','track1_package_from_parent.py','track1_review_humans.py','track1_cdo_hand_penetration_review.py','track1_cdo_render_candidate.py']]
        subprocess.run(['git','ls-files','--error-unmatch',*critical],cwd=root,check=True,stdout=subprocess.DEVNULL)
        subprocess.run(['git','diff','--quiet',commit,'--',*critical],cwd=root,check=True)
        assert not a.out.exists(),'Preserve existing packed archive'
        url='https://github.com/JavierMedel/video_to_data/commit/'+commit
        subprocess.run([sys.executable,'eval_reconstruction.py','--episodes',str(a.candidate_episodes.resolve()),'--sample','data/track_1_sample_submission.parquet','--commit',url,'--out',str(a.out.resolve())],cwd=kit,check=True)
        base=pd.read_parquet(a.parent_parquet).set_index('row_id');new=pd.read_parquet(a.out).set_index('row_id')
        assert len(base)==len(new)==740780
        changed,episodes=validate_packed_rows(base,new,allowed,{key:value[0] for key,value in frames.items()})
        for episode in np.unique(episodes[changed]): assert all(decisions['episodes'][str(episode)]['gates'].values())
        assert new.code_commit_url.nunique()==1 and new.code_commit_url.iloc[0]==url
        result.update(state='OFFICIAL_PACK_VALIDATED',rows=len(new),sha256=sha(a.out),commit=url,changed_rows=int(changed.sum()),changed_rows_by_episode={str(e):int(changed[episodes==e].sum()) for e in range(30)})
    write_json(a.report,result);print(result['state'])


if __name__=='__main__':main()
