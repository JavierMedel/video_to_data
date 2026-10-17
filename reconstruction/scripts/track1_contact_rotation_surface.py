"""Reselect checked orientation options with a fixed surface-motion objective.

All selected poses were already checked on the actual mesh. Separate outputs
preserve the first-order ablation and its exact hashes. No upload occurs.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import trimesh
from track1_v9_temporal import sha,write_json,PARENT_SHA
from track1_rotation_surface_selection import select_surface_path,surface_objective
from track1_contact_smoothing import summarize_selected_contacts,scored_motion_diagnostics


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--episodes',type=int,nargs='+',required=True);a=p.parse_args();root=a.root.resolve()
    data=root/'data/v2d_challenge/track_1';parent=data/'contact_smoothing_v8/submission_episodes';campaign=data/'v10_contact_rotation'
    assert sha(data/'submission_packages/track1_contact_smoothing_v8.parquet')==PARENT_SHA
    scored=json.loads((root/'incident_reports/track1_scored_frames.json').read_text());sys.path.insert(0,str(root/'modules/v2d_cari4d/lib/cari4d'))
    from prep.trajectory_diagnostics import rigid_motion_diagnostics
    for e in a.episodes:
        stem=f'episode_{e:06d}';source=campaign/stem;state_path=root/'incident_reports/track1_improvement/v10_contact_rotation'/stem/'status.json';original=json.loads(state_path.read_text());assert original['state']=='CANDIDATE_READY'
        assert all(sha(root/name)==value for name,value in original['inputs'].items())
        assert sha(source/'contact_costs.npz')==original['contact_costs_sha256'] and sha(source/'image_choices.npz')==original['image_choices_sha256']
        b=dict(np.load(parent/(stem+'.npz'),allow_pickle=False));mesh=trimesh.load(parent/(stem+'_object.glb'),force='mesh',process=False);v=mesh.vertices*float(b['object_scale']);sample=v[::max(1,int(np.ceil(len(v)/1024)))]
        choices=dict(np.load(source/'image_choices.npz',allow_pickle=False));costs=dict(np.load(source/'contact_costs.npz',allow_pickle=False));assert costs['done'].all()
        full,fps,gap=[costs[n] for n in ('full','fps','gap')];allowed=choices['eligible']&(full<=full[:,:1]+1e-10)&(fps<=fps[:,:1]+1e-10)&((gap[:,:1]>1.5)|(gap<=gap[:,:1]+.5));allowed[:,0]=True
        options=choices['rotations'];path=select_surface_path(options,b['object_translation'],sample,choices['ious'],allowed)
        before=surface_objective(options,b['object_translation'],sample,choices['ious'],np.zeros(len(path),int));after=surface_objective(options,b['object_translation'],sample,choices['ious'],path);assert after<=before+1e-8
        out=campaign/'surface_selection'/stem;out.mkdir(parents=True,exist_ok=True);candidate=dict(b);candidate['object_rotation']=options[np.arange(len(path)),path].copy()
        target=out/(stem+'.npz')
        if target.exists():
            saved=dict(np.load(target,allow_pickle=False));assert all(np.array_equal(saved[k],candidate[k]) for k in candidate)
        else:np.savez_compressed(target,**candidate)
        for key in b:
            if key!='object_rotation':assert np.array_equal(candidate[key],b[key])
        if not (out/(stem+'_object.glb')).exists():shutil.copy2(parent/(stem+'_object.glb'),out/(stem+'_object.glb'))
        motions={};sm={}
        for label,bundle in [('submitted_v8',b),('final_candidate',candidate)]:
            pose=np.tile(np.eye(4),(len(path),1,1));pose[:,:3,:3]=bundle['object_rotation'];pose[:,:3,3]=bundle['object_translation'];motions[label]=rigid_motion_diagnostics(pose,v)
            sm[label]=scored_motion_diagnostics(bundle['object_rotation'],bundle['object_translation'],v,np.array(scored[str(e)]))
        write_json(out/'review.json',{'control_label':'submitted_v8','held_out_frames':[],'geometry_review':{'held_out':{'depth_driven':False}},'summary':motions,'scope':'Fixed surface-acceleration objective, same-video observed-mask costs. Not held-out or hidden-reference scores.'})
        contacts=summarize_selected_contacts(full,fps,gap,costs['done'],path,np.array(scored[str(e)]));contacts.update(candidate_npz_sha256=sha(target),control_npz_sha256=sha(parent/(stem+'.npz')),all_scored_frames=True,frames=scored[str(e)],scope='Exact selected feasible option costs reused; visually ineligible options have unused copied baseline costs. Not independent hand rerender.')
        write_json(out/'fit_hand_review.json',contacts);np.savez_compressed(out/'selection.npz',path=path,allowed=allowed)
        report=root/'incident_reports/track1_improvement/v10_contact_rotation/surface_selection'/stem;report.mkdir(parents=True,exist_ok=True)
        inputs={str(p.relative_to(root)):sha(p) for p in [Path(__file__),Path(__file__).with_name('track1_rotation_surface_selection.py'),state_path,source/'contact_costs.npz',source/'image_choices.npz']}
        state={'state':'CANDIDATE_READY','episode':e,'inputs':inputs,'parent':'v8','surface_weight':.02,'objective_before':before,'objective_after':after,'selected_nonparent_frames':int((path!=0).sum()),'changed_scored_frames':int((path[np.array(scored[str(e)])]!=0).sum()),'candidate_npz_sha256':sha(target),'candidate_mesh_sha256':sha(out/(stem+'_object.glb')),'summary':motions,'scored_motion':sm,'framewise_contact_pass':True,'explicit_mesh_evaluations':int(len(path)+choices['eligible'][:,1:].sum()),'source_contact_cache_sha256':sha(source/'contact_costs.npz'),'uploaded':False}
        write_json(report/'status.json',state);print('SURFACE_SELECTED',e,'changed',state['selected_nonparent_frames'],'motion',motions,flush=True)


if __name__=='__main__':main()
