"""Review/stage a complete30-episode v8-parent rotation-only package, never upload."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from track1_v9_temporal import sha,write_json,PARENT_SHA
from track1_review_humans import framewise_contact_pass


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--stage',action='store_true');a=p.parse_args();root=a.root.resolve()
    data=root/'data/v2d_challenge/track_1';campaign=data/'v10_contact_rotation';report=root/'incident_reports/track1_improvement/v10_contact_rotation';parent=data/'contact_smoothing_v8/submission_episodes'
    assert sha(data/'submission_packages/track1_contact_smoothing_v8.parquet')==PARENT_SHA
    parent_hashes=json.loads((root/'incident_reports/track1_improvement/contact_smoothing_v8/package_validation.json').read_text())['artifact_sha256'];assert all(sha(parent/n)==v for n,v in parent_hashes.items())
    frames=json.loads((root/'incident_reports/track1_scored_frames.json').read_text());visual=json.loads((report/'visual_reviews.json').read_text()) if (report/'visual_reviews.json').exists() else {}
    decisions={str(e):{'verdict':'RETAIN_V8','source':str(parent.relative_to(root)),'gates':{'parent_hashes_match':True},'reason':'No accepted isolated change for this episode'} for e in range(30)};pending=[]
    for e in [7,18,20]:
        stem=f'episode_{e:06d}';folder=campaign/'surface_selection'/stem;sp=report/'surface_selection'/stem/'status.json'
        if not sp.exists():pending.append(e);decisions[str(e)]['reason']='Targeted fit pending';continue
        status=json.loads(sp.read_text());assert status['state']=='CANDIDATE_READY' and all(sha(root/n)==v for n,v in status['inputs'].items())
        assert sha(folder/(stem+'.npz'))==status['candidate_npz_sha256'] and sha(folder/(stem+'_object.glb'))==parent_hashes[stem+'_object.glb']
        before=dict(np.load(parent/(stem+'.npz'),allow_pickle=False));after=dict(np.load(folder/(stem+'.npz'),allow_pickle=False))
        assert set(before)==set(after) and all(np.array_equal(before[k],after[k]) for k in before if k!='object_rotation')
        image_path=folder/'all_scored_image_review.json'
        if not image_path.exists():pending.append(e);decisions[str(e)]['reason']='All-scored-frame image/depth validation pending';continue
        image=json.loads(image_path.read_text());assert image['frames']==frames[str(e)] and image['candidate_npz_sha256']==sha(folder/(stem+'.npz')) and image['control_npz_sha256']==parent_hashes[stem+'.npz']
        old,new=status['summary']['submitted_v8'],status['summary']['final_candidate'];so,sn=status['scored_motion']['submitted_v8'],status['scored_motion']['final_candidate']
        motion=(new['sampled_vertex_acceleration_cm_proxy']<=max(.1,old['sampled_vertex_acceleration_cm_proxy']*1.1) and sn['sampled_vertex_acceleration_cm_proxy']<=max(.1,so['sampled_vertex_acceleration_cm_proxy']*1.1))
        gain=image['summary']['candidate']['visible_iou_mean']-image['summary']['control']['visible_iou_mean']
        source=campaign/stem;costs=dict(np.load(source/'contact_costs.npz',allow_pickle=False));choices=dict(np.load(source/'image_choices.npz',allow_pickle=False));sel=dict(np.load(folder/'selection.npz',allow_pickle=False));path=sel['path'];rows=np.arange(len(path))
        assert costs['done'].all() and sel['allowed'][rows,path].all() and np.array_equal(after['object_rotation'],choices['rotations'][rows,path])
        assert np.all(costs['full'][rows,path]<=costs['full'][:,0]+1e-10) and np.all(costs['fps'][rows,path]<=costs['fps'][:,0]+1e-10)
        active=costs['gap'][:,0]<=1.5;contact=bool(np.all(costs['gap'][rows,path][active]<=costs['gap'][active,0]+.5+1e-10))
        gates={'mean_iou_gain_at_least_002':bool(gain>=.02),'full_image_alignment':bool(image['alignment_gate_pass']),'common_depth':bool(image['depth_gate_pass']),'surface_motion_both_scopes':bool(motion),'framewise_full_hand_fps_nonincrease':True,'framewise_contact_gap':contact,'only_rotation_changed':True}
        numerical=all(gates.values());hand_path=folder/'hand_penetration_review.sampled.json';hand_pass=False
        if numerical and hand_path.exists():
            hand=json.loads(hand_path.read_text());assert hand['candidate_npz_sha256']==sha(folder/(stem+'.npz')) and hand['control_npz_sha256']==parent_hashes[stem+'.npz']
            hand_pass,gap_pass=framewise_contact_pass(hand['summary']['control'],hand['summary']['candidate'],hand['grasp_proximity']['control_nearest_vertex_cm'],hand['grasp_proximity']['candidate_nearest_vertex_cm'],tolerance_cm=1e-6)
            hand_pass=hand_pass and gap_pass
        vis=visual.get(str(e),{});visual_pass=bool(vis.get('review_complete'))
        if visual_pass:
            assert all(sha(root/n)==v for n,v in vis['artifact_sha256'].items())
        gates.update(independent_sampled_hand_check=bool(hand_pass),visual_review=visual_pass)
        if numerical and not (hand_pass and visual_pass):pending.append(e)
        verdict='PROMOTE' if numerical and hand_pass and visual_pass else ('REVIEW_PENDING' if numerical else 'RETAIN_V8')
        decisions[str(e)]={'verdict':verdict,'source':str(folder.relative_to(root)) if verdict=='PROMOTE' else str(parent.relative_to(root)),'candidate':str(folder.relative_to(root)),'status':str(sp.relative_to(root)),'status_sha256':sha(sp),'gates':gates,'iou_gain':gain,'summary':status['summary'],'scored_motion':status['scored_motion'],'reason':'; '.join(k for k,v in gates.items() if not v) if verdict!='PROMOTE' else vis['reason']}
    promoted=[int(e) for e,d in decisions.items() if d['verdict']=='PROMOTE'];state='LOCAL_REVIEW_PENDING' if pending else 'LOCAL_VALIDATION_COMPLETE'
    result={'state':state,'parent':'v8','parent_parquet_sha256':PARENT_SHA,'episodes':decisions,'pending':pending,'promoted':promoted,'uploaded':False,'source_sha256':sha(__file__),'scope':'Three targeted orientation fits; complete30 package retains verified v8 elsewhere. Local checks do not establish an official score.'}
    write_json(report/'selection_review.json',result)
    lines=['# v10 contact-constrained rotation review','',f'State: {state}. Promoted: {promoted}. Pending: {pending}.','','| Episode | Decision | Mean IoU gain | Reasons |','|---:|---|---:|---|']
    for e in [7,18,20]:
        d=decisions[str(e)];lines.append(f"| {e} | {d['verdict']} | {d.get('iou_gain',float('nan')):.5f} | {d['reason']} |")
    lines+=['','All unpromoted episodes retain the exact v8 pair. Whole-package human/translation/scale/mesh arrays are unchanged. A candidate is not eligible merely because the general image-review tool says PASS: this campaign additionally requires0.02 mean IoU gain, framewise contact preservation, a motion guard, independent sampled hand review and explicit visual evidence.']
    (report/'final_review.md').write_text('\n'.join(lines)+'\n')
    if a.stage:
        assert state=='LOCAL_VALIDATION_COMPLETE' and promoted,'No complete, justified change to stage'
        out=campaign/'submission_episodes';out.mkdir(exist_ok=True)
        for e,d in decisions.items():
            stem=f'episode_{int(e):06d}';source=root/d['source']
            for suffix in ('.npz','_object.glb'):
                name=stem+suffix;target=out/name
                if target.exists():assert sha(target)==sha(source/name)
                else:shutil.copy2(source/name,target)
        print('COMPLETE_PACKAGE_STAGED',len(decisions),promoted)
    print(state,'promoted',promoted,'pending',pending)


if __name__=='__main__':main()
