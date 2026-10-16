"""Review necessary pilot gates; a failed pilot cannot stage/upload candidates."""
import json
from pathlib import Path
import numpy as np
from track1_v9_temporal import PILOT,LAMBDAS,sha,write_json,PARENT_SHA


def main():
    root=Path(__file__).resolve().parents[1];data=root/'data/v2d_challenge/track_1';out=data/'v9_temporal';report=root/'incident_reports/track1_improvement/v9_temporal'
    assert sha(data/'submission_packages/track1_contact_smoothing_v8.parquet')==PARENT_SHA
    validation=json.loads((root/'incident_reports/track1_improvement/contact_smoothing_v8/package_validation.json').read_text())
    parent=data/'contact_smoothing_v8/submission_episodes'
    assert all(sha(parent/name)==value for name,value in validation['artifact_sha256'].items())
    rows=[];inputs={};candidate_hashes={};human_pass=[];object_pass=[]
    for mode in ['plain','robust']:
        human_path=report/f'human_pilot_{mode}.json';human=json.loads(human_path.read_text());inputs[str(human_path.relative_to(root))]=sha(human_path)
        for lam in LAMBDAS:
            hr=[r for r in human['rows'] if r['lambda']==lam];assert sorted(r['episode'] for r in hr)==PILOT
            object_rows=[]
            for stream in ['human','object']:
                for episode in PILOT:
                    folder=out/stream/mode/'initial'/f'lam_{lam:g}'/f'episode_{episode:06d}';done=json.loads((folder/'done.json').read_text())
                    assert done['state']=='FIT_COMPLETE_NOT_PROMOTED'
                    for suffix,key in [('.npz','npz_sha256'),('_object.glb','mesh_sha256')]:
                        path=folder/(f'episode_{episode:06d}'+suffix);assert sha(path)==done[key];candidate_hashes[str(path.relative_to(root))]=done[key]
                    if stream=='object':object_rows.append(done['diagnostics'])
            hp=sum(r['pilot_displacement_and_gain_pass'] for r in hr)
            oa=float(np.mean([r['candidate_acceleration_cm_unaligned'] for r in object_rows]));ob=max(r['cd_o_bound_cm_per_unit_alignment_scale'] for r in object_rows)
            oh=oa<=.10 and ob<=2.
            row={'mode':mode,'lambda':lam,'human_passing_episodes':hp,'human_required_passing_episodes':6,
                 'human_max_p99_displacement_cm':max(r['p99_frame_max_vertex_displacement_cm'] for r in hr),
                 'human_equal_episode_mean_displacement_cm':float(np.mean([r['mean_vertex_displacement_cm'] for r in hr])),
                 'object_equal_episode_acceleration_cm_unaligned':oa,'object_max_cd_bound_cm_per_unit_alignment_scale':ob,
                 'object_necessary_motion_and_bound_pass':oh}
            rows.append(row)
            if hp>=6:human_pass.append(row)
            if oh:object_pass.append(row)
    # Further evaluation is required before promotion, even if these necessary gates pass.
    state='FURTHER_LOCAL_REVIEW_REQUIRED' if human_pass or object_pass else 'PILOT_REJECTED_NO_EXPANSION'
    result={'state':state,'episodes':PILOT,'fit_candidates':168,'decoded_human_candidates':84,'rows':rows,
            'input_reports_sha256':inputs,'source_sha256':sha(__file__),'candidate_sha256':candidate_hashes,
            'parent_parquet_sha256':PARENT_SHA,'all_parent_pairs_preserved':True,'promoted_episodes':[],
            'uploaded':False,'all_30_fits_started':False,
            'interpretation':'Necessary gate failure rejects expansion; no candidate has completed full contact/image/visual promotion review. Official scores are unchanged; these are unaligned local diagnostics.'}
    write_json(report/'pilot_gate_review.json',result)
    write_json(report/'selection_review.json',{'state':state,'parent':'v8','promoted':[],
        'episodes':{str(i):{'verdict':'RETAIN_V8','reason':'Temporal pilot failed expansion gates' if i in PILOT else 'No all30 expansion; not fitted in this round'} for i in range(30)},'uploaded':False})
    lines=['# v9 temporal pilot results','','State: **'+state+'**. No new submission or changed package. Immutable v8 remains current.','','Seven episodes × six lambdas × two robust modes × two streams =168 isolated fits;84 human candidates were decoded and compared to the actual parent. All60 parent pair hashes and the scored v8 Parquet hash match.','','| Mode | Lambda | Human passes /7 (need6) | Worst human p99 max displacement cm | Object mean acceleration cm, unaligned | Worst object bound cm per unit alignment scale |','|---|---:|---:|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['mode']} | {r['lambda']:g} | {r['human_passing_episodes']}/7 | {r['human_max_p99_displacement_cm']:.3f} | {r['object_equal_episode_acceleration_cm_unaligned']:.5f} | {r['object_max_cd_bound_cm_per_unit_alignment_scale']:.3f} |")
    lines+=['','## Decision','','No tested human configuration passes even one pilot episode under the combined gain/displacement conditions. No object configuration simultaneously reaches 0.10 cm unaligned acceleration and a 2 cm per-unit-alignment bound on every episode. Parameter-space smoothing is not accepted merely because its motion proxy decreases.','','The independent unwrap-only diagnostic changes decoded vertices by less than 0.0002 cm across the pilot: the observed large deformations come from smoothing these parameter trajectories, not the periodic branch normalization alone. There is no claim about hidden-reference score improvement.','','## Deferred checks','','Full-scored-frame contact, image, video promotion and anchor-repair expansion were not launched for candidates already failing necessary displacement/motion gates. A sampled changed-human checker smoke test and two decoded four-frame diagnostic clips (full scene and object crop) verify the upgraded tools; they do not promote those candidates. No actual official package is built without complete passing selection evidence.','','## Geometry precheck','','The independent all30 v8 geometry audit is in `../v10_geometry/risk_audit.md`. Hoops 0, 1, 2 already share exactly the same submitted GLB hash, scale and extents; Episodes 0 and 2 IoU are 0.7288 and 0.6721. The proposed missing-metric-hoop premise is false for the current artifacts, so its replacement experiment is skipped. Low-IoU rollers and remaining iron rotations remain diagnostic leads. Disconnected mesh components alone are not evidence of removable floaters.']
    lines += ['', 'The initial full-clip diagnostic render was stopped after80 logged frames because the candidate had failed promotion. Its partial temporary files remain local and are not completed-video evidence. The short diagnostic clips record DIAGNOSTIC_PASS and full_video_review:false. Only their frame2 full-scene preview was visually inspected; this is tool verification, not candidate acceptance.']
    (report/'pilot_results.md').write_text('\n'.join(lines)+'\n');print(state)


if __name__=='__main__':main()
