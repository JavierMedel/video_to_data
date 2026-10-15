"""Resume-safe sequential validation and visual evidence for the v7 sweep."""
import json
import subprocess
import sys
from pathlib import Path

root=Path('/repo');data=root/'data/v2d_challenge/track_1';report=root/'incident_reports/track1_improvement/cdo_v7'
experiment=data/'cdo_v7_experiments'


def run(script,*args):
    subprocess.run([sys.executable,str(root/'scripts'/script),*map(str,args)],check=True)


def folder(group,episode):
    return experiment/group/f'episode_{episode:06d}'


def hand_pass(path):
    d=json.loads(path.read_text())
    return d['hand_penetration_gate_pass'] and d['grasp_proximity_gate_pass']


def status(stage,**extra):
    p=report/'pipeline_status.json';d=json.loads(p.read_text());d['state']=stage;d.update(extra)
    if stage=='HAND_REFINEMENT':d['stages']['hand_review']='RUNNING'
    if stage=='FULL_HAND_VALIDATION':d['stages']['full_validation']='RUNNING'
    if stage=='FULL_IMAGE_AND_DEPTH_VALIDATION':d['stages']['hand_review']='COMPLETE'
    if stage in ('VISUAL_EVIDENCE','VISUAL_REVIEW_PENDING'):
        d['stages']['full_validation']='COMPLETE';d['stages']['visual_review']='RUNNING'
    p.write_text(json.dumps(d,indent=2)+'\n')


probes=[json.loads((folder('geometry_candidates',e)/'probe.json').read_text()) for e in range(30)]
candidates=[d['episode'] for d in probes if d['accepted']]
failed=[e for e in candidates if not hand_pass(folder('geometry_candidates',e)/'hand_penetration_review.sampled.json')]
status('HAND_REFINEMENT',geometry_candidates=candidates,initial_hand_failures=failed)
if failed:
    run('track1_cdo_hand_safe_candidates.py','--iteration',7,'--contact-aware','--parent-group','geometry_candidates','--output-group','hand_contact_candidates_v2','--episodes',*failed)
rescued=[e for e in failed if (folder('hand_contact_candidates_v2',e)/'review.json').exists()]
if rescued:
    run('track1_cdo_hand_penetration_review.py','--iteration',7,'--candidate-group','hand_contact_candidates_v2','--episodes',*rescued)
groups={e:('hand_contact_candidates_v2' if e in rescued else 'geometry_candidates') for e in candidates}
passing=[e for e in candidates if hand_pass(folder(groups[e],e)/'hand_penetration_review.sampled.json')]
sampled_pass=passing.copy()
status('FULL_HAND_VALIDATION',sampled_hand_pass=passing,rescued_candidates=rescued)
for group in ('geometry_candidates','hand_contact_candidates_v2'):
    episodes=[e for e in passing if groups[e]==group]
    if episodes:
        run('track1_cdo_hand_penetration_review.py','--iteration',7,'--candidate-group',group,'--episodes',*episodes,'--sample-count',0)
passing=[e for e in passing if hand_pass(folder(groups[e],e)/'hand_penetration_review.json')]
status('FULL_IMAGE_AND_DEPTH_VALIDATION',full_hand_pass=passing)
if passing:
    run('track1_cdo_full_image_review.py','--episodes',*passing)
final=[]
for e in passing:
    d=json.loads((folder(groups[e],e)/'all_scored_image_review.json').read_text())
    if d['image_gate_pass'] and d['depth_gate_pass']:final.append(e)
status('VISUAL_EVIDENCE',full_validation_pass=final)
for e in final:
    candidate=folder(groups[e],e)
    if not (candidate/'exact_candidate_comparison.jpg').exists():
        run('track1_cdo_render_candidate.py','--iteration',7,'--episode',e,'--mask-sheet')
    videos=experiment/'review_videos'/f'episode_{e:06d}'
    if not (videos/'validation.json').exists():
        run('track1_cdo_render_candidate.py','--iteration',7,'--episode',e)
    validation=json.loads((videos/'validation.json').read_text());assert validation['status']=='PASS'
    (report/f'episode_{e:06d}_video_validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    print('V7_VISUAL_EVIDENCE_READY',e,flush=True)
summary={'scope':list(range(30)),'geometry_candidates':candidates,'initial_hand_failures':failed,'rescued_candidates':rescued,
         'sampled_hand_pass':sampled_pass,'full_hand_pass':passing,'full_validation_pass':final,'visual_review_complete':False,
         'note':'Visual evidence generated and full videos decoded; human/model inspection must still be recorded before promotion.'}
(report/'validation_stage.json').write_text(json.dumps(summary,indent=2)+'\n')
status('VISUAL_REVIEW_PENDING',full_validation_pass=final)
print('V7_VALIDATION_READY',json.dumps(summary),flush=True)
