"""Read-only episode/stage status for the all-episode refinement sweep."""
import argparse
import json
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args()
report=a.root/'incident_reports/track1_improvement/cdo_v7';experiment=a.root/'data/v2d_challenge/track_1/cdo_v7_experiments'


def read(path):
    if not path.exists():return None
    try:return json.loads(path.read_text())
    except json.JSONDecodeError:return None  # A running stage may be writing.


def mark(value):
    return 'PASS' if value else 'FAIL'


status=read(report/'pipeline_status.json') or {};decisions=read(report/'promotion_decisions.json') or {}
print('Pipeline:',status.get('state','UNKNOWN'))
print('\n| Episode | Object | Audit | Geometry | Hand | Full hand | Image/depth | Video | Decision |')
print('|---:|---|---|---|---|---|---|---|---|')
for e in range(30):
    stem=f'episode_{e:06d}';audit=read(report/(stem+'_audit.json'));geometry=experiment/'geometry_candidates'/stem
    probe=read(geometry/'probe.json');correction=experiment/'hand_contact_candidates_v2'/stem;correction_probe=read(correction/'probe.json')
    candidate=correction if (correction/'review.json').exists() else geometry
    hoop=experiment/'temporal_hoop_candidates'/stem
    if (hoop/'review.json').exists():candidate=hoop
    hand=read(candidate/'hand_penetration_review.sampled.json');full=read(candidate/'hand_penetration_review.json');image=read(candidate/'all_scored_image_review.json')
    video=read(experiment/'review_videos'/stem/'validation.json');decision=decisions.get(str(e),{}).get('verdict','PENDING')
    if probe and not probe['accepted']:decision='RETAIN_V6'
    if correction_probe and not correction_probe['accepted']:decision='RETAIN_V6'
    skip=decision=='RETAIN_V6'
    hand_status=mark(hand['hand_penetration_gate_pass'] and hand['grasp_proximity_gate_pass']) if hand else ('—' if skip else 'PENDING')
    full_status=mark(full['hand_penetration_gate_pass'] and full['grasp_proximity_gate_pass']) if full else ('—' if skip else 'PENDING')
    if full and full_status=='FAIL':decision='RETAIN_V6';skip=True
    image_status=mark(image['image_gate_pass'] and image['depth_gate_pass']) if image else ('—' if skip else 'PENDING')
    if image and image_status=='FAIL':decision='RETAIN_V6'
    obj=audit['metadata']['object'] if audit else '?'
    print(f"| {e} | {obj} | {'DONE' if audit else 'PENDING'} | {mark(probe['accepted']) if probe else 'PENDING'} | {hand_status} | {full_status} | {image_status} | {'PASS' if video else ('—' if skip else 'PENDING')} | {decision} |")
