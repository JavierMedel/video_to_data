"""Visually audit baseline object selection before expanding the pilot."""
import json,sys
from pathlib import Path
import cv2
import h5py
import numpy as np
root=Path(sys.argv[1]); data=root/'data/v2d_challenge/track_1'
metadata=[json.loads(line) for line in (data/'meta/episodes_metadata.jsonl').read_text().splitlines()]
scored=json.loads((root/'incident_reports/track1_scored_frames.json').read_text())
panels=[]
for row in metadata:
    i=row['episode_index']; stem=f'episode_{i:06d}'; frame=scored[str(i)][0]
    cap=cv2.VideoCapture(str(data/'videos/chunk-000/observation.images.exo_camera'/(stem+'.mp4')))
    cap.set(cv2.CAP_PROP_POS_FRAMES,frame); ok,rgb=cap.read(); cap.release(); assert ok
    with h5py.File(data/stem/'cari4d_masks.h5','r') as f:
        group=f[next(iter(f.keys()))]; mask=group[f'{frame:06d}-k0.obj_rend_mask.png'][()]>0
    overlay=rgb.copy(); overlay[mask]=(overlay[mask]*.45+np.array([0,255,0])*.55).astype(np.uint8)
    overlay=cv2.resize(overlay,(512,384))
    cv2.putText(overlay,f'{i:02d} {row["object"]} f{frame}',(8,22),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
    panels.append(overlay)
for index,begin in enumerate((0,15)):
    selected=panels[begin:begin+15]; sheet=np.concatenate([np.concatenate(selected[j:j+3],axis=1) for j in range(0,15,3)],axis=0)
    path=root/'incident_reports/track1_improvement'/f'baseline_masks_{index}.jpg'; cv2.imwrite(str(path),sheet); print(path,flush=True)
