"""Run the official cached pipeline in the recovery container."""
import argparse
import os
import cv2
import h5py
from pathlib import Path
from v2d.cari4d.lib.run_inference import run_inference

p = argparse.ArgumentParser()
for name in ('video','masks','mesh','weights','output'):
    p.add_argument('--'+name, required=True)
a = p.parse_args()
cap = cv2.VideoCapture(a.video)
count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); ok, frame = cap.read(); cap.release()
assert ok and count > 0, 'Video cannot be decoded'
with h5py.File(a.masks,'r') as masks:
    print('MASK_KEYS',list(masks.keys()),'VIDEO_FRAMES',count,flush=True)
    group = masks[Path(a.video).name.removesuffix('.0.color.mp4')]
    assert len(group) == 2*count, 'Mask frame coverage mismatch'
    for t in range(count):
        for suffix in ('obj_rend_mask.png','person_mask.png'):
            ds = group[f'{t:06d}-k0.{suffix}']
            assert ds.shape == frame.shape[:2], 'Mask/video size mismatch'
run_inference(a.video,a.masks,a.mesh,a.weights,a.output,download_models=False,expected_frames=count,device='cuda',postopt_num_steps=1,postopt_batch_size=64,overwrite=False)
