#!/usr/bin/env python3
"""Run a video-derived Track 3 object pilot; no policy or submission is implied.

Detection is a separate command so its object identity can be reviewed before
segmentation and mesh reconstruction. Depth and SAM2 can run concurrently on
the 24 GB GPU; SAM3D runs exclusively after both finish.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw
import trimesh

from prepare_track3_campaign import ROOT, sha256


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def container(image, module, arguments):
    module_name = module.split('.')[1]
    cache = (['-e', 'TORCH_HOME=/repo/data/weights/sam3d/torch_home',
              '-e', 'HF_HOME=/repo/data/weights/sam3d/hf_home']
             if module_name == 'sam3d' else [])
    return ['docker', 'run', '--rm', '--gpus', 'all', '--user', f'{os.getuid()}:{os.getgid()}',
            '--entrypoint', 'python', '-e', 'HOME=/tmp', '-e', 'PYTHONUNBUFFERED=1',
            '-e', 'OMP_NUM_THREADS=4', '-e', 'OPENBLAS_NUM_THREADS=4',
            *cache,
            '-v', f'{ROOT}:/repo',
            '-v', f'{ROOT}/modules/v2d_{module_name}/lib:/workspace/v2d_{module_name}/lib:ro',
            image, '-m', module, *arguments]


def cpath(path):
    return '/repo/' + str(path.relative_to(ROOT))


def run_stage(name, command, outputs, inputs, report_dir):
    paths = lambda: sorted(p for output in outputs for p in
                           (output.rglob('*') if output.is_dir() else [output]) if p.is_file())
    image = command[command.index('-m') - 1]
    image_id = subprocess.check_output(['docker', 'image', 'inspect', image, '--format', '{{.Id}}'], text=True).strip()
    signature_data = {'command': command, 'image_id': image_id,
                      'input_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in inputs},
                      'runner_sha256': sha256(Path(__file__))}
    signature = hashlib.sha256(json.dumps(signature_data, sort_keys=True).encode()).hexdigest()
    record_path = report_dir / f'{name}.json'
    if record_path.exists():
        old = json.loads(record_path.read_text())
        actual = {str(p.relative_to(ROOT)): sha256(p) for p in paths()}
        if old.get('state') == 'COMPLETE' and old.get('signature') == signature and actual and actual == old.get('output_sha256'):
            print('REUSED', name, flush=True)
            return
    record = {'state': 'RUNNING', 'started_utc': datetime.now(timezone.utc).isoformat(),
              'signature': signature, **signature_data}
    save(record_path, record)
    print('START', name, flush=True)
    started = time.monotonic()
    with (report_dir / f'{name}.log').open('a') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    record.update(exit_code=result.returncode, elapsed_seconds=time.monotonic() - started,
                  state='COMPLETE' if result.returncode == 0 else 'FAILED',
                  output_sha256={str(p.relative_to(ROOT)): sha256(p) for p in paths()})
    save(record_path, record)
    if result.returncode:
        raise RuntimeError(f'{name} failed; see {report_dir / (name + ".log")}')
    if not record['output_sha256'] or any(not p.exists() for p in outputs):
        raise RuntimeError(f'{name} exited successfully with missing outputs')
    print('COMPLETE', name, round(record['elapsed_seconds'], 1), 'seconds', flush=True)


def telemetry(stop, path):
    with path.open('a') as output:
        while not stop.is_set():
            subprocess.run(['nvidia-smi', '--query-gpu=timestamp,utilization.gpu,memory.used,memory.total,power.draw',
                            '--format=csv,noheader,nounits'], stdout=output, stderr=subprocess.DEVNULL)
            output.flush()
            stop.wait(2)


def main(args):
    inputs = json.loads((ROOT / 'incident_reports/track3_baseline/inputs.json').read_text())
    episode = next(e for e in inputs['episodes'] if e['episode_index'] == 38)
    work = ROOT / 'data/track3/pilot/episode_000038'
    reports = ROOT / 'incident_reports/track3_baseline/episode_000038'
    work.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    video = ROOT / episode['videos']['ego_cam_a']['path']
    if sha256(video) != episode['videos']['ego_cam_a']['sha256']:
        raise ValueError('Pilot source video changed since inventory')
    lock = (work / 'pilot.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    reference = work / 'reference.png'
    if not reference.exists():
        cap = cv2.VideoCapture(str(video)); ok, frame = cap.read(); cap.release()
        if not ok: raise ValueError('Cannot decode reference frame')
        Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(reference)
    detection = work / 'detections.json'
    run_stage('detection', container('v2d_grounding_dino:blackwell', 'v2d.grounding_dino.lib.image_to_object_bboxes',
              ['--image_path', cpath(reference), '--output_path', cpath(detection), '--prompt', 'yellow dustpan.',
               '--model_dir', '/repo/data/weights/grounding_dino']), [detection],
              [reference, ROOT / 'data/weights/grounding_dino/groundingdino_swint_ogc.pth',
               ROOT / 'modules/v2d_grounding_dino/lib/image_to_object_bboxes.py'], reports)
    detections = json.loads(detection.read_text())
    if not detections: raise ValueError('No dustpan detected; review source before continuing')
    if args.stage == 'detect':
        image = Image.open(reference); draw = ImageDraw.Draw(image)
        for item in detections:
            box = item['box']; draw.rectangle((box['x0'], box['y0'], box['x1'], box['y1']), outline='red', width=5)
        image.save(work / 'detection_review.jpg')
        print('Review', work / 'detection_review.jpg', flush=True)
        return
    if not (work / 'detection_review.json').exists():
        raise ValueError('Record detection identity review in detection_review.json before reconstruction')
    review = json.loads((work / 'detection_review.json').read_text())
    if review.get('detections_sha256') != sha256(detection) or review.get('accepted') is not True:
        raise ValueError('Detection review does not match current detections')
    prompts = work / 'object_prompts.json'
    save(prompts, {'prompts': [{'frame_index': 0, 'object_id': 1, 'box': detections[0]['box']}]})
    masks, depth, intrinsics = work / 'masks', work / 'depth', work / 'intrinsics'
    state = {'state': 'RECONSTRUCTION_RUNNING', 'episode_index': 38, 'object': 'mini_dust_pan',
             'depth_batch': args.depth_batch, 'overlap_depth_masks': not args.sequential,
             'policy_trained': False, 'submission_ready': False, 'uploaded': False}
    save(reports / 'execution.json', state)
    jobs = [
        ('masks', container('v2d_sam2:blackwell', 'v2d.sam2.lib.video_to_masks',
            ['--video_path', cpath(video), '--prompts_path', cpath(prompts), '--masks_dir', cpath(masks),
             '--weights_dir', '/repo/data/weights/sam2']), [masks / '1'],
             [video, prompts, work / 'detection_review.json', ROOT / 'data/weights/sam2/sam2.1_hiera_large.pt',
              ROOT / 'modules/v2d_sam2/lib/video_to_masks.py', ROOT / 'modules/v2d_sam2/lib/sam2_utils.py']),
        ('depth', container('v2d_cari4d:blackwell', 'v2d.moge.lib.video_to_depth',
            ['--video_path', cpath(video), '--depth_folder', cpath(depth), '--intrinsics_folder', cpath(intrinsics),
             '--weights_path', '/repo/data/weights/moge', '--batch_size', str(args.depth_batch)]),
             [depth, intrinsics], [video, ROOT / 'data/weights/moge/model.pt', ROOT / 'modules/v2d_moge/lib/video_to_depth.py'])]
    stop = threading.Event(); monitor = threading.Thread(target=telemetry, args=(stop, reports / 'gpu.csv'), daemon=True); monitor.start()
    try:
        with ThreadPoolExecutor(max_workers=1 if args.sequential else 2) as pool:
            futures = [pool.submit(run_stage, *job, reports) for job in jobs]
            for future in futures: future.result()
        expected = {f'{f:06d}.png' for f in range(episode['frames'])}
        if {p.name for p in (masks / '1').glob('*.png')} != expected or {p.name for p in depth.glob('*.png')} != expected:
            raise ValueError('Incomplete depth/mask frame coverage')
        mask = masks / '1/000000.png'
        image = np.asarray(Image.open(reference).convert('RGB')).copy()
        selected = np.asarray(Image.open(mask)) > 0
        if selected.ndim != 2 or selected.sum() < 100: raise ValueError('Invalid object mask')
        image[selected] = (image[selected] * .5 + np.array([0, 255, 0]) * .5).astype(np.uint8)
        Image.fromarray(image).save(work / 'mask_review.jpg')
        mesh = work / 'mesh/mini_dust_pan.glb'; transform = mesh.with_name('transform.json'); mesh_k = mesh.with_name('intrinsics.json')
        run_stage('mesh', container('v2d_sam3d:blackwell', 'v2d.sam3d.lib.image_to_mesh',
            ['--image_path', cpath(reference), '--mask_path', cpath(mask), '--mesh_path', cpath(mesh),
             '--transform_path', cpath(transform), '--intrinsics_path', cpath(mesh_k),
             '--weights_dir', '/repo/data/weights/sam3d', '--depth_path', cpath(depth / '000000.png'),
             '--depth_intrinsics_path', cpath(intrinsics / '000000.json'), '--depth_mask_path', cpath(mask), '--seed', '42']),
            [mesh, transform, mesh_k], [reference, mask, depth / '000000.png', intrinsics / '000000.json',
             ROOT / 'modules/v2d_sam3d/lib/image_to_mesh.py'], reports)
        geometry = trimesh.load(mesh, force='scene').to_geometry()
        if not len(geometry.faces) or not np.isfinite(geometry.vertices).all() or geometry.area <= 0:
            raise ValueError('Invalid reconstructed geometry')
        state.update(state='OBJECT_RECONSTRUCTION_COMPLETE', frames=episode['frames'],
                     mesh_sha256=sha256(mesh), mesh_vertices=len(geometry.vertices), mesh_faces=len(geometry.faces),
                     scale_and_body_frame='SAM3D transform saved; validate/bake consistently with future tracking',
                     limitations=['Raw upright RGB; distortion not rectified in this first pilot.',
                                  'Camera/world motion, object 6D tracking, hand reconstruction and policy learning remain.'])
    except Exception as exc:
        state.update(state='FAILED', error=str(exc))
        raise
    finally:
        stop.set(); monitor.join(timeout=5)
        state['observed_utc'] = datetime.now(timezone.utc).isoformat()
        save(reports / 'execution.json', state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['detect', 'reconstruct'], required=True)
    parser.add_argument('--depth-batch', type=int, default=4)
    parser.add_argument('--sequential', action='store_true')
    main(parser.parse_args())
