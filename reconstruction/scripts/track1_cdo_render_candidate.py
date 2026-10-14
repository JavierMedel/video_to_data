"""Render the exact candidate NPZ/GLB pair beside the scored v3 prediction."""
import argparse
import json
import os
import sys
from pathlib import Path

import av
import cv2
import imageio.v2 as imageio
import numpy as np
import torch

p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, default=Path('/repo')); p.add_argument('--episode', type=int, default=12); p.add_argument('--mask-sheet', action='store_true'); p.add_argument('--iteration', type=int, choices=[4, 5], default=4); args = p.parse_args()
root = args.root; data = root / 'data/v2d_challenge/track_1'; stem = f'episode_{args.episode:06d}'
sys.path.insert(0, '/workspace/v2d_sam3d_body/lib')
sys.path.insert(0, '/workspace/v2d_foundation_pose/lib/FoundationPose')
sys.path.insert(0, str(root / 'modules/v2d_cari4d/lib/cari4d'))
sys.path.insert(0, str(root / 'v2d_submission_kit/v2d_submission_kit/tools/track1'))
from tools.mhr_mesh_renderer import NvdiffMeshRenderer
from tools.render_mhr_wild_inference import _overlay_frame, _header, _source_video_properties, HUMAN_ALBEDO
from prep.mhr_export_utils import camera_calibration, load_edex, read_mask, read_rgb
from lib_mhr import MHRLayer
from lib_mhr.human_texture import batched_vertex_normals
from mesh_to_mhr_params import MHR

candidate = None
experiment = f'cdo_v{args.iteration}'
control_label = f'Scored v{args.iteration-1}'
groups = ('pose_candidates', 'hoop_candidates', 'hoop_geometry', 'physics_hoop', 'physics_hoop_v2', 'contact_projection') if args.iteration == 4 else ('pose_candidates', 'hand_safe_candidates', 'hand_contact_candidates')
for group in groups:
    folder = data / f'{experiment}_experiments' / group / stem
    if (folder / 'review.json').exists():
        candidate = folder
assert candidate
control_dir = data/('improvement_rollout_v1/candidate_episodes' if args.iteration == 4 else 'cdo_v4_experiments/candidate_episodes')
folders = {control_label: control_dir, 'New candidate': candidate}
bundles = {label: dict(np.load(folder / (stem + '.npz'), allow_pickle=False)) for label, folder in folders.items()}
for key in ('pose', 'scales', 'shape'):
    assert np.array_equal(bundles[control_label][key], bundles['New candidate'][key])
work_group = 'improvement_pilot_v1' if args.episode in (6, 18, 28) else 'improvement_rollout_v1'
export = data / work_group / stem / 'run/export' / stem
video = data / 'videos/chunk-000/observation.images.exo_camera' / (stem + '.mp4')
fps, rate, total, width, height = _source_video_properties(video)
assert all(len(b['pose']) == total for b in bundles.values())
k, w2c = camera_calibration(load_edex(export), 0); assert np.allclose(w2c, np.eye(4))
if args.mask_sheet:
    import trimesh
    import nvdiffrast.torch as dr
    from types import SimpleNamespace
    from prep.run_foundationpose_mhr_export import _render_candidate_masks, _scaled_image_grid
    frames = np.array(json.loads((root/'incident_reports/track1_scored_frames.json').read_text())[str(args.episode)])
    frames = frames[np.linspace(0, len(frames)-1, 6).round().astype(int)]
    panels = []
    with torch.inference_mode():
        for label, bundle in bundles.items():
            mesh = trimesh.load(folders[label]/(stem+'_object.glb'), force='mesh', process=False)
            rend = SimpleNamespace(mesh_tensors={'pos': torch.as_tensor(mesh.vertices*float(bundle['object_scale']), device='cuda', dtype=torch.float32), 'faces': torch.as_tensor(mesh.faces, device='cuda', dtype=torch.int32)}, glctx=dr.RasterizeCudaContext())
            for frame in frames:
                obj = read_mask(export, 'object', 0, f'{frame:06d}')
                small_k, shape = _scaled_image_grid(k, obj.shape, 768)
                obj = cv2.resize(obj.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)>0
                pose = np.eye(4); pose[:3, :3] = bundle['object_rotation'][frame]; pose[:3, 3] = bundle['object_translation'][frame]
                mask = _render_candidate_masks(rend, pose[None], small_k, shape)[0]
                rgb = cv2.resize(read_rgb(export, 0, f'{frame:06d}'), (shape[1], shape[0]))
                rgb[obj] = (rgb[obj]*.6 + np.array([0, 255, 0])*.4).astype(np.uint8)
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(rgb, contours, -1, (255, 60, 60), 2)
                rgb = cv2.resize(rgb, (512, 384))
                panels.append(_header(rgb, f'{label} f{frame}'))
    output = candidate/'exact_candidate_comparison.jpg'
    cv2.imwrite(str(output), cv2.cvtColor(np.concatenate([np.concatenate(panels[:6], axis=1), np.concatenate(panels[6:], axis=1)], axis=0), cv2.COLOR_RGB2BGR))
    print('EXACT_CANDIDATE_MASK_SHEET', output, flush=True)
    raise SystemExit(0)
render_width = 768; render_height = int(round(height * render_width / width / 2)) * 2
scaled_k = k.copy(); scaled_k[0] *= render_width/width; scaled_k[1] *= render_height/height
layer = MHRLayer.from_mhr_assets(mhr_assets_root=os.environ['MHR_ASSETS_ROOT'], device='cuda')
faces = layer.mesh_faces(device='cuda').cpu().numpy().astype(np.int32)
del layer; torch.cuda.empty_cache()
model = MHR(str(root / 'data/weights/track1_submission_mhr/mhr_model.pt'), 'cuda', 2, 'float32')
renderer = NvdiffMeshRenderer(device='cuda')
out = data / f'{experiment}_experiments/review_videos' / stem; out.mkdir(parents=True, exist_ok=True)
video_stem = stem+f'_v{args.iteration-1}_vs_v{args.iteration}'
outputs = [out/(video_stem+'.mp4'), out/(video_stem+'_object_closeups.mp4')]
assert not any(path.exists() for path in outputs)
temporary = [path.with_name('.'+path.stem+'.tmp.mp4') for path in outputs]
writers = [imageio.get_writer(str(path), format='FFMPEG', fps=fps, codec='libx264', pixelformat='yuv420p', macro_block_size=None, input_params=['-r', rate], ffmpeg_params=['-crf', '18', '-preset', 'medium']) for path in temporary]
human = bundles[control_label]; identity = torch.as_tensor(np.r_[human['scales'], human['shape']][None], device='cuda', dtype=torch.float64)
written = 0
try:
    for start in range(0, total, 2):
        stop = min(start+2, total)
        body, _ = model.run(torch.as_tensor(human['pose'][start:stop], device='cuda', dtype=torch.float64), identity)
        vertices = (body/1000.).cpu().numpy().astype(np.float32)
        normals = batched_vertex_normals(vertices, faces)
        renders = {}
        for label, bundle in bundles.items():
            poses = np.tile(np.eye(4), (stop-start, 1, 1)).astype(np.float32)
            poses[:, :3, :3] = bundle['object_rotation'][start:stop] * float(bundle['object_scale'])
            poses[:, :3, 3] = bundle['object_translation'][start:stop]
            renders[label] = renderer.render_front_batch_constant_human_textured_object(vertices, faces, normals, str(folders[label]/(stem+'_object.glb')), poses, np.repeat(scaled_k[None], stop-start, axis=0), (render_height, render_width), HUMAN_ALBEDO)
        for j, frame in enumerate(range(start, stop)):
            rgb = cv2.resize(read_rgb(export, 0, f'{frame:06d}'), (render_width, render_height))
            panels = [(label, _overlay_frame(rgb, render[j], mask[j])) for label, (render, mask) in renders.items()]
            grid = _header(np.concatenate([_header(image, label) for label, image in panels], axis=1), f'Episode {args.episode} | frame {frame} | identical human parameters', height=40)
            writers[0].append_data(grid)
            observed = cv2.resize(read_mask(export, 'object', 0, f'{frame:06d}').astype(np.uint8), (render_width, render_height), interpolation=cv2.INTER_NEAREST)
            yy, xx = np.nonzero(observed)
            if len(xx):
                cx, cy = int((xx.min()+xx.max())/2), int((yy.min()+yy.max())/2)
                cw = int(min(render_width, max(180, (xx.max()-xx.min())*3, (yy.max()-yy.min())*4)))
                ch = int(min(render_height, cw*.75))
                x = max(0, min(render_width-cw, cx-cw//2)); y = max(0, min(render_height-ch, cy-ch//2))
            else:
                x, y, cw, ch = 0, 0, render_width, render_height
            crops = [_header(cv2.resize(image[y:y+ch, x:x+cw], (512, 384)), label) for label, image in panels]
            close = _header(np.concatenate(crops, axis=1), f'Episode {args.episode} | frame {frame} | matching object crops', height=40)
            writers[1].append_data(close)
            if frame == total//2:
                cv2.imwrite(str(out/'preview.jpg'), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(out/'closeup_preview.jpg'), cv2.cvtColor(close, cv2.COLOR_RGB2BGR))
            written += 1
        if stop % 40 == 0 or stop == total:
            print('CANDIDATE_RENDER_PROGRESS', stop, total, flush=True)
finally:
    for writer in writers:
        writer.close()
assert written == total
validation = []
for tmp, final in zip(temporary, outputs):
    with av.open(str(tmp)) as container:
        stream = container.streams.video[0]; count = sum(1 for _ in container.decode(stream))
        assert count == total and abs(float(stream.average_rate)-fps) < 1e-6
        validation.append({'path': str(final), 'frames': count, 'fps': str(stream.average_rate), 'width': stream.width, 'height': stream.height})
    os.replace(tmp, final)
(out/'validation.json').write_text(json.dumps({'status': 'PASS', 'videos': validation, 'candidate': str(candidate), 'human_arrays_identical': True}, indent=2)+'\n')
print('CANDIDATE_VIDEOS_READY', json.dumps(validation), flush=True)
