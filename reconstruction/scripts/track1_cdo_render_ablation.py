"""Render full-timeline HOI comparisons and matching object close-ups."""
import json
import os
from pathlib import Path

import av
import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from lib_mhr import MHRLayer
from lib_mhr.human_texture import batched_vertex_normals
from prep.mhr_export_utils import camera_calibration, load_edex, read_mask, read_rgb
from tools.mhr_mesh_renderer import NvdiffMeshRenderer
from tools.render_mhr_wild_inference import _params, _overlay_frame, _header, _source_video_properties, HUMAN_ALBEDO

root = Path('/repo'); stem = 'episode_000012'
data = root / 'data/v2d_challenge/track_1'
work = data / 'improvement_rollout_v1' / stem
export = work / 'run/export' / stem
experiments = data / 'cdo_v4_experiments/refinement' / stem
out = experiments / 'videos'; out.mkdir(parents=True, exist_ok=True)
paths = {label: experiments / name / 'refined.pth' for label, name in [('Control: 100 steps', 'control_100_batch64'), ('Rotation refinement: 100 steps', 'rotation_100_batch64'), ('Fixed rotation: 300 steps', 'steps300_batch64')]}
bundles = {label: torch.load(path, map_location='cpu', weights_only=False) for label, path in paths.items()}
names = next(iter(bundles.values()))['frames']
assert all(b['frames'] == names for b in bundles.values())
video = work / (stem + '.0.color.mp4')
fps, rate, total, width, height = _source_video_properties(video)
assert len(names) == total
k, w2c = camera_calibration(load_edex(export), 0)
assert np.allclose(w2c, np.eye(4), atol=1e-6)
render_width = 768
render_height = int(round(height * render_width / width / 2)) * 2
scaled_k = k.copy(); scaled_k[0] *= render_width / width; scaled_k[1] *= render_height / height
device = torch.device('cuda')
layer = MHRLayer.from_mhr_assets(mhr_assets_root=os.environ['MHR_ASSETS_ROOT'], device='cuda')
faces = layer.mesh_faces(device=device).detach().cpu().numpy().astype(np.int32)
renderer = NvdiffMeshRenderer(device='cuda')
mesh = work / 'run/export' / stem / 'object_mesh/output_aligned.glb'
outputs = [out / 'episode_000012_experiment_comparison.mp4', out / 'episode_000012_object_closeups.mp4']
assert not any(p.exists() for p in outputs), 'Preserve existing videos'
temporary = [p.with_name('.' + p.stem + '.tmp.mp4') for p in outputs]
writers = [imageio.get_writer(str(p), format='FFMPEG', mode='I', fps=fps, codec='libx264', pixelformat='yuv420p', macro_block_size=None, input_params=['-r', rate], ffmpeg_params=['-crf', '18', '-preset', 'medium']) for p in temporary]
written = 0
try:
    for start in range(0, total, 2):
        stop = min(start + 2, total); indices = slice(start, stop)
        renders = {}
        for label, bundle in bundles.items():
            with torch.inference_mode():
                body = layer.mhr_forward(_params(bundle['pr'], indices, device))
            vertices = body.vertices.detach().cpu().numpy().astype(np.float32)
            normals = batched_vertex_normals(vertices, faces)
            poses = np.asarray(bundle['pr'].get('pose_abs_postopt', bundle['pr']['pose_abs'])[indices], dtype=np.float32)
            renders[label] = renderer.render_front_batch_constant_human_textured_object(vertices, faces, normals, str(mesh), poses, np.repeat(scaled_k[None], stop - start, axis=0), (render_height, render_width), HUMAN_ALBEDO)
        for j, name in enumerate(names[start:stop]):
            rgb = cv2.resize(read_rgb(export, 0, str(name)), (render_width, render_height))
            panels = [('Source video', rgb)]
            for label, (rendered, mask) in renders.items():
                panels.append((label, _overlay_frame(rgb, rendered[j], mask[j])))
            full = [_header(frame, label) for label, frame in panels]
            grid = np.concatenate([np.concatenate(full[:2], axis=1), np.concatenate(full[2:], axis=1)], axis=0)
            grid = _header(grid, f'Episode 12 | frame {name} | experimental reconstructions', height=40)
            writers[0].append_data(grid)
            # Follow the observed object, using the same crop for every method.
            observed = cv2.resize(read_mask(export, 'object', 0, str(name)).astype(np.uint8), (render_width, render_height), interpolation=cv2.INTER_NEAREST)
            yy, xx = np.nonzero(observed)
            if len(xx):
                cx, cy = int((xx.min() + xx.max()) / 2), int((yy.min() + yy.max()) / 2)
                crop_width = int(min(render_width, max(180, (xx.max() - xx.min()) * 3, (yy.max() - yy.min()) * 4)))
                crop_height = int(min(render_height, crop_width * .75))
                x = max(0, min(render_width - crop_width, cx - crop_width // 2)); y = max(0, min(render_height - crop_height, cy - crop_height // 2))
            else:
                x, y, crop_width, crop_height = 0, 0, render_width, render_height
            crops = [_header(cv2.resize(frame[y:y + crop_height, x:x + crop_width], (512, 384)), label) for label, frame in panels]
            close = np.concatenate([np.concatenate(crops[:2], axis=1), np.concatenate(crops[2:], axis=1)], axis=0)
            writers[1].append_data(_header(close, f'Episode 12 | frame {name} | object close-ups', height=40))
            if written == total // 2:
                cv2.imwrite(str(out / 'comparison_preview.jpg'), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
            written += 1
        if stop % 20 == 0 or stop == total:
            print('ABLATION_RENDER_PROGRESS', stop, total, flush=True)
finally:
    for writer in writers:
        writer.close()
assert written == total
validation = []
for tmp, final in zip(temporary, outputs):
    with av.open(str(tmp)) as container:
        stream = container.streams.video[0]
        decoded = sum(1 for _ in container.decode(stream))
        assert decoded == total and abs(float(stream.average_rate) - fps) < 1e-6
        validation.append({'path': str(final), 'frames': decoded, 'fps': str(stream.average_rate), 'width': stream.width, 'height': stream.height})
    os.replace(tmp, final)
(out / 'render_validation.json').write_text(json.dumps({'status': 'PASS', 'source_frames': total, 'videos': validation, 'panels': ['Source video'] + list(paths), 'bundles': {k: str(v) for k, v in paths.items()}, 'scope': 'Episode 12 experimental outputs; not a new Kaggle submission'}, indent=2) + '\n')
print('ABLATION_VIDEOS_READY', json.dumps(validation), flush=True)
