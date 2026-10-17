"""Render sampled segmentation evidence and transform-aware mesh shape views."""
from pathlib import Path
import json

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import trimesh

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / 'data/track3/pilot/episode_000038'
VIDEO = ROOT / 'data/v2d_challenge/track_3/evaluation/videos/chunk-000/observation.images.ego_cam_a/episode_000038.mp4'

sheet = Image.new('RGB', (1280, 960))
draw = ImageDraw.Draw(sheet)
cap = cv2.VideoCapture(str(VIDEO))
areas = []
for path in sorted((WORK / 'masks/1').glob('*.png')):
    areas.append(int(np.count_nonzero(np.asarray(Image.open(path)))))
for index, frame in enumerate([0, 37, 75, 112, 150, 187, 225, 262]):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, rgb = cap.read()
    if not ok: raise RuntimeError(f'Cannot decode source frame {frame}')
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    mask = np.asarray(Image.open(WORK / f'masks/1/{frame:06d}.png')) > 0
    rgb[mask] = (rgb[mask] * .6 + np.array([0, 255, 0]) * .4).astype(np.uint8)
    image = Image.fromarray(rgb); image.thumbnail((320, 450))
    x, y = (index % 4) * 320, (index // 4) * 480
    sheet.paste(image, (x, y + 20)); draw.text((x, y), f'frame {frame}', fill='white')
cap.release()
sheet.save(WORK / 'mask_contact_sheet.jpg')

scene = trimesh.load(WORK / 'mesh/mini_dust_pan.glb', force='scene')
# dump() applies scene graph node transforms before concatenation.
geometry = trimesh.util.concatenate(tuple(scene.dump()))
mesh = geometry.simplify_quadric_decimation(face_count=6000)
vertices, faces = mesh.vertices, mesh.faces
center = (vertices.max(axis=0) + vertices.min(axis=0)) / 2
radius = float(np.ptp(vertices, axis=0).max()) / 2
fig = plt.figure(figsize=(12, 4))
for i, (elevation, azimuth) in enumerate([(30, 45), (75, 135), (5, 225)]):
    ax = fig.add_subplot(1, 3, i + 1, projection='3d')
    ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=faces,
                    color='gold', linewidth=0, antialiased=False, shade=True)
    ax.view_init(elev=elevation, azim=azimuth)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1)); ax.set_axis_off()
fig.suptitle('Episode 38 video-derived SAM3D candidate: shape views, metric scale not yet validated')
fig.tight_layout(); fig.savefig(WORK / 'mesh_shape_review.png', dpi=150); plt.close(fig)
(WORK / 'mask_geometry_checks.json').write_text(json.dumps({
    'mask_count': len(areas), 'empty_mask_count': sum(a == 0 for a in areas),
    'mask_area_min_px': min(areas), 'mask_area_max_px': max(areas),
    'mesh_vertices': len(geometry.vertices), 'mesh_faces': len(geometry.faces),
    'scene_node_transforms_applied': True, 'shape_preview_faces': len(faces),
    'raw_mesh_extents': geometry.extents.tolist(), 'metric_scale_verified': False,
    'localization_over_time_verified': False}, indent=2) + '\n')
