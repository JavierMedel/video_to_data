"""Read existing results and write display derivatives; never performs inference.

Run from the repository root with the reconstruction Python environment. FFmpeg
may be supplied with --ffmpeg; the ignored website-local imageio installation
is a convenience fallback. Source files are opened read-only.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw
import trimesh

ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / 'website'
OUT = SITE / 'public/media'
EP = ROOT / 'reconstruction/data/v2d_challenge/track_1/episode_000015'
INVENTORY = []


def record(source, dest, transform, extra=None):
    INVENTORY.append(dict(source=str(source.relative_to(ROOT)), output=str(dest.relative_to(SITE / 'public')),
                          sha256=hashlib.sha256(dest.read_bytes()).hexdigest(), bytes=dest.stat().st_size,
                          preparation=transform, **(extra or {})))


def still(source, name, image=None, note='Aspect-preserving WebP derivative; no enhancement.'):
    dest = OUT / name
    im = image if image is not None else Image.open(source).convert('RGB')
    im.thumbnail((1536, 1152))
    im.save(dest, quality=86, method=6)
    record(source, dest, note)
    return dest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ffmpeg', default=shutil.which('ffmpeg'))
    args = parser.parse_args()
    if not args.ffmpeg:
        sys.path.insert(0, str(SITE / '.media-venv/vendor'))
        import imageio_ffmpeg
        args.ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    OUT.mkdir(parents=True, exist_ok=True)
    videos = [
        ('t1-comparison', ROOT / 'reconstruction/data/v2d_challenge/track_1/cari4d_runs/episode_000001_stability/episode_000001/inference/episode_000001_step200000_before_after.mp4', 210, 1920),
        ('e15-input', next(EP.glob('*.mp4')), 154, 960),
        ('t3-hands', ROOT / 'reconstruction/data/track3/pilot/episode_000038/hamer_depth_aligned_overlay.mp4', 120, 960),
    ]
    for name, source, frame, width in videos:
        cap = cv2.VideoCapture(str(source))
        metadata = dict(sourceFps=cap.get(cv2.CAP_PROP_FPS), frameCount=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                        width=int(cap.get(3)), height=int(cap.get(4)), posterFrame=frame)
        if name == 't3-hands':
            input_path = ROOT / 'reconstruction/data/v2d_challenge/track_3/evaluation/videos/chunk-000/observation.images.ego_cam_a/episode_000038.mp4'
            input_capture = cv2.VideoCapture(str(input_path))
            metadata['inputVideoFps'] = input_capture.get(cv2.CAP_PROP_FPS)
            input_capture.release()
            metadata['inputVideoSource'] = str(input_path.relative_to(ROOT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, bgr = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f'Cannot decode {source.name}')
        rgb = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        still(source, name + '.webp', rgb.copy(), f'Decoded source frame {frame}; aspect-preserving WebP poster.')
        if name == 't1-comparison':
            still(source, 'hero.webp', rgb.copy(), f'Complete triptych at frame {frame}; aspect ratio preserved.')
        if name == 'e15-input':
            raw = np.asarray(rgb)
            mask_source = EP / 'object_masks/1/000154.png'
            mask = np.array(Image.open(mask_source)) > 0
            if mask.ndim == 3:
                mask = np.any(mask, axis=2)
            overlay = raw.copy()
            overlay[mask] = (0.55 * raw[mask] + 0.45 * np.array([118, 185, 0])).astype('uint8')
            still(mask_source, 'e15-segmentation.webp', Image.fromarray(overlay), 'Frame 154 RGB with SAM2 object mask in translucent green; no change to mask geometry.')
        dest = OUT / (name + '.mp4')
        if not dest.exists():
            subprocess.run([args.ffmpeg, '-hide_banner', '-loglevel', 'error', '-nostdin', '-i', str(source),
                            '-vf', f'scale={width}:-2', '-c:v', 'libx264', '-crf', '24', '-preset', 'fast',
                            '-pix_fmt', 'yuv420p', '-an', '-movflags', '+faststart', str(dest)], check=True)
        record(source, dest, f'H.264/yuv420p, CRF24, width {width}, full frame range and original frame rate, no audio; no crop or interpolation.', metadata)

    depth_source = EP / 'moge/depth/000154.png'
    depth = np.asarray(Image.open(depth_source)).astype(float)
    valid = np.isfinite(depth) & (depth > 0)
    lo, hi = np.percentile(depth[valid], [2, 98])
    gray = np.uint8(np.clip((depth - lo) / (hi - lo), 0, 1) * 255)
    gray[~valid] = 0
    still(depth_source, 'e15-depth.webp', Image.fromarray(gray).convert('RGB'), 'Relative-depth display: grayscale 2nd–98th percentile of positive values; dark near, light far, invalid black. No metric calibration asserted.')
    for name, source in [
        ('e15-scale.webp', EP / 'sam3d/scale_diagnostic_overlay.png'),
        ('e15-failure.webp', EP / 'sam3d/pose_check_000212.png'),
        ('e15-depth-jump.webp', EP / 'diagnostics/depth_jump_364_370.jpg'),
        ('t1-masks.webp', ROOT / 'reconstruction/incident_reports/track1_improvement/episode_000018_corrected_masks.jpg'),
    ]:
        still(source, name)

    mesh_source = EP / 'sam3d/object_simplified_backup.glb'
    dest = OUT / 'e15-display.glb'
    shutil.copyfile(mesh_source, dest)
    mesh = trimesh.load(mesh_source, force='mesh')
    original = trimesh.load(EP / 'sam3d/object.glb', force='mesh')
    record(mesh_source, dest, 'Existing simplified backup copied unchanged for display; original research mesh remains in place.',
           dict(vertices=len(mesh.vertices), faces=len(mesh.faces), originalVertices=len(original.vertices), originalFaces=len(original.faces)))
    vertices = mesh.vertices - mesh.bounds.mean(axis=0)
    a, b = 0.55, -0.35
    ry = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    rx = np.array([[1, 0, 0], [0, math.cos(b), -math.sin(b)], [0, math.sin(b), math.cos(b)]])
    vertices = vertices @ (rx @ ry).T
    vertices = vertices / np.ptp(vertices, axis=0).max() * 650
    im = Image.new('RGB', (960, 720), '#101010')
    draw = ImageDraw.Draw(im)
    for face in mesh.faces[np.argsort(vertices[mesh.faces].mean(axis=1)[:, 2])]:
        tri = vertices[face]
        normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        shade = 0.35 + 0.6 * abs(float(normal[2])) / max(np.linalg.norm(normal), 1e-9)
        color = tuple(int(x * shade) for x in [161, 183, 146])
        draw.polygon([(float(v[0] + 480), float(360 - v[1])) for v in tri], fill=color)
    still(mesh_source, 'e15-mesh.webp', im, 'Orthographic, flat-shaded rendering of the real display mesh. Neutral shading replaces texture for this poster only; arbitrary units.')

    pose_dir = EP / 'foundationpose_scale_diagnostic'
    poses = [(int(p.stem), json.loads(p.read_text())) for p in sorted(pose_dir.glob('*.json')) if p.stem.isdigit()]
    diagnostics = []
    for (f0, prev), (frame, pose) in zip(poses, poses[1:]):
        if frame != f0 + 1:
            continue
        q0, q1 = np.array(prev['rotation']), np.array(pose['rotation'])
        dot = abs(float(q0 @ q1 / (np.linalg.norm(q0) * np.linalg.norm(q1))))
        diagnostics.append(dict(frame=frame, translation=float(np.linalg.norm(np.array(pose['translation']) - np.array(prev['translation']))),
                                rotation=float(np.degrees(2 * np.arccos(np.clip(dot, 0, 1))))))
    dest = OUT / 'e15-motion.json'
    dest.write_text(json.dumps(diagnostics, indent=2) + '\n')
    record(pose_dir, dest, 'Adjacent saved-pose differences; translation in native estimated coordinates (scale unvalidated), rotation geodesic degrees. Independent registration files excluded.')
    dest = OUT / 'e15-motion.csv'
    with dest.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=['frame', 'translation', 'rotation'], lineterminator='\n')
        writer.writeheader()
        writer.writerows(diagnostics)
    record(pose_dir, dest, 'Same adjacent-pose differences as JSON; no smoothing.')
    coverage = dict(poseFiles=len(poses), objectMasks=len(list((EP / 'object_masks/1').glob('*.png'))),
                    depthFrames=len(list((EP / 'moge/depth').glob('*.png'))))
    (SITE / 'src/data/coverage.json').write_text(json.dumps(coverage, indent=2) + '\n')
    brand_sources = {
        'nvidia-logo.png': 'https://github.com/nvidia-isaac/video_to_data/blob/main/docs/hydra-0/img/nvidia-logo.png',
        'nvidia-logo-horz.svg': 'https://www.nvidia.com/content/nvidiaGDC/us/en_US/about-nvidia/legal-info/logo-brand-usage/_jcr_content/root/responsivegrid/nv_container_392921705/nv_container_412055486/nv_image.coreimg.svg/1776076920317/nvidia-logo-horz.svg',
    }
    for name, url in brand_sources.items():
        logo = SITE / 'public/brand' / name
        INVENTORY.append(dict(source=url, output='brand/' + name,
                              sha256=hashlib.sha256(logo.read_bytes()).hexdigest(), bytes=logo.stat().st_size,
                              preparation='Official NVIDIA logo, copied unchanged; trademark asset, not dataset evidence.'))
    inventory_text = json.dumps(INVENTORY, indent=2) + '\n'
    (SITE / 'asset-inventory.json').write_text(inventory_text)
    evidence = SITE / 'public/evidence'
    evidence.mkdir(parents=True, exist_ok=True)
    (evidence / 'asset-inventory.json').write_text(inventory_text)
    print(f'Prepared {len(INVENTORY)} derivatives. Coverage: {coverage}')


if __name__ == '__main__':
    main()
