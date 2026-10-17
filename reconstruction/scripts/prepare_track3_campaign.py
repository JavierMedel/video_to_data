#!/usr/bin/env python3
"""Stage the public Track 3 release and audit the exact scored video roster.

This prepares inputs only. Public development scans are never copied to the
reconstructed-mesh directory, and placeholder sample poses are never outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / 'v2d_submission_kit/v2d_submission_kit'
DATASET = 'nvidia/video_to_data_challenge'
REVISION = '5f68335f3acc802033d1e80728c1633197521de8'


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(download: bool = False) -> dict:
    source = ROOT / 'data/v2d_challenge'
    if download:
        from huggingface_hub import snapshot_download
        snapshot_download(DATASET, repo_type='dataset', revision=REVISION,
                          allow_patterns=['track_3/*'], local_dir=source, max_workers=4)
    source = source / 'track_3'
    roster_path = KIT / 'data/track_3_evaluation_objects.json'
    template_path = KIT / 'data/track_3_sample_submission.parquet'
    roster = json.loads(roster_path.read_text())['episodes']
    template = pd.read_parquet(template_path, columns=['row_id'])
    keys = template.row_id.str.extract(r'^track_3/e(\d{6})/w(\d{3})/f(\d{6})/o(\d+)$')
    if keys.isna().any().any() or template.row_id.duplicated().any():
        raise ValueError('Invalid or duplicate public template row IDs')
    keys = keys.astype(int)
    if set(keys[0]) != {int(e) for e in roster}:
        raise ValueError('Video roster and submission episode IDs disagree')
    episodes = []
    for episode_text, objects in roster.items():
        episode = int(episode_text)
        rows = keys[keys[0] == episode]
        frames = int(rows[2].max()) + 1
        expected = {(0, f, o) for f in range(frames) for o in range(len(objects))}
        if set(rows[[1, 2, 3]].itertuples(index=False, name=None)) != expected:
            raise ValueError(f'Incomplete public frame/world/object grid: {episode}')
        videos = {}
        for camera in ('ego_cam_a', 'ego_cam_b', 'ego_cam_c'):
            path = source / f'evaluation/videos/chunk-000/observation.images.{camera}/episode_{episode:06d}.mp4'
            cap = cv2.VideoCapture(str(path))
            try:
                count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                size = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]
                fps = cap.get(cv2.CAP_PROP_FPS)
                if count != frames or fps <= 0:
                    raise ValueError(f'Video/header roster mismatch: {path}: {count} != {frames}')
                for frame in (0, frames - 1):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
                    success, image = cap.read()
                    if not success or image.shape[:2] != (size[1], size[0]):
                        raise ValueError(f'Cannot decode boundary frame {frame}: {path}')
            finally:
                cap.release()
            videos[camera] = {'path': str(path.relative_to(ROOT)), 'sha256': sha256(path),
                              'frames': count, 'fps': fps, 'resolution': size,
                              'decode_check': 'first and last frame; full sequential decoding not asserted'}
        episodes.append({'episode_index': episode, 'objects_in_scored_order': objects,
                         'frames': frames, 'required_rows': len(rows), 'videos': videos})
    development = json.loads((source / 'public/meta/info.json').read_text())
    public_ids = set(development['episode_indices'])
    if public_ids & {e['episode_index'] for e in episodes}:
        raise ValueError('Development/evaluation episode IDs overlap')
    files = [p for p in source.rglob('*') if p.is_file()]
    report = {'observed_utc': datetime.now(timezone.utc).isoformat(),
              'state': 'INPUTS_VERIFIED', 'track': 'track_3', 'dataset': DATASET,
              'dataset_revision': REVISION, 'sample_sha256': sha256(template_path),
              'roster_sha256': sha256(roster_path), 'episode_count': len(episodes),
              'object_instance_count': sum(len(e['objects_in_scored_order']) for e in episodes),
              'scored_frames': sum(e['frames'] for e in episodes),
              'required_submission_rows': len(template),
              'development_episode_indices': sorted(public_ids),
              'downloaded_files': len(files), 'downloaded_bytes': sum(p.stat().st_size for p in files),
              'pilot_episode': 38, 'episodes': episodes,
              'input_policy': 'Reconstruct evaluation objects from released ego video; public scans are development assets, never submission meshes.',
              'coordinate_policy': 'Metres, continuous shared world for all objects; rollout quaternion xyzw; mesh body axes and poses must agree.',
              'camera_policy': 'Videos already upright; no additional 180 degree video rotation. Factory calibration has rotation metadata; verify its pixel convention before applying it.',
              'submission_ready': False, 'uploaded': False}
    destination = ROOT / 'incident_reports/track3_baseline/inputs.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('state', 'episode_count', 'object_instance_count', 'scored_frames', 'required_submission_rows', 'downloaded_files')}, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true')
    prepare(parser.parse_args().download)
