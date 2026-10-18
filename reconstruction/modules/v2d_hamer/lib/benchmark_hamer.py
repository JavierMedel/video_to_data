"""Compare real HaMeR crop batches with identical frame/mask inputs."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import statistics
import time

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import torch

from v2d.hamer.lib._hamer import get_model, run_hamer
from v2d.hamer.lib.masks_to_hands import masks_to_hands, _mask_centroid_and_size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work_dir', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--weights_dir', required=True)
    parser.add_argument('--batches', type=int, nargs='+', default=[1, 4, 8])
    parser.add_argument('--crops_per_hand', type=int, default=16)
    args = parser.parse_args()
    if not args.batches or args.batches[0] != 1 or min(args.batches) < 1 or args.crops_per_hand < 1:
        raise ValueError('Start with batch 1 and use positive batch/crop counts')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    work, output = args.work_dir, args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    tracks = json.loads((work / 'hand_tracks.json').read_text())['tracks']
    count = 0
    warmup = None
    for track in tracks:
        oid = str(track['object_id'])
        target = output / 'masks' / oid
        target.mkdir(parents=True, exist_ok=True)
        selected = 0
        for mask_path in sorted((work / 'hand_masks' / oid).glob('*.png')):
            mask = np.asarray(Image.open(mask_path)) > 0
            cx, cy, size, pixels = _mask_centroid_and_size(mask)
            if pixels < 256:
                continue
            shutil.copy2(mask_path, target / mask_path.name)
            if warmup is None:
                image = np.asarray(Image.open(work / 'frames' / (mask_path.stem + '.jpg')).convert('RGB'))
                warmup = (image, cx, cy, size * 1.7, bool(track['is_right']))
            count += 1
            selected += 1
            if selected == args.crops_per_hand:
                break
    if count != 2 * args.crops_per_hand:
        raise ValueError('Benchmark needs the requested valid crops for each of two hands')
    model, cfg = get_model(args.weights_dir)
    run_hamer(model, cfg, *warmup)
    torch.cuda.synchronize()
    results = []
    baseline = None
    for trial, batch in enumerate(args.batches * 2):
        destination = output / f'trial_{trial}_batch_{batch}'
        if destination.exists():
            raise ValueError('Use a fresh benchmark output directory')
        torch.cuda.reset_peak_memory_stats()
        started_utc = datetime.now(timezone.utc).isoformat()
        started = time.perf_counter()
        masks_to_hands(str(work / 'frames'), str(output / 'masks'),
                       str(work / 'hand_tracks.json'), str(destination), args.weights_dir,
                       batch_size=batch)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        records = {str(p.relative_to(destination)): json.loads(p.read_text())
                   for p in destination.glob('*/*.json')}
        if baseline is None:
            baseline = records
        if set(records) != set(baseline) or len(records) != count:
            raise ValueError('Batching changed frame/hand coverage')
        cam_error = beta_error = angle_error = 0.
        for key, record in records.items():
            original = baseline[key]
            for metadata in ('track_id', 'is_right', 'frame_idx', 'image_size', 'bbox'):
                if record[metadata] != original[metadata]:
                    raise ValueError('Batching changed crop metadata')
            cam_error = max(cam_error, float(np.linalg.norm(np.subtract(record['camera']['cam_t'], original['camera']['cam_t']))))
            beta_error = max(beta_error, float(np.max(np.abs(np.subtract(record['mano']['betas'], original['mano']['betas'])))))
            for field in ('global_orient', 'hand_pose'):
                actual = Rotation.from_rotvec(np.asarray(record['mano'][field]).reshape(-1, 3))
                expected = Rotation.from_rotvec(np.asarray(original['mano'][field]).reshape(-1, 3))
                angle_error = max(angle_error, float(np.rad2deg((actual * expected.inv()).magnitude()).max()))
        if cam_error > .001 or beta_error > .01 or angle_error > .1:
            raise ValueError(f'Batch numerical drift exceeds tolerance: {cam_error}, {beta_error}, {angle_error}')
        result = dict(batch=batch, seconds=elapsed, crops=count, crops_per_second=count / elapsed,
                      started_utc=started_utc,
                      torch_peak_mib=torch.cuda.max_memory_allocated() / 2**20,
                      torch_peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                      max_camera_error_m=cam_error, max_beta_error=beta_error, max_rotation_error_deg=angle_error)
        results.append(result)
        print('BENCHMARK', json.dumps(result), flush=True)
    medians = {batch: statistics.median(r['seconds'] for r in results if r['batch'] == batch) for batch in args.batches}
    best = min(medians, key=medians.get)
    report = dict(state='COMPLETE', trials=results, median_seconds=medians,
                  recommended_batch=best, speedup_over_batch1=medians[1] / medians[best],
                  includes_crop_preparation_and_json_writes=True, includes_model_loading=False)
    (output / 'benchmark.json').write_text(json.dumps(report, indent=2) + '\n')
    print('RESULT', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
