"""Atomic, input-bound recovery for quality-gated object tracking."""
import os
import pickle
from pathlib import Path
import numpy as np

SCHEMA = 'v2d.track1.quality_tracker_checkpoint.v1'


def save_checkpoint(path, signature, names, poses, records, previous):
    path = Path(path)
    payload = {'schema': SCHEMA, 'signature': signature, 'names': names,
               'poses': poses, 'records': records, 'previous': previous}
    validate_checkpoint(payload, signature, names)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with temporary.open('wb') as f:
        pickle.dump(payload, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def validate_checkpoint(payload, signature, names):
    if payload.get('schema') != SCHEMA or payload.get('signature') != signature or payload.get('names') != names:
        raise ValueError('Tracker checkpoint inputs or implementation changed; use a new output directory')
    poses, records, previous = payload['poses'], payload['records'], payload['previous']
    if len(poses) != len(records) or len(records) > len(names):
        raise ValueError('Tracker checkpoint has inconsistent frame coverage')
    if [r['frame'] for r in records] != names[:len(records)]:
        raise ValueError('Tracker checkpoint frames are not a contiguous prefix')
    for pose in poses:
        if pose is not None and (np.asarray(pose).shape != (4,4) or not np.isfinite(pose).all()):
            raise ValueError('Tracker checkpoint has an invalid pose')
    if previous is not None and (np.asarray(previous).shape != (4,4) or not np.isfinite(previous).all()):
        raise ValueError('Tracker checkpoint has an invalid trusted state')
    if bool(any(r['reliable'] for r in records)) != (previous is not None):
        raise ValueError('Tracker checkpoint trusted state disagrees with accepted frames')
    return poses, records, previous


def load_checkpoint(path, signature, names):
    with Path(path).open('rb') as f:
        payload = pickle.load(f)
    return validate_checkpoint(payload, signature, names)
