"""Local official Track 1 metric harness; never reads confidential references."""
import argparse
from pathlib import Path
import numpy as np
from cross_track.common import metric, sha, save
from cross_track.temporal import acceleration


def mesh_centroid(vertices, faces):
    triangles = np.asarray(vertices, float)[np.asarray(faces, int)]
    area = np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]), axis=1)/2
    if not np.isfinite(area).all() or area.sum() <= 0:
        raise ValueError('Mesh has no finite surface area')
    return np.average(triangles.mean(axis=1), axis=0, weights=area)


def recenter(vertices, faces, rotations, translation, scale):
    center = mesh_centroid(vertices, faces)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Invalid object scale')
    return np.asarray(vertices, float)-center, np.asarray(translation, float)+scale*np.einsum('tij,j->ti', rotations, center), center


def verify_recenter(vertices, faces, rotations, translation, scale, *, tolerance=1e-6):
    """Prove surface and PEN neutrality, not zero penetration or zero ACC error."""
    op = metric('track_1', 'ACC-O')
    moved, shifted, center = recenter(vertices, faces, rotations, translation, scale)
    before = op._sample_mesh_surface(np.asarray(vertices), faces, 512, 42)
    after = op._sample_mesh_surface(moved, faces, 512, 42)
    maximum = 0.
    for r, t, new_t in zip(rotations, translation, shifted):
        maximum = max(maximum, float(np.abs(scale*before@r.T+t-(scale*after@r.T+new_t)).max()))
    if maximum > tolerance:
        raise ValueError('Recentring changed world-space samples')
    return {'center': center.tolist(), 'max_world_sample_error_m': maximum,
            'faces_unchanged': True, 'scope': 'Float64 sampled-surface equivalence before serialization; verify packed output separately'}


def score_packed(reference, prediction, body_asset):
    """All official metrics on a participant-owned pseudo-reference.

    PEN is a property of the prediction, so self-PEN may be nonzero.
    The asset is an explicitly supplied participant-owned MHR decoder export.
    """
    import pandas as pd
    op = metric('track_1', 'ACC-O')
    body = op._load_body(body_asset)
    base, new = pd.read_parquet(reference), pd.read_parquet(prediction)
    if new.row_id.duplicated().any() or set(base.row_id) != set(new.row_id):
        raise ValueError('Prediction row grid differs from pseudo-reference')
    new = new.set_index('row_id').loc[base.row_id].reset_index()
    rows = []
    # Official helpers expect the enriched array_name/frame columns from the row IDs.
    from cross_track.common import kit_imports
    kit_imports()
    from v2dlb.mhr_submission import array_order
    for table in (base, new):
        parts = table.row_id.str.split('_', expand=True)
        if parts.shape[1] != 5 or not (parts[0] == 't1').all(): raise ValueError('Invalid Track 1 row IDs')
        table['episode_index'] = parts[1].astype(int)
        table['frame_index'] = parts[2].astype(int)
        table['array_name'] = parts[3].astype(int).map(dict(enumerate(array_order())))
        table['point_index'] = parts[4].astype(int)
    for episode, group in base.groupby('episode_index'):
        candidate = new[new.episode_index == episode]
        group = group.sort_values(['array_name', 'frame_index', 'point_index'])
        candidate = candidate.sort_values(['array_name', 'frame_index', 'point_index'])
        target, frames, _ = op._to_frames(group, ['x', 'y', 'z'], int(episode), body)
        pred, actual_frames, scene = op._to_frames(candidate, ['x', 'y', 'z'], int(episode), body)
        if not np.array_equal(frames, actual_frames): raise ValueError('Frame grid changed')
        values = op._episode_metrics(pred, target, frames, None)
        values['interpenetration_cm'] = op._episode_penetration(pred, scene, target)
        rows.append({'episode_index': int(episode), **values})
    return {'scope': 'Official metric functions against a participant-owned pseudo-reference, not hidden-reference accuracy',
            'episodes': rows, 'mean': {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != 'episode_index'},
            'official_metric_sha256': sha(op.__file__), 'body_asset_sha256': sha(body_asset)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--prediction', type=Path, required=True)
    p.add_argument('--body-asset', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    a = p.parse_args()
    save(a.report, score_packed(a.reference, a.prediction, a.body_asset))


if __name__ == '__main__': main()
