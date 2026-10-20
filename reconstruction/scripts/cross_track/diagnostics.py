"""CPU-only baseline tables; no metric scores are inferred from leaderboard ranks."""
import argparse
from pathlib import Path
import csv
import numpy as np
import trimesh
from .common import metric, sha, save, load, new_run, complete
from .temporal import acceleration
from .human import decompose
from .policy import diagnostics, checkpoint_metadata
from track1_metric_harness import mesh_centroid


def track1_rows(parent, scored, body_asset=None):
    body = metric('track_1', 'ACC-O')._load_body(body_asset) if body_asset else None
    rows = []
    for episode in range(30):
        stem = f'episode_{episode:06d}'; b = dict(np.load(Path(parent)/(stem+'.npz'), allow_pickle=False))
        mesh = trimesh.load(Path(parent)/(stem+'_object.glb'), force='mesh', process=False)
        center = mesh_centroid(mesh.vertices, mesh.faces)
        leak = float(b['object_scale'])*np.einsum('tij,j->ti', b['object_rotation'], center)
        parts = mesh.split(only_watertight=False)
        frames = np.asarray(scored[str(episode)])
        row = {'episode_index': episode, 'translation_acceleration_cm': acceleration(b['object_translation'], frames),
               'centroid_acceleration_cm': acceleration(b['object_translation']+leak, frames),
               'rotation_leak_acceleration_cm': acceleration(leak, frames),
               'mesh_origin_offset_m': float(np.linalg.norm(center)*float(b['object_scale'])),
               'mesh_area_outside_largest_component': float(1-max(p.area for p in parts)/mesh.area)}
        if body is not None: row.update(decompose(body, b['pose'], b['scales'], b['shape'], frames))
        else: row['human_diagnostic'] = 'requires_participant_owned_decoder_asset'
        rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--track', choices=['track_1', 'track_2_tier1', 'track_3'], required=True)
    p.add_argument('--manifest', type=Path, required=True, help='JSON object with source roots and episode receipt paths')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); spec = load(a.manifest)
    new_run(a.output, [a.manifest], {'track': a.track})
    if a.track == 'track_1': rows = track1_rows(spec['parent'], load(spec['scored']), spec.get('body_asset'))
    else:
        rows = []
        for item in spec['episodes']:
            row = diagnostics(item['trajectory'], a.track, item['episode_index'], reference_root=spec.get('reference_root'), checkpoint=item['checkpoint'])
            metadata = checkpoint_metadata(item['checkpoint'])
            row.update(trained_environment_steps=int(metadata['environment_steps']), curriculum_stage=metadata.get('curriculum_stage'))
            rows.append(row)
    save(a.output/'diagnostics.json', rows)
    scalar_keys = sorted({k for r in rows for k, v in r.items() if v is None or isinstance(v, (str, int, float, bool))})
    with (a.output/'diagnostics.csv').open('w') as stream:
        writer = csv.DictWriter(stream, scalar_keys, extrasaction='ignore'); writer.writeheader(); writer.writerows(rows)
    complete(a.output, 'DIAGNOSTICS_COMPLETE', episodes=len(rows), scope='Participant-owned inputs only; missing evidence remains explicitly missing')


if __name__ == '__main__': main()
