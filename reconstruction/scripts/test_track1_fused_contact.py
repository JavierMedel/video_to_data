"""Check fused exact-mesh depths against eager and time a dense real mesh."""

def main():
    import os
    os.environ.setdefault('LOGNAME', f'uid{os.getuid()}')
    os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS', '1')
    os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', f'/tmp/v2d-fused-contact-{os.getuid()}/inductor')
    os.environ.setdefault('TRITON_CACHE_DIR', f'/tmp/v2d-fused-contact-{os.getuid()}/triton')
    import json
    import sys
    import time
    from pathlib import Path

    import numpy as np
    import torch
    import trimesh

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root/'modules/v2d_cari4d/lib/cari4d'))
    from lib_mhr.mesh_penetration_review import MeshPenetrationReview
    from lib_mhr.fused_mesh_penetration_review import FusedMeshPenetrationReview

    meshes = [trimesh.creation.box(extents=[2., 2., 2.])]
    opened = meshes[0].copy(); opened.update_faces(np.arange(len(opened.faces)) >= 2); meshes.append(opened)
    reversed_mesh = meshes[0].copy(); reversed_mesh.faces = reversed_mesh.faces[:, ::-1]; meshes.append(reversed_mesh)
    control = root/'data/v2d_challenge/track_1/cdo_v6_experiments/candidate_episodes'
    meshes += [trimesh.load(control/f'episode_{i:06d}_object.glb', force='mesh', process=False) for i in (1, 16, 25)]
    maximum = 0.
    for i, mesh in enumerate(meshes):
        rng = np.random.default_rng(i+17)
        ids = np.linspace(0, len(mesh.vertices)-1, 32).round().astype(int)
        points = np.concatenate([mesh.vertices[ids]+rng.normal(size=(32, 3))*1e-6,
                                 rng.uniform(mesh.bounds[0]-.01, mesh.bounds[1]+.01, size=(128, 3))])
        eager = MeshPenetrationReview(mesh.vertices, mesh.faces)
        fused = FusedMeshPenetrationReview(mesh.vertices, mesh.faces)
        before = eager.depths(points).cpu().numpy(); after = fused.depths(points).cpu().numpy()
        error = float(abs(before-after).max()); maximum = max(maximum, error)
        np.testing.assert_allclose(before, after, atol=2e-7, rtol=1e-6)
        print('FUSED_DEPTH_MATCH', i, len(mesh.faces), error, flush=True)
    mesh = meshes[-2]; rng = np.random.default_rng(21)
    points = rng.uniform(mesh.bounds[0], mesh.bounds[1], size=(512, 3))
    eager = MeshPenetrationReview(mesh.vertices, mesh.faces); fused = FusedMeshPenetrationReview(mesh.vertices, mesh.faces)
    fused.depths(points); torch.cuda.synchronize()
    timings = {}
    for name, checker in [('eager', eager), ('fused', fused)]:
        start = time.perf_counter()
        for _ in range(3): checker.depths(points)
        torch.cuda.synchronize(); timings[name] = (time.perf_counter()-start)/3
    result = {'matched_meshes': len(meshes), 'max_depth_difference_m': maximum, 'seconds': timings,
              'speedup': timings['eager']/timings['fused'], 'scope': 'Six mesh/depth comparisons including open/reversed winding and dense real surfaces; timing under concurrent campaign load.'}
    print('FUSED_CONTACT_RESULT', json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
