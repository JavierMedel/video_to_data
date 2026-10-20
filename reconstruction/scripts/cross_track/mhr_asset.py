"""Export participant TorchScript buffers, with real three-decoder parity gates."""
import argparse
from pathlib import Path
import numpy as np
from scipy import sparse
from .common import new_run, save, sha, complete, metric


def script_model(path):
    import torch
    return torch.jit.load(str(path), map_location='cpu').eval()


def inventory(model_path, output):
    new_run(output, [model_path], {'method': 'participant_torchscript_inventory_v1'})
    model = script_model(model_path)
    tensors = {name: {'shape': list(t.shape), 'dtype': str(t.dtype), 'layout': str(t.layout)}
               for name, t in list(model.named_buffers())+list(model.named_parameters())}
    code = {name or 'forward': module.code for name, module in model.named_modules() if hasattr(module, 'code')}
    save(Path(output)/'torchscript_inventory.json', {'model_sha256': sha(model_path), 'tensors': tensors, 'code': code})
    complete(output, 'INVENTORIED', tensor_count=len(tensors))


def extract(model, hand_spec):
    """Known local model layout, not a shape-based tensor guessing heuristic."""
    tensors = dict(model.named_buffers()); tensors.update(dict(model.named_parameters()))
    mapping = {
        'joint_parents': 'character_torch.skeleton.joint_parents',
        'joint_offsets': 'character_torch.skeleton.joint_translation_offsets',
        'joint_prerotations': 'character_torch.skeleton.joint_prerotations',
        'parameter_transform': 'character_torch.parameter_transform.parameter_transform',
        'inverse_bind_pose': 'character_torch.linear_blend_skinning.inverse_bind_pose',
        'base_shape': 'character_torch.blend_shape.base_shape',
        'shape_vectors': 'character_torch.blend_shape.shape_vectors',
        'faces': 'character_torch.mesh.faces',
        'corr_w1_idx': 'pose_correctives_model.pose_dirs_predictor.0.sparse_indices',
        'corr_w1_val': 'pose_correctives_model.pose_dirs_predictor.0.sparse_weight',
    }
    asset = {key: tensors[name].detach().cpu().numpy().copy() for key, name in mapping.items()}
    transform = asset['parameter_transform']
    if transform.shape != (889, 249):
        raise ValueError('Unsupported scripted parameter transform; audit the forward before exporting')
    # Scripted forward appends 45 ZERO identity parameters before applying this matrix.
    asset['parameter_transform'] = transform[:, :204].copy()
    nv = asset['base_shape'].shape[0]
    skin = np.zeros((nv, 127), np.float64)
    prefix = 'character_torch.linear_blend_skinning.'
    vertex = tensors[prefix+'vert_indices_flattened'].cpu().numpy()
    joint = tensors[prefix+'skin_indices_flattened'].cpu().numpy()
    weight = tensors[prefix+'skin_weights_flattened'].cpu().numpy()
    np.add.at(skin, (vertex, joint), weight)
    if not np.isfinite(skin).all() or np.any(skin < 0) or not np.allclose(skin.sum(1), 1., atol=1e-6, rtol=0):
        raise ValueError('Invalid LBS weight normalization')
    asset['skin_dense'] = skin
    w2 = sparse.csr_matrix(tensors['pose_correctives_model.pose_dirs_predictor.2.weight'].detach().cpu().numpy())
    asset.update(corr_w1_shape=np.array([w2.shape[1], 125*6]), corr_w2_data=w2.data,
                 corr_w2_indices=w2.indices, corr_w2_indptr=w2.indptr, corr_w2_shape=np.array(w2.shape))
    with np.load(hand_spec, allow_pickle=False) as spec:
        hands = np.asarray(spec['vertex_indices'], dtype=np.int64)
    if hands.ndim != 2 or hands.shape[0] != 2 or np.any(hands < 0) or np.any(hands >= nv):
        raise ValueError('Hand surface specification does not match this topology')
    asset.update(role_left_hand=hands[0], role_right_hand=hands[1], role_body=np.arange(nv),
                 role_alignment=np.setdiff1d(np.arange(nv), hands.ravel()))
    if asset['joint_parents'].shape != (127,) or asset['shape_vectors'].shape != (45, nv, 3):
        raise ValueError('Unsupported skeleton/blendshape layout')
    for name, value in asset.items():
        if not np.isfinite(value).all(): raise ValueError('Nonfinite decoder buffer: '+name)
    return asset, mapping


def export(model_path, hand_spec, output):
    previous = new_run(output, [model_path, hand_spec], {'method': 'participant_scripted_mhr_export_v1'})
    path = Path(output)/'mhr_body_asset.npz'
    if previous['state'] == 'EXPORTED_REQUIRES_REAL_PARITY':
        if sha(path) != previous['asset_sha256']: raise ValueError('Exported body asset changed')
        return path
    model = script_model(model_path)
    asset, mapping = extract(model, hand_spec)
    np.savez_compressed(path, **asset)
    save(Path(output)/'mapping.json', {'buffers': mapping, 'source_parameter_transform_shape': [889, 249],
         'export_transform': 'first 204 columns; scripted remaining 45 inputs are zero',
         'role_source': 'participant_proxy_v1', 'hand_order': ['left_hand', 'right_hand'],
         'units': 'raw centimeters; official output /100 and [1,-1,-1]', 'model_sha256': sha(model_path),
         'hand_spec_sha256': sha(hand_spec), 'scope': 'Participant roles; not exact host alignment/CD-H/PEN'})
    complete(output, 'EXPORTED_REQUIRES_REAL_PARITY', asset_sha256=sha(path))
    return path


def parity(asset_path, model_path, bundles, output, count=100, minimum_episodes=10, tolerance=1e-5):
    """Stratified real frames; stream small chunks to bound host memory."""
    if count < 100 or minimum_episodes < 10: raise ValueError('Require >=100 real frames across >=10 episodes')
    paths = sorted(Path(bundles).glob('episode_*.npz'))
    if len(paths) < minimum_episodes: raise ValueError('Insufficient distinct real episodes for parity')
    chosen = paths[:minimum_episodes]
    new_run(output, [asset_path, model_path, *chosen], {'count': count, 'episodes': minimum_episodes, 'tolerance_m': tolerance})
    import torch
    from .mhr_torch import MHRBody
    official = metric('track_1', 'ACC-O')._load_body(asset_path)
    asset = dict(np.load(asset_path, allow_pickle=False))
    port = MHRBody(asset)
    scripted = script_model(model_path).to(torch.float64)
    maximum_v, maximum_j, gradients, samples = 0., 0., [], []
    for index, path in enumerate(chosen):
        with np.load(path, allow_pickle=False) as b:
            pose, scales, shape = b['pose'], b['scales'], b['shape']
            n = count//len(chosen) + int(index < count % len(chosen))
            if len(pose) < n: raise ValueError('Episode too short for distinct parity samples')
            # Include large articulation along with uniformly spread frames.
            extreme = int(np.argmax(np.linalg.norm(pose[:, 6:], axis=1)))
            order = [int(f) for f in list(dict.fromkeys([extreme, *np.linspace(0, len(pose)-1, n, dtype=int), *range(len(pose))]))[:n]]
            params = np.c_[pose[order], np.broadcast_to(scales, (n, 68))].astype(np.float64)
        for start in range(0, n, 2):
            p = torch.tensor(params[start:start+2], dtype=torch.float64, requires_grad=True)
            sh = torch.tensor(shape, dtype=torch.float64)
            expected = official(p.detach().numpy(), shape, chunk=2)
            with torch.no_grad():
                raw_v, state = scripted(sh[None].expand(len(p), -1), p.detach(), torch.zeros((len(p), 72), dtype=torch.float64), True)
                raw = (raw_v.numpy()/100*np.array([1, -1, -1]), state[..., :3].numpy()/100*np.array([1, -1, -1]))
                actual = port(p.detach(), sh)
            maximum_v = max(maximum_v, float(np.linalg.norm(expected[0]-raw[0], axis=-1).max()), float(np.linalg.norm(expected[0]-actual[0].numpy(), axis=-1).max()))
            maximum_j = max(maximum_j, float(np.linalg.norm(expected[1]-raw[1], axis=-1).max()), float(np.linalg.norm(expected[1]-actual[1].numpy(), axis=-1).max()))
            loss = port.joints(p).square().mean(); loss.backward()
            if not torch.isfinite(p.grad).all() or not torch.any(p.grad != 0): raise ValueError('Real-pose gradients invalid')
            gradients.append(float(p.grad.abs().max()))
        samples.append({'bundle': str(path.resolve()), 'sha256': sha(path), 'frames': order})
    passed = maximum_v <= tolerance and maximum_j <= tolerance
    record = {'passed': passed, 'maximum_vertex_error_m': maximum_v, 'maximum_joint_error_m': maximum_j,
              'frames': count, 'episodes': len(chosen), 'samples': samples, 'gradient_maxima': gradients,
              'asset_sha256': sha(asset_path), 'model_path': str(Path(model_path).resolve()), 'model_sha256': sha(model_path),
              'scope': 'Three-decoder parity; roles remain participant proxies'}
    save(Path(output)/'decoder_parity.json', record)
    complete(output, 'REAL_PARITY_PASSED' if passed else 'REAL_PARITY_FAILED', **record)
    if not passed: raise ValueError(f'Real decoder parity failed: vertices={maximum_v:g}, joints={maximum_j:g} m')
    return record


def verify_report(path, asset_path):
    from .common import load, REPO
    record = load(path)
    if (not record.get('passed') or record.get('frames', 0) < 100 or record.get('episodes', 0) < 10
            or record.get('asset_sha256') != sha(asset_path)
            or record.get('maximum_vertex_error_m', float('inf')) > 1e-5
            or record.get('maximum_joint_error_m', float('inf')) > 1e-5):
        raise ValueError('Participant decoder lacks passing real three-way parity')
    def local(p):
        p = Path(p)
        return REPO/p.relative_to('/repo') if p.is_relative_to('/repo') else p
    samples = record.get('samples', [])
    if len({str(local(s['bundle']).resolve()) for s in samples}) < 10 or sum(len(set(s['frames'])) for s in samples) < 100:
        raise ValueError('Parity sampling coverage is incomplete')
    for item in samples:
        if sha(local(item['bundle'])) != item['sha256']: raise ValueError('Parity real-frame source changed')
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    action = __import__('sys').argv[0].split()[-1]
    if action == 'mhr-export': p.add_argument('--hand-spec', type=Path, required=True)
    if action == 'mhr-parity':
        p.add_argument('--asset', type=Path, required=True); p.add_argument('--bundles', type=Path, required=True)
        p.add_argument('--frames', type=int, default=100)
    a = p.parse_args()
    if action == 'mhr-inventory': inventory(a.model, a.output)
    elif action == 'mhr-export': export(a.model, a.hand_spec, a.output)
    else: parity(a.asset, a.model, a.bundles, a.output, a.frames)
