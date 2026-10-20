"""CPU decoder parity and gradient checks on independent synthetic assets."""
from pathlib import Path
import sys
import numpy as np
import pytest
from scipy import sparse

torch = pytest.importorskip('torch')
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cross_track.mhr_torch import MHRBody
from cross_track.common import metric


def asset():
    rng = np.random.default_rng(4); vertices = 12
    w1 = sparse.random(8, 750, density=.1, random_state=4).tocoo()
    w2 = sparse.random(vertices*3, 8, density=.2, random_state=4).tocsr()
    skin = rng.uniform(size=(vertices, 127)); skin /= skin.sum(axis=1, keepdims=True)
    q = np.tile([0., 0., 0., 1.], (127, 1))
    return {'joint_parents': np.r_[-1, np.arange(126)], 'joint_offsets': rng.normal(0, .1, (127, 3)),
            'joint_prerotations': q, 'parameter_transform': rng.normal(0, .001, (889, 204)),
            'inverse_bind_pose': np.c_[np.zeros((127, 3)), q, np.ones(127)],
            'base_shape': rng.normal(size=(vertices, 3)), 'shape_vectors': rng.normal(0, .1, (45, vertices, 3)),
            'skin_dense': skin, 'corr_w1_idx': np.array([w1.row, w1.col]), 'corr_w1_val': w1.data, 'corr_w1_shape': w1.shape,
            'corr_w2_data': w2.data, 'corr_w2_indices': w2.indices, 'corr_w2_indptr': w2.indptr, 'corr_w2_shape': w2.shape,
            **{'role_'+role: np.arange(4) for role in ('alignment', 'body', 'left_hand', 'right_hand')}}


def test_torch_decoder_matches_official_numpy_vertices_and_joints():
    arrays = asset(); official = metric('track_1', 'ACC-O')._MHRBody(arrays); model = MHRBody(arrays)
    rng = np.random.default_rng(42); params = rng.normal(0, .05, (5, 204)); shape = rng.normal(0, .01, 45)
    actual = model(torch.tensor(params), torch.tensor(shape))
    expected = official(params, shape)
    for a, b in zip(actual, expected): assert np.max(np.abs(a.detach().numpy()-b)) < 1e-12


def test_differentiable_correctives_and_fk_match_finite_difference():
    model = MHRBody(asset())
    params = torch.tensor(np.random.default_rng(42).normal(0, .05, (2, 204)), dtype=torch.float64, requires_grad=True)
    shape = torch.zeros(45, dtype=torch.float64)
    vertices, joints = model(params, shape); loss = vertices.square().sum()+joints.square().sum(); loss.backward()
    eps = 1e-5; index = 3
    with torch.no_grad():
        a, b = params.clone(), params.clone(); a[0, index] += eps; b[0, index] -= eps
        va, ja = model(a, shape); vb, jb = model(b, shape)
        finite = ((va.square().sum()+ja.square().sum())-(vb.square().sum()+jb.square().sum()))/(2*eps)
    assert float(params.grad[0, index]) == pytest.approx(float(finite), rel=2e-4, abs=1e-8)


def test_scripted_export_handles_zero_padded_transform_and_flattened_skin(tmp_path):
    from cross_track.mhr_asset import extract
    from types import SimpleNamespace
    a = asset(); nv = len(a['base_shape'])
    names = {'joint_parents': 'character_torch.skeleton.joint_parents',
             'joint_offsets': 'character_torch.skeleton.joint_translation_offsets',
             'joint_prerotations': 'character_torch.skeleton.joint_prerotations',
             'inverse_bind_pose': 'character_torch.linear_blend_skinning.inverse_bind_pose',
             'base_shape': 'character_torch.blend_shape.base_shape', 'shape_vectors': 'character_torch.blend_shape.shape_vectors',
             'corr_w1_idx': 'pose_correctives_model.pose_dirs_predictor.0.sparse_indices',
             'corr_w1_val': 'pose_correctives_model.pose_dirs_predictor.0.sparse_weight'}
    tensors = {name: torch.tensor(a[key]) for key, name in names.items()}
    transform = np.c_[a['parameter_transform'], np.ones((889, 45))*99]
    tensors['character_torch.parameter_transform.parameter_transform'] = torch.tensor(transform)
    tensors['character_torch.mesh.faces'] = torch.tensor([[0, 1, 2]])
    vertex, joint = np.nonzero(a['skin_dense'])
    for key, value in [('vert_indices_flattened', vertex), ('skin_indices_flattened', joint), ('skin_weights_flattened', a['skin_dense'][vertex, joint])]:
        tensors['character_torch.linear_blend_skinning.'+key] = torch.tensor(value)
    w2 = sparse.csr_matrix((a['corr_w2_data'], a['corr_w2_indices'], a['corr_w2_indptr']), shape=a['corr_w2_shape'])
    tensors['pose_correctives_model.pose_dirs_predictor.2.weight'] = torch.tensor(w2.toarray())
    source = SimpleNamespace(named_buffers=lambda: tensors.items(), named_parameters=lambda: [])
    spec = tmp_path/'hands.npz'; np.savez_compressed(spec, vertex_indices=np.array([[0, 1], [2, 3]]))
    actual, _ = extract(source, spec)
    assert actual['parameter_transform'].shape == (889, 204)
    op = metric('track_1', 'ACC-O')
    params = np.random.default_rng(42).normal(0, .01, (2, 204)); shape = np.zeros(45)
    before, after = op._MHRBody(a)(params, shape), op._MHRBody(actual)(params, shape)
    for first, second in zip(before, after): assert np.allclose(first, second, atol=1e-12)
    tensors['character_torch.linear_blend_skinning.skin_weights_flattened'] *= .9
    with pytest.raises(ValueError, match='normalization'): extract(source, spec)
