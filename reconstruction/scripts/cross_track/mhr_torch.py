"""Differentiable float64 port of the official NumPy MHR decoder.

Requires a participant-owned decoder asset export; never fetches host assets.
Real-data use requires verify_parity against the official NumPy decoder first.
"""
import numpy as np
import torch
from torch import nn


def euler_xyz(e):
    cx, cy, cz = (torch.cos(e[..., i]) for i in range(3))
    sx, sy, sz = (torch.sin(e[..., i]) for i in range(3))
    return torch.stack([cy*cz, -cx*sz+sx*sy*cz, sx*sz+cx*sy*cz,
                        cy*sz, cx*cz+sx*sy*sz, -sx*cz+cx*sy*sz,
                        -sy, sx*cy, cx*cy], -1).reshape(*e.shape[:-1], 3, 3)


class MHRBody(nn.Module):
    def __init__(self, asset):
        super().__init__()
        from .common import metric
        official = metric('track_1', 'ACC-O')
        self.parents = np.asarray(asset['joint_parents'], dtype=int).tolist()
        if len(self.parents) != 127 or any(parent >= j for j, parent in enumerate(self.parents)):
            raise ValueError('Expected 127 topologically ordered joints')
        def buffer(key, value): self.register_buffer(key, torch.as_tensor(np.asarray(value).copy(), dtype=torch.float64))
        for key in ('parameter_transform', 'joint_offsets', 'skin_dense'):
            buffer(key, asset[key])
        buffer('pre_rotation', official._quat_to_mat(asset['joint_prerotations']))
        bind = asset['inverse_bind_pose']
        buffer('bind_t', bind[:, :3]); buffer('bind_R', official._quat_to_mat(bind[:, 3:7])); buffer('bind_s', bind[:, 7])
        buffer('base_shape', asset['base_shape'].reshape(-1))
        buffer('shape_vectors', asset['shape_vectors'].reshape(45, -1))
        indices = torch.tensor(np.asarray(asset['corr_w1_idx']), dtype=torch.int64)
        self.register_buffer('w1', torch.sparse_coo_tensor(indices, torch.tensor(asset['corr_w1_val'], dtype=torch.float64), tuple(asset['corr_w1_shape'])).coalesce())
        from scipy.sparse import csr_matrix
        w2 = csr_matrix((asset['corr_w2_data'], asset['corr_w2_indices'], asset['corr_w2_indptr']), shape=tuple(asset['corr_w2_shape'])).tocoo()
        self.register_buffer('w2', torch.sparse_coo_tensor(torch.tensor(np.array([w2.row, w2.col])), torch.tensor(w2.data, dtype=torch.float64), w2.shape).coalesce())
        buffer('flip', [1., -1., -1.])
        self.roles = {name: np.asarray(asset['role_'+name], dtype=int) for name in ('alignment', 'body', 'left_hand', 'right_hand')}

    def skeleton(self, params):
        joint = (params@self.parameter_transform.T).reshape(-1, 127, 7)
        local_t = joint[..., :3]+self.joint_offsets
        local_r = self.pre_rotation[None]@euler_xyz(joint[..., 3:6])
        local_s = torch.exp2(joint[..., 6])
        wt, wr, ws = [], [], []
        for j, parent in enumerate(self.parents):
            if parent < 0:
                wt.append(local_t[:, j]); wr.append(local_r[:, j]); ws.append(local_s[:, j])
            else:
                wt.append(wt[parent]+ws[parent][:, None]*(wr[parent]@local_t[:, j, :, None])[..., 0])
                wr.append(wr[parent]@local_r[:, j]); ws.append(ws[parent]*local_s[:, j])
        return joint, torch.stack(wt, 1), torch.stack(wr, 1), torch.stack(ws, 1)

    def joints(self, params):
        return self.skeleton(params)[1]/100*self.flip

    def forward(self, params, shape):
        joint, wt, wr, ws = self.skeleton(params)
        rotation = euler_xyz(joint[:, 2:, 3:6])
        features = torch.cat([rotation[..., :, 0], rotation[..., :, 1]], -1)
        identity = torch.tensor([1., 0., 0., 0., 1., 0.], dtype=params.dtype, device=params.device)
        features = features-identity
        hidden = torch.relu(torch.sparse.mm(self.w1, features.reshape(len(params), -1).T))
        rest = self.base_shape + shape@self.shape_vectors + torch.sparse.mm(self.w2, hidden).T
        rest = rest.reshape(len(params), -1, 3)
        skin_r = (ws[..., None, None]*wr)@(self.bind_s[:, None, None]*self.bind_R)
        skin_t = wt+ws[..., None]*torch.einsum('njab,jb->nja', wr, self.bind_t)
        blend = torch.cat([skin_r, skin_t[..., None]], -1).reshape(len(params), 127, 12)
        mixed = self.skin_dense@blend.transpose(0, 1).reshape(127, len(params)*12)
        mixed = mixed.reshape(-1, len(params), 3, 4).permute(1, 0, 2, 3)
        vertices = torch.einsum('nvab,nvb->nva', mixed[..., :3], rest)+mixed[..., 3]
        return vertices/100*self.flip, wt/100*self.flip


def verify_parity(model, official, params, shape, tolerance=1e-5):
    """Require >=100 distinct real frames, float64 vertices AND joints."""
    if len(params) < 100: raise ValueError('Parity requires at least 100 real frames')
    maximum = 0.
    for start in range(0, len(params), 8):
        block = params[start:start+8]
        expected = official(block, shape)
        with torch.no_grad():
            actual = model(torch.as_tensor(block, dtype=torch.float64, device=model.flip.device),
                           torch.as_tensor(shape, dtype=torch.float64, device=model.flip.device))
        maximum = max(maximum, *(float(np.max(np.abs(a.detach().cpu().numpy()-b))) for a, b in zip(actual, expected)))
    if maximum > tolerance: raise ValueError(f'Decoder parity failed: {maximum:g} m')
    return maximum
