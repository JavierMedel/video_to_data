"""Derive a labelled left MANO asset from the existing HaMeR right model.

HaMeR predicts both sides with the right model. This preserves that convention,
with an exact reflected skinning model for the native two-side FK loader.
Run only inside the dedicated licensed-model inference image.
"""
import copy
import hashlib
import json
from pathlib import Path
import pickle

import chumpy as ch
import numpy as np
import torch
from manotorch.manolayer import ManoLayer

ROOT = Path('/repo/reconstruction/data/weights/track3_mano')
source = ROOT / 'models/MANO_RIGHT.pkl'
target = ROOT / 'models/MANO_LEFT.pkl'
model = pickle.loads(source.read_bytes(), encoding='latin1')
derived = copy.deepcopy(model)
reflect = np.array([-1., 1., 1.])
axis_sign = -reflect
arr = lambda x: np.asarray(x.r if hasattr(x, 'r') else x).copy()
for key in ('v_template', 'J'):
    if key in model: derived[key] = arr(model[key]) * reflect
# Keep Chumpy wrappers where manotorch expects .r.
derived['shapedirs'] = ch.array(arr(model['shapedirs']) * reflect[None, :, None])
feature_sign = np.tile(np.outer(reflect, reflect).reshape(-1), 15)
derived['posedirs'] = ch.array(arr(model['posedirs']) * reflect[None, :, None] * feature_sign[None, None, :])
for key in ('hands_components', 'hands_mean'):
    if key in model: derived[key] = arr(model[key]) * np.tile(axis_sign, 15)
derived['f'] = arr(model['f'])[:, [0, 2, 1]]
target.write_bytes(pickle.dumps(derived, protocol=4))
right = ManoLayer(use_pca=False, side='right', center_idx=None, flat_hand_mean=True, mano_assets_root=str(ROOT))
left = ManoLayer(use_pca=False, side='left', center_idx=None, flat_hand_mean=True, mano_assets_root=str(ROOT))
torch.manual_seed(42)
poses = torch.randn(5, 48) * .15
betas = torch.randn(5, 10) * .05
r = right(poses, betas)
l = left(poses * torch.tensor(np.tile(axis_sign, 16), dtype=torch.float32), betas)
expected_joints = r.joints.clone()
# manotorch deliberately uses vertex 445 for the left middle tip, 444 for right.
expected_joints[:, 12] = r.verts[:, 445]
error = max(float((l.verts - r.verts * torch.tensor(reflect)).abs().max()),
            float((l.joints - expected_joints * torch.tensor(reflect)).abs().max()))
if error > 1e-5:
    target.unlink()
    raise ValueError(f'Mirrored MANO FK equivalence failed: {error}')
record = {'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
          'derived_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
          'derived_asset': True, 'official_left_model': False,
          'method': 'Reflect existing right model across x including blend-shape pose features and face winding',
          'fk_random_pose_max_error_m': error, 'purpose': 'Match HaMeR right-model mirrored-left prediction convention'}
(ROOT / 'mirrored_left_manifest.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2))
