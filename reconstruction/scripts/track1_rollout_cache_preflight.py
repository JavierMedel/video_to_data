"""Validate reusable cache timelines, schema, and pinned depth provenance."""
import argparse
import json
import sys
from pathlib import Path
import cv2
import h5py
import joblib
import numpy as np
from track1_rollout_common import ORDER, paths

p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); a=p.parse_args()
sys.path.insert(0,str(a.root/'modules/v2d_cari4d/lib/cari4d'))
from lib_mhr import MHR_PARAM_DIMS, assert_mhr_schema, load_mhr_result
from prep.mhr_depth_backend import MOGE2_MODEL_ID, MOGE2_MODEL_REVISION, MOGE2_SOURCE_COMMIT
from prep.mhr_depth_h5 import DEPTH_ALIGNMENT_INPUT_IDENTITY_ATTRIBUTE
from prep.mhr_export_utils import read_depth_m

rows=[]
for i in ORDER:
    q=paths(a.root,i)
    try:
        cap=cv2.VideoCapture(str(q['video'])); count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
        init=load_mhr_result(q['init']); assert_mhr_schema(init,require_geometry=True)
        assert [str(v) for v in init['frames']]==[f'{f:06d}' for f in range(count)], 'Human/video frame mismatch'
        for key in MHR_PARAM_DIMS: assert np.isfinite(init[key]).all(), key
        intrinsics=joblib.load(q['intrinsics'])
        assert intrinsics['model_id']==MOGE2_MODEL_ID and intrinsics['model_revision']==MOGE2_MODEL_REVISION and intrinsics['source_commit']==MOGE2_SOURCE_COMMIT
        with h5py.File(q['depth'],'r') as f:
            identity=f.attrs.get(DEPTH_ALIGNMENT_INPUT_IDENTITY_ATTRIBUTE)
            if isinstance(identity,bytes): identity=identity.decode()
            identity=json.loads(str(identity))
        assert identity['depth_backend']=='moge2' and identity['depth_model_id']==MOGE2_MODEL_ID and identity['depth_model_revision']==MOGE2_MODEL_REVISION and identity['depth_source_commit']==MOGE2_SOURCE_COMMIT, 'Aligned depth provenance mismatch'
        depth_ranges=[]
        for frame in (0,count-1):
            depth=read_depth_m(q['baseline_export'],0,f'{frame:06d}',depth_root=q['depth'])
            assert depth.shape==(1152,1536) and np.isfinite(depth).all() and (depth>0).any()
            valid=depth[depth>0]; depth_ranges.append([float(valid.min()),float(valid.max())])
        row={'episode':i,'status':'PASS','frames':count,'depth_ranges_first_last_m':depth_ranges,'cache_paths':{k:str(q[k]) for k in ('init','depth','intrinsics')}}
    except Exception as e:
        row={'episode':i,'status':'FAIL','error':f'{type(e).__name__}: {e}'}
    rows.append(row); print('ROLLOUT_CACHE_PREFLIGHT',json.dumps(row),flush=True)
q['report_dir'].mkdir(parents=True,exist_ok=True)
(q['report_dir']/'cache_preflight.json').write_text(json.dumps(rows,indent=2)+'\n')
if any(r['status']!='PASS' for r in rows): raise SystemExit(1)
