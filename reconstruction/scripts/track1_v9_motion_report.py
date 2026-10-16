"""GPU decoder pilot diagnostics with official temporal inclusion and episode weights."""
import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
from track1_v9_temporal import PILOT, LAMBDAS, sha, write_json
from track1_whittaker import scored_acceleration_magnitude


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/repo'))
    p.add_argument('--probe',action='store_true')
    p.add_argument('--episodes',type=int,nargs='+',default=PILOT)
    p.add_argument('--lambdas',type=float,nargs='+',default=LAMBDAS)
    p.add_argument('--robust',action='store_true')
    a=p.parse_args(); root=a.root; data=root/'data/v2d_challenge/track_1';out=data/'v9_temporal'
    os.environ.setdefault('LOGNAME','uid1000')
    sys.path.insert(0,'/workspace/v2d_sam3d_body/lib')
    sys.path.insert(0,str(root/'v2d_submission_kit/v2d_submission_kit/tools/track1'))
    sys.path.insert(0,str(root/'v2d_submission_kit/v2d_submission_kit'))
    import torch
    from mesh_to_mhr_params import MHR
    from v2dlb.mhr_metrics import MHR_TABLE3_BODY_JOINT_INDICES
    model_path=root/'data/weights/track1_submission_mhr/mhr_model.pt'
    model=MHR(str(model_path),'cuda',8,'float32')
    parent=data/'contact_smoothing_v8/submission_episodes'
    if a.probe:
        b=dict(np.load(parent/'episode_000007.npz',allow_pickle=False))
        names=list(model.model.get_parameter_names()); pose=torch.as_tensor(b['pose'][[100]],device='cuda',dtype=torch.float64)
        z=torch.as_tensor(np.r_[b['scales'],b['shape']][None],device='cuda',dtype=torch.float64)
        base,_=model.run(pose,z); trials=pose.repeat(127,1)
        trials[torch.arange(127,device='cuda'),torch.arange(3,130,device='cuda')]+=2*np.pi
        verts,_=model.run(trials,z)
        difference=torch.linalg.norm(verts-base,dim=-1).max(dim=1).values.cpu().numpy()
        periodic=[i+3 for i,d in enumerate(difference) if d<.02]
        write_json(out/'parameter_probe.json',{'parameter_names':names,'model_sha256':sha(model_path),'periodic_pose_columns':periodic,
            'two_pi_max_vertex_displacement_mm':dict(zip(map(str,range(3,130)),map(float,difference))),
            'root_translation_units':'cm; decoder converts vertices to mm and joints to m',
            'periodicity_scope':'Numerical +2pi probe on Episode7 frame100; angle branch mapping remains locally tested, not universal Euler equivalence.',
            'internal_lengths':list(range(130,136))})
        print('PARAMETER_PROBE',len(periodic),'of127 angular-named columns numerically periodic',flush=True);return
    scored=json.loads((root/'incident_reports/track1_scored_frames.json').read_text())
    report=root/'incident_reports/track1_improvement/v9_temporal'; signature={'source':sha(__file__),'model':sha(model_path),'solver':sha(Path(__file__).with_name('track1_whittaker.py'))}
    rows=[]
    for episode in a.episodes:
        stem=f'episode_{episode:06d}';b=dict(np.load(parent/(stem+'.npz'),allow_pickle=False));frames=np.array(scored[str(episode)])
        to_tensor=lambda x:torch.as_tensor(x,device='cuda',dtype=torch.float64)
        identity=to_tensor(np.r_[b['scales'],b['shape']][None])
        with torch.inference_mode():
            v,j=model.run(to_tensor(b['pose']),identity)
            base_vertices=(v/1000).float().cpu().numpy(); base_joints=j.cpu().numpy();del v,j
        control=scored_acceleration_magnitude(base_joints[:,MHR_TABLE3_BODY_JOINT_INDICES],frames)
        for lam in a.lambdas:
            folder=out/'human'/('robust' if a.robust else 'plain')/'initial'/f'lam_{lam:g}'/stem
            dest=folder/'motion_report.json'; candidate=folder/(stem+'.npz')
            fingerprints={**signature,'candidate':sha(candidate),'control':sha(parent/candidate.name)}
            if dest.exists():
                r=json.loads(dest.read_text()); assert r['inputs']==fingerprints;rows.append(r);continue
            c=dict(np.load(candidate,allow_pickle=False));assert np.array_equal(c['pose'][frames[0]],b['pose'][frames[0]])
            means=[];maxima=[];joints=[]
            with torch.inference_mode():
                for start in range(0,len(c['pose']),8):
                    v,j=model.run(to_tensor(c['pose'][start:start+8]),identity)
                    vertex=(v/1000).cpu().numpy(); displacement=np.linalg.norm(vertex-base_vertices[start:start+8],axis=-1)*100
                    means.extend(displacement.mean(axis=1));maxima.extend(displacement.max(axis=1));joints.append(j.cpu().numpy())
            acc=scored_acceleration_magnitude(np.concatenate(joints)[:,MHR_TABLE3_BODY_JOINT_INDICES],frames)
            r={'inputs':fingerprints,'episode':episode,'lambda':lam,'robust':a.robust,
                'control_acceleration_cm_unaligned':control,'candidate_acceleration_cm_unaligned':acc,'acceleration_reduction':1-acc/control,
                'mean_vertex_displacement_cm':float(np.mean(means)),'p99_frame_max_vertex_displacement_cm':float(np.quantile(maxima,.99)),
                'max_vertex_displacement_cm':float(np.max(maxima)),
                'cd_h_bound_cm_per_unit_alignment_scale':float(2*np.array(maxima)[frames].mean()),
                'scope':'Exact scorer endpoint replication and gap inclusion; per-episode magnitude, not hidden-reference error. Bound remains multiplied by unknown alignment scale.'}
            r['pilot_displacement_and_gain_pass']=r['acceleration_reduction']>=.7 and r['mean_vertex_displacement_cm']<=1. and r['p99_frame_max_vertex_displacement_cm']<=4.
            write_json(dest,r);rows.append(r);print('MOTION_REPORT',episode,lam,'gain',round(r['acceleration_reduction'],4),'maxp99',round(r['p99_frame_max_vertex_displacement_cm'],3),flush=True)
    write_json(report/('human_pilot_robust.json' if a.robust else 'human_pilot_plain.json'),{'inputs':signature,'rows':rows,'episodes':a.episodes,'lambdas':a.lambdas})


if __name__=='__main__':main()
