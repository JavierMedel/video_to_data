"""Full-source-video comparison of actual object meshes, with normal shading.

Humans are unchanged and remain source RGB, occluding objects via observed masks.
This is not a reconstructed-human video. No predicted hand/body geometry is hidden
behind a claim of full human rendering; exact hand checks are separate evidence.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np
from track1_v9_temporal import sha,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('/repo'));p.add_argument('--episode',type=int,required=True)
    p.add_argument('--parent',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();root=a.root;stem=f'episode_{a.episode:06d}';data=root/'data/v2d_challenge/track_1'
    sys.path[:0]=['/workspace/v2d_foundation_pose/lib/FoundationPose',str(root/'modules/v2d_cari4d/lib/cari4d')]
    import torch
    import nvdiffrast.torch as dr
    import cv2
    import trimesh
    import av
    import imageio.v2 as imageio
    import Utils
    from track1_mask_render import _scaled_image_grid
    from prep.mhr_export_utils import camera_calibration,load_edex,read_rgb,read_mask
    from tools.render_mhr_wild_inference import _source_video_properties
    bundles=[dict(np.load(folder/(stem+'.npz'),allow_pickle=False)) for folder in (a.parent,a.candidate)]
    assert all(np.array_equal(bundles[0][key],bundles[1][key]) for key in bundles[0] if key!='object_rotation')
    assert sha(a.parent/(stem+'_object.glb'))==sha(a.candidate/(stem+'_object.glb'))
    work=data/('improvement_pilot_v1' if a.episode in (6,18,28) else 'improvement_rollout_v1')/stem;export=work/'run/export'/stem
    k,w2c=camera_calibration(load_edex(export),0);assert np.allclose(w2c,np.eye(4))
    video=data/'videos/chunk-000/observation.images.exo_camera'/(stem+'.mp4');fps,rate,total,width,height=_source_video_properties(video)
    assert all(len(b['pose'])==total for b in bundles)
    mesh=trimesh.load(a.parent/(stem+'_object.glb'),force='mesh',process=False);vertices=torch.as_tensor(mesh.vertices*float(bundles[0]['object_scale']),device='cuda',dtype=torch.float32)
    normals=torch.as_tensor(mesh.vertex_normals.copy(),device='cuda',dtype=torch.float32);faces=torch.as_tensor(mesh.faces,device='cuda',dtype=torch.int32);ctx=dr.RasterizeCudaContext()
    sk,shape=_scaled_image_grid(k,(height,width),768);h,w=shape
    projection=torch.as_tensor(Utils.projection_matrix_from_intrinsics(sk,height=h,width=w,znear=.001,zfar=100.),device='cuda',dtype=torch.float32)
    cvgl=torch.as_tensor(Utils.glcam_in_cvcam,device='cuda',dtype=torch.float32)
    a.out.mkdir(parents=True,exist_ok=True);outputs=[a.out/(stem+'_object_rotation_comparison.mp4'),a.out/(stem+'_object_rotation_closeups.mp4')]
    assert not any(path.exists() for path in outputs),'Preserve completed visual evidence'
    temps=[path.with_name('.'+path.stem+'.tmp.mp4') for path in outputs];assert not any(path.exists() for path in temps)
    writers=[imageio.get_writer(str(path),format='FFMPEG',fps=fps,codec='libx264',pixelformat='yuv420p',macro_block_size=None,ffmpeg_params=['-crf','18','-preset','fast']) for path in temps]
    scored=json.loads((root/'incident_reports/track1_scored_frames.json').read_text())[str(a.episode)];samples={scored[i] for i in np.linspace(0,len(scored)-1,6).round().astype(int)}
    def header(image,text):
        panel=np.zeros((image.shape[0]+40,image.shape[1],3),dtype=np.uint8);panel[40:]=image;cv2.putText(panel,text,(10,27),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2);return panel
    with torch.inference_mode():
        try:
            for frame in range(total):
                name=f'{frame:06d}';rgb=cv2.resize(read_rgb(export,0,name),(w,h));human=cv2.resize(read_mask(export,'human',0,name).astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST)>0
                observed=cv2.resize(read_mask(export,'object',0,name).astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST)>0;panels=[]
                for label,bundle in zip(('v8 parent','Rotation candidate'),bundles):
                    rotation=torch.as_tensor(bundle['object_rotation'][frame],device='cuda',dtype=torch.float32);translation=torch.as_tensor(bundle['object_translation'][frame],device='cuda',dtype=torch.float32)
                    camera=vertices@rotation.T+translation;clip=Utils.to_homo_torch(camera)@(projection@cvgl).T
                    raster,_=dr.rasterize(ctx,clip[None],faces,resolution=shape)
                    n,_=dr.interpolate((normals@rotation.T)[None],raster,faces)
                    shade=torch.flip(.3+.7*n[0,...,2].abs().clamp(0,1),[0]).cpu().numpy();mask=torch.flip(raster[0,...,3]>0,[0]).cpu().numpy() & ~human
                    image=rgb.copy();color=shade[...,None]*np.array([255,150,45]);image[mask]=(image[mask]*.35+color[mask]*.65).astype(np.uint8)
                    contours,_=cv2.findContours(observed.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(image,contours,-1,(20,255,20),1)
                    panels.append(image)
                full=np.concatenate([header(image,label) for image,label in zip(panels,('v8 parent','Candidate: actual object mesh; source RGB humans'))],axis=1)
                yy,xx=np.nonzero(observed)
                if len(xx):
                    cw=int(min(w,max(180,(xx.max()-xx.min())*3,(yy.max()-yy.min())*4)));ch=int(min(h,cw*.75));cx,cy=int((xx.max()+xx.min())/2),int((yy.max()+yy.min())/2);x=max(0,min(w-cw,cx-cw//2));y=max(0,min(h-ch,cy-ch//2))
                else:x,y,cw,ch=0,0,w,h
                crops=np.concatenate([header(cv2.resize(image[y:y+ch,x:x+cw],(512,384)),label) for image,label in zip(panels,('v8 parent','Rotation candidate'))],axis=1)
                writers[0].append_data(full);writers[1].append_data(crops)
                if frame in samples:
                    cv2.imwrite(str(a.out/f'frame_{frame:06d}.jpg'),cv2.cvtColor(full,cv2.COLOR_RGB2BGR));cv2.imwrite(str(a.out/f'crop_{frame:06d}.jpg'),cv2.cvtColor(crops,cv2.COLOR_RGB2BGR))
                if frame%100==0:print('OBJECT_VIDEO',a.episode,frame,total,flush=True)
        finally:
            for writer in writers:writer.close()
    validation=[]
    for tmp,destination in zip(temps,outputs):
        with av.open(str(tmp)) as container:
            stream=container.streams.video[0];count=sum(1 for _ in container.decode(stream));assert count==total and abs(float(stream.average_rate)-fps)<1e-6
            validation.append({'frames':count,'fps':str(stream.average_rate),'path':str(destination),'sha256':sha(tmp)})
        tmp.replace(destination)
    inputs={str(path):sha(path) for path in [Path(__file__),a.parent/(stem+'.npz'),a.candidate/(stem+'.npz'),a.parent/(stem+'_object.glb'),export/'edex']}
    write_json(a.out/'validation.json',{'state':'COMPLETE','videos':validation,'input_sha256':inputs,'sample_frames':sorted(samples),'scope':'Actual normal-shaded object projection on full source RGB; observed human masks occlude mesh. Humans/meshes/translation/scale unchanged. No reconstructed-human rendering or exhaustive visual inspection claim.'})
    print('OBJECT_VIDEOS_COMPLETE',a.episode,total,flush=True)


if __name__=='__main__':main()
