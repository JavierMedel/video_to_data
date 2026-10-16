"""Read-only v8 geometry audit; writes diagnostics, never mutates candidates."""
import argparse
import json
from pathlib import Path
import numpy as np
import trimesh
from track1_v9_temporal import sha, write_json, PARENT_SHA


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args();root=a.root.resolve()
    data=root/'data/v2d_challenge/track_1';parent=data/'contact_smoothing_v8/submission_episodes'
    assert sha(data/'submission_packages/track1_contact_smoothing_v8.parquet')==PARENT_SHA
    reports=root/'incident_reports/track1_improvement';campaign=json.loads((reports/'contact_smoothing_v8/campaign.json').read_text())
    selection=json.loads((reports/'contact_smoothing_v8/selection_review.json').read_text());metadata={r['episode_index']:r for r in map(json.loads,(data/'meta/episodes_metadata.jsonl').read_text().splitlines())}
    frames=json.loads((root/'incident_reports/track1_scored_frames.json').read_text());geometry_cache={};rows=[]
    for e in range(30):
        stem=f'episode_{e:06d}';b=dict(np.load(parent/(stem+'.npz'),allow_pickle=False));mh=sha(parent/(stem+'_object.glb'));nh=sha(parent/(stem+'.npz'))
        status_path=root/campaign['episodes'][str(e)]['status'];status=json.loads(status_path.read_text());image=status['image_review']
        promoted=selection['episodes'][str(e)]['verdict']=='PROMOTE';label='candidate' if promoted else 'control'
        assert image[label+'_npz_sha256']==nh and image[label+'_mesh_sha256']==mh,'Image evidence must match submitted v8'
        if mh not in geometry_cache:
            mesh=trimesh.load(parent/(stem+'_object.glb'),force='mesh',process=False)
            # process=False GLBs may contain unwelded vertices; connectivity needs welding.
            welded=mesh.copy();welded.merge_vertices();welded.remove_unreferenced_vertices()
            pieces=welded.split(only_watertight=False);areas=np.array([x.area for x in pieces])
            geometry_cache[mh]={'extents_m_unscaled':mesh.extents.tolist(),'area_m2_unscaled':float(mesh.area),
                'components_after_vertex_weld':len(pieces),'area_outside_largest_component_fraction':float(1-areas.max()/areas.sum()) if len(areas) and areas.sum()>0 else None,
                'watertight_after_vertex_weld':bool(welded.is_watertight),'vertices':len(mesh.vertices),'faces':len(mesh.faces),'centroid':mesh.centroid.tolist()}
        g=geometry_cache[mh];f=np.array(frames[str(e)]);r=b['object_rotation'][f];relative=np.einsum('tji,tjk->tik',r[:-1],r[1:]);angles=np.rad2deg(np.arccos(np.clip((np.trace(relative,axis1=1,axis2=2)-1)/2,-1,1)));angles=angles[np.diff(f)==1]
        rows.append({'episode':e,'object':metadata[e]['object'],'npz_sha256':nh,'mesh_sha256':mh,'image_evidence':str(status_path.relative_to(root)),'image_evidence_sha256':sha(status_path),
            'visible_iou_mean':image['summary'][label]['visible_iou_mean'],'scaled_extents_m':(np.array(g['extents_m_unscaled'])*float(b['object_scale'])).tolist(),
            'origin_centroid_offset_m':float(np.linalg.norm(g['centroid'])*float(b['object_scale'])),'rotation_jumps_over_45_consecutive_scored_frames':int((angles>45).sum()),'geometry':g})
        print('RISK_AUDIT',e,flush=True)
    for row in rows:
        same=[x for x in rows if x['object']==row['object']];median=np.median([np.sort(x['scaled_extents_m']) for x in same],axis=0)
        row['sorted_extent_ratio_to_label_group_median']=(np.sort(row['scaled_extents_m'])/np.maximum(median,1e-12)).tolist()
        row['scope']='Object labels group episodes; physical identity and common local orientation not established by this ratio. Extents/IoU/components are risk signals, not hidden CD-O.'
    out=reports/'v10_geometry';write_json(out/'risk_audit.json',{'parent':'v8','parent_parquet_sha256':PARENT_SHA,'source_sha256':sha(__file__),'episodes':rows})
    lines=['# v8 geometry risk audit','','Diagnostics only; no geometry modified or promoted. Cached image evidence is hash-matched to the exact submitted v8 pairs. Equal episode weighting does not reveal hidden per-episode distance.','','| Episode | Object | IoU | Extents m | Area outside largest component | >45 degree jumps |','|---:|---|---:|---|---:|---:|']
    for row in sorted(rows,key=lambda x:x['visible_iou_mean']):
        fraction=row['geometry']['area_outside_largest_component_fraction'];lines.append(f"| {row['episode']} | {row['object']} | {row['visible_iou_mean']:.4f} | {', '.join(f'{x:.3f}' for x in row['scaled_extents_m'])} | {fraction:.4f} | {row['rotation_jumps_over_45_consecutive_scored_frames']} |")
    lines+=['','## Hoop precheck','','Bounding extents are not a fitted ring diameter. The same-label metric hoop must be checked for physical identity, local pose/origin, thin-tube geometry and image agreement before substitution.']
    for e in (0,1,2):
        row=rows[e];lines.append(f"- Episode {e}: IoU {row['visible_iou_mean']:.4f}, scaled extents {row['scaled_extents_m']}, mesh SHA `{row['mesh_sha256']}`.")
    (out/'risk_audit.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
