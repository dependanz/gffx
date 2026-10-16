"""Bounded original-mesh inspection. No dense render or accuracy admission."""
import argparse,hashlib,json,os,time
from pathlib import Path
import numpy as np
from tools import material_loss_cost as cost

def candidate_counts(vertices,faces,height,width):
    """Exact closed-AABB predicate used by native candidate admission, not coverage."""
    v=np.asarray(vertices);f=np.asarray(faces)
    if type(height)!=int or type(width)!=int or not 1<=height<=1080 or not 1<=width<=1920:raise ValueError('bounded image extent')
    if v.ndim!=2 or v.shape[1]<2 or len(v)>10000 or not np.isfinite(v).all():raise ValueError('finite bounded vertices')
    if f.ndim!=2 or f.shape[1]!=3 or len(f)>20000 or not np.issubdtype(f.dtype,np.integer) or (f.size and (f.min()<0 or f.max()>=len(v))):raise ValueError('bounded triangle indices')
    diff=np.zeros((height+1,width+1),dtype=np.int32)
    if not len(f):return diff[:-1,:-1]
    q=v[f,:2];lo=q.min(1);hi=q.max(1)
    take=(hi[:,0]>=0)&(lo[:,0]<=width)&(hi[:,1]>=0)&(lo[:,1]<=height)
    lo,hi=lo[take],hi[take]
    x0=np.clip(np.ceil(lo[:,0]-1),0,width).astype(np.int64);x1=np.clip(np.floor(hi[:,0])+1,0,width).astype(np.int64)
    y0=np.clip(np.ceil(lo[:,1]-1),0,height).astype(np.int64);y1=np.clip(np.floor(hi[:,1])+1,0,height).astype(np.int64)
    np.add.at(diff,(y0,x0),1);np.add.at(diff,(y0,x1),-1);np.add.at(diff,(y1,x0),-1);np.add.at(diff,(y1,x1),1)
    return diff.cumsum(0,dtype=np.int32).cumsum(1,dtype=np.int32)[:-1,:-1]

def selected_faces(pixels,faces,x,y):
    q=pixels[faces,:2];return np.flatnonzero((q[:,:,0].max(1)>=x)&(q[:,:,0].min(1)<=x+1)&(q[:,:,1].max(1)>=y)&(q[:,:,1].min(1)<=y+1))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('fixtures','output','wheel'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--sha256',required=True);p.add_argument('--worker',action='store_true');a=p.parse_args()
    if not a.worker:return cost.supervise(a.fixtures,a.output,worker_module='tools.opaque_mesh_preflight',worker_args=('--wheel',str(a.wheel),'--sha256',a.sha256))
    r=dict(status='working',worker_pid=os.getpid(),render_executed=False,profile=dict(threads=1,host_bytes=1073741824,active_bytes=67108864,cooperative_seconds=120,watchdog_seconds=180));start=time.perf_counter();deadline=start+120
    try:
        import torch
        from tools.corpus_registry import load_entry
        from tools.shading_cuda_closeout import identity,digest
        from tools.deferred_flame import create_scene
        from gffx.torch.camera import Camera
        from gffx.torch._opaque_scene import _project
        from gffx.torch import render
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        r['wheel']=identity(a.wheel,a.sha256);entry=load_entry(a.fixtures,'synthetic_flame_m1',allow_development=True);blob=entry['files']['scene.json'];r['scene_sha256']=hashlib.sha256(blob).hexdigest();data=json.loads(blob);model,cam=create_scene(data,'principled')
        r['source_sha256']={str(q):digest(q) for q in map(Path,('tools/opaque_mesh_preflight.py','tools/deferred_flame.py','tools/synthetic_flame_fixture.py'))}
        v=model.prims[0].vertices.numpy();f=model.prims[0].faces.numpy();n=model.prims[0].attributes['normals'].values.numpy();view=np.array([0.,0.,3.])-v;l=np.array([1.,1.,2.]);l/=np.linalg.norm(l)
        nv=(n*view).sum(-1)/np.linalg.norm(n,axis=-1)/np.linalg.norm(view,axis=-1);nl=n@l/np.linalg.norm(n,axis=-1)
        mat=model.prims[1];cn=np.array(mat.parameters['budget']._envelope['payload']['cosine_nodes']);qnv=nv[f];qnl=nl[f]
        r['material']=dict(roughness=float(mat.parameters['roughness']),f0=float(mat.parameters['f0']),albedo=mat.parameters['albedo'].tolist(),f0_shape=list(mat.parameters['f0'].shape),opaque_binding_requires='expand scalar F0 to identical RGB constant; no value change',texture=False,two_sided=False,vertex_normal_lengths=[float(np.linalg.norm(n,axis=1).min()),float(np.linalg.norm(n,axis=1).max())],corner_nv_range=[float(nv.min()),float(nv.max())],corner_nl_range=[float(nl.min()),float(nl.max())],faces_with_corner_light_sign_crossing=int(((qnl.min(1)<0)&(qnl.max(1)>0)).sum()),faces_with_corner_view_sign_crossing=int(((qnv.min(1)<0)&(qnv.max(1)>0)).sum()),faces_with_corner_light_angular_knot_crossing=int((np.searchsorted(cn,qnl.min(1),side='right')<np.searchsorted(cn,qnl.max(1),side='left')).sum()),range_scope='Vertex observations prove some whole-face crossings, not visible-pixel error enclosures.')
        r['geometry']=dict(vertices=len(v),faces=len(f),recipe_verified=True);r['frames']=[]
        for h,w in [(128,128),(256,256),(512,512),(1080,1920)]:
            cost.check_deadline(deadline);camera=Camera.from_intrinsics(cam.fx*h/128,cam.fy*h/128,cam.cx*w/128,cam.cy*h/128,h,w,.1,10.,pose=cam.pose)
            packed,camera,pixels,keep,clip=_project(model,camera,render.CULL_BACK);pv=pixels.numpy();ff=f[keep.numpy()];kept=np.flatnonzero(keep.numpy());counts=candidate_counts(pv,ff,h,w);unique,total=np.unique(counts,return_counts=True);samples=[]
            for x,y in [(w//2,h//2),(w//2+w//8,h//2),(w//2,h//2-h//4),(w//2-w//8,h//2)]:
                local=selected_faces(pv,ff,x,y);samples.append(dict(origin=[x,y],candidate_count=len(local),face_ids=kept[local].tolist()))
            interior=[]
            for y,x in np.argwhere(counts==1):
                if not .35*w<x<.65*w or not .35*h<y<.65*h:continue
                loc=selected_faces(pv,ff,int(x),int(y));face=ff[loc[0]];tri=pv[face,:2];a0,b,c=tri;matrix=np.stack((b-a0,c-a0),axis=1)
                beta=np.linalg.solve(matrix,(np.array([[x,y],[x+1,y],[x+1,y+1],[x,y+1]])-a0).T).T
                if np.all(beta>0) and np.all(beta.sum(1)<1):interior.append(dict(origin=[int(x),int(y)],face_id=int(kept[loc[0]])))
                if len(interior)==3:break
            r['frames'].append(dict(height=h,width=w,retained_faces=len(ff),candidate_histogram={str(k):int(vv) for k,vv in zip(unique,total)},maximum_candidates=int(counts.max()),pixels_over_candidate_cap=int((counts>32).sum()),candidate_cap=32,full_frame_candidate_admitted=not bool((counts>32).any()),samples=samples,fully_inside_single_face_pixels=interior,all_vertices_in_clip=bool(((clip[:,3]>0)&(pixels[:,2]>=-1)&(pixels[:,2]<=1)).all()),full_scene_output_reserve_bytes=1048576+65536+h*w*4*8*3))
        r.update(status='passed',seconds=time.perf_counter()-start,scope='Preflight only. Candidate counts are closed-AABB admission, not final visible triangles or shader error. No full frame or performance admitted.')
    except Exception as e:r.update(status='failed',error=repr(e));raise
    finally:cost.write_report(a.output,r)
    return 0
if __name__=='__main__':raise SystemExit(main())
