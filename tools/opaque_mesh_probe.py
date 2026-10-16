"""One original-mesh pixel: independent enclosure or installed bounded crop.
No whole-frame execution, alternate mesh, changed camera or accuracy admission.
"""
import argparse,hashlib,json,os,time
from pathlib import Path
from tools import material_loss_cost as cost

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('fixtures','output','wheel'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--sha256',required=True);p.add_argument('--mode',choices=('domain','reference','reference-corrected','cpu','cuda'),required=True)
    for n in ('width','height','x','y'):p.add_argument('--'+n,type=int,required=True)
    p.add_argument('--max-cells',type=int,default=256);p.add_argument('--worker',action='store_true');a=p.parse_args()
    if not 1<=a.max_cells<=256:raise ValueError('reference cell ceiling remains256')
    if a.mode=='cuda':cost.PROFILE['sampled_working_set_limit_bytes']=2*1024**3
    if not a.worker:
        args=sum((['--'+n,str(getattr(a,n.replace('-','_')))] for n in ('wheel','sha256','mode','width','height','x','y','max-cells')),[])
        return cost.supervise(a.fixtures,a.output,worker_module='tools.opaque_mesh_probe',worker_args=args)
    r=dict(status='working',worker_pid=os.getpid(),profile=dict(width=a.width,height=a.height,origin=[a.x,a.y],crop=[1,1],max_cells=a.max_cells,mode=a.mode,roughness=1.,f0=[.04]*3,albedo=[.2,.4,.6],tangent='unit first albedo channel',threads=1,host_bytes=cost.PROFILE['sampled_working_set_limit_bytes'],active_bytes=67108864,cooperative_seconds=120,watchdog_seconds=180));start=time.perf_counter();deadline=start+120
    try:
        import torch,numpy as np
        from fractions import Fraction as Q
        from copy import copy
        from dataclasses import replace
        from gffx.torch.scene import Scene
        from tools.corpus_registry import load_entry
        from tools.shading_cuda_closeout import identity,digest
        from tools.deferred_flame import create_scene
        from tools.opaque_mesh_preflight import selected_faces
        from gffx.torch.camera import Camera
        from gffx.torch import render
        from gffx.torch._opaque_scene import _project,shade
        from gffx.torch._deferred import _plan_workload
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        r['wheel']=identity(a.wheel,a.sha256);blob=load_entry(a.fixtures,'synthetic_flame_m1',allow_development=True)['files']['scene.json'];r['scene_sha256']=hashlib.sha256(blob).hexdigest();data=json.loads(blob)
        device='cuda' if a.mode=='cuda' else 'cpu';model,cam=create_scene(data,'principled',device=device);mat=model.prims[1];mat=replace(mat,parameters=dict(mat.parameters,f0=mat.parameters['f0'].repeat(3)));model=Scene([model.prims[0],mat,model.prims[2]]);albedo=mat.parameters['albedo'];albedo.requires_grad_(a.mode not in ('reference','reference-corrected','domain'))
        camera=Camera.from_intrinsics(cam.fx*a.height/128,cam.fy*a.height/128,cam.cx*a.width/128,cam.cy*a.height/128,a.height,a.width,.1,10.,pose=cam.pose)
        r['source_sha256']={n:digest(n) for n in ('tools/opaque_mesh_probe.py','tools/opaque_mesh_preflight.py','tools/deferred_flame.py','tools/synthetic_flame_fixture.py','tools/opaque_principled_reference.py','tools/opaque_shading_intervals.py','tools/opaque_loss_extension.py')}
        if not 0<=a.x<a.width or not 0<=a.y<a.height:raise ValueError('pixel outside declared frame')
        if a.mode in ('reference','reference-corrected','domain'):
            from tools.opaque_principled_reference import PreparedPrincipled,perspective_field,integrate_pixel
            from tools.opaque_shading_intervals import Unresolved
            packed,_,pv,keep,clip=_project(model,camera,render.CULL_BACK);v=pv.tolist();f=packed.faces.tolist();kept=np.flatnonzero(keep.numpy());candidates=kept[selected_faces(np.array(v),np.array(f)[kept],a.x,a.y)];used=sorted({i for j in candidates for i in f[j]});mapping={v:i for i,v in enumerate(used)}
            vertices=[[(Q(v[i][0])-a.x),(Q(v[i][1])-a.y),Q(v[i][2])] for i in used];faces=[[mapping[i] for i in f[j]] for j in candidates]
            if len(faces)>32 or len(used)>64 or any(abs(q)>16 for row in vertices for q in row):raise ValueError('reference geometry finite-profile refusal')
            normals=model.prims[0].attributes['normals'].values.tolist();world=packed.vertices.tolist();clip_w=clip[:,3].tolist();budget=mat.parameters['budget'];reference_start=time.perf_counter();curve=PreparedPrincipled({k:v.tolist() for k,v in budget._data.items()},1.);fields=[];corrections=[];preparers=[]
            for j,face in zip(candidates,faces):
                indices=f[j];attrs=dict(inverse_w=[Q(1)/Q(clip_w[i]) for i in indices],normals=[normals[i] for i in indices],views=[[-world[i][0],-world[i][1],3-world[i][2]] for i in indices],albedo=[[.2,.4,.6]]*3,f0=[.04]*3,direction=[1.,1.,2.],tangent={'albedo':[[1.,0,0]]*3});tri=[vertices[i] for i in face]
                if a.mode=='reference-corrected':
                    from tools.opaque_loss_extension import LossExtension,correction_field
                    from tools.opaque_shading_intervals import Jet
                    from types import SimpleNamespace
                    extension=LossExtension(curve);field=perspective_field(extension,tri,**attrs);field(Jet(.5),Jet(.5));fields.append(field);corrections.append(correction_field(extension,tri,**attrs))
                    def prepare(x,y,e=extension,fn=field):
                        e.indices=[None,None];fn(Jet(x),Jet(y))
                    preparers.append(prepare)
                else:fields.append(perspective_field(curve,tri,**attrs))
            r['input']=dict(vertices=vertices,faces=faces,original_face_ids=candidates.tolist(),original_vertex_ids=used);r['input']=json.loads(json.dumps(r['input'],default=str));r['input_sha256']=hashlib.sha256(json.dumps(r['input'],sort_keys=True).encode()).hexdigest()
            if a.mode=='domain':
                from tools.opaque_shading_intervals import Jet,I
                branches=[]
                for j,field in zip(candidates,fields):
                    try:
                        values=field(Jet(I(0,1)),Jet(I(0,1)));branches.append(dict(face=int(j),smooth_extension=True,value=[str(q[0,0]) for q in values]))
                    except ValueError as e:branches.append(dict(face=int(j),smooth_extension=False,reason=str(e)))
                r.update(status='passed',domain_checked=True,smooth_extension=all(q['smooth_extension'] for q in branches),branches=branches,accuracy_admitted=False,scope='Whole pixel-box field extension branch check only, not an integral error bound.');r['seconds']=time.perf_counter()-start
                return 0
            def check():
                cost.check_deadline(deadline)
                if time.perf_counter()-reference_start>90:raise Unresolved('whole reference deadline')
            try:r['reference']=integrate_pixel(vertices,faces,fields,checkpoint=check,max_cells=a.max_cells,corrections=corrections if a.mode=='reference-corrected' else None,preparers=preparers if a.mode=='reference-corrected' else None);r['reference']=json.loads(json.dumps(r['reference'],default=str));r.update(status='passed',accuracy_admitted=True)
            except Unresolved as e:r.update(status='not_admitted',accuracy_admitted=False,reason=str(e),details=e.details)
            r['reference_seconds']=time.perf_counter()-reference_start
        else:
            proxy=copy(camera);proxy.image_height=1;proxy.image_width=800
            workload=_plan_workload(model,proxy,1,67108864-1048576,('colour','mask'));r['workload']=workload
            if device=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();baseline=torch.cuda.memory_allocated()
            start_render=time.perf_counter();out=shade(model,camera,outputs=('colour','mask'),cull_mode=render.CULL_BACK,background=None,workspace_limit_bytes=67108864,_region=(a.y,a.x,1,1),_workload=workload)
            if device=='cuda':torch.cuda.synchronize()
            r['forward_seconds']=time.perf_counter()-start_render;start_backward=time.perf_counter();grad=torch.stack([torch.autograd.grad(out['colour'][0,0,0,c],albedo,retain_graph=c<2)[0] for c in range(3)])
            if device=='cuda':torch.cuda.synchronize();r['cuda_incremental_bytes']=torch.cuda.max_memory_allocated()-baseline;assert torch.cuda.max_memory_allocated()<=67108864
            r['backward_seconds']=time.perf_counter()-start_backward;r['image']={k:v.detach().cpu().tolist() for k,v in out.items()};r['albedo_jacobian']=grad.cpu().tolist();r['bytes']={k:v.detach().cpu().contiguous().numpy().tobytes().hex() for k,v in {**out,'albedo_jacobian':grad}.items()};assert all(torch.isfinite(v).all() for v in [*out.values(),grad]);r.update(status='passed',render_executed=True,accuracy_admitted=False)
        cost.check_deadline(deadline);r['seconds']=time.perf_counter()-start
    except Exception as e:r.update(status='failed',error=repr(e));raise
    finally:cost.write_report(a.output,r)
    return 0
if __name__=='__main__':raise SystemExit(main())
