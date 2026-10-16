"""Bounded original opaque-frame qualification, through installed public windows.

D504 permits128/256startup; higher extents require matching full-frame anchors.
No partial frame becomes an anchor. Imports/setup/storage/replay share120seconds.
"""
import argparse,hashlib,json,os,time,shutil
from pathlib import Path
from tools import material_loss_cost as cost
from tools import opaque_original_frame_admission as admission


def inspect_frame(receipt):
    import numpy as np
    from tools.opaque_frame_store import Reader
    reader=Reader(receipt['root'],receipt['manifest_sha256'],receipt['identity']);pixels=covered=0
    for offset,count,values in reader.blocks():
        for name in ('colour','mask'):
            a=np.frombuffer(values[name],dtype='<f8')
            if not np.isfinite(a).all():raise ValueError('nonfinite frame output')
            if name=='mask':
                if not ((a>=0)&(a<=1+1e-12)).all():raise ValueError('invalid frame coverage')
                covered+=int((a>0).sum())
        pixels+=count
    m=reader.manifest;arrays={}
    for name,channels in [('colour',3),('mask',1)]:
        arrays[name]=dict(path=str(Path(receipt['root'])/(name+'.f64')),shape=[1,m['height'],m['width']]+([3] if channels==3 else []),dtype='<f8',**m['streams'][name])
    return dict(pixels=pixels,covered_pixels=covered,arrays=arrays)


def _execute_frame(execute,model,camera,root,row,args,seconds):
    cot=(1/(args.width*args.height*3),)*3+(0.,)
    receipt=execute(model,camera,root,identity=dict(wheel=args.sha256,scene=row['scene_sha256'],profile=row['profile']),cotangent=cot,seconds=seconds,cull_mode=2,window_pixels=args.window_pixels,request_pixels=args.request_pixels,albedo_replay=args.albedo_replay)
    if receipt['identity']['profile']!=row['ownership_profile']:raise ValueError('frame receipt ownership differs from declared execution profile')
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker',action='store_true')
    for key in ('fixtures','output','wheel','prerequisites'):p.add_argument('--'+key,type=Path,required=True)
    for key in ('sha256','prerequisites-sha256'):p.add_argument('--'+key,required=True)
    p.add_argument('--width',type=int,required=True);p.add_argument('--height',type=int,required=True)
    p.add_argument('--device',choices=('cpu','cuda'),required=True);p.add_argument('--backward',action='store_true')
    p.add_argument('--albedo-replay',action='store_true')
    p.add_argument('--window-pixels',type=int,choices=range(1,17),default=1)
    p.add_argument('--request-pixels',type=int,choices=range(1,257))
    p.add_argument('--anchor',nargs=2,action='append',default=[]);p.add_argument('--started',type=float)
    p.add_argument('--preflight-only',action='store_true');a=p.parse_args()
    if a.albedo_replay and (not a.backward or a.request_pixels is None):raise ValueError('albedo replay needs backward and explicit bounded requests')
    if not a.worker:
        if a.output.exists() or a.output.with_suffix('.frame').exists():raise FileExistsError('preserve previous frame attempt')
        cost.PROFILE=dict(cost.PROFILE,sampled_working_set_limit_bytes=(2 if a.device=='cuda' else 1)*1024**3)
        extra=[]
        for k in ('wheel','sha256','prerequisites','prerequisites_sha256','width','height','device','window_pixels'):extra+=['--'+k.replace('_','-'),str(getattr(a,k))]
        if a.request_pixels is not None:extra+=['--request-pixels',str(a.request_pixels)]
        for path,sha in a.anchor:extra+=['--anchor',path,sha]
        if a.backward:extra+=['--backward']
        if a.albedo_replay:extra+=['--albedo-replay']
        if a.preflight_only:extra+=['--preflight-only']
        extra+=['--started',str(time.perf_counter())]
        return cost.supervise(a.fixtures,a.output,worker_module='tools.opaque_original_frame_probe',worker_args=extra)
    if a.started is None or not 0<=time.perf_counter()-a.started<120:raise ValueError('supervised whole-worker start required')
    deadline=a.started+120;root=a.output.with_suffix('.frame')
    r=dict(status='working',worker_pid=os.getpid(),render_executed=False,ownership_profile='public-window-pixel-replay-v2',profile=dict(width=a.width,height=a.height,coverage='opaque',material='principled',dtype='float64',device=a.device,vertices=5002,faces=10000,outputs=['colour','mask'],fitted_parameters=['constant_RGB_albedo'] if a.backward else [],backward=a.backward,cull_mode=2,workspace_limit_bytes=67108864,threads=1),estimate=dict(principled_point_block_limit=4,principled_group_point_limit=128,meaning='Planner upper limits; actual windows downselect inside unchanged storage budget'))
    if a.window_pixels!=1:r['ownership_profile']='public-window-pixel-replay-v3';r['profile']['window_pixels']=a.window_pixels
    if a.request_pixels is not None:r['ownership_profile']='public-window-pixel-replay-v4';r['profile'].update(window_pixels=a.window_pixels,request_pixels=a.request_pixels)
    if a.albedo_replay:r['ownership_profile']='public-window-albedo-replay-v5'
    cost.write_report(a.output,r)
    try:
        import torch
        from dataclasses import replace
        from tools.shading_cuda_closeout import identity,digest
        from tools.corpus_registry import load_entry
        from tools.deferred_flame import create_scene
        from tools.opaque_mesh_preflight import candidate_counts
        from tools.opaque_frame_execution import execute,replay
        from tools.deferred_progressive_admission import load_anchor
        from gffx.torch.scene import Scene
        from gffx.torch.camera import Camera
        from gffx.torch._opaque_scene import _project
        from gffx.torch import render
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        r['wheel']=identity(a.wheel,a.sha256)
        r['prerequisites']=admission.load_prerequisites(a.prerequisites,a.prerequisites_sha256,wheel_sha256=a.sha256)
        sources=('tools/opaque_frame_execution.py','tools/opaque_frame_store.py','tools/deferred_flame.py','tools/synthetic_flame_fixture.py','tools/opaque_original_frame_admission.py','tools/opaque_mesh_preflight.py')
        r['driver_sha256']=digest(__file__);r['source_sha256']={q:digest(q) for q in sources}
        blob=load_entry(a.fixtures,'synthetic_flame_m1',allow_development=True)['files']['scene.json'];r['scene_sha256']=hashlib.sha256(blob).hexdigest()
        if r['scene_sha256']!=admission.SCENE_SHA:raise ValueError('original scene changed')
        model,cam=create_scene(json.loads(blob),'principled',device=a.device);m=model.prims[1];m=replace(m,parameters=dict(m.parameters,f0=m.parameters['f0'].repeat(3)));model=Scene([model.prims[0],m,model.prims[2]])
        camera=Camera.from_intrinsics(cam.fx*a.height/128,cam.fy*a.height/128,cam.cx*a.width/128,cam.cy*a.height/128,a.height,a.width,.1,10.,pose=cam.pose)
        albedo=m.parameters['albedo'];albedo.requires_grad_(a.backward)
        with torch.no_grad():
            packed,_,pixels,keep,_=_project(model,camera,render.CULL_BACK)
            counts=candidate_counts(pixels.cpu().numpy(),packed.faces[keep].cpu().numpy(),a.height,a.width)
        r['maximum_candidates']=int(counts.max());del packed,pixels,keep,counts
        if r['maximum_candidates']>32:raise ValueError('original frame candidate cap exceeded')
        required=a.width*a.height*64+1048576
        if shutil.disk_usage(root.parent).free<required:raise OSError('insufficient bounded frame storage')
        r['storage_admission']=dict(required_bytes=required,max_block_pixels=256,payload_bytes=16384,full_frame_tensor=False)
        r['time_admission']=admission.time_admission([load_anchor(path,sha) for path,sha in a.anchor],r,remaining_seconds=deadline-time.perf_counter())
        r['active_memory_policy']='Public opaque_window enforces64MiB including input/output/geometry/replay reserves for every pixel;supervisor enforces CPU1GiB/CUDA2GiB host;actual GPU peak checked'
        if a.preflight_only:r.update(status='preflight_only');return 0
        cost.check_deadline(deadline);cost.write_report(a.output,r)
        if a.device=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter();r['render_executed']=True
        receipt=_execute_frame(execute,model,camera,root,r,a,deadline-time.perf_counter())
        r['receipt']=receipt
        if a.backward:
            gradients=replay(model,camera,receipt,seconds=deadline-time.perf_counter())
            if len(gradients)!=1 or not torch.isfinite(gradients[0]).all() or not bool((gradients[0]!=0).all()):raise ValueError('invalid full-frame albedo gradient')
            r['albedo_gradient']=gradients[0].cpu().tolist()
        if a.device=='cuda':
            torch.cuda.synchronize();r['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
            if r['peak_allocated_bytes']>67108864:raise MemoryError('actual CUDA allocation exceeded64MiB')
        r.update(inspect_frame(receipt))
        if r['pixels']!=a.width*a.height or not 0<r['covered_pixels']<r['pixels']:raise ValueError('incomplete or empty original frame')
        r['render_storage_seconds']=time.perf_counter()-start
        cost.check_deadline(deadline);r['measured_seconds']=time.perf_counter()-a.started
        r.update(status='passed',scope='Complete original frame execution/storage and optional canonical constant-albedo VJP;representative independent pixels,not an oracle certificate for every pixel')
    except Exception as e:
        r.update(status='stopped',error=f'{type(e).__name__}: {e}')
        if root.exists():r['partial_storage_bytes']={p.name:p.stat().st_size for p in root.iterdir() if p.is_file()}
        raise
    finally:
        r['worker_seconds']=time.perf_counter()-a.started;cost.write_report(a.output,r)
    return 0

if __name__=='__main__':raise SystemExit(main())
