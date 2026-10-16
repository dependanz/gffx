"""One CPU/CUDA whole-frame profile on the fixed nondegenerate scale head.

Valid 5002V/10000F geometry; independent pixel correctness remains a separate gate.
No automatic sweep, tile scheduling, numerical tolerance or material-limit change.
"""
import argparse,hashlib,json,os,time,math
from pathlib import Path
from tools import material_loss_cost as cost,shading_cuda_closeout as artifact
from tools.deferred_vram_budget import choose_budget


def forecast(rows,*,width,height,material,soft,backward,device,wheel,driver,scene,partition):
    if len(rows)!=2:raise ValueError('both32/64 calibrations required')
    rates=[]
    for row,expected_size in zip(rows,(32,64)):
        expected=dict(width=expected_size,height=expected_size,material=material,soft=soft,backward=backward,device=device)
        if row.get('status')!='passed' or row.get('wheel',{}).get('sha256')!=wheel or row.get('driver_sha256')!=driver or any(row.get('profile',{}).get(k)!=v for k,v in expected.items()):raise ValueError('calibration profile/identity differs')
        if row.get('scene_sha256')!=scene or row.get('estimate',{}).get('principled_point_block_limit')!=partition:raise ValueError('calibration scene/partition differs')
        if row.get('supervisor',{}).get('exit_code')!=0 or row['supervisor'].get('failure'):raise ValueError('calibration supervision failed')
        seconds=row.get('measured_seconds')
        if type(seconds) not in (int,float) or not math.isfinite(seconds) or seconds<=0:raise ValueError('invalid measured time')
        rates.append(seconds/(expected_size**2))
    if len(rates)!=2:raise ValueError('both32/64 calibrations required')
    return max(rates)*width*height*2+30


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker',action='store_true')
    for key in ('fixtures','output','wheel'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--sha256',required=True);p.add_argument('--width',type=int,required=True);p.add_argument('--height',type=int,required=True);p.add_argument('--device',choices=('cpu','cuda'),required=True)
    p.add_argument('--material',choices=('classic','principled'),required=True)
    p.add_argument('--soft',type=float,choices=(0.,.2),default=0.);p.add_argument('--backward',action='store_true')
    p.add_argument('--calibration32',type=Path);p.add_argument('--calibration64',type=Path)
    p.add_argument('--preflight-only',action='store_true');a=p.parse_args()
    if (a.width,a.height) not in ((32,32),(64,64),(128,128),(256,256),(512,512),(1920,1080)):raise ValueError('unsupported fixed profile')
    if not a.worker:
        if a.output.exists():raise FileExistsError(a.output)
        cost.PROFILE['sampled_working_set_limit_bytes']=(2 if a.device=='cuda' else 1)*1024**3
        extra=[]
        for key in ('wheel','sha256','width','height','device','material','soft','calibration32','calibration64'):
            value=getattr(a,key)
            if value is not None:extra+=['--'+key.replace('_','-'),str(value)]
        if a.backward:extra+=['--backward']
        if a.preflight_only:extra+=['--preflight-only']
        return cost.supervise(a.fixtures,a.output,worker_module='tools.deferred_flame_probe',worker_args=extra)
    row=dict(status='working',worker_pid=os.getpid(),driver_sha256=artifact.digest(__file__),profile=dict(width=a.width,height=a.height,material=a.material,soft=a.soft,backward=a.backward,dtype='float64',device=a.device,vertices=5002,faces=10000,light_samples=1,outputs=['colour','mask'],fitted_parameters=['constant_RGB_albedo'] if a.backward else []))
    cost.write_report(a.output,row);deadline=time.perf_counter()+120
    try:
        import torch
        from tools.corpus_registry import load_entry
        from tools.deferred_flame import create_scene
        from gffx.torch import render
        from gffx.torch.camera import Camera
        from gffx.torch._deferred import _plan_workload
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        if a.device=='cuda':torch.cuda.init()
        row['wheel']=artifact.identity(a.wheel,a.sha256)
        entry=load_entry(a.fixtures,'synthetic_flame_m1',allow_development=True);blob=entry['files']['scene.json'];row['scene_sha256']=hashlib.sha256(blob).hexdigest()
        model,cam=create_scene(json.loads(blob),a.material,device=a.device)
        camera=Camera.from_intrinsics(cam.fx*a.height/128,cam.fy*a.height/128,cam.cx*a.width/128,cam.cy*a.height/128,image_height=a.height,image_width=a.width,near=.1,far=10.,pose=cam.pose)
        albedo=next(p for p in model.prims if p.name=='skin').parameters['albedo'];albedo.requires_grad_(a.backward)
        if a.device=='cuda':
            free,total=torch.cuda.mem_get_info();row['budget']=choose_budget(free_bytes=free,total_bytes=total,allocated_bytes=torch.cuda.memory_allocated());limit=row['budget']['limit_bytes']
        else:row['budget']=dict(limit_bytes=64*1024**2,allocated_bytes=0,policy='unchanged CPU64MiB workspace');limit=64*1024**2
        row['source_sha256']={n:artifact.digest(n) for n in ('tools/deferred_vram_budget.py','tools/deferred_flame.py','tools/synthetic_flame_fixture.py')}
        with torch.set_grad_enabled(a.backward):
            # Reserve the entire current tensor baseline in addition to the scene
            # estimate. This deliberately double-counts borrowed scene inputs.
            render_limit=limit-row['budget']['allocated_bytes']
            row['render_limit_bytes']=render_limit
            try:row['estimate']=_plan_workload(model,camera,1,render_limit,('colour','mask'),soft=a.soft)
            except ValueError as error:
                if 'workspace' not in str(error):raise
                row.update(status='not_admitted',reason=str(error),render_executed=False,memory_admitted=False);return 0
            row['estimated_total_with_baseline_bytes']=row['estimate']['estimated_bytes']+row['budget']['allocated_bytes']
            row['memory_admitted']=True
            if max(a.width,a.height)>64:
                if not a.calibration32 or not a.calibration64:raise ValueError('larger frame requires calibration')
                rows=[json.loads(f.read_text(encoding='utf-8')) for f in (a.calibration32,a.calibration64)]
                row['forecast_seconds']=forecast(rows,width=a.width,height=a.height,material=a.material,soft=a.soft,backward=a.backward,device=a.device,wheel=a.sha256,driver=row['driver_sha256'],scene=row['scene_sha256'],partition=row['estimate'].get('principled_point_block_limit'))
                row['calibration_sha256']=[artifact.digest(f) for f in (a.calibration32,a.calibration64)]
                if row['forecast_seconds']>deadline-time.perf_counter():
                    row.update(status='not_admitted',reason='prospective complete-frame time exceeds remaining120s allowance',render_executed=False);return 0
            if a.preflight_only:
                row.update(status='preflight_only',render_executed=False);return 0
            cost.write_report(a.output,row);cost.check_deadline(deadline)
            def sync():
                if a.device=='cuda':torch.cuda.synchronize()
            if a.device=='cuda':torch.cuda.reset_peak_memory_stats()
            sync();start=time.perf_counter()
            image=render(model,camera,outputs=('colour','mask'),soft=a.soft,workspace_limit_bytes=render_limit);sync();row['forward_seconds']=time.perf_counter()-start
            if a.backward:
                start_back=time.perf_counter();gradient,=torch.autograd.grad(image['colour'].mean(),albedo);sync();row['backward_seconds']=time.perf_counter()-start_back
                assert torch.isfinite(gradient).all() and bool((gradient!=0).any());row['albedo_gradient']=gradient.tolist()
            row['measured_seconds']=time.perf_counter()-start;row['peak_allocated_bytes']=torch.cuda.max_memory_allocated() if a.device=='cuda' else None;row['reserved_bytes']=torch.cuda.memory_reserved() if a.device=='cuda' else None
            if a.device=='cuda' and row['peak_allocated_bytes']>limit:raise MemoryError('recorded CUDA budget exceeded')
            assert torch.isfinite(image['colour']).all() and torch.isfinite(image['mask']).all()
            row['covered_pixels']=int((image['mask']>0).sum());assert 0<row['covered_pixels']<a.width*a.height
            if list(image['colour'].shape)!=[1,a.height,a.width,3] or list(image['mask'].shape)!=[1,a.height,a.width]:
                raise ValueError('rendered arrays differ from declared profile extent')
            arrays={}
            for name,value in image.items():
                path=a.output.with_suffix('.'+name+'.f64');blob=value.detach().cpu().contiguous().numpy().astype('<f8',copy=False).tobytes()
                with path.open('xb') as handle:handle.write(blob)
                arrays[name]=dict(path=str(path),shape=list(value.shape),bytes=len(blob),sha256=hashlib.sha256(blob).hexdigest(),dtype='<f8')
            row['arrays']=arrays
            cost.check_deadline(deadline)
            row.update(status='passed',render_executed=True,scope='One full frame, one measured evaluation, no warmup or sustained-throughput claim. Finite values/coverage and optional albedo VJP only; not independent image accuracy or sustained throughput.')
    except Exception as e:row.update(status='stopped',error=str(e));raise
    finally:cost.write_report(a.output,row)
    return 0
if __name__=='__main__':raise SystemExit(main())
