"""One full-frame CUDA memory/time profile on the fixed synthetic head.

Calibration evidence, not FLAME-scale or independent whole-render acceptance.
No automatic sweep, tile scheduling, numerical tolerance or material-limit change.
"""
import argparse,hashlib,json,os,time
from pathlib import Path
from tools import material_loss_cost as cost,shading_cuda_closeout as artifact
from tools.deferred_vram_budget import choose_budget


def forecast(rows,*,size,material,soft,backward,wheel,driver):
    rates=[]
    for row,expected_size in zip(rows,(32,64)):
        expected=dict(size=expected_size,material=material,soft=soft,backward=backward)
        if row.get('status')!='passed' or row.get('wheel',{}).get('sha256')!=wheel or row.get('driver_sha256')!=driver or any(row.get('profile',{}).get(k)!=v for k,v in expected.items()):raise ValueError('calibration profile/identity differs')
        if row.get('supervisor',{}).get('exit_code')!=0 or row['supervisor'].get('failure'):raise ValueError('calibration supervision failed')
        rates.append(row['measured_seconds']/(expected_size**2))
    if len(rates)!=2:raise ValueError('both32/64 calibrations required')
    return max(rates)*size*size*2+30


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker',action='store_true')
    for key in ('fixtures','output','wheel'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--sha256',required=True);p.add_argument('--size',type=int,choices=(32,64,128,256,512),required=True)
    p.add_argument('--material',choices=('classic','principled'),required=True)
    p.add_argument('--soft',type=float,choices=(0.,.2),default=0.);p.add_argument('--backward',action='store_true')
    p.add_argument('--calibration32',type=Path);p.add_argument('--calibration64',type=Path)
    p.add_argument('--preflight-only',action='store_true');a=p.parse_args()
    if not a.worker:
        cost.PROFILE['sampled_working_set_limit_bytes']=2*1024**3
        extra=[]
        for key in ('wheel','sha256','size','material','soft','calibration32','calibration64'):
            value=getattr(a,key)
            if value is not None:extra+=['--'+key.replace('_','-'),str(value)]
        if a.backward:extra+=['--backward']
        if a.preflight_only:extra+=['--preflight-only']
        return cost.supervise(a.fixtures,a.output,worker_module='tools.deferred_vram_probe',worker_args=extra)
    row=dict(status='working',worker_pid=os.getpid(),driver_sha256=artifact.digest(__file__),profile=dict(size=a.size,width=a.size,height=a.size,material=a.material,soft=a.soft,backward=a.backward,dtype='float64',device='cuda',vertices=1986,faces=3968,light_samples=1,outputs=['colour','mask'],fitted_parameters=['constant_RGB_albedo'] if a.backward else []))
    cost.write_report(a.output,row);deadline=time.perf_counter()+120
    try:
        import torch
        from tools.corpus_registry import load_entry
        from tools.deferred_head import create_scene
        from gffx.torch import render
        from gffx.torch.camera import Camera
        from gffx.torch._deferred import _plan_workload
        torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.cuda.init()
        row['wheel']=artifact.identity(a.wheel,a.sha256)
        entry=load_entry(a.fixtures,'synthetic_face_m1',allow_development=True);blob=entry['files']['scene.json'];row['scene_sha256']=hashlib.sha256(blob).hexdigest()
        model,cam=create_scene(json.loads(blob),a.material,device='cuda')
        camera=Camera.from_intrinsics(cam.fx*a.size/128,cam.fy*a.size/128,cam.cx*a.size/128,cam.cy*a.size/128,a.size,a.size,.1,10.,pose=cam.pose)
        albedo=next(p for p in model.prims if p.name=='skin').parameters['albedo'];albedo.requires_grad_(a.backward)
        free,total=torch.cuda.mem_get_info();row['budget']=choose_budget(free_bytes=free,total_bytes=total,allocated_bytes=torch.cuda.memory_allocated());limit=row['budget']['limit_bytes']
        row['source_sha256']={n:artifact.digest(n) for n in ('tools/deferred_vram_budget.py','tools/deferred_head.py')}
        with torch.set_grad_enabled(a.backward):
            # Reserve the entire current tensor baseline in addition to the scene
            # estimate. This deliberately double-counts borrowed scene inputs.
            render_limit=limit-row['budget']['allocated_bytes']
            row['render_limit_bytes']=render_limit
            row['estimate']=_plan_workload(model,camera,1,render_limit,('colour','mask'),soft=a.soft)
            row['estimated_total_with_baseline_bytes']=row['estimate']['estimated_bytes']+row['budget']['allocated_bytes']
            row['memory_admitted']=True
            if a.size>64:
                if not a.calibration32 or not a.calibration64:raise ValueError('larger frame requires calibration')
                rows=[json.loads(f.read_text(encoding='utf-8')) for f in (a.calibration32,a.calibration64)]
                row['forecast_seconds']=forecast(rows,size=a.size,material=a.material,soft=a.soft,backward=a.backward,wheel=a.sha256,driver=row['driver_sha256'])
                row['calibration_sha256']=[artifact.digest(f) for f in (a.calibration32,a.calibration64)]
                if row['forecast_seconds']>deadline-time.perf_counter():
                    row.update(status='not_admitted',reason='prospective complete-frame time exceeds remaining120s allowance',render_executed=False);return 0
            if a.preflight_only:
                row.update(status='preflight_only',render_executed=False);return 0
            cost.write_report(a.output,row);cost.check_deadline(deadline)
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
            image=render(model,camera,outputs=('colour','mask'),soft=a.soft,workspace_limit_bytes=render_limit);torch.cuda.synchronize();row['forward_seconds']=time.perf_counter()-start
            if a.backward:
                start_back=time.perf_counter();gradient,=torch.autograd.grad(image['colour'].mean(),albedo);torch.cuda.synchronize();row['backward_seconds']=time.perf_counter()-start_back
                assert torch.isfinite(gradient).all() and bool((gradient!=0).any());row['albedo_gradient']=gradient.tolist()
            row['measured_seconds']=time.perf_counter()-start;row['peak_allocated_bytes']=torch.cuda.max_memory_allocated();row['reserved_bytes']=torch.cuda.memory_reserved()
            if row['peak_allocated_bytes']>limit:raise MemoryError('recorded CUDA budget exceeded')
            assert torch.isfinite(image['colour']).all() and torch.isfinite(image['mask']).all()
            row['covered_pixels']=int((image['mask']>0).sum());assert 0<row['covered_pixels']<a.size*a.size
            cost.check_deadline(deadline)
            row.update(status='passed',render_executed=True,scope='One full frame, one measured evaluation, no warmup or sustained-throughput claim. Finite values/coverage and optional albedo VJP only; not independent image accuracy or FLAME-scale qualification.')
    except Exception as e:row.update(status='stopped',error=str(e));raise
    finally:cost.write_report(a.output,row)
    return 0
if __name__=='__main__':raise SystemExit(main())
