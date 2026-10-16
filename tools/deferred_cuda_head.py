"""One admitted 64-row fixed-head CUDA partition, exact against its CPU partition."""
import argparse,json,os,time,hashlib
from pathlib import Path
from tools import material_loss_cost as cost,shading_cuda_closeout as artifact

def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true')
    for key in ('fixtures','output','wheel','scale16','scale32','cpu_report'):p.add_argument('--'+key.replace('_','-'),type=Path,required=True)
    p.add_argument('--sha256',required=True);p.add_argument('--cpu-sha256',required=True)
    p.add_argument('--row-start',type=int,choices=(0,64),required=True)
    p.add_argument('--material',choices=('classic','principled'),required=True)
    p.add_argument('--albedo',choices=('constant','vertex','gradient','checker'),required=True);a=p.parse_args()
    if not a.worker:
        cost.PROFILE['sampled_working_set_limit_bytes']=2*1024**3
        extra=[]
        for key in ('wheel','sha256','scale16','scale32','cpu_report','cpu_sha256','row_start','material','albedo'):extra+=['--'+key.replace('_','-'),str(getattr(a,key))]
        return cost.supervise(a.fixtures,a.output,worker_module='tools.deferred_cuda_head',worker_args=extra)
    row={'status':'working','worker_pid':os.getpid(),'profile':{'height':64,'width':128,'row_start':a.row_start,'material':a.material,'albedo':a.albedo,'device':'cuda','dtype':'float64','backward':False}};cost.write_report(a.output,row);deadline=time.perf_counter()+120
    try:
        import torch
        from tools import corpus_registry,shading_image_bundle,principled_direct_images as reference
        from tools.deferred_head import create_scene,projected_seconds
        from tools.deferred_image_partition import plan_partitioned
        from gffx.torch import render
        torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.cuda.init();torch.cuda.reset_peak_memory_stats()
        row['wheel']=artifact.identity(a.wheel,a.sha256);row['driver_sha256']=artifact.digest(__file__)
        row['sources']={n:artifact.digest(n) for n in ('tools/deferred_head.py','tools/deferred_image_partition.py','tools/deferred_cuda_scale.py')}
        assert artifact.digest(a.cpu_report)==a.cpu_sha256
        cpu=json.loads(a.cpu_report.read_text(encoding='utf-8'));assert cpu['status']=='passed' and cpu['partitioned'] and cpu['wheel']['sha256']==a.sha256
        assert cpu['material']==a.material and cpu['profile']['albedo']==a.albedo
        assert cpu['sources']['tools/deferred_image_partition.py']==row['sources']['tools/deferred_image_partition.py']
        descriptor=cpu['arrays']['actual'];file=a.cpu_report.parent/descriptor['path'];blob=file.read_bytes();assert len(blob)==descriptor['bytes'] and hashlib.sha256(blob).hexdigest()==descriptor['sha256']
        expected=shading_image_bundle.tensor_from_bytes(blob,(128,128,3),'float64')[a.row_start:a.row_start+64]
        row['cpu_report_sha256']=a.cpu_sha256
        parent=corpus_registry.load_entry(a.fixtures,'synthetic_face_m1',allow_development=True);data=json.loads(parent['files']['scene.json'])
        row['scene_sha256']=hashlib.sha256(parent['files']['scene.json']).hexdigest()
        bundle=shading_image_bundle.load_bundle(a.fixtures,path=reference.MANIFEST,sha256=reference.MANIFEST_HASH)
        covered=bundle['buffers']['covered'][a.row_start:a.row_start+64];row['covered_pixels']=int(covered.sum())
        with torch.no_grad():
            model,camera=create_scene(data,a.material,albedo=a.albedo,device='cuda')
            cameras,plans=plan_partitioned(model,camera,tile_rows=64)
            crop=cameras[a.row_start//64];plan=plans[a.row_start//64];row['plan']=plan
            small,large=[json.loads(f.read_text(encoding='utf-8')) for f in (a.scale16,a.scale32)]
            assert small['driver_sha256']==large['driver_sha256']==row['sources']['tools/deferred_cuda_scale.py']
            row['calibration_sha256']=[artifact.digest(f) for f in (a.scale16,a.scale32)]
            row['forecast_seconds']=projected_seconds(small,large,a.sha256,point_block_limit=plan['estimate']['principled_point_block_limit'],device='cuda',covered_pixels=row['covered_pixels'])
            cost.write_report(a.output,row);cost.check_deadline(deadline)
            if row['forecast_seconds']>deadline-time.perf_counter():raise ValueError('forecast exceeds remaining allowance')
            torch.cuda.synchronize();start=time.perf_counter();result=render(model,crop,outputs=('colour','mask'),workspace_limit_bytes=plan['allowance']);torch.cuda.synchronize()
            row['render_seconds']=time.perf_counter()-start;row['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
            if row['peak_allocated_bytes']>67108864:raise MemoryError('CUDA allocation cap')
            actual=result['colour'][0].cpu();mask=result['mask'][0].cpu()
            row['mask_exact']=torch.equal(mask,covered.to(torch.float64))
            raw=shading_image_bundle.tensor_bytes(actual);reference_bytes=shading_image_bundle.tensor_bytes(expected)
            row['cpu_bytes_exact']=raw==reference_bytes;row['max_cpu_absolute_error']=float((actual-expected).abs().max())
            row['different_channels']=int((actual!=expected).sum());row['actual_sha256']=hashlib.sha256(raw).hexdigest()
            with a.output.with_suffix('.actual.bin').open('xb') as f:f.write(raw)
            if not row['mask_exact'] or not row['cpu_bytes_exact']:raise ValueError('fixed-head CPU/CUDA reference bytes differ')
        cost.check_deadline(deadline);row['status']='passed'
        row['scope']='One separately supervised64-row crop of fixed128x128 synthetic head. Exact corresponding CPU partition output and hard mask. No full-frame single-call, backward,other-resolution or throughput claim.'
    except Exception as e:row.update(status='stopped',error=str(e));raise
    finally:cost.write_report(a.output,row)
    return 0
if __name__=='__main__':raise SystemExit(main())
