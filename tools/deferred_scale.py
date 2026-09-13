"""Prospectively bounded16/32 CPU renderer profiles, never an automatic sweep."""
import argparse
import os
from pathlib import Path
import time
from tools import material_loss_cost as cost,shading_cuda_closeout as artifact


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--worker',action='store_true')
    for key in ('fixtures','output','wheel'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--sha256',required=True)
    parser.add_argument('--size',type=int,choices=(16,32),required=True)
    parser.add_argument('--backward',action='store_true')
    parser.add_argument('--workspace-limit-bytes',type=int,default=67108864)
    parser.add_argument('--expect-point-block',type=int,choices=(1,2,4),default=4)
    args=parser.parse_args()
    if not 0<args.workspace_limit_bytes<=67108864:raise ValueError('unchanged64MiB maximum')
    if not args.worker:
        extra=['--wheel',str(args.wheel),'--sha256',args.sha256,'--size',str(args.size),
               '--workspace-limit-bytes',str(args.workspace_limit_bytes),'--expect-point-block',str(args.expect_point_block)]
        if args.backward:extra.append('--backward')
        return cost.supervise(args.fixtures,args.output,worker_module='tools.deferred_scale',worker_args=extra)
    row=dict(status='working',worker_pid=os.getpid(),driver_sha256=artifact.digest(__file__),
             profile=dict(width=args.size,height=args.size,faces=2,light_samples=1,dtype='float64',device='cpu',
                          fitted_parameters=['roughness'] if args.backward else [],
                          workspace_limit_bytes=args.workspace_limit_bytes,expected_point_block_limit=args.expect_point_block))
    cost.write_report(args.output,row);deadline=time.perf_counter()+120
    try:
        import torch
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        row['wheel']=artifact.identity(args.wheel,args.sha256);row['torch_version']=torch.__version__
        from gffx.torch import scene,material,light,render
        from gffx.torch.camera import Camera
        from gffx.torch._deferred import _plan_workload
        def t(value):return torch.tensor(value,dtype=torch.float64)
        roughness=t(.613).requires_grad_(args.backward)
        budget=material.PrincipledBudget.bundled('ggx-smith-loss-cpu-f64-v1',device='cpu',dtype=torch.float64)
        vertices=t([[-2.,-2.,-2.],[2.,-2.,-2.],[2.,2.,-2.],[-2.,2.,-2.]])
        faces=torch.tensor([[0,1,2],[0,2,3]],dtype=torch.int32)
        mat=scene.Material('p',material.principled,dict(albedo=t([.2,.3,.4]),roughness=roughness,f0=t(.071),budget=budget))
        lamp=scene.Light('key',light.directional,dict(directions=t([[.2,.3,1.]]),irradiance=t([[1.,1.,1.]])))
        model=scene.Scene([scene.Mesh('q',vertices,faces,material='p'),mat,lamp])
        camera=Camera.from_intrinsics(t(float(args.size)),t(float(args.size)),t(args.size/2),t(args.size/2),
                                       args.size,args.size,.1,10.)
        with torch.set_grad_enabled(args.backward):
            row['estimate']=_plan_workload(model,camera,1,args.workspace_limit_bytes,('colour','mask'))
            if row['estimate']['principled_point_block_limit']!=args.expect_point_block:
                raise ValueError('calibration point partition differs from requested profile')
            cost.write_report(args.output,row);cost.check_deadline(deadline)
            start=time.perf_counter();image=render(model,camera,workspace_limit_bytes=args.workspace_limit_bytes)
            row['forward_seconds']=time.perf_counter()-start
            cost.check_deadline(deadline)
            assert torch.isfinite(image['colour']).all() and bool((image['colour']>0).all())
            assert bool((image['mask']==1).all())
            row['covered_pixels']=int(image['mask'].sum())
            if args.backward:
                start=time.perf_counter()
                gradient,=torch.autograd.grad(image['colour'].sum(),roughness)
                row['backward_seconds']=time.perf_counter()-start
                assert torch.isfinite(gradient) and gradient!=0
                row['roughness_gradient']=float(gradient)
                cost.check_deadline(deadline)
        row['status']='passed'
    except Exception as error:
        row.update(status='stopped',error=f'{type(error).__name__}: {error}')
        raise
    finally:cost.write_report(args.output,row)
    return 0


if __name__=='__main__':raise SystemExit(main())
