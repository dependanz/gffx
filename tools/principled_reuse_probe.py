"""Finite exact-wheel CPU costs and one pinned environment point; no image sweep."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
import torch
from tools.material_loss_cost import check_deadline,supervise,write_report,working_set_reader
from tools.shading_image_probe import producer_identity
from tools.environment_candidate_point import sample

REFERENCE_HASH='1b35d8b12c2cf3d6d194199447001bc8195dd34b860e8d12e3399cbc68d857e0'


def costs(budget,deadline,*,large=False):
    from gffx.torch import material
    rows=[]
    read,close=working_set_reader(os.getpid())
    try:
        cases=((1,1024,False),(8,128,False),(1,1024,True),(4,256,True)) if large else ((1,10,False),(8,1,False),(4,1,True))
        for points,lights,backward in cases:
            def t(x):return torch.tensor(x,dtype=torch.float64,requires_grad=backward)
            n=t([[0.,0.,1.]]*points);v=t([[.2,0.,1.]]*points)
            l=t([[.2,.3,1.]]*lights);a=t([.2,.4,.6]);r=t(.827);f=t([.07,.47,.93])
            args=(n,v,l,a,r,f)
            def run():
                check_deadline(deadline)
                with torch.set_grad_enabled(backward):
                    out=material.principled(n,v,l,albedo=a,roughness=r,f0=f,budget=budget)
                    if not torch.isfinite(out).all():raise ValueError('nonfinite material')
                    if backward:
                        gradients=torch.autograd.grad(out.sum(),args)
                        if not all(torch.isfinite(g).all() for g in gradients):raise ValueError('nonfinite gradient')
                return out.detach()
            run();durations=[]
            for _ in range(3):
                start=time.perf_counter();run();durations.append(time.perf_counter()-start)
            saved={}
            def pack(x):
                storage=x.untyped_storage(); saved[storage.data_ptr()]=storage.nbytes();return x
            with torch.autograd.graph.saved_tensors_hooks(pack,lambda x:x):run()
            rss=[];reference=run()
            for _ in range(12):
                out=run()
                if not torch.equal(reference,out):raise ValueError('repeated-call values changed')
                rss.append(read())
            count=points*3
            declared=(3*budget.workspace(count,backward=backward) if large else
                      2*budget.workspace(count,backward=backward)+budget.workspace(count*lights,backward=backward))
            declared+=8*points*lights*(192 if backward else 64)
            rows.append(dict(points=points,lights=lights,backward=backward,
                median_seconds=statistics.median(durations),durations_seconds=durations,
                declared_material_workspace_bytes=declared,saved_unique_storage_bytes=sum(saved.values()),
                saved_storage_scope='autograd hooks, includes shared inputs/constants; not scratch peak',
                repeated_calls=12,working_set_bytes=rss,
                finite_repetition_only=True))
        try:
            n=torch.tensor([[0.,0.,1.]]*8,dtype=torch.float64,requires_grad=True)
            material.principled(n,n,n[:1],albedo=a,roughness=r,f0=f,budget=budget)
        except ValueError as error:
            if 'workspace' not in str(error):raise
        else:raise ValueError('old eight-point backward cap not enforced')
        return rows
    finally:close()


def worker(args):
    from gffx.torch.material import PrincipledBudget
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    deadline=time.perf_counter()+120
    report=dict(status='working',worker_pid=os.getpid(),mode=args.mode,large=args.large)
    try:
        producer=producer_identity(args.wheel)
        if producer['wheel']['sha256']!=args.wheel_hash:raise ValueError('wheel identity mismatch')
        report.update(producer=producer,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        budget=PrincipledBudget.bundled('ggx-smith-loss-cpu-f64-v1',device='cpu',dtype=torch.float64)
        report['resource']=budget.metadata
        write_report(args.output,report)
        if args.mode=='costs':report['costs']=costs(budget,deadline,large=args.large)
        else:
            path=args.fixtures/'public-principled-0.json';blob=path.read_bytes()
            if hashlib.sha256(blob).hexdigest()!=REFERENCE_HASH:raise ValueError('reference identity mismatch')
            old=json.loads(blob);report['baseline_sha256']=REFERENCE_HASH
            report['baseline_seconds']=old['result']['elapsed_seconds']
            def progress(row):report['progress']=row;write_report(args.output,report)
            result=sample(old['point']['cosine'],model='principled',order=old['order'],deadline=deadline,budget=budget,progress=progress,chunk_size=1024 if args.large else 10)
            errors=[[abs(x-y) for x,y in zip(row,old['reference_radiance'])] for row in result['estimates']]
            report.update(result=result,errors=errors,limits=old['limits'])
            if any(x>limit for row in errors for x,limit in zip(row,old['limits'])):raise ValueError('independent reference mismatch')
            if result['calls']!=(64 if args.large else 6554) or result['directions']!=65536:raise ValueError('workload changed')
        report['status']='passed bounded CPU reuse check; not full-image acceptance'
    except Exception as error:
        report.update(status='stopped',error=f'{type(error).__name__}: {error}')
        write_report(args.output,report);raise
    write_report(args.output,report)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixtures',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--wheel',type=Path,required=True);p.add_argument('--wheel-hash',required=True)
    p.add_argument('--mode',choices=['costs','environment'],required=True);p.add_argument('--worker',action='store_true')
    p.add_argument('--large',action='store_true',help='Qualify the explicit 1024-pair material lane')
    a=p.parse_args()
    if a.worker:worker(a);return 0
    return supervise(a.fixtures,a.output,worker_module='tools.principled_reuse_probe',
        worker_args=('--wheel',str(a.wheel),'--wheel-hash',a.wheel_hash,'--mode',a.mode)+ (('--large',) if a.large else ()))


if __name__=='__main__':raise SystemExit(main())
