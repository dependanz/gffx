"""One explicitly selected bounded image recipe profile; no full-frame allocations or services."""
import argparse
import gc
import json
import os
from pathlib import Path
import time

from tools import material_loss_cost as cost,shading_cuda_closeout as artifact

PROFILE=dict(sizes=(32,128,256),warmups=1,repetitions=3,dtype='float32',threads=1,
             cooperative_seconds=120,watchdog_seconds=180,cpu_host_bytes=1024**3,
             cuda_host_bytes=2*1024**3,cuda_allocated_bytes=64*1024**2)


def check_previous(side,cuda,previous,source_hash,wheel_hash):
    if type(side) is not int or side not in PROFILE['sizes']:
        raise ValueError('undeclared image size')
    if side==32:
        if previous is not None:raise ValueError('initial size must not reuse a previous stage')
        return None
    if previous is None:raise ValueError('larger size requires an explicitly reviewed previous stage')
    prior=PROFILE['sizes'][PROFILE['sizes'].index(side)-1]
    supervision=previous.get('supervisor',{})
    if (previous.get('status')!='passed' or previous.get('side')!=prior or
        previous.get('device')!=('cuda' if cuda else 'cpu') or
        previous.get('driver_sha256')!=source_hash or
        previous.get('wheel',{}).get('sha256')!=wheel_hash or
        supervision.get('exit_code')!=0 or supervision.get('failure')):
        raise ValueError('previous stage scope, identity or supervision differs')
    return previous


def worker(args):
    row=dict(status='working',worker_pid=os.getpid(),side=args.side,
             device='cuda' if args.cuda else 'cpu',profile=PROFILE,iterations=[])
    cost.write_report(args.output,row)
    deadline=time.perf_counter()+120;close=lambda:None
    try:
        import torch
        from gffx.torch import image
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        row['wheel']=artifact.identity(args.wheel,args.sha256)
        row['driver_sha256']=artifact.digest(__file__)
        row['backend_sha256']=artifact.digest(image.__file__)
        if args.previous is not None:row['previous_sha256']=artifact.digest(args.previous)
        device=row['device']
        if args.cuda and not torch.cuda.is_available():raise ValueError('CUDA required')
        read,close=cost.working_set_reader(os.getpid())
        n=args.side
        # Host-generated fixed inputs are transferred unchanged to the chosen device.
        y,x=torch.meshgrid(torch.arange(n,dtype=torch.float32),torch.arange(n,dtype=torch.float32),indexing='ij')
        rgb=torch.stack((.03+.2*x/n,.02+.15*y/n,.04+.1*(x+y)/(2*n)),-1)
        source=torch.cat((rgb,torch.full((n,n,1),.85)),-1).to(device)
        fixed_mask=(.3+.6*(x+y)/(2*n)).unsqueeze(-1).to(device)
        background=torch.cat((torch.full((n,n,3),.03),torch.ones(n,n,1)),-1).to(device)
        initial=torch.tensor([.23,-.17,.13,.97,1.03,.72,.21,.83],dtype=torch.float32).to(device)
        pivot=torch.tensor([(n-1)/2,(n-1)/2],dtype=torch.float32).to(device)
        del x,y,rgb
        if args.cuda:torch.cuda.reset_peak_memory_stats()
        for iteration in range(PROFILE['warmups']+PROFILE['repetitions']):
            cost.check_deadline(deadline)
            if args.cuda:torch.cuda.synchronize()
            start=time.perf_counter()
            pixels=source.clone().requires_grad_();mask=fixed_mask.clone().requires_grad_()
            p=initial.clone().requires_grad_()
            layer=image.adjust(pixels,exposure=p[6])
            layer=image.apply_mask(layer,mask,opacity=p[5])
            layer=image.gaussian_blur(layer,p[7],radius=2)
            matrix=image.affine_matrix(p[:2],p[2],p[3:5],pivot)
            layer=image.warp_affine(layer,matrix,(n,n))
            result=image.composite([layer],background)
            loss=(result-background).square().mean()
            gradients=torch.autograd.grad(loss,(pixels,mask,p))
            if not all(bool(torch.isfinite(value).all()) for value in (result,loss,*gradients)):
                raise ValueError('nonfinite recipe output or derivative')
            if args.cuda:torch.cuda.synchronize()
            elapsed=time.perf_counter()-start
            measurement=dict(index=iteration,warmup=iteration==0,forward_backward_seconds=elapsed,
                             loss=float(loss.detach()),working_set_with_results_bytes=read())
            if args.cuda:
                measurement['allocated_with_results_bytes']=torch.cuda.memory_allocated()
                measurement['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
                if measurement['peak_allocated_bytes']>PROFILE['cuda_allocated_bytes']:
                    raise MemoryError('CUDA allocation cap')
            # Do not retain output/gradient/history references or empty the caching allocator.
            del pixels,mask,p,layer,matrix,result,loss,gradients
            gc.collect()
            if args.cuda:torch.cuda.synchronize()
            measurement['working_set_after_release_bytes']=read()
            if args.cuda:
                measurement['allocated_after_release_bytes']=torch.cuda.memory_allocated()
                measurement['reserved_after_release_bytes']=torch.cuda.memory_reserved()
            row['iterations'].append(measurement)
            cost.write_report(args.output,row)
            cost.check_deadline(deadline)
        measured=row['iterations'][1:]
        row['measured_seconds']=[v['forward_backward_seconds'] for v in measured]
        released=[v['working_set_after_release_bytes'] for v in measured]
        row['released_working_set_first_to_last_delta_bytes']=released[-1]-released[0]
        row['scope']='One warmup plus three fresh full-recipe graphs. Cost/memory observations, not FPS, convergence, plateau acceptance, CUDA byte identity or full1080p evidence.'
        row['status']='passed'
    except Exception as error:
        row.update(status='stopped',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        close();cost.write_report(args.output,row)
    return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--cuda',action='store_true')
    for name in ('fixtures','output','wheel'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--sha256',required=True)
    parser.add_argument('--side',type=int,choices=PROFILE['sizes'],required=True)
    parser.add_argument('--previous',type=Path)
    args=parser.parse_args()
    previous=json.loads(args.previous.read_text(encoding='utf-8')) if args.previous else None
    check_previous(args.side,args.cuda,previous,artifact.digest(__file__),args.sha256)
    if args.worker:return worker(args)
    cost.PROFILE['sampled_working_set_limit_bytes']=PROFILE['cuda_host_bytes'] if args.cuda else PROFILE['cpu_host_bytes']
    extra=['--side',str(args.side),'--wheel',str(args.wheel),'--sha256',args.sha256]
    if args.cuda:extra.append('--cuda')
    if args.previous:extra+=['--previous',str(args.previous)]
    return cost.supervise(args.fixtures,args.output,worker_module='tools.image_cost_probe',worker_args=extra)


if __name__=='__main__':raise SystemExit(main())
