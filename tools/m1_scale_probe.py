"""Prospective bounded camera and image cost profiles; explicit caller-side partitions.

No implementation lives here: every numerical operation calls the installed public API.
Partitioned profiles are not full-frame/device-resident graph or real-time claims.
"""
import argparse
import gc
import os
from pathlib import Path
import time
from tools import material_loss_cost as cost,shading_cuda_closeout as artifact


def admission(mode,size):
    if type(size) is not int or mode not in ('camera','image'):
        raise ValueError('undeclared profile')
    if mode=='camera':
        if size not in (65536,1000000):raise ValueError('undeclared point count')
        return dict(points=size,chunk_points=65536,dtype='float32',fitted_inputs=False)
    if size not in (64,1080):raise ValueError('undeclared height')
    return dict(height=size,width=1920 if size==1080 else 96,tile_rows=16,factor=2,dtype='float32',layers=2)


def image_inputs(mode,height,width,start,stop):
    import torch
    factor=2 if mode=='reconstruct' else 1
    y,x=torch.meshgrid(torch.arange(start*factor,stop*factor,dtype=torch.float32),torch.arange(width*factor,dtype=torch.float32),indexing='ij')
    a=(.2+.5*(x+y).remainder(31)/31).unsqueeze(-1)
    rgb=torch.stack((.05+.1*x/(width*factor),.04+.1*y/(height*factor),.03+.05*(x+y)/(width*factor+height*factor)),-1)
    top=torch.cat((rgb*a,a),-1)
    if mode=='reconstruct':return (top,)
    bottom=torch.cat((rgb.flip(-1)*.6,torch.full_like(a,.6)),-1)
    background=torch.cat((torch.full_like(rgb,.03),torch.ones_like(a)),-1)
    return top,bottom,background


def image_operation(mode,inputs):
    from gffx.torch import image
    return image.reconstruct(inputs[0],2) if mode=='reconstruct' else image.composite([inputs[1],inputs[0]],inputs[2])


def image_tiles(mode,height,width,tile_rows,device,cot):
    import torch
    values=[];gradients=[]
    for start in range(0,height,tile_rows):
        stop=min(height,start+tile_rows)
        inputs=tuple(v.to(device).requires_grad_() for v in image_inputs(mode,height,width,start,stop))
        output=image_operation(mode,inputs)
        derivative=torch.autograd.grad(output,inputs,cot[start:stop].to(device))
        values.append(output.detach().cpu());gradients.append(tuple(g.cpu() for g in derivative))
    return torch.cat(values),tuple(torch.cat(parts) for parts in zip(*gradients))


def worker(args):
    profile=admission(args.mode,args.size)
    row=dict(status='working',worker_pid=os.getpid(),profile=profile,device='cuda' if args.cuda else 'cpu',mode=args.mode,iterations=[])
    cost.write_report(args.output,row);deadline=time.perf_counter()+120
    try:
        import torch
        from gffx.torch.camera import Camera
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        row['wheel']=artifact.identity(args.wheel,args.sha256);row['driver_sha256']=artifact.digest(__file__)
        device=row['device'];sync=lambda:torch.cuda.synchronize() if args.cuda else None
        if args.cuda and not torch.cuda.is_available():raise ValueError('CUDA required')
        if args.cuda:torch.cuda.reset_peak_memory_stats()
        if args.mode=='camera':
            t=lambda value:torch.tensor(value,dtype=torch.float32,device=device)
            camera=Camera.from_intrinsics(t(800.),t(800.),t(500.),t(500.),1000,1000,.1,10.)
            for repeat in range(4):
                cost.check_deadline(deadline);total=time.perf_counter();times={'project':0.,'unproject':0.};maximum=0.
                with torch.no_grad():
                    for start in range(0,args.size,profile['chunk_points']):
                        cost.check_deadline(deadline)
                        k=torch.arange(start,min(args.size,start+profile['chunk_points']),dtype=torch.float32)
                        points=torch.stack(((k.remainder(1000)/1000-.5)*.4,((k//1000)/1000-.5)*.4,torch.full_like(k,-2.)),-1).to(device)
                        sync();begin=time.perf_counter();projected=camera.project(points);sync();times['project']+=time.perf_counter()-begin
                        begin=time.perf_counter();recovered=camera.unproject_points(projected.pixels,projected.view_depth,kind='view');sync();times['unproject']+=time.perf_counter()-begin
                        if not bool(projected.valid.all()):raise ValueError('camera invalid point')
                        error=float((recovered-points).abs().max());maximum=max(maximum,error)
                        if error>2e-6:raise ValueError('camera roundtrip differs from existing float32 control')
                        del points,projected,recovered,k
                gc.collect();sync()
                result=dict(warmup=repeat==0,operation_seconds=times,including_inputs_and_transfers_seconds=time.perf_counter()-total,max_roundtrip_error=maximum)
                if args.cuda:result.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),allocated_after_release_bytes=torch.cuda.memory_allocated())
                row['iterations'].append(result);cost.write_report(args.output,row)
        else:
            for mode in ('composite','reconstruct'):
                for repeat in range(4):
                    cost.check_deadline(deadline);total=time.perf_counter();forward=backward=0.;tiles=0;maximum=0.
                    for start in range(0,profile['height'],profile['tile_rows']):
                        cost.check_deadline(deadline);stop=min(profile['height'],start+profile['tile_rows'])
                        inputs=tuple(v.to(device).requires_grad_() for v in image_inputs(mode,profile['height'],profile['width'],start,stop))
                        sync();begin=time.perf_counter();output=image_operation(mode,inputs);sync();forward+=time.perf_counter()-begin
                        cot=torch.full_like(output,.125)
                        begin=time.perf_counter();gradients=torch.autograd.grad(output,inputs,cot);sync();backward+=time.perf_counter()-begin
                        if not all(bool(torch.isfinite(v).all()) for v in (output,*gradients)):raise ValueError('nonfinite image profile')
                        maximum=max(maximum,float(output.abs().max()));tiles+=1
                        del inputs,output,cot,gradients
                    gc.collect();sync()
                    result=dict(operation=mode,warmup=repeat==0,tiles=tiles,forward_seconds=forward,backward_seconds=backward,including_inputs_and_transfers_seconds=time.perf_counter()-total,max_output_magnitude=maximum)
                    if args.cuda:result.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),allocated_after_release_bytes=torch.cuda.memory_allocated())
                    row['iterations'].append(result);cost.write_report(args.output,row)
        cost.check_deadline(deadline)
        if args.cuda and torch.cuda.max_memory_allocated()>64*1024**2:raise MemoryError('GPU cap')
        row['status']='passed';row['scope']='One warmup/three samples, serial caller-side partitions. Camera fixed-parameter forward operations; image full output extent and input VJPs in independent 16-row tiles. Inputs generated per tile; no full-frame device-resident graph, full image recipe, optimizer or sustained throughput claim.'
    except Exception as error:
        row.update(status='stopped',error=f'{type(error).__name__}: {error}');raise
    finally:cost.write_report(args.output,row)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker',action='store_true');p.add_argument('--cuda',action='store_true')
    for name in ('fixtures','output','wheel'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--sha256',required=True);p.add_argument('--mode',choices=('camera','image'),required=True);p.add_argument('--size',type=int,required=True)
    p.add_argument('--previous',type=Path)
    a=p.parse_args();admission(a.mode,a.size)
    if a.size in (1000000,1080):
        import json
        if a.previous is None:raise ValueError('large profile requires reviewed small report')
        prior=json.loads(a.previous.read_text());expected=65536 if a.mode=='camera' else 64
        if prior.get('status')!='passed' or prior.get('mode')!=a.mode or prior.get('device')!=('cuda' if a.cuda else 'cpu') or prior.get('profile')!=admission(a.mode,expected) or prior.get('wheel',{}).get('sha256')!=a.sha256 or prior.get('driver_sha256')!=artifact.digest(__file__) or prior.get('supervisor',{}).get('failure') or prior.get('supervisor',{}).get('exit_code')!=0:raise ValueError('previous profile identity/scope differs')
    if a.worker:return worker(a)
    cost.PROFILE['sampled_working_set_limit_bytes']=(2 if a.cuda else 1)*1024**3
    extra=['--wheel',str(a.wheel),'--sha256',a.sha256,'--mode',a.mode,'--size',str(a.size)]
    if a.cuda:extra+=['--cuda']
    if a.previous:extra+=['--previous',str(a.previous)]
    return cost.supervise(a.fixtures,a.output,worker_module='tools.m1_scale_probe',worker_args=extra)

if __name__=='__main__':raise SystemExit(main())
