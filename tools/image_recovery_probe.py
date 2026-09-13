"""Supervised, finite capture of the fixed image recovery and its inspection board.

No publication mode: inspect the cached figure before separately sending it.
"""
import argparse
import ctypes
import hashlib
import math
import os
from pathlib import Path
import time

from tools import material_loss_cost as cost, shading_cuda_closeout as artifact
from tools.testing_dashboard.image_recovery_demo import PROFILE


def validate_metadata(value):
    if value.get('profile')!=PROFILE or value.get('target_parameters')!=[.4,-.3,.8]:
        raise ValueError('recovery profile differs from the approved fixed fixture')
    if value.get('source_unchanged') is not True or value.get('background_unchanged') is not True:
        raise ValueError('locked recovery resources changed')
    if value.get('independent_target')!='scalar-bilinear-over-v1':
        raise ValueError('independent target identity differs')
    losses=value.get('loss_history',[]);parameters=value.get('parameter_history',[])
    if len(losses)!=101 or len(parameters)!=101:
        raise ValueError('all 101 optimizer states are required')
    finite=lambda x:type(x) in (int,float) and math.isfinite(x)
    if not all(finite(x) and x>=0 for x in losses):
        raise ValueError('loss history is nonfinite or negative')
    if not losses[0]>0 or not losses[-1]<=losses[0]*.01:
        raise ValueError('fixed recovery improvement gate failed')
    if not all(isinstance(p,list) and len(p)==3 and all(finite(x) for x in p) and 0<=p[2]<=1 for p in parameters):
        raise ValueError('parameter history is malformed or nonfinite')


def _raw(directory,data,suffix):
    directory.mkdir(parents=True,exist_ok=True)
    digest=hashlib.sha256(data).hexdigest();path=directory/(digest+suffix)
    if path.exists():
        if path.is_symlink() or artifact.digest(path)!=digest:
            raise ValueError('existing cache identity differs')
    else:
        path.write_bytes(data)
    return dict(raw_path=str(path.resolve()),raw_sha256=digest,bytes=len(data))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',action='store_true')
    for name in ('fixtures','output','wheel'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--sha256',required=True)
    args=parser.parse_args()
    if not args.worker:
        return cost.supervise(args.fixtures,args.output,worker_module='tools.image_recovery_probe',
                              worker_args=['--wheel',str(args.wheel),'--sha256',args.sha256])
    row=dict(status='working',worker_pid=os.getpid(),mode='fixed-recovery')
    cost.write_report(args.output,row);deadline=time.perf_counter()+120
    try:
        import torch
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        row['wheel']=artifact.identity(args.wheel,args.sha256)
        paths=('tools/image_recovery_probe.py','tools/testing_dashboard/image_recovery_demo.py',
               'tools/testing_dashboard/image_showcase.py','tests/python/test_image_recovery.py')
        row['sources']={path:artifact.digest(path) for path in paths}
        from tools.testing_dashboard import image_recovery_demo as demo
        from tools.testing_dashboard.presentation import AssetWriter,png_rgb
        result=demo.capture(checkpoint=lambda:cost.check_deadline(deadline))
        metadata={key:value for key,value in result.items() if not isinstance(value,torch.Tensor)}
        validate_metadata(metadata);row['measurements']=metadata
        row['evidence']=[]
        for name in ('target','before','after','residual_before','residual_after'):
            pixels=result[name].contiguous()
            if pixels.dtype!=torch.float64 or not bool(torch.isfinite(pixels).all()) or pixels.requires_grad:
                raise ValueError('evidence must be finite detached CPU float64 values')
            raw=ctypes.string_at(pixels.data_ptr(),pixels.numel()*pixels.element_size())
            entry=_raw(args.output.parent/'evidence',raw,'.f64')
            entry.update(label=name,dtype='float64',shape=list(pixels.shape))
            row['evidence'].append(entry)
        cost.check_deadline(deadline)
        row['images']=[];row['logical_output_bytes']=0
        for name,create,shape in (('recovery',demo.board,(800,608,3)),('history',demo.history_chart,(280,608,3))):
            pixels=create(result).contiguous()
            if pixels.dtype!=torch.uint8 or tuple(pixels.shape)!=shape:
                raise ValueError('recovery presentation shape differs')
            raw=ctypes.string_at(pixels.data_ptr(),pixels.numel())
            entry=_raw(args.output.parent/'images',raw,'.rgb')
            entry.update(AssetWriter(args.output.parent/'images').put(png_rgb(shape[1],shape[0],raw)))
            entry.update(label=name,shape=list(pixels.shape))
            row['images'].append(entry);row['logical_output_bytes']+=pixels.numel()
        row['scope']=demo.GUIDE
        cost.check_deadline(deadline);row['status']='passed'
    except Exception as error:
        row.update(status='stopped',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        cost.write_report(args.output,row)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
