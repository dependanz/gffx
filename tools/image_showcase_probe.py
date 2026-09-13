"""Render-only finite CHROMA presentation worker with existing process supervision.

Run diagnostic first, then render with --preflight pointing to its passed report.
Publication is deliberately not a mode; inspect saved PNGs before a separate publisher.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import time

from tools import material_loss_cost as cost, shading_cuda_closeout as artifact


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--worker',action='store_true')
    for name in ('fixtures','output','wheel'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--sha256',required=True)
    parser.add_argument('--mode',choices=('diagnostic','render'),required=True)
    parser.add_argument('--preflight',type=Path)
    args=parser.parse_args()
    if not args.worker:
        extra=['--wheel',str(args.wheel),'--sha256',args.sha256,'--mode',args.mode]
        if args.preflight is not None:
            extra+=['--preflight',str(args.preflight)]
        return cost.supervise(args.fixtures,args.output,worker_module='tools.image_showcase_probe',worker_args=extra)
    row=dict(status='working',worker_pid=os.getpid(),mode=args.mode)
    cost.write_report(args.output,row)
    deadline=time.perf_counter()+120
    try:
        import torch
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        row['wheel']=artifact.identity(args.wheel,args.sha256)
        row['sources']={name:artifact.digest(name) for name in
                        ('tools/image_showcase_probe.py','tools/testing_dashboard/image_showcase.py')}
        if args.mode=='render':
            if args.preflight is None:
                raise ValueError('full render requires an explicit passed diagnostic report')
            previous=json.loads(args.preflight.read_text(encoding='utf-8'))
            if previous.get('status')!='passed' or previous.get('mode')!='diagnostic':
                raise ValueError('diagnostic did not pass')
            if previous.get('sources')!=row['sources'] or previous.get('wheel',{}).get('sha256')!=args.sha256:
                raise ValueError('diagnostic source/wheel identity differs')
            if previous.get('supervisor',{}).get('failure') or previous.get('supervisor',{}).get('exit_code')!=0:
                raise ValueError('diagnostic supervision failed')
            row['preflight_sha256']=artifact.digest(args.preflight)
        from tools.testing_dashboard import image_showcase as demo
        from tools.testing_dashboard.presentation import AssetWriter,png_rgb
        writer=AssetWriter(args.output.parent/'images')
        size=(180,320) if args.mode=='diagnostic' else (720,1280)
        row['images']=[]
        row['logical_output_bytes']=0
        for label,create in (('hero',lambda:demo.hero(size)),('transparency',demo.transparency),('sampling',demo.sampling)):
            cost.check_deadline(deadline)
            pixels=create().contiguous()
            if pixels.dtype!=torch.uint8 or pixels.ndim!=3 or pixels.shape[-1]!=3:
                raise ValueError('presentation output must be uint8 HWC RGB')
            data=ctypes.string_at(pixels.data_ptr(),pixels.numel())
            digest=hashlib.sha256(data).hexdigest()
            raw=args.output.parent/'images'/(digest+'.rgb')
            if raw.exists():
                if raw.is_symlink() or artifact.digest(raw)!=digest:
                    raise ValueError('existing raw image identity mismatch')
            else:
                raw.write_bytes(data)
            entry=writer.put(png_rgb(pixels.shape[1],pixels.shape[0],data))
            entry.update(label=label,shape=list(pixels.shape),raw_path=str(raw.resolve()),raw_sha256=digest)
            row['images'].append(entry)
            row['logical_output_bytes']+=pixels.numel()
            del pixels,data
            cost.check_deadline(deadline)
        row['status']='passed'
        row['scope']='Three finite display images; no test acceptance, publication or runtime performance claim'
    except Exception as error:
        row.update(status='stopped',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        cost.write_report(args.output,row)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
