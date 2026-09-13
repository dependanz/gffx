"""Supervised32-pixel studio smoke and finite diagnostic image export.

This is not a high-resolution beauty qualification. No network publication. The
128-pixel demo remains preflight-only until separately reviewed admission.
"""
import argparse
import ctypes
import hashlib
import os
from pathlib import Path
import time
from tools import material_loss_cost as cost,shading_cuda_closeout as artifact


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',action='store_true')
    for name in ('fixtures','output','wheel'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--sha256',required=True)
    args=parser.parse_args()
    if not args.worker:
        if args.output.exists():raise ValueError('preserve prior studio evidence')
        return cost.supervise(args.fixtures,args.output,worker_module='tools.deferred_showcase_probe',
            worker_args=['--wheel',str(args.wheel),'--sha256',args.sha256])
    row=dict(status='working',worker_pid=os.getpid(),mode='render',size=32)
    cost.write_report(args.output,row);deadline=time.perf_counter()+120
    try:
        import torch
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        row['wheel']=artifact.identity(args.wheel,args.sha256)
        row['sources']={name:artifact.digest(name) for name in
            ('tools/deferred_showcase_probe.py','tools/testing_dashboard/deferred_demo.py',
             'tools/testing_dashboard/presentation.py')}
        from tools.testing_dashboard import deferred_demo as demo
        from tools.testing_dashboard.presentation import AssetWriter,png_rgb
        from tools.deferred_showcase_cached import LABELS
        start=time.perf_counter();payload=demo.payload(32,deadline=deadline)
        row['render_seconds']=time.perf_counter()-start;row['estimates']=payload['estimates']
        row['angles']=payload['angles'];row['faces']=payload['faces'];row['scope']=payload['scope']
        cost.check_deadline(deadline)
        frames=payload['frames']
        if len(frames)!=3:raise ValueError('expected three fixed camera views')
        linear=[frame['colour'][0].clamp(0,1) for frame in frames]
        display=[torch.where(rgb<=.0031308,12.92*rgb,1.055*rgb.pow(1/2.4)-.055) for rgb in linear]
        centre=frames[1];mask=centre['mask'][0];normal=centre['normal'][0]
        display.append((normal*.5+.5)*mask[...,None])
        depth=centre['depth'][0];covered=mask>0
        if not bool(covered.any()):raise ValueError('studio centre is empty')
        near,far=depth[covered].min(),depth[covered].max()
        denominator=far-near
        safe=torch.where(covered,depth,near)
        grey=torch.where(covered,1-(safe-near)/denominator if denominator>0 else torch.ones_like(depth),0.)
        display.append(grey[...,None].expand(-1,-1,3));display.append(mask[...,None].expand(-1,-1,3))
        row['display']=dict(rgb='clip linear[0,1], then sRGB',normal='(unit world normal+1)/2; background zero',
            depth='covered NDC range mapped near-white/far-black; background zero',
            depth_range=[float(near),float(far)],coverage='linear grayscale')
        writer=AssetWriter(args.output.parent/'images');row['images']=[];row['logical_output_bytes']=0
        for label,pixels in zip(LABELS,display):
            cost.check_deadline(deadline)
            if pixels.shape!=(32,32,3) or not torch.isfinite(pixels).all():raise ValueError('invalid studio display')
            pixels=(pixels.clamp(0,1)*255).round().to(torch.uint8).contiguous()
            data=ctypes.string_at(pixels.data_ptr(),pixels.numel());sha=hashlib.sha256(data).hexdigest()
            raw=args.output.parent/'images'/(sha+'.rgb')
            if raw.exists():
                if raw.is_symlink() or artifact.digest(raw)!=sha:raise ValueError('existing image identity differs')
            else:raw.write_bytes(data)
            entry=writer.put(png_rgb(32,32,data));entry.update(label=label,shape=[32,32,3],
                raw_path=str(raw.resolve()),raw_sha256=sha)
            row['images'].append(entry);row['logical_output_bytes']+=len(data)
        row['status']='passed'
    except Exception as error:
        row.update(status='stopped',error=f'{type(error).__name__}: {error}');raise
    finally:cost.write_report(args.output,row)
    return 0


if __name__=='__main__':raise SystemExit(main())
