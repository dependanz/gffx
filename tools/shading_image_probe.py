"""Supervised CPU image references; create-only sidecars and local preview only."""
import argparse
import hashlib
import importlib
import io
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import zipfile

from tools import corpus_registry as registry
from tools.material_loss_cost import check_deadline,supervise,write_report

PROFILE=dict(name='classic-images-v1',width=128,height=128,dtype='float64',device='cpu',threads=1,
             cooperative_seconds=120,watchdog_seconds=180,sampled_working_set_limit_bytes=1024**3)


def producer_identity(wheel):
    if not wheel.is_file() or wheel.suffix!='.whl':raise ValueError('exact existing wheel required')
    raw=registry._read(wheel,registry.MAX_FILE_BYTES)
    import gffx
    import torch
    root=Path(gffx.__file__).parent.resolve()
    hashes={}
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        entries=archive.infolist()
        if len(entries)>256 or len({x.filename for x in entries})!=len(entries):raise ValueError('wheel entry count/duplicates')
        if sum(x.file_size for x in entries)>registry.MAX_TOTAL_BYTES:raise ValueError('wheel expanded byte limit')
        for entry in entries:
            if entry.filename.startswith('gffx/') and not entry.is_dir():
                if entry.file_size>registry.MAX_FILE_BYTES:raise ValueError('wheel file byte limit')
                name=entry.filename.removeprefix('gffx/')
                installed=registry._read(registry._safe(root,name),registry.MAX_FILE_BYTES)
                if installed!=archive.read(entry):raise ValueError('installed files differ from wheel: '+name)
                hashes[name]=hashlib.sha256(installed).hexdigest()
    actual={str(p.relative_to(root)).replace('\\','/') for p in root.rglob('*')
            if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.pyd','.dll')}
    if not actual.issubset(hashes):raise ValueError('extra executable files outside wheel')
    modules={}
    for name in ('gffx.torch.camera','gffx.torch.render','gffx.torch.mesh','gffx._torch'):
        location=Path(importlib.import_module(name).__file__).resolve()
        if not location.is_relative_to(root):raise ValueError('mixed installed module roots')
        modules[name]=str(location)
    if not any(x.endswith('.dll') for x in hashes):raise ValueError('missing installed native core identity')
    return dict(kind='isolated-installed-cpu-wheel',versions=dict(python=platform.python_version(),torch=torch.__version__,gffx=gffx.__version__),
                installed_files=hashes,installed_root=str(root),modules=modules,
                wheel=dict(path=str(wheel.resolve()),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw)),
                platform=platform.platform(),host=platform.node())


def direct_controls(buffers,images):
    """Separate vector expression checks scalar reference lighting, not public code."""
    import torch
    from tools.shading_image_reference import compare_images
    n,v=buffers['normal'],buffers['view'];mask=buffers['covered']
    light=torch.tensor([1.,1.,2.],dtype=torch.float64);light/=light.norm()
    half=v+light;half/=half.norm(dim=-1,keepdim=True)
    nl=(n*light).sum(-1).clamp_min(0);nh=(n*half).sum(-1).clamp_min(0)
    active=mask&((n*v).sum(-1)>0)
    rows={}
    for name,albedo in buffers['albedos'].items():
        for model,specular in (('diffuse',0.),('classic',.1)):
            expected=(albedo/math.pi+specular*10/(8*math.pi)*nh[...,None]**2)*nl[...,None]
            expected=torch.where(active[...,None],expected,torch.zeros_like(expected))
            rows[model+'/'+name]=compare_images(images[model+'/'+name],expected,mask)
    if not all(row['passed'] for row in rows.values()):raise ValueError('independent direct-light control failed')
    return rows


def worker(root,output,wheel):
    import torch
    from tools import shading_image_reference as reference
    from tools import shading_image_bundle as bundle
    from tools.testing_dashboard.image_reference_demo import presentation
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    deadline=time.perf_counter()+PROFILE['cooperative_seconds']
    report=dict(status='working',profile=PROFILE,worker_pid=os.getpid(),public_images='blocked: material/light API absent',principled_images='blocked: no admitted physical resource')
    try:
        producer=producer_identity(wheel);report['producer']=producer
        loaded=registry.load_entry(root,'synthetic_face_m1',allow_development=True)
        scene=registry._decode(loaded['files']['scene.json']);check_deadline(deadline)
        buffers=reference.build_buffers(scene);check_deadline(deadline)
        images=reference.reference_images(buffers,scene);check_deadline(deadline)
        controls=direct_controls(buffers,images)
        points=reference.environment_controls(buffers,deadline=deadline);check_deadline(deadline)
        if not all(row['status']=='converged' and row['max_error']<1e-9 for row in points):raise ValueError('environment controls failed')
        details=dict(camera=buffers['camera'],constant_controls=buffers['controls'],direct_controls=controls,
                     environment_controls=points,covered_pixels=int(buffers['covered'].sum()),reference_image_count=len(images))
        report['measurements']=details;write_report(output,report)
        flat=bundle.flatten_buffers(buffers)
        record=bundle.write_bundle(root,flat,images,producer=producer,details=details)
        report['bundle']=record;write_report(output,report);check_deadline(deadline)
        restored=bundle.load_bundle(root,**record)
        if any(not torch.equal(values,restored[group][name]) for group,items in (('buffers',flat),('images',images)) for name,values in items.items()):
            raise ValueError('serialized arrays changed')
        plate=presentation(restored);check_deadline(deadline)
        png=output.with_suffix('.png')
        with png.open('xb') as handle:handle.write(plate['png'])
        report['presentation']=dict(path=str(png),sha256=hashlib.sha256(plate['png']).hexdigest(),bytes=len(plate['png']),
                                    width=plate['width'],height=plate['height'],panels=plate['panels'],limits=plate['limits'])
        report['status']='measured development references; not public admission'
    except Exception as error:
        report.update(status='stopped',error=f'{type(error).__name__}: {error}');write_report(output,report)
        raise
    write_report(output,report)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--wheel',type=Path,required=True);parser.add_argument('--worker',action='store_true')
    args=parser.parse_args()
    if args.worker:worker(args.fixtures,args.output,args.wheel);return 0
    if args.output.with_suffix('.png').exists():raise ValueError('refusing to replace a previous preview')
    return supervise(args.fixtures,args.output,worker_module='tools.shading_image_probe',worker_args=('--wheel',str(args.wheel)))


if __name__=='__main__':raise SystemExit(main())
