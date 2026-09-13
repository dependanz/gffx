"""Fixed-reference setup reuse and six prospectively selected affine image tiles.

Only development tooling. Previous numerical source remains unchanged; public
sampling still calls the frozen complete-material integrator for every pixel.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import time

import torch

from tools import affine_environment as old
from tools import environment_full_images as fixed
from tools.material_loss_cost import check_deadline,supervise,write_report
from tools.principled_direct_images import PreparedCurve,RESOURCE
from tools.shading_independent_evaluator import LossCurve
from tools.corpus_registry import _read

AFFINE_HASH='30a18861cf24fef9e29299f1ac06f00838456882b2481843a68578412c775f5b'
PROFILE=dict(name='affine-image-tile-qualification-v1',point_profile=old.PROFILE,
    selected_tiles=[0,26,51],points_per_tile=128,pixels_per_material=264,
    maximum_samples=161488896,aggregate_seconds=900,max_tile_bytes=4*1024**2)
COMPLETE='passed six affine tiles; not complete images or M1 acceptance'


class Prepared:
    """Worker-owned, fixed-material reference data; never a runtime graph cache."""
    def __init__(self,model,curve,*,deadline):
        check_deadline(deadline)
        if model not in old.PROFILE['materials']:raise ValueError('unknown material')
        self.model=model;self._curve=copy.deepcopy(curve) if model=='principled' else None
        self._moments=old.moments(self._curve,deadline=deadline) if model=='principled' else None
        self.moment_samples=0 if self._moments is None else self._moments['evaluations']

    def point(self,normal,view,*,deadline):
        """Same arithmetic/refinement as the frozen point oracle; no pixel reuse."""
        check_deadline(deadline);model=self.model;b=old.bands_checked(old.BANDS)
        c,basis=old.frame(normal,view);deadline=min(deadline,time.perf_counter()+30)
        radial=copy.deepcopy(self._moments);matte=torch.zeros(3,dtype=torch.float64)
        if model=='principled':
            m,j=radial['values'];lv=old.scalar._number(self._curve.loss(c,1.,.04))
            matte=torch.tensor(old.PROFILE['albedo'],dtype=torch.float64)*lv*(m+(torch.tensor(b,dtype=torch.float64)@basis[2])*j)/radial['adaptive_mean']
        history=[];count=0;previous=None;streak=0
        for order in (8,16,32,64,128):
            check_deadline(deadline)
            if count+8*order*order>174592:
                raise old.scalar.ReferenceConvergenceError(dict(reason='work_limit',evaluations=count,history=history))
            estimates=[]
            for split in (.5,.25):
                q,w=old.quadrature(order,split);brdf=fixed.reference_brdf(c,q,model=model)
                estimate=(brdf*old.field(q@basis,b)*(q[:,2]*w)[:,None]).sum(0)+matte
                if not bool(torch.isfinite(estimate).all()) or bool((estimate<0).any()):raise ValueError('invalid reference sample')
                estimates.append(estimate);count+=len(q);check_deadline(deadline)
            estimates=torch.stack(estimates);mean=estimates.mean(0);limits=1e-11+1e-9*mean.abs()
            delta=(estimates[0]-estimates[1]).abs();change=None
            if previous is not None:
                change=(estimates-previous).abs().amax(0)
                streak=streak+1 if bool(((change<=limits)&(delta<=limits)).all()) else 0
            history.append(dict(n_mu=order,estimates=estimates.tolist(),radiance=mean.tolist(),threshold=limits.tolist(),
                offset_change=delta.tolist(),change=None if change is None else change.tolist(),streak=streak,evaluations=count))
            if streak==2:
                return dict(status='converged',model=model,bands=[list(x) for x in b],
                    normal=list(old.scalar._normal(normal)),view=list(old.scalar._normal(view)),
                    radiance=mean.tolist(),history=history,evaluations=count,moments=radial)
            previous=estimates
        raise old.scalar.ReferenceConvergenceError(dict(reason='level_limit',evaluations=count,history=history))


def schedule():return [(m,i) for m in old.PROFILE['materials'] for i in PROFILE['selected_tiles']]


def tile_range(index):return fixed.tile_range(index)


def sources():
    result=old.sources()
    if result['affine_environment.py']!=AFFINE_HASH:raise ValueError('frozen affine source changed')
    result[Path(__file__).name]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return result


def inputs(buffers,index):
    _,ids,_=fixed.pixel_layout(buffers);start,end=tile_range(index)
    return [(i,buffers['normal'].reshape(-1,3)[i].tolist(),buffers['view'].reshape(-1,3)[i].tolist())
            for i in ids[start:end]]


def validate_pixel(pixel,index,normal,view,model):
    ref=pixel['reference'];actual=pixel['actual'];old.validate_reference(ref)
    if (pixel['flat_index']!=index or ref['model']!=model or ref['bands']!=old.PROFILE['bands']
        or ref['normal']!=list(old.scalar._normal(normal)) or ref['view']!=list(old.scalar._normal(view))):
        raise ValueError('pixel ownership/geometry mismatch')
    order=ref['history'][-1]['n_mu'];values=actual['estimates']
    if (len(values)!=2 or any(len(x)!=3 for x in values) or actual['chunk_size']!=1024
        or actual['directions']!=8*order*order or actual['calls']!=2*math.ceil(4*order*order/1024)
        or not 0<actual['elapsed_seconds']<120):raise ValueError('pixel work/shape mismatch')
    limits=[1e-10+1e-8*abs(x) for x in ref['radiance']]
    errors=[[abs(a-b) for a,b in zip(value,ref['radiance'])] for value in values]
    if (actual['errors']!=errors or actual['limits']!=limits
        or any(not math.isfinite(x) or x>t for error in errors for x,t in zip(error,limits))):
        raise ValueError('pixel numerical evidence mismatch')


def counts(rows,moment_samples):
    return dict(reference_directions=sum(p['reference']['evaluations'] for p in rows),
                candidate_directions=sum(p['actual']['directions'] for p in rows),
                calls=sum(p['actual']['calls'] for p in rows),moment_samples=moment_samples)


def worker(root,output,wheel,model,index):
    from gffx.torch.material import PrincipledBudget
    torch.set_num_threads(1);torch.set_num_interop_threads(1);deadline=time.perf_counter()+120
    report=dict(status='working',worker_pid=os.getpid(),profile=PROFILE,model=model,index=index,rows=[])
    try:
        if (model,index) not in schedule():raise ValueError('unqualified tile')
        report.update(sources=sources(),producer=fixed.producer_identity(wheel),manifest_sha256=fixed.MANIFEST_HASH)
        if report['producer']['wheel']['sha256']!=fixed.WHEEL_HASH:raise ValueError('wheel identity')
        bundle=fixed.load_bundle(root,path=fixed.MANIFEST,sha256=fixed.MANIFEST_HASH)
        points=inputs(bundle['buffers'],index)
        started=time.perf_counter();budget=None;curve=None
        if model=='principled':
            budget=PrincipledBudget.bundled(RESOURCE,device='cpu',dtype=torch.float64)
            curve=PreparedCurve(LossCurve(budget.to_envelope()['payload']),roughness=1.)
        prepared=Prepared(model,curve,deadline=deadline)
        report.update(setup_seconds=time.perf_counter()-started,setup_moment_samples=prepared.moment_samples,
                      reference_seconds=0.,candidate_seconds=0.)
        write_report(output,report)
        for flat_index,normal,view in points:
            check_deadline(deadline);started=time.perf_counter()
            ref=prepared.point(normal,view,deadline=deadline)
            report['reference_seconds']+=time.perf_counter()-started
            actual=old.candidate(normal,view,model=model,budget=budget,ref=ref,deadline=deadline)
            report['candidate_seconds']+=actual['elapsed_seconds']
            pixel=dict(flat_index=flat_index,reference=ref,actual=actual)
            validate_pixel(pixel,flat_index,normal,view,model)
            report['rows'].append(pixel)
        report['counts']=counts(report['rows'],prepared.moment_samples)
        check_deadline(deadline);report['status']='passed affine tile; not a complete image'
    except Exception as error:
        report.update(status='stopped',error=f'{type(error).__name__}: {error}')
        if isinstance(error,old.scalar.ReferenceConvergenceError):report['unresolved']=error.report
        raise
    finally:write_report(output,report)


def read_tile(path,digest,buffers):
    blob=_read(Path(path),PROFILE['max_tile_bytes'])
    if hashlib.sha256(blob).hexdigest()!=digest:raise ValueError('tile identity mismatch')
    row=json.loads(blob);s=row['supervisor']
    if (row['status']!='passed affine tile; not a complete image' or row['profile']!=PROFILE
        or row['sources']!=sources() or row['manifest_sha256']!=fixed.MANIFEST_HASH
        or row['producer']['wheel']['sha256']!=fixed.WHEEL_HASH or (row['model'],row['index']) not in schedule()
        or s['failure'] is not None or s['exit_code']!=0 or s['monitored_pid']!=row['worker_pid']
        or not 0<s['elapsed_seconds']<180 or not 0<s['sampled_peak_working_set_bytes']<=1024**3):raise ValueError('unqualified tile')
    points=inputs(buffers,row['index'])
    if len(row['rows'])!=len(points):raise ValueError('incomplete tile')
    for pixel,(index,n,v) in zip(row['rows'],points):validate_pixel(pixel,index,n,v,row['model'])
    samples=0 if row['model']=='classic' else row['rows'][0]['reference']['moments']['evaluations']
    if (row['setup_moment_samples']!=samples or row['counts']!=counts(row['rows'],samples)
        or any(p['reference']['moments']!=row['rows'][0]['reference']['moments'] for p in row['rows'])
        or any(not math.isfinite(row[k]) or not 0<=row[k]<120 for k in ('setup_seconds','reference_seconds','candidate_seconds'))
        or row['candidate_seconds']!=sum(p['actual']['elapsed_seconds'] for p in row['rows'])):raise ValueError('tile setup/work mismatch')
    return row


def replay(root,digest,corpus):
    root=Path(root);blob=_read(root/'cohort.json',65536)
    if hashlib.sha256(blob).hexdigest()!=digest:raise ValueError('cohort identity mismatch')
    record=json.loads(blob)
    if (record['profile']!=PROFILE or record['status']!=COMPLETE or record['sources']!=sources()
        or not 0<record['elapsed_seconds']<=900 or len(record['tiles'])!=6 or 'images' in record):raise ValueError('unqualified partial cohort')
    bundle=fixed.load_bundle(corpus,path=fixed.MANIFEST,sha256=fixed.MANIFEST_HASH);tiles=[]
    for descriptor,(model,index) in zip(record['tiles'],schedule()):
        if descriptor['path']!=f'{model}-{index:02d}.json':raise ValueError('tile ordering/path mismatch')
        row=read_tile(root/descriptor['path'],descriptor['sha256'],bundle['buffers'])
        if row['model']!=model or row['index']!=index:raise ValueError('tile ownership mismatch')
        tiles.append(row)
    return dict(record=record,sha256=digest,tiles=tiles)


def launch_guard(started):check_deadline(started+900-180)


def run(root,output,wheel):
    output=Path(output)
    if output.exists():raise FileExistsError('fresh tile-qualification directory required')
    source=sources();started=time.perf_counter();output.mkdir(parents=True)
    record=dict(profile=PROFILE,status='working',sources=source,tiles=[]);write_report(output/'cohort.json',record)
    try:
        bundle=fixed.load_bundle(root,path=fixed.MANIFEST,sha256=fixed.MANIFEST_HASH)
        for model,index in schedule():
            launch_guard(started)
            if sources()!=source:raise ValueError('tile source changed during execution')
            path=output/f'{model}-{index:02d}.json'
            code=supervise(root,path,worker_module='tools.affine_tiles',worker_args=('--wheel',str(wheel),'--model',model,'--index',str(index)))
            digest=hashlib.sha256(path.read_bytes()).hexdigest();record['tiles'].append(dict(path=path.name,sha256=digest))
            write_report(output/'cohort.json',record)
            if code:raise RuntimeError('tile qualification stopped; no retry')
            read_tile(path,digest,bundle['buffers'])
        check_deadline(started+900);record['status']=COMPLETE
    except Exception as error:
        record.update(status='stopped',error=f'{type(error).__name__}: {error}');raise
    finally:
        record['elapsed_seconds']=time.perf_counter()-started;write_report(output/'cohort.json',record)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixtures',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--wheel',type=Path,required=True);p.add_argument('--worker',action='store_true')
    p.add_argument('--model');p.add_argument('--index',type=int)
    a=p.parse_args()
    if a.worker:worker(a.fixtures,a.output,a.wheel,a.model,a.index)
    else:run(a.fixtures,a.output,a.wheel)


if __name__=='__main__':main()
