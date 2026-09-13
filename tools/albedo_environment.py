"""Bounded independent reference and public integration for saved albedo fields.

Development evidence only: selected surface points are not complete images.
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
from tools import affine_tiles
from tools.material_loss_cost import check_deadline, supervise, write_report
from tools.corpus_registry import _read

PROFILE=dict(name='albedo-environment-preflight-v1',albedos=['vertex','gradient','checker'],
    environments=['uniform','affine'],materials=['classic','principled'],points=8,comparisons=96,
    maximum_samples=29343744,aggregate_seconds=600,cooperative_seconds=120,watchdog_seconds=180,
    sampled_working_set_limit_bytes=1024**3,reference_directions=174592,candidate_directions=131072,
    moment_samples=32768,dtype='float64',device='cpu',threads=1,chunk_size=1024,
    reference_atol=1e-11,reference_rtol=1e-9,channel_atol=1e-10,channel_rtol=1e-8)
COMPLETE='passed albedo environment preflight; not full images or M1 acceptance'
WORKER_COMPLETE='passed 24 albedo points; not images'


def colour(value):
    if (not isinstance(value,(list,tuple)) or len(value)!=3
        or any(type(x) not in (int,float) or not math.isfinite(x) or not 0<=x<=1 for x in value)):
        raise ValueError('albedo must contain three finite numbers in [0,1]')
    return tuple(float(x) for x in value)


def bands(environment):
    if environment not in PROFILE['environments']:raise ValueError('unknown environment')
    return old.BANDS if environment=='affine' else ((0.,0.,0.),)*3


class Prepared:
    """Independent worker-owned moment setup; each point refines independently."""
    def __init__(self,model,curve,environment,*,deadline):
        check_deadline(deadline);bands(environment)
        if model not in PROFILE['materials']:raise ValueError('unknown material')
        self.model=model;self.environment=environment
        self.curve=copy.deepcopy(curve) if model=='principled' else None
        if model=='principled' and self.curve is None:raise ValueError('independent curve required')
        self.radial=old.moments(self.curve,deadline=deadline) if model=='principled' else None
        self.moment_samples=0 if self.radial is None else self.radial['evaluations']

    def point(self,normal,view,albedo,*,deadline,max_evaluations=174592):
        check_deadline(deadline);a=colour(albedo);b=bands(self.environment)
        if type(max_evaluations) is not int or not 0<max_evaluations<=174592:raise ValueError('reference work cap')
        c,basis=old.frame(normal,view);deadline=min(deadline,time.perf_counter()+30)
        radial=copy.deepcopy(self.radial);matte=torch.zeros(3,dtype=torch.float64)
        if self.model=='principled':
            m,j=radial['values'];lv=old.scalar._number(self.curve.loss(c,1.,.04))
            matte=torch.tensor(a,dtype=torch.float64)*lv*(m+(torch.tensor(b,dtype=torch.float64)@basis[2])*j)/radial['adaptive_mean']
        history=[];count=0;previous=None;streak=0
        for order in (8,16,32,64,128):
            check_deadline(deadline)
            if count+8*order*order>max_evaluations:
                raise old.scalar.ReferenceConvergenceError(dict(reason='work_limit',evaluations=count,history=history))
            estimates=[]
            for split in (.5,.25):
                q,w=old.quadrature(order,split)
                if self.model=='classic':
                    dot=math.sqrt((1-c)*(1+c))*q[:,0]+c*q[:,2]
                    nh2=(c+q[:,2]).square()/(2*(1+dot))
                    brdf=torch.tensor(a,dtype=torch.float64)/math.pi+(.1*10/(8*math.pi)*nh2)[:,None]
                else:brdf=old.fixed.reference_brdf(c,q,model='principled')
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
                return dict(status='converged',model=self.model,environment=self.environment,albedo=list(a),
                    bands=[list(x) for x in b],normal=list(old.scalar._normal(normal)),view=list(old.scalar._normal(view)),
                    radiance=mean.tolist(),history=history,evaluations=count,moments=radial)
            previous=estimates
        raise old.scalar.ReferenceConvergenceError(dict(reason='level_limit',evaluations=count,history=history))


def validate_reference(ref):
    colour(ref['albedo'])
    if ref['bands']!=[list(x) for x in bands(ref['environment'])]:raise ValueError('environment identity')
    old.validate_reference(ref)


def candidate(normal,view,albedo,*,model,environment,budget,ref,deadline):
    from gffx.torch import material,light
    validate_reference(ref);check_deadline(deadline);a=colour(albedo);b=bands(environment)
    if (ref['model']!=model or ref['environment']!=environment or ref['albedo']!=list(a)
        or ref['normal']!=list(old.scalar._normal(normal)) or ref['view']!=list(old.scalar._normal(view))):
        raise ValueError('reference input identity')
    _,basis=old.frame(normal,view)
    n=torch.tensor([normal],dtype=torch.float64);v=torch.tensor([view],dtype=torch.float64)
    estimates=[];calls=0;count=0;started=time.perf_counter()
    with torch.no_grad():
        for split in (.5,.25):
            q,w=old.quadrature(ref['history'][-1]['n_mu'],split);world=q@basis
            total=torch.zeros(3,dtype=torch.float64)
            for start in range(0,len(q),1024):
                check_deadline(deadline);d=world[start:start+1024];size=len(d)
                args=dict(albedo=torch.tensor(a,dtype=torch.float64))
                if model=='classic':
                    brdf=material.classic(n,v,d,specular=torch.tensor(.1,dtype=torch.float64),exponent=torch.tensor(2.,dtype=torch.float64),**args)
                else:
                    brdf=material.principled(n,v,d,roughness=torch.tensor(1.,dtype=torch.float64),f0=torch.tensor(.04,dtype=torch.float64),budget=budget,**args)
                packet=dict(measure='solid_angle',direction=d[None],radiance=old.field(d,b)[None],
                    valid=torch.ones((1,size),dtype=torch.bool),solid_angle=w[start:start+1024][None])
                total+=light.integrate(n,brdf,packet)[0];count+=size;calls+=1
            estimates.append(total.tolist())
    check_deadline(deadline)
    limits=[1e-10+1e-8*abs(x) for x in ref['radiance']]
    errors=[[abs(a-b) for a,b in zip(row,ref['radiance'])] for row in estimates]
    if any(not math.isfinite(x) or x>t for row in errors for x,t in zip(row,limits)):raise ValueError('public albedo reference tolerance failed')
    return dict(estimates=estimates,errors=errors,limits=limits,directions=count,calls=calls,chunk_size=1024,
        elapsed_seconds=time.perf_counter()-started)


def schedule():return [(m,e) for m in PROFILE['materials'] for e in PROFILE['environments']]


def sources():
    result=old.sources()
    if result['affine_environment.py']!=affine_tiles.AFFINE_HASH:raise ValueError('frozen affine source changed')
    result[Path(__file__).name]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return result


def inputs(buffers):
    from tools.principled_direct_images import validate_buffers
    selected=json.loads(json.dumps(old.select_pixels(buffers)))
    rows=[]
    for name in PROFILE['albedos']:
        validate_buffers(buffers,name)
        for point in selected:
            a=colour(buffers['albedo/'+name].reshape(-1,3)[point['flat_index']].tolist())
            rows.append(dict(point=point,albedo_name=name,albedo=list(a)))
    return rows


def worker(root,output,wheel,model,environment):
    from gffx.torch.material import PrincipledBudget
    torch.set_num_threads(1);torch.set_num_interop_threads(1);deadline=time.perf_counter()+120
    report=dict(status='working',worker_pid=os.getpid(),profile=PROFILE,model=model,environment=environment,rows=[])
    try:
        if (model,environment) not in schedule():raise ValueError('worker identity')
        report.update(sources=sources(),producer=old.fixed.producer_identity(wheel),manifest_sha256=old.fixed.MANIFEST_HASH)
        if report['producer']['wheel']['sha256']!=old.fixed.WHEEL_HASH:raise ValueError('wheel identity')
        bundle=old.fixed.load_bundle(root,path=old.fixed.MANIFEST,sha256=old.fixed.MANIFEST_HASH)
        budget=PrincipledBudget.bundled(old.RESOURCE,device='cpu',dtype=torch.float64) if model=='principled' else None
        curve=old.PreparedCurve(old.LossCurve(budget.to_envelope()['payload']),roughness=1.) if budget is not None else None
        prepared=Prepared(model,curve,environment,deadline=deadline)
        report['moment_samples']=prepared.moment_samples;write_report(output,report)
        for entry in inputs(bundle['buffers']):
            check_deadline(deadline);p=entry['point'];a=entry['albedo']
            ref=prepared.point(p['normal'],p['view'],a,deadline=deadline)
            actual=candidate(p['normal'],p['view'],a,model=model,environment=environment,budget=budget,ref=ref,deadline=deadline)
            report['rows'].append(dict(entry,reference=ref,actual=actual));write_report(output,report)
        check_deadline(deadline);report['status']=WORKER_COMPLETE
    except Exception as error:
        report.update(status='stopped',error=f'{type(error).__name__}: {error}')
        if isinstance(error,old.scalar.ReferenceConvergenceError):report['unresolved']=error.report
        raise
    finally:write_report(output,report)


def validate_row(row,model,environment):
    ref=row['reference'];validate_reference(ref);a=colour(row['albedo']);point=row['point'];actual=row['actual']
    if (ref['model']!=model or ref['environment']!=environment or ref['albedo']!=list(a)
        or ref['normal']!=list(old.scalar._normal(point['normal'])) or ref['view']!=list(old.scalar._normal(point['view']))):
        raise ValueError('row input identity')
    order=ref['history'][-1]['n_mu'];values=actual['estimates']
    if (len(values)!=2 or any(len(x)!=3 for x in values) or actual['chunk_size']!=1024
        or actual['directions']!=8*order*order or actual['calls']!=2*math.ceil(4*order*order/1024)
        or not 0<actual['elapsed_seconds']<120):raise ValueError('candidate shape/work mismatch')
    limits=[1e-10+1e-8*abs(x) for x in ref['radiance']]
    errors=[[abs(a-b) for a,b in zip(x,ref['radiance'])] for x in values]
    if (actual['errors']!=errors or actual['limits']!=limits
        or any(not math.isfinite(x) or x>t for error in errors for x,t in zip(error,limits))):raise ValueError('candidate evidence mismatch')


def read_worker(path,digest):
    blob=_read(Path(path),4*1024**2)
    if hashlib.sha256(blob).hexdigest()!=digest:raise ValueError('worker hash mismatch')
    row=json.loads(blob);s=row['supervisor']
    if (row['status']!=WORKER_COMPLETE or row['profile']!=PROFILE or row['sources']!=sources()
        or row['manifest_sha256']!=old.fixed.MANIFEST_HASH or row['producer']['wheel']['sha256']!=old.fixed.WHEEL_HASH
        or s['failure'] is not None or s['exit_code']!=0 or s['monitored_pid']!=row['worker_pid']
        or not 0<s['elapsed_seconds']<180 or not 0<s['sampled_peak_working_set_bytes']<=1024**3
        or (row['model'],row['environment']) not in schedule() or len(row['rows'])!=24):raise ValueError('unqualified worker')
    radial=row['rows'][0]['reference']['moments']
    if row['moment_samples']!=(0 if radial is None else radial['evaluations']):raise ValueError('moment work mismatch')
    for item in row['rows']:
        validate_row(item,row['model'],row['environment'])
        if item['reference']['moments']!=radial:raise ValueError('worker moment identity')
    return row


def replay(root,digest,corpus):
    root=Path(root);blob=_read(root/'cohort.json',262144)
    if hashlib.sha256(blob).hexdigest()!=digest:raise ValueError('cohort hash mismatch')
    record=json.loads(blob)
    if (record['profile']!=PROFILE or record['status']!=COMPLETE or record['sources']!=sources()
        or not 0<record['elapsed_seconds']<=600 or len(record['workers'])!=4):raise ValueError('unqualified cohort')
    expected=inputs(old.fixed.load_bundle(corpus,path=old.fixed.MANIFEST,sha256=old.fixed.MANIFEST_HASH)['buffers'])
    workers=[];samples=0
    for (model,environment),descriptor in zip(schedule(),record['workers']):
        if descriptor['path']!=f'{model}-{environment}.json':raise ValueError('worker path/ownership mismatch')
        row=read_worker(root/descriptor['path'],descriptor['sha256'])
        if (row['model'],row['environment'])!=(model,environment):raise ValueError('worker identity')
        for entry,wanted in zip(row['rows'],expected):
            if {k:entry[k] for k in wanted}!=wanted:raise ValueError('saved albedo/selected point identity')
            samples+=entry['reference']['evaluations']+entry['actual']['directions']
        samples+=row['moment_samples'];workers.append(row)
    if samples>PROFILE['maximum_samples']:raise ValueError('cohort work cap')
    return dict(record=record,sha256=digest,workers=workers,samples=samples)


def launch_guard(started):check_deadline(started+600-180)


def run(root,output,wheel):
    output=Path(output)
    if output.exists():raise FileExistsError('fresh albedo cohort directory required')
    started=time.perf_counter();output.mkdir(parents=True)
    record=dict(profile=PROFILE,status='working',workers=[],sources=sources());write_report(output/'cohort.json',record)
    try:
        for model,environment in schedule():
            launch_guard(started);path=output/f'{model}-{environment}.json'
            code=supervise(root,path,worker_module='tools.albedo_environment',
                worker_args=('--wheel',str(wheel),'--model',model,'--environment',environment))
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            record['workers'].append(dict(path=path.name,sha256=digest));write_report(output/'cohort.json',record)
            if code:raise RuntimeError('albedo worker failed; no retry')
            read_worker(path,digest)
        check_deadline(started+600);record['status']=COMPLETE
    except Exception as error:
        record.update(status='stopped',error=f'{type(error).__name__}: {error}');raise
    finally:
        record['elapsed_seconds']=time.perf_counter()-started;write_report(output/'cohort.json',record)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixtures',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--wheel',type=Path,required=True);p.add_argument('--worker',action='store_true')
    p.add_argument('--model');p.add_argument('--environment');a=p.parse_args()
    if a.worker:worker(a.fixtures,a.output,a.wheel,a.model,a.environment)
    else:run(a.fixtures,a.output,a.wheel)


if __name__=='__main__':main()
