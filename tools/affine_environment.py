"""Bounded world-space affine-lighting preflight; development tooling, not an API.

Reference: independent fixed-profile glossy formulas plus segment-integrated matte
moments. Candidate: complete public material and light.integrate in world space.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import torch

from tools import environment_full_images as fixed
from tools import shading_environment_reference as scalar
from tools.environment_continuation import check_sources, FROZEN_SOURCES
from tools.environment_image_preflight import select_pixels
from tools.material_loss_cost import check_deadline, supervise, write_report
from tools.principled_direct_images import PreparedCurve, RESOURCE
from tools.shading_independent_evaluator import LossCurve
from tools.corpus_registry import _read

BANDS=((.3,.2,.1),(-.2,.35,.1),(.15,-.25,.4))
PROFILE=dict(name='affine-world-environment-v1',bands=[list(x) for x in BANDS],
    materials=['classic','principled'],points=8,albedo=[.2,.4,.6],roughness=1.,f0=.04,
    specular=.1,exponent=2.,reference_directions=174592,candidate_directions=131072,
    moment_samples=32768,aggregate_seconds=600,cooperative_seconds=120,
    watchdog_seconds=180,sampled_working_set_limit_bytes=1024**3,
    dtype='float64',device='cpu',threads=1,reference_atol=1e-11,reference_rtol=1e-9,
    channel_atol=1e-10,channel_rtol=1e-8)
COMPLETE='passed affine point preflight; not full images or M1 acceptance'


def bands_checked(bands):
    if not isinstance(bands,(tuple,list)) or len(bands)!=3:raise ValueError('three coefficient triples required')
    result=tuple(scalar._triple(row) for row in bands)
    if any(math.hypot(*row)>=1 for row in result):raise ValueError('strict positive-radiance profile required')
    return result


def field(directions,bands=BANDS):
    b=bands_checked(bands)
    if (not isinstance(directions,torch.Tensor) or directions.ndim!=2 or directions.shape[1]!=3
        or directions.dtype!=torch.float64 or directions.device.type!='cpu'
        or not bool(torch.isfinite(directions).all())):raise ValueError('finite CPU float64 directions required')
    return 1+directions@torch.tensor(b,dtype=torch.float64).T


def coefficients(bands=BANDS):
    b=bands_checked(bands);c1=math.sqrt(3/(4*math.pi))
    return [[math.sqrt(4*math.pi)]*3]+[[row[j]/c1 for row in b] for j in (1,2,0)]+[[0.]*3 for _ in range(5)]


def frame(normal,view):
    n,v=scalar._normal(normal),scalar._normal(view);c=scalar._dot(n,v)
    if c<=0:raise ValueError('positive view hemisphere required')
    c=min(c,1.)
    tangent=tuple(x-c*y for x,y in zip(v,n))
    if math.hypot(*tangent)<1e-12:
        helper=(0.,1.,0.) if abs(n[2])>.9 else (0.,0.,1.)
        tangent=scalar._cross(helper,n)
    t=scalar._normal(tangent);b=scalar._normal(scalar._cross(n,t))
    t=scalar._cross(b,n)
    return c,torch.tensor([t,b,n],dtype=torch.float64)


def quadrature(order,split):
    if type(order) is not int:raise ValueError('integer order required')
    q,w=fixed.quadrature(order,split)
    reflected=q.clone();reflected[:,1]*=-1
    return torch.cat((q,reflected)),torch.cat((w/2,w/2))


def moments(curve,*,deadline,max_evaluations=32768):
    """Independent segment quadrature of 2*loss(mu)*mu and *mu^2."""
    if type(max_evaluations) is not int or not 0<max_evaluations<=32768:raise ValueError('moment work cap')
    knots=tuple(scalar._number(x) for x in curve.c)
    if (not 2<=len(knots)<=513 or knots[0]!=0 or knots[-1]!=1
        or any(a>=b for a,b in zip(knots,knots[1:]))):raise ValueError('moment curve knots')
    deadline=min(deadline,time.perf_counter()+20);history=[];count=0;previous=None;streak=0
    for order in (8,16,32):
        check_deadline(deadline)
        if count+order*(len(knots)-1)>max_evaluations:
            raise scalar.ReferenceConvergenceError(dict(reason='work_limit',evaluations=count,history=history))
        nodes,weights=scalar.gauss_legendre(order);terms=[[],[]]
        for a,b in zip(knots,knots[1:]):
            for u,w in zip(nodes,weights):
                check_deadline(deadline)
                mu=a+(b-a)*u;jac=b-a
                loss=scalar._number(curve.loss(mu,1.,.04))
                if not 0<=loss<=1:raise ValueError('invalid loss moment sample')
                terms[0].append(2*w*jac*loss*mu);terms[1].append(2*w*jac*loss*mu*mu)
                count+=1
        values=[math.fsum(x) for x in terms]
        limits=[1e-13+1e-11*abs(x) for x in values]
        change=None if previous is None else [abs(a-b) for a,b in zip(values,previous)]
        if change is not None:streak=streak+1 if all(a<=b for a,b in zip(change,limits)) else 0
        history.append(dict(order=order,values=values,change=change,limits=limits,streak=streak,evaluations=count))
        previous=values
    if streak!=2:raise scalar.ReferenceConvergenceError(dict(reason='moment_level_limit',evaluations=count,history=history))
    mean=scalar._number(curve.mean(1.,.04))
    if mean<=0 or abs(values[0]-mean)>1e-11:raise ValueError('independent loss mean mismatch')
    check_deadline(deadline)
    return dict(values=values,adaptive_mean=mean,history=history,evaluations=count,segments=len(knots)-1)


def reference(normal,view,*,model,curve,deadline,bands=BANDS,max_evaluations=174592):
    check_deadline(deadline)
    if model not in PROFILE['materials']:raise ValueError('unknown material')
    if type(max_evaluations) is not int or not 0<max_evaluations<=174592:raise ValueError('reference work cap')
    b=bands_checked(bands);c,basis=frame(normal,view)
    deadline=min(deadline,time.perf_counter()+30)
    radial=None;matte=torch.zeros(3,dtype=torch.float64)
    if model=='principled':
        if curve is None:raise ValueError('independent curve required')
        radial=moments(curve,deadline=deadline)
        m,j=radial['values'];lv=scalar._number(curve.loss(c,1.,.04))
        matte=torch.tensor(PROFILE['albedo'],dtype=torch.float64)*lv*(m+(torch.tensor(b,dtype=torch.float64)@basis[2])*j)/radial['adaptive_mean']
    history=[];count=0;previous=None;streak=0
    for order in (8,16,32,64,128):
        check_deadline(deadline)
        if count+8*order*order>max_evaluations:
            raise scalar.ReferenceConvergenceError(dict(reason='work_limit',evaluations=count,history=history))
        estimates=[]
        for split in (.5,.25):
            q,w=quadrature(order,split)
            brdf=fixed.reference_brdf(c,q,model=model)
            estimate=(brdf*field(q@basis,b)*(q[:,2]*w)[:,None]).sum(0)+matte
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
                normal=list(scalar._normal(normal)),view=list(scalar._normal(view)),
                radiance=mean.tolist(),history=history,evaluations=count,moments=radial)
        previous=estimates
    raise scalar.ReferenceConvergenceError(dict(reason='level_limit',evaluations=count,history=history))


def validate_reference(ref):
    """Recheck every refinement; never trust a stored success label alone."""
    if ref['status']!='converged' or not 3<=len(ref['history'])<=5:raise ValueError('unresolved reference')
    bands_checked(ref['bands']);frame(ref['normal'],ref['view'])
    previous=None;count=0;streak=0
    for i,row in enumerate(ref['history']):
        order=(8,16,32,64,128)[i];count+=8*order*order
        values=torch.tensor(row['estimates'],dtype=torch.float64)
        if values.shape!=(2,3) or not bool(torch.isfinite(values).all()) or bool((values<0).any()):raise ValueError('reference RGB shape/domain')
        mean=values.mean(0);limits=1e-11+1e-9*mean.abs();delta=(values[0]-values[1]).abs()
        change=None if previous is None else (values-previous).abs().amax(0)
        if change is not None:streak=streak+1 if bool(((change<=limits)&(delta<=limits)).all()) else 0
        if (row['n_mu']!=order or row['evaluations']!=count or row['streak']!=streak
            or row['radiance']!=mean.tolist() or row['threshold']!=limits.tolist()
            or row['offset_change']!=delta.tolist() or row['change']!=(None if change is None else change.tolist())
            or (streak==2 and i!=len(ref['history'])-1)):raise ValueError('reference history mismatch')
        previous=values
    if streak!=2 or ref['evaluations']!=count or ref['radiance']!=mean.tolist():raise ValueError('reference final mismatch')
    if ref['model']=='principled':
        m=ref['moments'];last=None;streak=0;count=0
        if (len(m['history'])!=3 or type(m['segments']) is not int or not 1<=m['segments']<=512
            or not math.isfinite(m['adaptive_mean']) or not 0<m['adaptive_mean']<=1):raise ValueError('moment history shape/domain')
        for order,row in zip((8,16,32),m['history']):
            count+=order*m['segments']
            values=row['values']
            if len(values)!=2 or any(not math.isfinite(x) or not 0<x<=1 for x in values):raise ValueError('moment domain')
            limits=[1e-13+1e-11*abs(x) for x in values]
            change=None if last is None else [abs(a-b) for a,b in zip(values,last)]
            if change is not None:streak=streak+1 if all(a<=b for a,b in zip(change,limits)) else 0
            if (row['order']!=order or row['change']!=change or row['limits']!=limits
                or row['streak']!=streak or row['evaluations']!=count):raise ValueError('moment history mismatch')
            last=values
        if (streak!=2 or m['values']!=last or m['evaluations']!=count or not 0<count<=32768
            or abs(last[0]-m['adaptive_mean'])>1e-11):raise ValueError('moment final mismatch')
    elif ref['model']!='classic' or ref['moments'] is not None:raise ValueError('reference model mismatch')


def candidate(normal,view,*,model,budget,ref,deadline,bands=BANDS):
    from gffx.torch import material,light
    validate_reference(ref);check_deadline(deadline)
    b=bands_checked(bands);_,basis=frame(normal,view)
    if (ref['model']!=model or ref['bands']!=[list(x) for x in b]
        or ref['normal']!=list(scalar._normal(normal)) or ref['view']!=list(scalar._normal(view))):raise ValueError('reference input identity')
    n=torch.tensor([normal],dtype=torch.float64);v=torch.tensor([view],dtype=torch.float64)
    estimates=[];calls=0;count=0;started=time.perf_counter()
    with torch.no_grad():
        for split in (.5,.25):
            q,w=quadrature(ref['history'][-1]['n_mu'],split);world=q@basis
            total=torch.zeros(3,dtype=torch.float64)
            for start in range(0,len(q),1024):
                check_deadline(deadline);d=world[start:start+1024];size=len(d)
                args=dict(albedo=torch.tensor(PROFILE['albedo'],dtype=torch.float64))
                if model=='classic':
                    brdf=material.classic(n,v,d,specular=torch.tensor(.1,dtype=torch.float64),exponent=torch.tensor(2.,dtype=torch.float64),**args)
                else:
                    brdf=material.principled(n,v,d,roughness=torch.tensor(1.,dtype=torch.float64),f0=torch.tensor(.04,dtype=torch.float64),budget=budget,**args)
                packet=dict(measure='solid_angle',direction=d[None],radiance=field(d,b)[None],
                    valid=torch.ones((1,size),dtype=torch.bool),solid_angle=w[start:start+1024][None])
                total+=light.integrate(n,brdf,packet)[0];count+=size;calls+=1
            estimates.append(total.tolist())
    check_deadline(deadline)
    limits=[1e-10+1e-8*abs(x) for x in ref['radiance']]
    errors=[[abs(a-b) for a,b in zip(row,ref['radiance'])] for row in estimates]
    if any(not math.isfinite(x) or x>t for row in errors for x,t in zip(row,limits)):raise ValueError('public affine reference tolerance failed')
    return dict(estimates=estimates,errors=errors,limits=limits,directions=count,calls=calls,chunk_size=1024,
                elapsed_seconds=time.perf_counter()-started)


def sources():
    check_sources()
    paths=[Path(__file__),Path(__file__).with_name('environment_image_preflight.py'),
           Path(__file__).with_name('material_loss_cost.py')]
    return dict(FROZEN_SOURCES,**{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


def worker(root,output,wheel,model,index):
    from gffx.torch.material import PrincipledBudget
    torch.set_num_threads(1);torch.set_num_interop_threads(1);deadline=time.perf_counter()+120
    report=dict(status='working',worker_pid=os.getpid(),profile=PROFILE,model=model,index=index)
    try:
        if model not in PROFILE['materials'] or type(index) is not int or not 0<=index<8:raise ValueError('point identity')
        report.update(sources=sources(),producer=fixed.producer_identity(wheel),manifest_sha256=fixed.MANIFEST_HASH)
        if report['producer']['wheel']['sha256']!=fixed.WHEEL_HASH:raise ValueError('wheel identity')
        bundle=fixed.load_bundle(root,path=fixed.MANIFEST,sha256=fixed.MANIFEST_HASH)
        point=select_pixels(bundle['buffers'])[index];report['point']=point
        budget=PrincipledBudget.bundled(RESOURCE,device='cpu',dtype=torch.float64)
        curve=PreparedCurve(LossCurve(budget.to_envelope()['payload']),roughness=1.)
        write_report(output,report)
        ref=reference(point['normal'],point['view'],model=model,curve=curve,deadline=deadline)
        report['reference']=ref;write_report(output,report)
        report['actual']=candidate(point['normal'],point['view'],model=model,budget=budget,ref=ref,deadline=deadline)
        report['status']='passed affine point; not an image'
    except Exception as error:
        report.update(status='stopped',error=f'{type(error).__name__}: {error}')
        if isinstance(error,scalar.ReferenceConvergenceError):report['unresolved']=error.report
        raise
    finally:write_report(output,report)


def read_point(path,digest):
    blob=_read(Path(path),262144)
    if hashlib.sha256(blob).hexdigest()!=digest:raise ValueError('point identity mismatch')
    row=json.loads(blob);s=row['supervisor']
    if (row['status']!='passed affine point; not an image' or row['profile']!=PROFILE or row['sources']!=sources()
        or row['manifest_sha256']!=fixed.MANIFEST_HASH or row['producer']['wheel']['sha256']!=fixed.WHEEL_HASH
        or s['failure'] is not None or s['exit_code']!=0 or s['monitored_pid']!=row['worker_pid']
        or not 0<s['elapsed_seconds']<180 or not 0<s['sampled_peak_working_set_bytes']<=1024**3):raise ValueError('unqualified point')
    ref=row['reference'];validate_reference(ref);actual=row['actual'];order=ref['history'][-1]['n_mu']
    if (ref['model']!=row['model'] or ref['bands']!=PROFILE['bands']
        or ref['normal']!=list(scalar._normal(row['point']['normal']))
        or ref['view']!=list(scalar._normal(row['point']['view']))
        or actual['directions']!=8*order*order or actual['calls']!=2*math.ceil(4*order*order/1024)
        or actual['chunk_size']!=1024 or not 0<actual['elapsed_seconds']<120):raise ValueError('point work/geometry mismatch')
    values=actual['estimates']
    if len(values)!=2 or any(len(x)!=3 for x in values):raise ValueError('candidate RGB shape')
    limits=[1e-10+1e-8*abs(x) for x in ref['radiance']]
    errors=[[abs(a-b) for a,b in zip(x,ref['radiance'])] for x in values]
    if (actual['errors']!=errors or actual['limits']!=limits
        or any(not math.isfinite(x) or x>t for error in errors for x,t in zip(error,limits))):raise ValueError('candidate evidence mismatch')
    return row


def replay(root,digest,corpus):
    root=Path(root);blob=_read(root/'cohort.json',262144)
    if hashlib.sha256(blob).hexdigest()!=digest:raise ValueError('cohort identity mismatch')
    record=json.loads(blob)
    if (record['profile']!=PROFILE or record['status']!=COMPLETE or record['sources']!=sources()
        or not 0<record['elapsed_seconds']<=600 or len(record['points'])!=16):raise ValueError('unqualified affine cohort')
    bundle=fixed.load_bundle(corpus,path=fixed.MANIFEST,sha256=fixed.MANIFEST_HASH)
    selected=json.loads(json.dumps(select_pixels(bundle['buffers'])));rows=[]
    for position,descriptor in enumerate(record['points']):
        model=PROFILE['materials'][position//8];index=position%8
        if descriptor['path']!=f'{model}-{index}.json':raise ValueError('point path/ownership mismatch')
        row=read_point(root/descriptor['path'],descriptor['sha256'])
        if row['model']!=model or row['index']!=index or row['point']!=selected[index]:raise ValueError('selected point identity mismatch')
        rows.append(row)
    return dict(record=record,sha256=digest,rows=rows)


def launch_guard(started):check_deadline(started+600-180)


def run(root,output,wheel):
    output=Path(output)
    if output.exists():raise FileExistsError('fresh affine cohort directory required')
    started=time.perf_counter();output.mkdir(parents=True)
    record=dict(profile=PROFILE,status='working',points=[],sources=sources())
    write_report(output/'cohort.json',record)
    try:
        for model in PROFILE['materials']:
            for index in range(8):
                launch_guard(started);path=output/f'{model}-{index}.json'
                code=supervise(root,path,worker_module='tools.affine_environment',
                    worker_args=('--wheel',str(wheel),'--model',model,'--index',str(index)))
                digest=hashlib.sha256(path.read_bytes()).hexdigest()
                record['points'].append(dict(path=path.name,sha256=digest));write_report(output/'cohort.json',record)
                if code:raise RuntimeError('affine worker failed; no retry')
                read_point(path,digest)
        check_deadline(started+600);record['status']=COMPLETE
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
