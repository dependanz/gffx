"""Bounded two-pass environment adjoints; selected pixels, not full images."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
import torch

from tools import albedo_environment as base
from tools import image_loss_gradients as direct
from tools import image_loss_reference as ref
from tools.material_loss_cost import check_deadline, supervise, write_report

PASSED = 'environment adjoint preflight passed; two pixels per profile only'
PROFILE = dict(name='environment-loss-preflight-v1', dtype='float64', device='cpu', threads=1,
               pixels=2, chunk_size=1024, passes=2, splits=[.5,.25], step_size=5.,
               cooperative_seconds=120, watchdog_seconds=180, sampled_working_set_limit_bytes=1024**3,
               aggregate_seconds=900, maximum_calls=4096, maximum_directions=4194304)


def sources():
    result = base.sources(); result.update(direct.sources())
    result['environment_loss.py'] = direct.digest(Path(__file__))
    return result


def schedule(): return [(m,e) for m in ('classic','principled') for e in ('uniform','affine')]


def streamed_scalar_example(value, target, coefficients):
    """Worked teaching control: total first, then its cotangent, not chunk losses."""
    complete = math.fsum(value*c for c in coefficients)
    cotangent = 2*(complete-target)/3
    return math.fsum(cotangent*c for c in coefficients)


def selected(root):
    buffers = direct.load_bundle(root, path=direct.MANIFEST, sha256=direct.MANIFEST_HASH)['buffers']
    points = base.old.select_pixels(buffers)
    ids = [points[i]['flat_index'] for i in (0,4)]
    if len(set(ids)) != 2: raise ValueError('distinct preflight pixels required')
    return ids, [buffers[name].reshape(-1, c)[ids].contiguous() for name,c in (('normal',3),('view',3),('uv',2))]


def slope(normal, view, model, environment, prepared):
    bands = base.bands(environment); c,basis = base.old.frame(normal, view)
    dot = [math.fsum(x*y for x,y in zip(b, basis[2].tolist())) for b in bands]
    if model == 'classic': return [1+2*x/3 for x in dot]
    if model != 'principled' or prepared is None: raise ValueError('independent principled setup required')
    m,j = prepared.radial['values']; mean = prepared.radial['adaptive_mean']
    lv = prepared.curve.loss(c,1.,.04)
    return [lv*(m+x*j)/mean for x in dot]


def reference_inputs(n,v,uv,model,environment,budget,deadline):
    curve = base.old.PreparedCurve(base.old.LossCurve(budget.to_envelope()['payload']),roughness=1.) if model=='principled' else None
    prepared = base.Prepared(model,curve,environment,deadline=deadline)
    weights = [ref.stencil(32,32,u) for u in uv.tolist()]
    tex = direct.texture().tolist(); references = []; b = []; s = []
    for nn,vv,taps in zip(n.tolist(),v.tolist(),weights):
        a = [math.fsum(tex[y][x][c]*w for y,x,w in taps) for c in range(3)]
        rr = prepared.point(nn,vv,a,deadline=deadline); base.validate_reference(rr)
        ss = slope(nn,vv,model,environment,prepared)
        references.append(rr); s.append(ss); b.append([r-x*y for r,x,y in zip(rr['radiance'],ss,a)])
    target = ref.evaluate((direct.texture()*.85).tolist(),weights,b,s,[[0.]*3 for _ in weights])['values']
    return dict(references=references,intercepts=b,slopes=s,target=target,
                moment_samples=prepared.moment_samples,reference_directions=sum(r['evaluations'] for r in references))


def contribution(tex,n,v,uv,d,w,model,environment,budget):
    from gffx.torch import material,light,render
    pyramid,offsets = render.texture_pyramid(tex,levels=1)
    a = render.texture(pyramid,offsets,32,32,uv.contiguous(),channel_count=3,
                       filter=render.FILTER_BILINEAR,mip_filter=render.MIP_NEAREST,
                       wrap_u=render.WRAP_REPEAT,wrap_v=render.WRAP_REPEAT)
    if model=='classic':
        brdf = material.classic(n,v,d,albedo=a,specular=torch.tensor(.1,dtype=torch.float64),exponent=torch.tensor(2.,dtype=torch.float64))
    else:
        brdf = material.principled(n,v,d,albedo=a,roughness=torch.tensor(1.,dtype=torch.float64),f0=torch.tensor(.04,dtype=torch.float64),budget=budget)
    packet = dict(measure='solid_angle',direction=d[None],radiance=base.old.field(d,base.bands(environment))[None],
                  valid=torch.ones((1,len(d)),dtype=torch.bool),solid_angle=w[None])
    return light.integrate(n,brdf,packet)


def angular_pass(tex,n,v,uv,target,*,model,environment,budget,orders,split,deadline,chunk_size=1024):
    direct.validate(n,v,uv,model,deadline); base.bands(environment)
    if len(n)>2 or len(orders)!=len(n) or any(type(k) is not int or k not in (8,16,32,64,128) for k in orders):
        raise ValueError('bounded angular orders/pixels required')
    if split not in (.5,.25) or type(chunk_size) is not int or not 1<=chunk_size<=1024: raise ValueError('angular work limit')
    if (tex.shape!=(32,32,3) or tex.dtype!=torch.float64 or tex.device.type!='cpu'
            or not bool(torch.isfinite(tex).all()) or not bool(((tex>=0)&(tex<=1)).all())
            or target.shape!=n.shape or target.dtype!=torch.float64 or target.device.type!='cpu'
            or target.requires_grad or not bool(torch.isfinite(target).all())): raise ValueError('texture/target domain')
    values=[]; gradient=torch.zeros_like(tex); calls=0; directions=0
    for i,order in enumerate(orders):
        q,w=base.old.quadrature(order,split); d=q@base.old.frame(n[i].tolist(),v[i].tolist())[1]
        total=torch.zeros((1,3),dtype=torch.float64)
        with torch.no_grad():
            for start in range(0,len(q),chunk_size):
                check_deadline(deadline); end=min(start+chunk_size,len(q))
                total+=contribution(tex,n[i:i+1],v[i:i+1],uv[i:i+1],d[start:end],w[start:end],model,environment,budget)
                calls+=1; directions+=end-start
        cotangent=2*(total-target[i:i+1])/(3*len(n))
        for start in range(0,len(q),chunk_size):
            check_deadline(deadline); end=min(start+chunk_size,len(q))
            leaf=tex.detach().requires_grad_(True)
            rgb=contribution(leaf,n[i:i+1],v[i:i+1],uv[i:i+1],d[start:end],w[start:end],model,environment,budget)
            grad,=torch.autograd.grad((rgb*cotangent).sum(),leaf)
            gradient+=grad.detach(); calls+=1; directions+=end-start
            del grad,rgb,leaf
        values.append(total[0])
    values=torch.stack(values)
    loss=float((values-target).square().sum()/(3*len(n)))
    check_deadline(deadline)
    return dict(values=values,gradient=gradient,loss=loss,calls=calls,directions=directions)


def inspect(tex,weights,inputs,values,gradient,loss):
    expected=ref.evaluate(tex,weights,inputs['intercepts'],inputs['slopes'],inputs['target'])
    metrics=dict(rgb=direct._metric(values,expected['values'],1e-10,1e-8),
                 gradient=direct._metric(gradient,expected['gradient'],1e-11,1e-7))
    epsilon=[[1e-10+1e-8*abs(x) for x in row] for row in expected['values']]
    bound=1e-12+math.fsum(2*abs(r-t)*e+e*e for rr,tt,ee in zip(expected['values'],inputs['target'],epsilon)
                         for r,t,e in zip(rr,tt,ee))/(3*len(weights))
    error=abs(loss-expected['loss'])
    metrics['loss']=dict(passed=math.isfinite(loss) and error<=bound,absolute_error=error,allowance=bound)
    touched={(y,x) for taps in weights for y,x,w in taps if w!=0}
    zero=all(gradient[y][x]==[0.,0.,0.] for y in range(32) for x in range(32) if (y,x) not in touched)
    nonzero=any(x!=0 for row in gradient for rgb in row for x in rgb)
    if not all(m['passed'] for m in metrics.values()) or not zero or not nonzero: raise ValueError('environment adjoint failed: '+str(metrics))
    return dict(metrics=metrics,untouched_zero=zero,nonzero=nonzero,reference_loss=expected['loss'])


def evaluate(n,v,uv,model,environment,*,deadline):
    direct.validate(n,v,uv,model,deadline); base.bands(environment)
    if len(n)!=2: raise ValueError('exact two-pixel preflight required')
    from gffx.torch.material import PrincipledBudget
    budget=PrincipledBudget.bundled(direct.RESOURCE,device='cpu',dtype=torch.float64)
    inputs=reference_inputs(n,v,uv,model,environment,budget,deadline)
    orders=[r['history'][-1]['n_mu'] for r in inputs['references']]
    weights=[ref.stencil(32,32,u) for u in uv.tolist()]
    target=torch.tensor(inputs['target'],dtype=torch.float64); tex=direct.texture(); passes=[]; calls=0; directions=0
    for iteration in range(2):
        rows=[]; average=torch.zeros_like(tex)
        for split in (.5,.25):
            result=angular_pass(tex,n,v,uv,target,model=model,environment=environment,budget=budget,orders=orders,split=split,deadline=deadline)
            calls+=result['calls']; directions+=result['directions']
            values=result['values'].tolist(); gradient=result['gradient'].tolist()
            checked=inspect(tex.tolist(),weights,inputs,values,gradient,result['loss'])
            rows.append(dict(split=split,values=values,gradient=gradient,loss=result['loss'],**checked))
            average+=result['gradient']/2
        passes.append(dict(texture=tex.tolist(),offsets=rows))
        if iteration==0:
            expected=ref.evaluate(tex.tolist(),weights,inputs['intercepts'],inputs['slopes'],inputs['target'])
            fd=ref.finite_differences(tex.tolist(),weights,inputs['intercepts'],inputs['slopes'],inputs['target'],expected['gradient'])
            if not all(x['passed'] for x in fd): raise ValueError('reference directional FD failed')
            tex=(tex-5*average).contiguous()
            if not bool(((tex>=0)&(tex<=1)).all()): raise ValueError('fitting update domain')
    if any(b['loss']>=a['loss'] for a,b in zip(passes[0]['offsets'],passes[1]['offsets'])): raise ValueError('loss did not decrease')
    check_deadline(deadline)
    return dict(inputs=inputs,passes=passes,finite_differences=fd,calls=calls,directions=directions,orders=orders,budget=budget.metadata)


def worker(root,output,wheel,model,environment):
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    start=time.perf_counter(); report=dict(status='working',worker_pid=os.getpid(),profile=PROFILE,
        model=model,environment=environment,corpus_root=str(root.resolve()),sources=sources(),manifest_sha256=direct.MANIFEST_HASH)
    try:
        if (model,environment) not in schedule(): raise ValueError('unknown profile')
        report['producer']=direct.producer_identity(wheel)
        if report['producer']['wheel']['sha256']!=direct.WHEEL_HASH: raise ValueError('wheel identity')
        report['indices'],(n,v,uv)=selected(root)
        report['result']=evaluate(n,v,uv,model,environment,deadline=start+120)
        report['elapsed_seconds']=time.perf_counter()-start; report['status']=PASSED
        if len(json.dumps(report,allow_nan=False).encode())>16*1024**2: raise ValueError('report size limit')
    except Exception as error:
        report.update(status='stopped',error=type(error).__name__+': '+str(error));write_report(output,report);raise
    write_report(output,report)


def run(root,output,wheel):
    output.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    cohort=dict(status='working',profile=PROFILE,sources=sources(),workers=[])
    write_report(output/'cohort.json',cohort)
    for index,(model,environment) in enumerate(schedule()):
        if time.perf_counter()-start>720:
            cohort.update(status='stopped',error='aggregate reserve exhausted');write_report(output/'cohort.json',cohort)
            raise TimeoutError('aggregate reserve exhausted')
        path=output/f'worker-{index}.json'
        code=supervise(root,path,worker_module='tools.environment_loss',worker_args=('--wheel',str(wheel),'--model',model,'--environment',environment))
        cohort['workers'].append(dict(path=path.name,sha256=direct.digest(path),model=model,environment=environment))
        cohort.update(status='stopped' if code else 'working',elapsed_seconds=time.perf_counter()-start)
        write_report(output/'cohort.json',cohort)
        if code: raise RuntimeError('stopped environment adjoint; no automatic retry')
    cohort['status']=PASSED;write_report(output/'cohort.json',cohort);return cohort


def replay(root,sha256,*,blob=None):
    if blob is None and (root/'cohort.json').stat().st_size>65536: raise ValueError('cohort size')
    raw=(root/'cohort.json').read_bytes() if blob is None else blob
    if len(raw)>65536 or hashlib.sha256(raw).hexdigest()!=sha256: raise ValueError('cohort hash/size')
    cohort=json.loads(raw)
    if (cohort['status']!=PASSED or cohort['profile']!=PROFILE or cohort['sources']!=sources()
            or len(cohort['workers'])!=4 or not 0<cohort['elapsed_seconds']<=900): raise ValueError('cohort identity')
    from gffx.torch.material import PrincipledBudget
    budget=PrincipledBudget.bundled(direct.RESOURCE,device='cpu',dtype=torch.float64)
    for index,(entry,(model,environment)) in enumerate(zip(cohort['workers'],schedule())):
        if (entry['path']!=f'worker-{index}.json' or entry['model']!=model or entry['environment']!=environment): raise ValueError('schedule identity')
        path=root/entry['path']
        if path.stat().st_size>16*1024**2 or direct.digest(path)!=entry['sha256']: raise ValueError('worker bytes changed')
        r=json.loads(path.read_text());sup=r['supervisor']; result=r['result']
        if (r['status']!=PASSED or r['profile']!=PROFILE or r['sources']!=sources() or r['model']!=model or r['environment']!=environment
                or r['manifest_sha256']!=direct.MANIFEST_HASH or r['producer']['wheel']['sha256']!=direct.WHEEL_HASH
                or r['producer']!=direct.producer_identity(Path(r['producer']['wheel']['path'])) or result['budget']!=budget.metadata): raise ValueError('worker identity')
        if (sup['failure'] is not None or sup['exit_code']!=0 or sup['monitored_pid']!=r['worker_pid']
                or not 0<sup['sampled_peak_working_set_bytes']<=1024**3 or not 0<sup['elapsed_seconds']<=180
                or not 0<r['elapsed_seconds']<=120): raise ValueError('worker limits')
        ids,(n,v,uv)=selected(Path(r['corpus_root']))
        if ids!=r['indices']: raise ValueError('pixel ownership')
        inputs=reference_inputs(n,v,uv,model,environment,budget,time.perf_counter()+120)
        if inputs!=result['inputs']: raise ValueError('independent reference changed')
        orders=[x['history'][-1]['n_mu'] for x in inputs['references']]
        if (result['orders']!=orders or result['directions']!=8*sum(4*k*k for k in orders)
                or result['calls']!=8*sum(math.ceil(4*k*k/1024) for k in orders) or len(result['passes'])!=2): raise ValueError('work counts')
        tex=direct.texture();weights=[ref.stencil(32,32,u) for u in uv.tolist()]
        for iteration,p in enumerate(result['passes']):
            if p['texture']!=tex.tolist() or len(p['offsets'])!=2: raise ValueError('texture/pass identity')
            average=torch.zeros_like(tex)
            for split,row in zip((.5,.25),p['offsets']):
                if row['split']!=split: raise ValueError('offset identity')
                checked=inspect(p['texture'],weights,inputs,row['values'],row['gradient'],row['loss'])
                if any(row[k]!=value for k,value in checked.items()): raise ValueError('replayed metrics differ')
                average+=torch.tensor(row['gradient'],dtype=torch.float64)/2
            if iteration==0:
                expected=ref.evaluate(tex.tolist(),weights,inputs['intercepts'],inputs['slopes'],inputs['target'])
                fd=ref.finite_differences(tex.tolist(),weights,inputs['intercepts'],inputs['slopes'],inputs['target'],expected['gradient'])
                if fd!=result['finite_differences'] or not all(x['passed'] for x in fd): raise ValueError('FD evidence')
                tex-=5*average
        if any(b['loss']>=a['loss'] for a,b in zip(result['passes'][0]['offsets'],result['passes'][1]['offsets'])): raise ValueError('no fitting improvement')
    return cohort


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('fixtures','output','wheel'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--worker',action='store_true');p.add_argument('--model');p.add_argument('--environment')
    a=p.parse_args()
    if a.worker:worker(a.fixtures,a.output,a.wheel,a.model,a.environment)
    else:run(a.fixtures,a.output,a.wheel)
    return 0


if __name__=='__main__': raise SystemExit(main())
