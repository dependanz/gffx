"""Measure bounded multi-point public integration against frozen tile evidence."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
import torch
from tools import albedo_tiles as tiles
from tools import albedo_environment as base
from tools.material_loss_cost import check_deadline,supervise,write_report
from tools.corpus_registry import _read

PARENT_HASHES=('b385d80ee856e5fb859bb5560fe6f64595763d43a2ce6c0bb775fad08bfeab99',
 '4ffaeeb65410bb863776ae91e596db82b78687745d71ef411131aec80404df74',
 '0b2167f48d34d51d1fc751921654036e1d7ce7a72347377567e87417e8942660',
 '717f95987f7a8640d0d6b2b758dbc8c3bef66ad56cf4ce6c1baa45f2606c9ae0',
 'a4f123efd32219545a8aa242c6663a64df2eaf558864cc3c4660b4de71b21f64',
 '199349c72ee23d4857ae141442bc8e0724e62a3ea04aeabe51d111028d091758')
PROFILE=dict(name='albedo-multipoint-screen-v1',pixels=72,pixels_per_worker=18,
    warmups=1,repetitions=3,width=8,pairs=1024,maximum_directions=75497472,
    aggregate_seconds=900,cooperative_seconds=120,watchdog_seconds=180,host_bytes=1024**3,
    parity_atol=1e-12,parity_rtol=1e-11)
COMPLETE='passed sampling comparison; speed decision is profile-specific'

def groups(rows,width):
    if type(width) is not int or not 1<=width<=8:raise ValueError('width must be 1 through 8')
    if not isinstance(rows,list) or not 1<=len(rows)<=128:raise ValueError('1 through 128 rows required')
    by_order={}
    for i,row in enumerate(rows):by_order.setdefault(row['reference']['history'][-1]['n_mu'],[]).append(i)
    result=[]
    for order,indices in sorted(by_order.items()):
        if order not in (8,16,32,64,128):raise ValueError('unsupported order')
        for start in range(0,len(indices),width):
            ids=indices[start:start+width];result.append(dict(order=order,indices=ids,chunk=1024//len(ids)))
    return result

def sample(rows,*,model,environment,budget,deadline,width=8):
    from gffx.torch import material,light
    check_deadline(deadline);plan=groups(rows,width);b=base.bands(environment)
    for row in rows:base.validate_row(row,model,environment)
    started=time.perf_counter();estimates=[None]*len(rows);calls=0;directions=0;max_pairs=0;max_points=0
    with torch.no_grad():
        for group in plan:
            ids=group['indices'];size=len(ids);chunk=group['chunk']
            selected=[rows[i] for i in ids]
            n=torch.tensor([r['point']['normal'] for r in selected],dtype=torch.float64)
            v=torch.tensor([r['point']['view'] for r in selected],dtype=torch.float64)
            a=torch.tensor([r['albedo'] for r in selected],dtype=torch.float64)
            bases=[base.old.frame(r['point']['normal'],r['point']['view'])[1] for r in selected];values=[]
            for split in (.5,.25):
                q,w=base.old.quadrature(group['order'],split);total=torch.zeros((size,3),dtype=torch.float64)
                for start in range(0,len(q),chunk):
                    check_deadline(deadline)
                    d=torch.stack([q[start:start+chunk]@basis for basis in bases]);length=d.shape[1]
                    args=dict(albedo=a)
                    if model=='classic':
                        brdf=material.classic(n,v,d,specular=torch.tensor(.1,dtype=torch.float64),exponent=torch.tensor(2.,dtype=torch.float64),**args)
                    else:
                        brdf=material.principled(n,v,d,roughness=torch.tensor(1.,dtype=torch.float64),f0=torch.tensor(.04,dtype=torch.float64),budget=budget,**args)
                    packet=dict(measure='solid_angle',direction=d,radiance=base.old.field(d.reshape(-1,3),b).reshape(size,length,3),
                        valid=torch.ones((size,length),dtype=torch.bool),solid_angle=w[start:start+chunk][None].expand(size,-1))
                    total+=light.integrate(n,brdf,packet);calls+=1;directions+=size*length
                    max_pairs=max(max_pairs,size*length);max_points=max(max_points,size)
                values.append(total)
            for j,i in enumerate(ids):estimates[i]=[x[j].tolist() for x in values]
    check_deadline(deadline)
    result=dict(estimates=estimates,groups=plan,width=width,calls=calls,directions=directions,
        max_pairs=max_pairs,max_points=max_points,seconds=time.perf_counter()-started)
    validate(result,rows);return result

def validate(result,rows):
    plan=groups(rows,result['width']);calls=0;directions=0;max_pairs=0;max_points=0
    for group in plan:
        k=len(group['indices']);q=4*group['order']**2;chunk=group['chunk']
        calls+=2*math.ceil(q/chunk);directions+=2*k*q
        max_pairs=max(max_pairs,k*min(q,chunk));max_points=max(max_points,k)
    if (result['groups']!=plan or result['calls']!=calls or result['directions']!=directions
        or result['max_pairs']!=max_pairs or result['max_points']!=max_points
        or len(result['estimates'])!=len(rows) or not 0<result['seconds']<120):raise ValueError('sampling work identity')
    for values,row in zip(result['estimates'],rows):
        if len(values)!=2 or any(len(v)!=3 for v in values):raise ValueError('sampling RGB shape')
        for estimate,baseline in zip(values,row['actual']['estimates']):
            for actual,reference,previous in zip(estimate,row['reference']['radiance'],baseline):
                if (not math.isfinite(actual) or abs(actual-reference)>1e-10+1e-8*abs(reference)
                    or abs(actual-previous)>1e-12+1e-11*abs(previous)):raise ValueError('sampling numerical parity')

def sources():
    result=tiles.sources()
    if result['albedo_tiles.py']!='b72a9a675d1af128415ab30e7c13ec41809251ffe084cbcec9fadd0980b9bf9b':raise ValueError('frozen tile source changed')
    result[Path(__file__).name]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();return result

def load_rows(parent,corpus,model,environment):
    wanted=[k for k in tiles.schedule() if k[:2]==(model,environment)];selected=[]
    for batch in range(6):
        if any(k in wanted for k in tiles.batch_schedule(batch)):
            result=tiles.replay(Path(parent)/f'batch-{batch}',PARENT_HASHES[batch],corpus)
            for tile in result['tiles']:
                if tiles.identity(tile) in wanted:selected.extend((tile['rows'][0],tile['rows'][-1]))
    if len(selected)!=18:raise ValueError('selected regression ownership')
    return selected

def baseline(rows,model,environment,budget,deadline):
    started=time.perf_counter();values=[];calls=0;directions=0
    for r in rows:
        p=r['point'];a=base.candidate(p['normal'],p['view'],r['albedo'],model=model,environment=environment,budget=budget,ref=r['reference'],deadline=deadline)
        values.append(a['estimates']);calls+=a['calls'];directions+=a['directions']
    result=dict(estimates=values,groups=groups(rows,1),width=1,calls=calls,directions=directions,
        max_pairs=max(min(1024,4*r['reference']['history'][-1]['n_mu']**2) for r in rows),max_points=1,seconds=time.perf_counter()-started)
    validate(result,rows);return result

def worker(corpus,output,wheel,parent,model,environment):
    from gffx.torch.material import PrincipledBudget
    torch.set_num_threads(1);torch.set_num_interop_threads(1);deadline=time.perf_counter()+120
    report=dict(status='working',worker_pid=os.getpid(),profile=PROFILE,model=model,environment=environment,passes=[])
    try:
        if (model,environment) not in base.schedule():raise ValueError('worker profile')
        report.update(sources=sources(),parent_hashes=list(PARENT_HASHES),producer=base.old.fixed.producer_identity(wheel))
        if report['producer']['wheel']['sha256']!=base.old.fixed.WHEEL_HASH:raise ValueError('wheel identity')
        rows=load_rows(parent,corpus,model,environment);report['rows']=rows
        budget=PrincipledBudget.bundled(base.old.RESOURCE,device='cpu',dtype=torch.float64) if model=='principled' else None
        write_report(output,report)
        for repetition in range(4):
            for method in (('single','batched') if repetition%2==0 else ('batched','single')):
                check_deadline(deadline)
                started=time.perf_counter()
                result=baseline(rows,model,environment,budget,deadline) if method=='single' else sample(rows,model=model,environment=environment,budget=budget,deadline=deadline)
                report['passes'].append(dict(method=method,repetition=repetition,result=result,duration_seconds=time.perf_counter()-started));write_report(output,report)
        medians={m:statistics.median(p['duration_seconds'] for p in report['passes'] if p['method']==m and p['repetition']>0) for m in ('single','batched')}
        report.update(status=COMPLETE,medians=medians,prefer_batched=medians['batched']<medians['single'])
        check_deadline(deadline)
    except Exception as error:report.update(status='stopped',error=f'{type(error).__name__}: {error}');raise
    finally:write_report(output,report)

def read_worker(path,sha,expected_rows):
    blob=_read(Path(path),4*1024**2)
    if hashlib.sha256(blob).hexdigest()!=sha:raise ValueError('worker hash')
    r=json.loads(blob);s=r['supervisor']
    if (r['status']!=COMPLETE or r['profile']!=PROFILE or r['sources']!=sources() or r['parent_hashes']!=list(PARENT_HASHES)
        or r['producer']['wheel']['sha256']!=base.old.fixed.WHEEL_HASH or (r['model'],r['environment']) not in base.schedule()
        or len(r['rows'])!=18 or r['rows']!=expected_rows or s['exit_code']!=0 or s['failure'] is not None or s['monitored_pid']!=r['worker_pid']
        or not 0<s['elapsed_seconds']<180 or not 0<s['sampled_peak_working_set_bytes']<=1024**3):raise ValueError('unqualified sampling worker')
    for row in r['rows']:base.validate_row(row,r['model'],r['environment'])
    expected=[(m,i) for i in range(4) for m in (('single','batched') if i%2==0 else ('batched','single'))]
    if [(p['method'],p['repetition']) for p in r['passes']]!=expected:raise ValueError('benchmark ordering')
    for p in r['passes']:
        if p['result']['width']!=(1 if p['method']=='single' else 8) or not 0<p['duration_seconds']<120:raise ValueError('method width/time')
        validate(p['result'],r['rows'])
    medians={m:statistics.median(p['duration_seconds'] for p in r['passes'] if p['method']==m and p['repetition']>0) for m in ('single','batched')}
    if r['medians']!=medians or r['prefer_batched']!=(medians['batched']<medians['single']):raise ValueError('timing decision')
    return r

def replay(root,sha,parent,corpus):
    root=Path(root);blob=_read(root/'screen.json',65536)
    if hashlib.sha256(blob).hexdigest()!=sha:raise ValueError('screen hash')
    record=json.loads(blob)
    if (record['status']!=COMPLETE or record['profile']!=PROFILE or record['sources']!=sources()
        or len(record['workers'])!=4 or not 0<record['seconds']<=900):raise ValueError('unqualified screen')
    workers=[]
    for key,d in zip(base.schedule(),record['workers']):
        if d['path']!='-'.join(key)+'.json':raise ValueError('worker path')
        row=read_worker(root/d['path'],d['sha256'],load_rows(parent,corpus,*key))
        if (row['model'],row['environment'])!=key:raise ValueError('worker ownership')
        workers.append(row)
    if sum(p['result']['directions'] for w in workers for p in w['passes'])>PROFILE['maximum_directions']:raise ValueError('screen work limit')
    return dict(record=record,workers=workers,sha256=sha)

def run(corpus,output,parent,wheel):
    output=Path(output)
    if output.exists():raise FileExistsError('fresh sampling screen required')
    started=time.perf_counter();source=sources();output.mkdir(parents=True)
    record=dict(profile=PROFILE,status='working',sources=source,workers=[]);write_report(output/'screen.json',record)
    try:
        for model,environment in base.schedule():
            check_deadline(started+900-180)
            if sources()!=source:raise ValueError('source changed')
            path=output/f'{model}-{environment}.json'
            code=supervise(corpus,path,worker_module='tools.albedo_sampling',worker_args=('--wheel',str(wheel),'--parent',str(parent),'--model',model,'--environment',environment))
            sha=hashlib.sha256(path.read_bytes()).hexdigest();record['workers'].append(dict(path=path.name,sha256=sha));write_report(output/'screen.json',record)
            if code:raise RuntimeError('sampling screen stopped; no retry')
            read_worker(path,sha,load_rows(parent,corpus,model,environment))
        check_deadline(started+900);record['status']=COMPLETE
    except Exception as error:record.update(status='stopped',error=f'{type(error).__name__}: {error}');raise
    finally:record['seconds']=time.perf_counter()-started;write_report(output/'screen.json',record)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('fixtures','output','wheel','parent'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--worker',action='store_true');p.add_argument('--model');p.add_argument('--environment');a=p.parse_args()
    if a.worker:worker(a.fixtures,a.output,a.wheel,a.parent,a.model,a.environment)
    else:run(a.fixtures,a.output,a.parent,a.wheel)

if __name__=='__main__':main()
