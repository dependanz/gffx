"""Finite, supervised whole-image source-texture adjoints with strict tile ownership."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
import torch
from tools import environment_image_schedule as schedule
from tools import environment_image_adjoint as method
from tools import environment_loss as old
from tools.shading_cuda_closeout import identity

PASSED='passed complete fixed environment image adjoints; not all shading or M1'
TILE_PASSED='passed bounded image adjoint tile; not complete image'


def sources():
    result=schedule.sources()
    result[Path(__file__).name]=old.direct.digest(Path(__file__))
    return result


def read_schedule(path, expected):
    raw=Path(path).read_bytes()
    if len(raw)>8*1024**2 or hashlib.sha256(raw).hexdigest()!=expected: raise ValueError('schedule hash/size')
    row=json.loads(raw); sup=row['supervisor']
    if (row['status']!=schedule.PASSED or row['sources']!=schedule.sources()
            or row['progress_sha256']!=schedule.PROGRESS_HASH or row['manifest_sha256']!=old.direct.MANIFEST_HASH
            or (row['model'],row['environment']) not in old.schedule()
            or len(row['points'])!=6536 or row['groups']!=schedule.partition([p['order'] for p in row['points']])
            or sup['failure'] is not None or sup['exit_code']!=0 or sup['monitored_pid']!=row['worker_pid']
            or not 0<sup['sampled_peak_working_set_bytes']<=1024**3
            or not 0<sup['elapsed_seconds']<=180 or not 0<row['elapsed_seconds']<=120):
        raise ValueError('unqualified image schedule')
    return row


def launch_guard(elapsed, allowance):
    if not math.isfinite(elapsed) or elapsed<0 or allowance!=6000. or elapsed>allowance-180:
        raise TimeoutError('image aggregate watchdog reserve exhausted')


def combine(parts, total):
    if type(total) is not int or not 1<=total<=6536: raise ValueError('bounded image required')
    end=0; values=[]; losses=[]; gradient=torch.zeros((32,32,3),dtype=torch.float64)
    for part in parts:
        start,stop=part['start'],part['end']
        if type(start) is not int or type(stop) is not int or start!=end or not start<stop<=total:
            raise ValueError('gap, overlap or wrong tile range')
        rgb=torch.tensor(part['values'],dtype=torch.float64); grad=torch.tensor(part['gradient'],dtype=torch.float64)
        if (rgb.shape!=(stop-start,3) or grad.shape!=(32,32,3)
                or not bool(torch.isfinite(rgb).all()) or not bool(torch.isfinite(grad).all())
                or not math.isfinite(part['loss']) or part['loss']<0): raise ValueError('nonfinite or malformed tile')
        weight=(stop-start)/total
        values.append(rgb); gradient+=weight*grad; losses.append(weight*part['loss']);end=stop
    if end!=total: raise ValueError('incomplete image')
    return dict(values=torch.cat(values),gradient=gradient,loss=math.fsum(losses))


def inputs(points):
    weights=[old.ref.stencil(32,32,p['uv']) for p in points]
    data=dict(intercepts=[p['intercept'] for p in points],slopes=[p['slopes'] for p in points],target=[p['target'] for p in points])
    return weights,data


def evaluate_tile(plan, group, tex, deadline):
    from gffx.torch.material import PrincipledBudget
    points=plan['points'][group['start']:group['end']]; count=len(points)
    weights,data=inputs(points)
    budget=PrincipledBudget.bundled(old.direct.RESOURCE,device='cpu',dtype=torch.float64)
    offsets=[]; counts=dict(calls=0,directions=0,colour_backwards=0,texture_backwards=0)
    for split in (.5,.25):
        values=[]; losses=[]; gradient=torch.zeros_like(tex)
        for p in points:
            fields=[torch.tensor([p[k]],dtype=torch.float64) for k in ('normal','view','uv','target')]
            result=method.angular_pass(tex,*fields,model=plan['model'],environment=plan['environment'],
                budget=budget,orders=[p['order']],split=split,deadline=deadline)
            for key in counts: counts[key]+=result[key]
            values.append(result['values'][0].tolist()); gradient+=result['gradient']/count; losses.append(result['loss']/count)
        loss=math.fsum(losses)
        checked=old.inspect(tex.tolist(),weights,data,values,gradient.tolist(),loss)
        offsets.append(dict(start=group['start'],end=group['end'],split=split,values=values,
            gradient=gradient.tolist(),loss=loss,**checked))
    wanted=dict(calls=group['calls'],directions=group['directions'],
                colour_backwards=3*group['calls'],texture_backwards=2*count)
    if counts!=wanted: raise ValueError('tile work count')
    return dict(offsets=offsets,counts=counts)


def worker(path, output, expected, wheel, texture, texture_hash, index):
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    start=time.perf_counter()
    row=dict(status='working',worker_pid=os.getpid(),sources=sources(),schedule_sha256=expected,
             texture_sha256=texture_hash,index=index)
    try:
        plan=read_schedule(path,expected)
        if type(index) is not int or not 0<=index<len(plan['groups']): raise ValueError('unknown group')
        row['producer']=identity(wheel,old.direct.WHEEL_HASH)
        raw=texture.read_bytes()
        if len(raw)>256*1024 or hashlib.sha256(raw).hexdigest()!=texture_hash: raise ValueError('texture hash/size')
        tex=torch.tensor(json.loads(raw)['texture'],dtype=torch.float64)
        row['result']=evaluate_tile(plan,plan['groups'][index],tex,start+120)
        old.check_deadline(start+120)
        row.update(status=TILE_PASSED,elapsed_seconds=time.perf_counter()-start)
    except Exception as error:
        row.update(status='stopped',error=type(error).__name__+': '+str(error));old.write_report(output,row);raise
    old.write_report(output,row)


def replay_tile(path, expected, plan, index, texture_hash, tex):
    if path.stat().st_size>2*1024**2 or old.direct.digest(path)!=expected: raise ValueError('tile hash/size')
    row=json.loads(path.read_text());sup=row['supervisor'];group=plan['groups'][index]
    if (row['status']!=TILE_PASSED or row['sources']!=sources() or row['index']!=index
            or row['texture_sha256']!=texture_hash or row['producer']['sha256']!=old.direct.WHEEL_HASH
            or sup['failure'] is not None or sup['exit_code']!=0 or sup['monitored_pid']!=row['worker_pid']
            or not 0<sup['sampled_peak_working_set_bytes']<=1024**3 or not 0<sup['elapsed_seconds']<=180
            or not 0<row['elapsed_seconds']<=120): raise ValueError('unqualified tile worker')
    count=group['end']-group['start'];result=row['result']
    if result['counts']!=dict(calls=group['calls'],directions=group['directions'],colour_backwards=3*group['calls'],texture_backwards=2*count):
        raise ValueError('recorded work mismatch')
    if len(result['offsets'])!=2: raise ValueError('both offsets required')
    weights,data=inputs(plan['points'][group['start']:group['end']])
    for split,r in zip((.5,.25),result['offsets']):
        if (r['split'],r['start'],r['end'])!=(split,group['start'],group['end']): raise ValueError('offset ownership')
        checked=old.inspect(tex.tolist(),weights,data,r['values'],r['gradient'],r['loss'])
        if any(r[k]!=v for k,v in checked.items()): raise ValueError('tile metrics changed')
    return row


def run(path, output, expected, wheel):
    torch.set_num_threads(1)
    plan=read_schedule(path,expected);source=sources();start=time.perf_counter()
    output.mkdir(parents=True,exist_ok=False)
    cohort=dict(status='working',sources=source,schedule_sha256=expected,model=plan['model'],
                environment=plan['environment'],workers=[],passes=[],aggregate_seconds=6000,
                declared_counts=plan['counts'],started_unix=time.time())
    old.write_report(output/'cohort.json',cohort)
    tex=old.direct.texture();weights,data=inputs(plan['points'])
    try:
        for iteration in range(2):
            texture=output/f'texture-{iteration}.json';old.write_report(texture,dict(texture=tex.tolist()))
            texture_hash=old.direct.digest(texture);tiles=[]
            for index,group in enumerate(plan['groups']):
                launch_guard(time.perf_counter()-start,6000.)
                if sources()!=source: raise ValueError('source changed during cohort')
                worker_path=output/f'iteration-{iteration}-tile-{index}.json'
                code=old.supervise(path,worker_path,worker_module='tools.environment_image_runner',
                    worker_args=('--schedule-hash',expected,'--wheel',str(wheel),'--texture',str(texture),
                                 '--texture-hash',texture_hash,'--index',str(index)))
                digest=old.direct.digest(worker_path)
                cohort['workers'].append(dict(path=worker_path.name,sha256=digest,iteration=iteration,index=index))
                cohort['elapsed_seconds']=time.perf_counter()-start
                old.write_report(output/'cohort.json',cohort)
                if code: raise RuntimeError('image worker stopped; no automatic retry')
                tile=replay_tile(worker_path,digest,plan,index,texture_hash,tex)
                if tile['schedule_sha256']!=expected: raise ValueError('tile schedule identity')
                tiles.append(tile)
                old.write_report(output/f'checkpoint-{len(cohort["workers"]):03}.json',dict(
                    status='validated partial image',schedule_sha256=expected,sources=source,
                    workers=cohort['workers'],elapsed_seconds=time.perf_counter()-start))
            offsets=[];average=torch.zeros_like(tex)
            for offset in range(2):
                combined=combine([t['result']['offsets'][offset] for t in tiles],6536)
                checked=old.inspect(tex.tolist(),weights,data,combined['values'].tolist(),combined['gradient'].tolist(),combined['loss'])
                offsets.append(dict(split=(.5,.25)[offset],values=combined['values'].tolist(),
                    gradient=combined['gradient'].tolist(),loss=combined['loss'],**checked))
                average+=combined['gradient']/2
            cohort['passes'].append(dict(texture_sha256=texture_hash,offsets=offsets))
            if iteration==0:
                reference=old.ref.evaluate(tex.tolist(),weights,data['intercepts'],data['slopes'],data['target'])
                fd=old.ref.finite_differences(tex.tolist(),weights,data['intercepts'],data['slopes'],data['target'],reference['gradient'])
                if not all(r['passed'] for r in fd): raise ValueError('whole-image independent finite differences')
                cohort['finite_differences']=fd
                tex=(tex-5*average).contiguous()
                if not bool(((tex>=0)&(tex<=1)).all()): raise ValueError('whole-image update domain')
            old.write_report(output/'cohort.json',cohort)
        if any(b['loss']>=a['loss'] for a,b in zip(cohort['passes'][0]['offsets'],cohort['passes'][1]['offsets'])):
            raise ValueError('whole-image loss did not decrease')
        old.check_deadline(start+6000)
        cohort.update(status=PASSED,elapsed_seconds=time.perf_counter()-start)
    except Exception as error:
        cohort.update(status='stopped',error=type(error).__name__+': '+str(error),elapsed_seconds=time.perf_counter()-start)
        old.write_report(output/'cohort.json',cohort);raise
    old.write_report(output/'cohort.json',cohort)
    return cohort


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('fixtures','output','wheel'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--schedule-hash',required=True);p.add_argument('--worker',action='store_true')
    p.add_argument('--texture',type=Path);p.add_argument('--texture-hash');p.add_argument('--index',type=int)
    a=p.parse_args()
    if a.worker:worker(a.fixtures,a.output,a.schedule_hash,a.wheel,a.texture,a.texture_hash,a.index)
    else:run(a.fixtures,a.output,a.schedule_hash,a.wheel)
    return 0


if __name__=='__main__':raise SystemExit(main())
