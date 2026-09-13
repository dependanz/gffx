"""Create-only selected textured-image tile batches; never full-image acceptance."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
import torch
from tools import albedo_environment as base
from tools.material_loss_cost import check_deadline,supervise,write_report
from tools.corpus_registry import _read

BASE_HASH='5a189d9037beac4eb72688b92092b16e507a6219de82f216f6db45df6ea731d7'
PROFILE=dict(name='albedo-selected-tiles-v1',point_profile=base.PROFILE,selected_tiles=[0,26,51],
    tiles=36,batches=6,tiles_per_batch=6,pixels=3168,pixels_per_profile=264,
    maximum_samples=968933376,aggregate_seconds=900,total_seconds=5400,max_tile_bytes=4*1024**2)
PASSED='passed selected albedo tile; not complete image'
COMPLETE='passed six selected albedo tiles; not complete images'

def schedule():
    return [(m,e,a,i) for m,e in base.schedule() for a in base.PROFILE['albedos'] for i in (0,26,51)]

def batch_schedule(batch):
    if type(batch) is not int or not 0<=batch<6:raise ValueError('invalid batch')
    return schedule()[batch*6:(batch+1)*6]

def identity(row):return (row['model'],row['environment'],row['albedo_name'],row['index'])

def filename(key):return '-'.join(str(x) for x in key)+'.json'

def sources():
    result=base.sources()
    if result['albedo_environment.py']!=BASE_HASH:raise ValueError('frozen albedo source changed')
    result[Path(__file__).name]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return result

def inputs(buffers,name,index):
    if name not in base.PROFILE['albedos'] or type(index) is not int or index not in (0,26,51):raise ValueError('unqualified field/tile')
    from tools.principled_direct_images import validate_buffers
    validate_buffers(buffers,name);fixed=base.old.fixed
    _,ids,_=fixed.pixel_layout(buffers);start,end=fixed.tile_range(index)
    return [dict(albedo_name=name,albedo=list(base.colour(buffers['albedo/'+name].reshape(-1,3)[i].tolist())),
                 point=dict(flat_index=i,normal=buffers['normal'].reshape(-1,3)[i].tolist(),
                            view=buffers['view'].reshape(-1,3)[i].tolist())) for i in ids[start:end]]

def counts(rows,moments):
    return dict(reference_directions=sum(r['reference']['evaluations'] for r in rows),
        candidate_directions=sum(r['actual']['directions'] for r in rows),
        calls=sum(r['actual']['calls'] for r in rows),moment_samples=moments)

def worker(root,output,wheel,model,environment,albedo,index):
    from gffx.torch.material import PrincipledBudget
    torch.set_num_threads(1);torch.set_num_interop_threads(1);deadline=time.perf_counter()+120
    report=dict(status='working',worker_pid=os.getpid(),profile=PROFILE,model=model,
        environment=environment,albedo_name=albedo,index=index,rows=[])
    try:
        if identity(report) not in schedule():raise ValueError('tile identity')
        report.update(sources=sources(),producer=base.old.fixed.producer_identity(wheel),manifest_sha256=base.old.fixed.MANIFEST_HASH)
        if report['producer']['wheel']['sha256']!=base.old.fixed.WHEEL_HASH:raise ValueError('wheel identity')
        buffers=base.old.fixed.load_bundle(root,path=base.old.fixed.MANIFEST,sha256=base.old.fixed.MANIFEST_HASH)['buffers']
        points=inputs(buffers,albedo,index);started=time.perf_counter()
        budget=PrincipledBudget.bundled(base.old.RESOURCE,device='cpu',dtype=torch.float64) if model=='principled' else None
        curve=base.old.PreparedCurve(base.old.LossCurve(budget.to_envelope()['payload']),roughness=1.) if budget is not None else None
        prepared=base.Prepared(model,curve,environment,deadline=deadline)
        report.update(setup_seconds=time.perf_counter()-started,moment_samples=prepared.moment_samples,reference_seconds=0.,candidate_seconds=0.)
        write_report(output,report)
        for entry in points:
            check_deadline(deadline);p=entry['point'];a=entry['albedo'];started=time.perf_counter()
            ref=prepared.point(p['normal'],p['view'],a,deadline=deadline)
            report['reference_seconds']+=time.perf_counter()-started
            actual=base.candidate(p['normal'],p['view'],a,model=model,environment=environment,budget=budget,ref=ref,deadline=deadline)
            report['candidate_seconds']+=actual['elapsed_seconds']
            row=dict(entry,reference=ref,actual=actual);base.validate_row(row,model,environment);report['rows'].append(row)
        report['counts']=counts(report['rows'],prepared.moment_samples)
        check_deadline(deadline);report['status']=PASSED
    except Exception as error:
        report.update(status='stopped',error=f'{type(error).__name__}: {error}')
        if isinstance(error,base.old.scalar.ReferenceConvergenceError):report['unresolved']=error.report
        raise
    finally:write_report(output,report)

def read_tile(path,digest,buffers,*,blob=None):
    blob=_read(Path(path),PROFILE['max_tile_bytes']) if blob is None else blob
    if len(blob)>PROFILE['max_tile_bytes'] or hashlib.sha256(blob).hexdigest()!=digest:raise ValueError('tile hash/size mismatch')
    row=json.loads(blob);s=row['supervisor']
    if (row['status']!=PASSED or row['profile']!=PROFILE or row['sources']!=sources()
        or row['manifest_sha256']!=base.old.fixed.MANIFEST_HASH or row['producer']['wheel']['sha256']!=base.old.fixed.WHEEL_HASH
        or identity(row) not in schedule() or s['failure'] is not None or s['exit_code']!=0
        or s['monitored_pid']!=row['worker_pid'] or not 0<s['elapsed_seconds']<180
        or not 0<s['sampled_peak_working_set_bytes']<=1024**3):raise ValueError('unqualified tile')
    wanted=inputs(buffers,row['albedo_name'],row['index'])
    if len(row['rows'])!=len(wanted):raise ValueError('incomplete tile')
    radial=row['rows'][0]['reference']['moments'];samples=0 if radial is None else radial['evaluations']
    for entry,expected in zip(row['rows'],wanted):
        if {k:entry[k] for k in expected}!=expected:raise ValueError('tile pixel identity mismatch')
        base.validate_row(entry,row['model'],row['environment'])
        if entry['reference']['moments']!=radial:raise ValueError('moment identity mismatch')
    if (row['moment_samples']!=samples or row['counts']!=counts(row['rows'],samples)
        or any(not math.isfinite(row[k]) or not 0<=row[k]<120 for k in ('setup_seconds','reference_seconds','candidate_seconds'))
        or row['candidate_seconds']!=sum(r['actual']['elapsed_seconds'] for r in row['rows'])):raise ValueError('tile work mismatch')
    return row

def replay(root,digest,corpus):
    root=Path(root);blob=_read(root/'batch.json',65536)
    if hashlib.sha256(blob).hexdigest()!=digest:raise ValueError('batch hash mismatch')
    record=json.loads(blob);keys=batch_schedule(record['batch'])
    if (record['profile']!=PROFILE or record['status']!=COMPLETE or record['sources']!=sources()
        or not 0<record['elapsed_seconds']<=900 or len(record['tiles'])!=6 or 'images' in record):raise ValueError('unqualified batch')
    buffers=base.old.fixed.load_bundle(corpus,path=base.old.fixed.MANIFEST,sha256=base.old.fixed.MANIFEST_HASH)['buffers'];tiles=[]
    for descriptor,key in zip(record['tiles'],keys):
        if descriptor['path']!=filename(key):raise ValueError('tile path/order mismatch')
        row=read_tile(root/descriptor['path'],descriptor['sha256'],buffers)
        if identity(row)!=key:raise ValueError('tile ownership mismatch')
        tiles.append(row)
    return dict(record=record,sha256=digest,tiles=tiles)

def combine(results):
    """Aggregate the six strictly replayed batches; never claim complete images."""
    if len(results)!=6 or [r['record']['batch'] for r in results]!=list(range(6)):raise ValueError('all six unique ordered batches required')
    tiles=[]
    for batch,result in enumerate(results):
        record=result['record']
        if (record['profile']!=PROFILE or record['sources']!=sources() or record['status']!=COMPLETE
            or not 0<record['elapsed_seconds']<=900 or [identity(t) for t in result['tiles']]!=batch_schedule(batch)):
            raise ValueError('unqualified combined batch')
        tiles.extend(result['tiles'])
    samples=sum(t['counts'][k] for t in tiles for k in ('reference_directions','candidate_directions','moment_samples'))
    if samples>PROFILE['maximum_samples'] or sum(len(t['rows']) for t in tiles)!=3168:raise ValueError('combined work mismatch')
    return dict(tiles=tiles,batches=[dict(batch=r['record']['batch'],sha256=r['sha256']) for r in results],
        elapsed_seconds=sum(r['record']['elapsed_seconds'] for r in results),samples=samples)

def launch_guard(started):check_deadline(started+900-180)

def run(root,output,wheel,batch):
    output=Path(output)
    if output.exists():raise FileExistsError('fresh tile batch directory required')
    keys=batch_schedule(batch);source=sources();started=time.perf_counter();output.mkdir(parents=True)
    record=dict(profile=PROFILE,status='working',batch=batch,sources=source,tiles=[]);write_report(output/'batch.json',record)
    try:
        buffers=base.old.fixed.load_bundle(root,path=base.old.fixed.MANIFEST,sha256=base.old.fixed.MANIFEST_HASH)['buffers']
        for key in keys:
            launch_guard(started)
            if sources()!=source:raise ValueError('source changed during batch')
            model,environment,albedo,index=key;path=output/filename(key)
            code=supervise(root,path,worker_module='tools.albedo_tiles',worker_args=('--wheel',str(wheel),
                '--model',model,'--environment',environment,'--albedo',albedo,'--index',str(index)))
            digest=hashlib.sha256(path.read_bytes()).hexdigest();record['tiles'].append(dict(path=path.name,sha256=digest))
            write_report(output/'batch.json',record)
            if code:raise RuntimeError('tile batch stopped; no retry')
            read_tile(path,digest,buffers)
        check_deadline(started+900);record['status']=COMPLETE
    except Exception as error:
        record.update(status='stopped',error=f'{type(error).__name__}: {error}');raise
    finally:
        record['elapsed_seconds']=time.perf_counter()-started;write_report(output/'batch.json',record)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixtures',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--wheel',type=Path,required=True);p.add_argument('--batch',type=int);p.add_argument('--worker',action='store_true')
    p.add_argument('--model');p.add_argument('--environment');p.add_argument('--albedo');p.add_argument('--index',type=int)
    a=p.parse_args()
    if a.worker:worker(a.fixtures,a.output,a.wheel,a.model,a.environment,a.albedo,a.index)
    else:run(a.fixtures,a.output,a.wheel,a.batch)

if __name__=='__main__':main()
