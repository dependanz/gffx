"""Bounded, checkpointed completion of the fixed affine world-light image pair."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import torch

from tools import affine_tiles as tiles
from tools.corpus_registry import _read
from tools.material_loss_cost import check_deadline,supervise,write_report

fixed=tiles.fixed
PARENT_HASH='0b368869260d70b57a2d8dce7cd5b94bf918360700ee9004d401dd5fecefadb2'
TILE_SOURCE='5dc6f5e2a1d07913821738f7156186f1331fa4f883fa27634337f5d0e7717547'
PROFILE=dict(name='affine-complete-images-v1',parent_sha256=PARENT_HASH,
    batch_seconds=900,tiles_per_batch=6,new_tiles=98,new_pixels=12544,
    maximum_new_samples=3835854848,point_profile=tiles.old.PROFILE)
COMPLETE='passed two fixed affine images; not arbitrary environment or M1 acceptance'


def remaining():return [(m,i) for m in ('classic','principled') for i in range(52) if (m,i) not in tiles.schedule()]


def batches():
    todo=remaining();return [todo[i:i+6] for i in range(0,len(todo),6)]


def launch_guard(started):check_deadline(started+900-180)


def sources():
    result=tiles.sources()
    if result['affine_tiles.py']!=TILE_SOURCE:raise ValueError('frozen tile source changed')
    result[Path(__file__).name]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return result


def parent(root,corpus):return tiles.replay(root,PARENT_HASH,corpus)


def worker(root,output,wheel,model,index):
    from gffx.torch.material import PrincipledBudget
    torch.set_num_threads(1);torch.set_num_interop_threads(1);deadline=time.perf_counter()+120
    report=dict(status='working',worker_pid=os.getpid(),profile=PROFILE,model=model,index=index,rows=[])
    try:
        if (model,index) not in remaining():raise ValueError('unqualified tile')
        report.update(sources=sources(),producer=fixed.producer_identity(wheel),manifest_sha256=fixed.MANIFEST_HASH)
        if report['producer']['wheel']['sha256']!=fixed.WHEEL_HASH:raise ValueError('wheel identity')
        bundle=fixed.load_bundle(root,path=fixed.MANIFEST,sha256=fixed.MANIFEST_HASH)
        points=tiles.inputs(bundle['buffers'],index)
        started=time.perf_counter();budget=None;curve=None
        if model=='principled':
            budget=PrincipledBudget.bundled(tiles.RESOURCE,device='cpu',dtype=torch.float64)
            curve=tiles.PreparedCurve(tiles.LossCurve(budget.to_envelope()['payload']),roughness=1.)
        prepared=tiles.Prepared(model,curve,deadline=deadline)
        report.update(setup_seconds=time.perf_counter()-started,setup_moment_samples=prepared.moment_samples,
                      reference_seconds=0.,candidate_seconds=0.)
        write_report(output,report)
        for flat_index,normal,view in points:
            check_deadline(deadline);started=time.perf_counter()
            ref=prepared.point(normal,view,deadline=deadline)
            report['reference_seconds']+=time.perf_counter()-started
            actual=tiles.old.candidate(normal,view,model=model,budget=budget,ref=ref,deadline=deadline)
            report['candidate_seconds']+=actual['elapsed_seconds']
            pixel=dict(flat_index=flat_index,reference=ref,actual=actual)
            tiles.validate_pixel(pixel,flat_index,normal,view,model)
            report['rows'].append(pixel)
            if len(report['rows'])%16==0:write_report(output,report)
        report['counts']=tiles.counts(report['rows'],prepared.moment_samples)
        check_deadline(deadline);report['status']='passed affine tile; not a complete image'
    except Exception as error:
        report.update(status='stopped',error=f'{type(error).__name__}: {error}')
        if isinstance(error,tiles.old.scalar.ReferenceConvergenceError):report['unresolved']=error.report
        raise
    finally:write_report(output,report)


def read_tile(path,digest,buffers):
    blob=_read(Path(path),tiles.PROFILE['max_tile_bytes'])
    if hashlib.sha256(blob).hexdigest()!=digest:raise ValueError('tile identity mismatch')
    row=json.loads(blob);s=row['supervisor']
    if (row['status']!='passed affine tile; not a complete image' or row['profile']!=PROFILE
        or row['sources']!=sources() or row['manifest_sha256']!=fixed.MANIFEST_HASH
        or row['producer']['wheel']['sha256']!=fixed.WHEEL_HASH or (row['model'],row['index']) not in remaining()
        or s['failure'] is not None or s['exit_code']!=0 or s['monitored_pid']!=row['worker_pid']
        or not 0<s['elapsed_seconds']<180 or not 0<s['sampled_peak_working_set_bytes']<=1024**3):raise ValueError('unqualified tile')
    points=tiles.inputs(buffers,row['index'])
    if len(row['rows'])!=len(points):raise ValueError('incomplete tile')
    for pixel,(index,n,v) in zip(row['rows'],points):tiles.validate_pixel(pixel,index,n,v,row['model'])
    samples=0 if row['model']=='classic' else row['rows'][0]['reference']['moments']['evaluations']
    if (row['setup_moment_samples']!=samples or row['counts']!=tiles.counts(row['rows'],samples)
        or any(p['reference']['moments']!=row['rows'][0]['reference']['moments'] for p in row['rows'])
        or any(not math.isfinite(row[k]) or not 0<=row[k]<120 for k in ('setup_seconds','reference_seconds','candidate_seconds'))
        or row['candidate_seconds']!=sum(p['actual']['elapsed_seconds'] for p in row['rows'])):raise ValueError('tile setup/work mismatch')
    return row


def validate_batch(row,index):
    if (type(index) is not int or not 0<=index<len(batches()) or row['index']!=index
        or row['status']!='passed' or row['profile']!=PROFILE or row['sources']!=sources()
        or not 0<row['elapsed_seconds']<=900):raise ValueError('unqualified batch')
    expected=[f'{m}-{i:02d}.json' for m,i in batches()[index]]
    if [x['path'] for x in row['tiles']]!=expected:raise ValueError('batch ownership mismatch')
    if any(len(x['sha256'])!=64 for x in row['tiles']):raise ValueError('invalid tile identity')


def assemble(records,mask,model):
    if mask.shape!=(128,128) or mask.dtype!=torch.bool or int(mask.sum())!=6536:raise ValueError('image mask mismatch')
    ids=mask.flatten().nonzero().flatten().tolist();actual=torch.zeros((128,128,3),dtype=torch.float64)
    expected=torch.zeros_like(actual);seen=0;count=0
    totals=dict(reference_directions=0,candidate_directions=0,calls=0,moment_samples=0)
    for index,row in enumerate(records):
        if index>=52 or row['index']!=index or row['model']!=model or row['status']!='passed affine tile; not a complete image':
            raise ValueError('tile order/model/status mismatch')
        start,end=tiles.tile_range(index)
        if len(row['rows'])!=end-start:raise ValueError('tile coverage mismatch')
        for offset,pixel in enumerate(row['rows'],start):
            if pixel['flat_index']!=ids[offset]:raise ValueError('pixel coverage mismatch')
            ref=torch.tensor(pixel['reference']['radiance'],dtype=torch.float64)
            values=torch.tensor(pixel['actual']['estimates'],dtype=torch.float64)
            if values.shape!=(2,3) or not bool(torch.isfinite(values).all()) or not bool(torch.isfinite(ref).all()) or bool(((values-ref).abs()>1e-10+1e-8*ref.abs()).any()):raise ValueError('pixel tolerance mismatch')
            actual.reshape(-1,3)[ids[offset]]=values.mean(0);expected.reshape(-1,3)[ids[offset]]=ref;seen+=1
        for k in totals:totals[k]+=row['counts'][k]
        count+=1
    if count!=52 or seen!=6536:raise ValueError('complete tile set required')
    metrics=fixed.compare_images(actual,expected,mask)
    if not metrics['passed']:raise ValueError('image tolerance mismatch')
    return dict(actual=actual,reference=expected,metrics=metrics,counts=totals,pixels=seen)


def all_descriptors(parent_result,batch_records):
    result={(r['model'],r['index']):('parent',d) for r,d in zip(parent_result['tiles'],parent_result['record']['tiles'])}
    for batch in batch_records:
        for key,d in zip(batches()[batch['index']],batch['tiles']):
            if key in result:raise ValueError('duplicate tile')
            result[key]=('new',d)
    if len(result)!=104:raise ValueError('incomplete image ownership')
    return result


def records_for(model,descriptors,parent_root,root,buffers):
    for i in range(52):
        origin,d=descriptors[(model,i)]
        read,base=(tiles.read_tile,parent_root) if origin=='parent' else (read_tile,root)
        row=read(Path(base)/d['path'],d['sha256'],buffers)
        if row['model']!=model or row['index']!=i:raise ValueError('descriptor ownership mismatch')
        yield row


def run(parent_root,output,corpus,wheel):
    output=Path(output)
    if output.exists():raise FileExistsError('fresh full-image output required')
    verified=parent(parent_root,corpus);source=sources()
    bundle=fixed.load_bundle(corpus,path=fixed.MANIFEST,sha256=fixed.MANIFEST_HASH)
    started=time.perf_counter();output.mkdir(parents=True)
    record=dict(status='working',profile=PROFILE,sources=source,parent_sha256=PARENT_HASH,
                original_seconds=verified['record']['elapsed_seconds'],batches=[],images={})
    write_report(output/'cohort.json',record);batch_records=[]
    try:
        for batch_index,group in enumerate(batches()):
            # Previous completed batch records are immutable; recheck all hashes before continuation.
            for d in record['batches']:
                if hashlib.sha256((output/d['path']).read_bytes()).hexdigest()!=d['sha256']:raise ValueError('completed batch changed')
            if hashlib.sha256((Path(parent_root)/'cohort.json').read_bytes()).hexdigest()!=PARENT_HASH:raise ValueError('parent changed')
            batch_start=time.perf_counter();batch=dict(status='working',index=batch_index,profile=PROFILE,sources=source,tiles=[])
            path=output/f'batch-{batch_index:02d}.json';write_report(path,batch)
            try:
                for model,index in group:
                    launch_guard(batch_start)
                    if sources()!=source:raise ValueError('source changed')
                    tile_path=output/f'{model}-{index:02d}.json'
                    code=supervise(corpus,tile_path,worker_module='tools.affine_images',worker_args=('--wheel',str(wheel),'--model',model,'--index',str(index)))
                    d=dict(path=tile_path.name,sha256=hashlib.sha256(tile_path.read_bytes()).hexdigest())
                    batch['tiles'].append(d);write_report(path,batch)
                    if code:raise RuntimeError('worker failed; preserve checkpoint and diagnose, no automatic retry')
                    read_tile(tile_path,d['sha256'],bundle['buffers'])
                check_deadline(batch_start+900);batch['status']='passed'
            except Exception as error:
                batch.update(status='stopped',error=f'{type(error).__name__}: {error}');raise
            finally:
                batch['elapsed_seconds']=time.perf_counter()-batch_start;write_report(path,batch)
                record['batches'].append(dict(path=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
                record['additional_seconds']=time.perf_counter()-started
                record['cumulative_seconds']=record['original_seconds']+record['additional_seconds']
                write_report(output/'cohort.json',record)
            validate_batch(batch,batch_index);batch_records.append(batch)
            print(json.dumps(dict(batch=batch_index,status='passed',completed_new_tiles=sum(len(x['tiles']) for x in batch_records),additional_seconds=record['additional_seconds'])),flush=True)
        descriptors=all_descriptors(verified,batch_records);assembly_deadline=time.perf_counter()+180
        for model in ('classic','principled'):
            result=assemble(records_for(model,descriptors,parent_root,output,bundle['buffers']),bundle['buffers']['covered'],model)
            image=dict(metrics=result['metrics'],counts=result['counts'],pixels=result['pixels'],arrays={})
            for key in ('actual','reference'):
                blob=fixed.tensor_bytes(result[key]);path=output/f'{model}-{key}.bin'
                with path.open('xb') as stream:stream.write(blob)
                image['arrays'][key]=dict(path=path.name,sha256=hashlib.sha256(blob).hexdigest(),bytes=len(blob))
            record['images'][model]=image;write_report(output/'cohort.json',record)
            check_deadline(assembly_deadline)
        record['status']=COMPLETE
    except Exception as error:
        record.update(status='stopped',error=f'{type(error).__name__}: {error}');raise
    finally:
        record['additional_seconds']=time.perf_counter()-started
        record['cumulative_seconds']=record['original_seconds']+record['additional_seconds'];write_report(output/'cohort.json',record)


def replay(root,digest,parent_root,corpus):
    root=Path(root);blob=_read(root/'cohort.json',65536)
    if hashlib.sha256(blob).hexdigest()!=digest:raise ValueError('cohort identity mismatch')
    record=json.loads(blob);verified=parent(parent_root,corpus)
    if (record['status']!=COMPLETE or record['profile']!=PROFILE or record['sources']!=sources()
        or record['parent_sha256']!=PARENT_HASH or len(record['batches'])!=17 or set(record['images'])!={'classic','principled'}
        or record['original_seconds']!=verified['record']['elapsed_seconds']
        or not 0<record['additional_seconds']<17*900+180
        or record['cumulative_seconds']!=record['original_seconds']+record['additional_seconds']):raise ValueError('unqualified full cohort')
    rows=[]
    for index,d in enumerate(record['batches']):
        if d['path']!=f'batch-{index:02d}.json':raise ValueError('batch path/order mismatch')
        raw=_read(root/d['path'],65536)
        if hashlib.sha256(raw).hexdigest()!=d['sha256']:raise ValueError('batch identity mismatch')
        row=json.loads(raw);validate_batch(row,index);rows.append(row)
    if sum(x['elapsed_seconds'] for x in rows)>record['additional_seconds']:raise ValueError('batch elapsed accounting mismatch')
    bundle=fixed.load_bundle(corpus,path=fixed.MANIFEST,sha256=fixed.MANIFEST_HASH)
    descriptors=all_descriptors(verified,rows);images={}
    for model in ('classic','principled'):
        result=assemble(records_for(model,descriptors,parent_root,root,bundle['buffers']),bundle['buffers']['covered'],model)
        stored=record['images'][model]
        if any(result[k]!=stored[k] for k in ('metrics','counts','pixels')):raise ValueError('saved image metrics mismatch')
        for key in ('actual','reference'):
            d=stored['arrays'][key]
            if d['path']!=f'{model}-{key}.bin' or d['bytes']!=393216:raise ValueError('image descriptor mismatch')
            raw=_read(root/d['path'],393216)
            if hashlib.sha256(raw).hexdigest()!=d['sha256'] or raw!=fixed.tensor_bytes(result[key]):raise ValueError('saved image identity/reconstruction mismatch')
        images[model]=result
    return dict(record=record,sha256=digest,images=images)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixtures',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--wheel',type=Path,required=True);p.add_argument('--parent',type=Path)
    p.add_argument('--worker',action='store_true');p.add_argument('--model');p.add_argument('--index',type=int)
    a=p.parse_args()
    if a.worker:worker(a.fixtures,a.output,a.wheel,a.model,a.index)
    else:run(a.parent,a.output,a.fixtures,a.wheel)


if __name__=='__main__':main()
