"""Strict complete-profile assembly from immutable selected and new image tiles."""
import hashlib
import json
from pathlib import Path
import time
import torch
from tools import albedo_images as images
from tools import albedo_tiles as selected
from tools.corpus_registry import _read
from tools.material_loss_cost import check_deadline,write_report

fixed=images.base.old.fixed
PASSED='passed complete fixed textured image; not all profiles or M1'

def sources():
    result=images.sources()
    if result['albedo_images.py']!='4a86a0c4d3ca84fd3f4393bf3242c0d107d74b08efa17e5398f52fccc024e099':raise ValueError('frozen image source changed')
    result[Path(__file__).name]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return result

def descriptors(root,parent,hashes,key):
    if (not isinstance(hashes,list) or not 1<=len(hashes)<=98 or tuple(key) not in
        [(m,e,a) for m,e in images.base.schedule() for a in images.base.PROFILE['albedos']]):raise ValueError('explicit image profile/batch hashes required')
    found={}
    def batch(module,base,index,digest,origin):
        raw=_read(base/f'batch-{index}'/'batch.json',65536)
        if hashlib.sha256(raw).hexdigest()!=digest:raise ValueError('batch hash mismatch')
        record=json.loads(raw);keys=module.batch_schedule(index)
        if (record['batch']!=index or record['status']!=module.COMPLETE or record['profile']!=module.PROFILE
            or record['sources']!=module.sources() or not 0<record['elapsed_seconds']<=900
            or len(record['tiles'])!=6 or 'images' in record):raise ValueError('unqualified batch')
        for d,k in zip(record['tiles'],keys):
            if d['path']!=module.filename(k) or len(d['sha256'])!=64:raise ValueError('tile descriptor identity')
            if k[:3]==tuple(key):
                if k[3] in found:raise ValueError('duplicate tile ownership')
                found[k[3]]=dict(origin=origin,batch=index,**d)
    for i,digest in enumerate(images.sampling.assessed.PARENT_HASHES):
        if any(k[:3]==tuple(key) for k in selected.batch_schedule(i)):batch(selected,Path(parent),i,digest,'selected')
    for i,digest in enumerate(hashes):batch(images,Path(root),i,digest,'new')
    if set(found)!=set(range(52)):raise ValueError('all52 unique tiles required')
    return [found[i] for i in range(52)]

def records(descriptions,root,parent,buffers):
    for d in descriptions:
        module,base=(selected,Path(parent)) if d['origin']=='selected' else (images,Path(root))
        yield module.read_tile(base/f'batch-{d["batch"]}'/d['path'],d['sha256'],buffers)

def assemble(tiles,mask,key):
    if mask.shape!=(128,128) or mask.dtype!=torch.bool or int(mask.sum())!=6536:raise ValueError('image mask mismatch')
    ids=mask.flatten().nonzero().flatten().tolist();actual=torch.zeros((128,128,3),dtype=torch.float64)
    reference=torch.zeros_like(actual);seen=0;count=0
    totals=dict(reference_directions=0,candidate_directions=0,calls=0,moment_samples=0)
    for index,tile in enumerate(tiles):
        status=selected.PASSED if index in (0,26,51) else images.PASSED
        if index>=52 or images.identity(tile)!=(*key,index) or tile['status']!=status:raise ValueError('tile order/profile/status mismatch')
        start,end=fixed.tile_range(index)
        if len(tile['rows'])!=end-start:raise ValueError('tile coverage mismatch')
        values_by_pixel=tile['candidate']['estimates'] if 'candidate' in tile else [r['actual']['estimates'] for r in tile['rows']]
        if len(values_by_pixel)!=len(tile['rows']):raise ValueError('candidate pixel count')
        for offset,(row,estimates) in enumerate(zip(tile['rows'],values_by_pixel),start):
            i=row['point']['flat_index']
            if type(i) is not int or i!=ids[offset]:raise ValueError('pixel ownership mismatch')
            ref=torch.tensor(row['reference']['radiance'],dtype=torch.float64);values=torch.tensor(estimates,dtype=torch.float64)
            if (ref.shape!=(3,) or values.shape!=(2,3) or not bool(torch.isfinite(ref).all())
                or not bool(torch.isfinite(values).all()) or bool((ref<0).any())
                or bool(((values-ref).abs()>1e-10+1e-8*ref.abs()).any())):raise ValueError('pixel tolerance mismatch')
            actual.reshape(-1,3)[i]=values.mean(0);reference.reshape(-1,3)[i]=ref;seen+=1
        for name in totals:totals[name]+=tile['counts'][name]
        count+=1
    if count!=52 or seen!=6536:raise ValueError('complete tile set required')
    metrics=fixed.compare_images(actual,reference,mask)
    if not metrics['passed']:raise ValueError('image tolerance mismatch')
    return dict(actual=actual,reference=reference,metrics=metrics,counts=totals,pixels=seen)

def reconstruct(root,parent,corpus,hashes,key):
    desc=descriptors(root,parent,hashes,key)
    buffers=fixed.load_bundle(corpus,path=fixed.MANIFEST,sha256=fixed.MANIFEST_HASH)['buffers']
    return desc,assemble(records(desc,root,parent,buffers),buffers['covered'],key)

def run(root,parent,corpus,output,hashes,key):
    output=Path(output)
    if output.exists():raise FileExistsError('fresh assembly output required')
    started=time.perf_counter();output.mkdir(parents=True)
    report=dict(status='working',profile=list(key),sources=sources(),batch_hashes=hashes,
                parent_hashes=list(images.sampling.assessed.PARENT_HASHES),manifest_sha256=fixed.MANIFEST_HASH,wheel_sha256=fixed.WHEEL_HASH)
    try:
        desc,result=reconstruct(root,parent,corpus,hashes,key)
        report.update(tiles=desc,pixels=result['pixels'],counts=result['counts'],metrics=result['metrics'],arrays={})
        for name in ('actual','reference'):
            raw=fixed.tensor_bytes(result[name]);path=output/(name+'.bin')
            with path.open('xb') as stream:stream.write(raw)
            report['arrays'][name]=dict(path=path.name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        check_deadline(started+180);report['status']=PASSED
    except Exception as error:
        report.update(status='stopped',error=f'{type(error).__name__}: {error}');raise
    finally:
        report['assembly_seconds']=time.perf_counter()-started;write_report(output/'image.json',report)

def replay(output,digest,root,parent,corpus,*,blob=None):
    output=Path(output);blob=_read(output/'image.json',65536) if blob is None else blob
    if hashlib.sha256(blob).hexdigest()!=digest:raise ValueError('image report hash mismatch')
    report=json.loads(blob)
    if (report['status']!=PASSED or report['sources']!=sources() or report['parent_hashes']!=list(images.sampling.assessed.PARENT_HASHES)
        or report['manifest_sha256']!=fixed.MANIFEST_HASH or report['wheel_sha256']!=fixed.WHEEL_HASH
        or not 0<report['assembly_seconds']<180 or set(report['arrays'])!={'actual','reference'}):raise ValueError('unqualified image report')
    desc,result=reconstruct(root,parent,corpus,report['batch_hashes'],tuple(report['profile']))
    if report['tiles']!=desc or any(report[k]!=result[k] for k in ('pixels','counts','metrics')):raise ValueError('image reconstruction mismatch')
    for name in ('actual','reference'):
        d=report['arrays'][name]
        if d['path']!=name+'.bin' or d['bytes']!=393216:raise ValueError('image descriptor mismatch')
        raw=_read(output/d['path'],393216)
        if hashlib.sha256(raw).hexdigest()!=d['sha256'] or raw!=fixed.tensor_bytes(result[name]):raise ValueError('image bytes mismatch')
    return dict(record=report,sha256=digest,image=result)
