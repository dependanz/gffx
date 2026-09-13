"""One approved 900-second continuation of the frozen constant-environment cohort.

Only principled tiles32..51 execute. Parent reports and classic arrays remain
in their original directory; a separate record owns the additional budget.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import torch

from tools import environment_full_images as original
from tools.corpus_registry import _read, _safe
from tools.material_loss_cost import check_deadline, supervise, write_report
from tools.testing_dashboard import environment_full_demo as saved

PARENT_HASH='4c037730a7bd34fea4f8cc886685a5bd543cf86fe427f8e6f0be5c8f30bb1897'
FROZEN_SOURCES={
    'environment_full_images.py':'6d8b0054b6848bb70bda3a9f53126c7bf759f4e14971f589f1c7e45280cb9798',
    'environment_candidate_point.py':'a2478a05f31d94a35823a1422484240e516373be7fb0706aa2707d4ab58ee8df',
    'constant_environment_reference.py':'897d4533d4ed41c8bba39939fc46a6f8e2eaf150d00044c2e26e44a31b1822cc',
    'principled_direct_images.py':'5a38f5b98d855248364db64e30f517a9544a5a91307fa9c4a2afba1aa391cffd',
    'shading_independent_evaluator.py':'8e1dd0a473034264e79617930b433e1d0ef33aeab98da32154215271034d641c',
    'shading_environment_reference.py':'ca0d0b9e0a34f64a37deacfe27c0cceda2b37e5a44060a007128a86a976814ce',
}
PROFILE=dict(name='constant-environment-continuation-v1',parent_sha256=PARENT_HASH,
    additional_seconds=900,first_tile=32,last_tile=51,additional_pixels=2440,
    maximum_additional_evaluations=372910080,worker_watchdog_seconds=180)
COMPLETE='passed continued constant-environment images; not general environment or M1 acceptance'


def remaining(): return list(range(32,52))


def launch_guard(started): check_deadline(started+900-180)


def check_sources():
    if original.source_identities()!=FROZEN_SOURCES:raise ValueError('frozen image source identity changed')


def validate_parent_layout(row):
    if (row['status']!='stopped' or row['profile']!=original.PROFILE
        or row['error']!='TimeoutError: cooperative measurement deadline reached'
        or not math.isfinite(row['elapsed_seconds']) or not 0<row['elapsed_seconds']<=1800
        or set(row['images'])!={'classic'} or set(row['tiles'])!={'classic','principled'}):
        raise ValueError('unqualified stopped parent')
    for model,count in (('classic',52),('principled',32)):
        descriptors=row['tiles'][model]
        if len(descriptors)!=count:raise ValueError('incomplete parent prefix')
        for index,descriptor in enumerate(descriptors):
            if descriptor['path']!=f'{model}-{index:02d}.json':raise ValueError('parent tile ownership/path mismatch')


def parent_manifest(root):
    raw=_read(Path(root)/'cohort.json',65536)
    if hashlib.sha256(raw).hexdigest()!=PARENT_HASH:raise ValueError('parent identity mismatch')
    result=json.loads(raw);validate_parent_layout(result)
    return result


def checked_tile(root,descriptor,index,layout):
    if descriptor['path']!=f'principled-{index:02d}.json':raise ValueError('continuation tile path/order')
    row=original.read_tile(_safe(Path(root).resolve(),descriptor['path']),descriptor['sha256'])
    start,end=original.tile_range(index);_,ids,cosines=layout
    if (row['model']!='principled' or row['index']!=index or row['start']!=start
        or row['end']!=end or len(row['rows'])!=end-start or row['sources']!=FROZEN_SOURCES):
        raise ValueError('continuation tile ownership/source')
    for offset,pixel in enumerate(row['rows'],start):
        if pixel['flat_index']!=ids[offset] or pixel['cosine']!=cosines[offset]:
            raise ValueError('pixel geometry identity mismatch')
        ref,value=pixel['reference'],pixel['actual'];saved.validate_reference(ref)
        order=ref['history'][-1]['n_mu']
        if (value['directions']!=4*order*order or value['calls']!=2*math.ceil(2*order*order/1024)
            or value['chunk_size']!=1024):raise ValueError('pixel work identity mismatch')
        estimates=value['estimates'];rgb=ref['radiance']
        if len(estimates)!=2 or any(len(e)!=3 for e in estimates):raise ValueError('pixel RGB shape')
        if any(not math.isfinite(x) or abs(x-y)>1e-10+1e-8*abs(y)
               for estimate in estimates for x,y in zip(estimate,rgb)):
            raise ValueError('saved pixel tolerance failed')
    return row


def validate_parent(root,buffers):
    check_sources();record=parent_manifest(root);layout=original.pixel_layout(buffers)
    verified=saved.load(root,PARENT_HASH,layout[0],require_complete=False)
    prefix=[checked_tile(root,d,i,layout) for i,d in enumerate(record['tiles']['principled'])]
    return dict(record=record,classic=verified['images']['classic'],prefix=prefix,layout=layout)


def run(parent,output,corpus,wheel):
    output=Path(output)
    if output.exists():raise FileExistsError('fresh continuation output required')
    started=time.perf_counter()
    record=dict(profile=PROFILE,status='working',parent_sha256=PARENT_HASH,
                sources=FROZEN_SOURCES,tiles=[],images={})
    output.mkdir(parents=True)
    write_report(output/'continuation.json',record)
    try:
        bundle=original.load_bundle(Path(corpus),path=original.MANIFEST,sha256=original.MANIFEST_HASH)
        verified=validate_parent(parent,bundle['buffers'])
        producer=original.producer_identity(wheel)
        if producer['wheel']['sha256']!=original.WHEEL_HASH:raise ValueError('continuation wheel identity')
        record.update(producer=producer,original_elapsed_seconds=verified['record']['elapsed_seconds'],
            runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            reused_tiles={k:len(v) for k,v in verified['record']['tiles'].items()})
        write_report(output/'continuation.json',record)
        rows=verified['prefix']
        for index in remaining():
            launch_guard(started);check_sources()
            # Recheck the parent identity before each launch, without regenerating it.
            parent_manifest(parent)
            path=output/f'principled-{index:02d}.json'
            code=supervise(Path(corpus),path,worker_module='tools.environment_full_images',
                worker_args=('--wheel',str(wheel),'--model','principled','--index',str(index)))
            descriptor=dict(path=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            record['tiles'].append(descriptor)
            write_report(output/'continuation.json',record)
            if code:raise RuntimeError('continuation worker failed; no retry')
            rows.append(checked_tile(output,descriptor,index,verified['layout']))
        result=original.assemble(rows,verified['layout'][0])
        image=dict(metrics=result['metrics'],counts=result['counts'],arrays={})
        for name in ('actual','reference'):
            blob=original.tensor_bytes(result[name]);path=output/f'principled-{name}.bin'
            with path.open('xb') as handle:handle.write(blob)
            image['arrays'][name]=dict(path=path.name,sha256=hashlib.sha256(blob).hexdigest(),bytes=len(blob))
        record['images']['principled']=image
        check_deadline(started+900)
        parent_manifest(parent)
        record['status']=COMPLETE
    except Exception as error:
        record.update(status='stopped',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        record['additional_elapsed_seconds']=time.perf_counter()-started
        record['cumulative_elapsed_seconds']=(record['original_elapsed_seconds']+record['additional_elapsed_seconds']
            if 'original_elapsed_seconds' in record else None)
        write_report(output/'continuation.json',record)


def replay(root,digest,parent,buffers):
    root=Path(root).resolve();blob=_read(root/'continuation.json',65536)
    if hashlib.sha256(blob).hexdigest()!=digest:raise ValueError('continuation identity mismatch')
    record=json.loads(blob)
    if (record['status']!=COMPLETE or record['profile']!=PROFILE or record['parent_sha256']!=PARENT_HASH
        or record['sources']!=FROZEN_SOURCES or record['producer']['wheel']['sha256']!=original.WHEEL_HASH
        or record['reused_tiles']!={'classic':52,'principled':32}
        or not 0<record['additional_elapsed_seconds']<=900 or len(record['tiles'])!=20
        or set(record['images'])!={'principled'}):raise ValueError('unqualified continuation')
    verified=validate_parent(parent,buffers)
    if (record['original_elapsed_seconds']!=verified['record']['elapsed_seconds']
        or record['cumulative_elapsed_seconds']!=record['original_elapsed_seconds']+record['additional_elapsed_seconds']):
        raise ValueError('continuation elapsed provenance mismatch')
    rows=verified['prefix']+[checked_tile(root,d,i,verified['layout']) for i,d in zip(remaining(),record['tiles'])]
    result=original.assemble(rows,verified['layout'][0]);image=record['images']['principled']
    if result['metrics']!=image['metrics'] or result['counts']!=image['counts']:raise ValueError('continued image metrics mismatch')
    for name in ('actual','reference'):
        descriptor=image['arrays'][name]
        if descriptor['path']!=f'principled-{name}.bin' or descriptor['bytes']!=393216:
            raise ValueError('continued array descriptor mismatch')
        payload=_read(_safe(root,descriptor['path']),393216)
        if len(payload)!=393216 or hashlib.sha256(payload).hexdigest()!=descriptor['sha256']:
            raise ValueError('continued image hash mismatch')
        if not torch.equal(original.tensor_from_bytes(payload,[128,128,3],'float64'),result[name]):
            raise ValueError('continued image reconstruction mismatch')
    result['supervision']=dict(workers=52,maximum_worker_seconds=max(r['supervisor']['elapsed_seconds'] for r in rows),
        maximum_sampled_worker_bytes=max(r['supervisor']['sampled_peak_working_set_bytes'] for r in rows),all_exit_zero=True)
    return dict(record=record,sha256=digest,parent=verified['record'],
                images={'classic':verified['classic'],'principled':result})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--corpus',type=Path,required=True);p.add_argument('--wheel',type=Path,required=True)
    a=p.parse_args();run(a.parent,a.output,a.corpus,a.wheel)


if __name__=='__main__':main()
