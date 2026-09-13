"""Strict finite replay of constant-environment image evidence; no rendering."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from tools.environment_full_images import (
    PROFILE,WHEEL_HASH,MANIFEST,MANIFEST_HASH,read_tile,assemble,load_bundle,tensor_from_bytes)
from tools.corpus_registry import _read,_safe
from .image_reference_demo import display_rgb


def same(a,b):
    return math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,rel_tol=1e-14,abs_tol=1e-25)


def validate_reference(ref):
    """Recompute the finite rule from stored estimates, not success booleans."""
    try:
        if ref['status']!='converged' or not 3<=len(ref['history'])<=5:
            raise ValueError('incomplete reference history')
        if not isinstance(ref['radiance'],list) or len(ref['radiance'])!=3:
            raise ValueError('reference RGB shape')
        previous=None;streak=0;evaluations=0
        for i,row in enumerate(ref['history']):
            order=(8,16,32,64,128)[i];evaluations+=4*order*order
            for key in ('radiance','threshold','offset_change'):
                if not isinstance(row[key],list) or len(row[key])!=3:raise ValueError('reference RGB history shape')
            if row['change'] is not None and (not isinstance(row['change'],list) or len(row['change'])!=3):
                raise ValueError('reference change shape')
            estimates=row['estimates']
            if len(estimates)!=2 or any(len(e)!=3 for e in estimates):raise ValueError('reference shape')
            if any(not math.isfinite(x) or x<0 for e in estimates for x in e):raise ValueError('reference finite domain')
            mean=[(a+b)/2 for a,b in zip(*estimates)]
            limits=[1e-11+1e-9*abs(x) for x in mean]
            delta=[abs(a-b) for a,b in zip(*estimates)]
            change=None if previous is None else [max(abs(estimates[j][k]-previous[j][k]) for j in range(2)) for k in range(3)]
            if previous is not None:
                streak=streak+1 if all(a<=t and b<=t for a,b,t in zip(change,delta,limits)) else 0
            if (row['n_mu']!=order or row['evaluations']!=evaluations or row['streak']!=streak
                or (streak==2 and i!=len(ref['history'])-1)
                or not all(same(a,b) for a,b in zip(mean,row['radiance']))
                or not all(same(a,b) for a,b in zip(limits,row['threshold']))
                or not all(same(a,b) for a,b in zip(delta,row['offset_change']))):raise ValueError('reference history mismatch')
            if (change is None and row['change'] is not None) or (change is not None and (
                row['change'] is None or not all(same(a,b) for a,b in zip(change,row['change'])))):
                raise ValueError('reference transition mismatch')
            previous=estimates
        if streak!=2 or ref['evaluations']!=evaluations or not all(same(a,b) for a,b in zip(mean,ref['radiance'])):
            raise ValueError('reference final mismatch')
    except (KeyError,TypeError,IndexError,OverflowError) as error:
        raise ValueError('malformed reference evidence') from error


def load(root,digest,mask,*,require_complete=True):
    root=Path(root).resolve();blob=_read(root/'cohort.json',65536)
    if hashlib.sha256(blob).hexdigest()!=digest:raise ValueError('cohort identity mismatch')
    cohort=json.loads(blob)
    complete='passed two constant-environment images; not general environment or M1 acceptance'
    if cohort['profile']!=PROFILE or (require_complete and cohort['status']!=complete):
        raise ValueError('complete qualified cohort required')
    if cohort['status'] not in (complete,'stopped') or not 0<cohort['elapsed_seconds']<=1800:
        raise ValueError('terminal bounded cohort required')
    if (not set(cohort['images']).issubset(PROFILE['materials'])
        or (require_complete and set(cohort['images'])!=set(PROFILE['materials']))):
        raise ValueError('complete known image set required')
    result=dict(cohort=cohort,sha256=digest,images={})
    for model,image in cohort['images'].items():
        descriptors=cohort['tiles'][model]
        if len(descriptors)!=52:raise ValueError('complete image tiles required')
        rows=[]
        for i,descriptor in enumerate(descriptors):
            if descriptor['path']!=f'{model}-{i:02d}.json':raise ValueError('tile path identity')
            row=read_tile(_safe(root,descriptor['path']),descriptor['sha256'])
            if row['model']!=model:raise ValueError('mixed model tile')
            for pixel in row['rows']:
                ref,value=pixel['reference'],pixel['actual'];validate_reference(ref)
                order=ref['history'][-1]['n_mu']
                if (not math.isfinite(pixel['cosine']) or not 0<pixel['cosine']<=1
                    or value['directions']!=4*order*order or value['calls']!=2*math.ceil(2*order*order/1024)
                    or value['chunk_size']!=1024):raise ValueError('pixel work mismatch')
            rows.append(row)
        rebuilt=assemble(rows,mask)
        rebuilt['supervision']=dict(workers=len(rows),
            maximum_worker_seconds=max(r['supervisor']['elapsed_seconds'] for r in rows),
            maximum_sampled_worker_bytes=max(r['supervisor']['sampled_peak_working_set_bytes'] for r in rows),
            all_exit_zero=all(r['supervisor']['exit_code']==0 for r in rows))
        if rebuilt['metrics']!=image['metrics'] or rebuilt['counts']!=image['counts']:
            raise ValueError('image measurements mismatch')
        for name in ('actual','reference'):
            description=image['arrays'][name]
            if description['path']!=f'{model}-{name}.bin' or description['bytes']!=393216:
                raise ValueError('image descriptor mismatch')
            payload=_read(_safe(root,description['path']),393216)
            if len(payload)!=393216 or hashlib.sha256(payload).hexdigest()!=description['sha256']:
                raise ValueError('image payload identity')
            actual=tensor_from_bytes(payload,[128,128,3],'float64')
            if not torch.equal(actual,rebuilt[name]):raise ValueError('assembled image changed')
        result['images'][model]=rebuilt
    return result


def publish(address,run,result):
    from gffx.dashboard import ArrayView,connect
    from .reporter import Reporter
    reporter=Reporter(address,run)
    def send(event,**values):
        if not reporter.send(event,**values):raise RuntimeError(reporter.error)
    send('start',kind='demo',provenance=dict(artifact_sha256=WHEEL_HASH))
    artifacts=[];cohort=result['cohort']
    with connect(address,run=run,wait_acks=True) as dash:
        message=(f"Cohort status: {cohort['status']}. Complete images shown: {', '.join(result['images']) or 'none'}. "
                 "Each displayed image has 6536 independently checked visible pixels. Reference, actual public result, "
                 "and error divided by the unchanged tolerance are shown separately. Linear float64 evidence is "
                 "unchanged by display-only sRGB/clipping. Uniform unit lighting and constant colour only; "
                 "not arbitrary environment maps, new raster accuracy, GPU, image gradients or M1 completion. "
                 "Finite saved replay, no live rendering. "+cohort.get('error',''))
        dash.text('full-environment/scope',message,step=0)
        artifacts.append(dict(kind='text',path='full-environment/scope',label='Completed evidence and remaining scope'))
        for model,row in result['images'].items():
            error=(row['actual']-row['reference']).abs()/(1e-10+1e-8*row['reference'].abs())
            for name,pixels in (('reference',display_rgb(row['reference'])),('public',display_rgb(row['actual'])),
                                ('error-over-tolerance',(255*error.clamp(0,1)).round().byte())):
                path=f'full-environment/{model}/{name}'
                dash.image(path,ArrayView('uint8',(128,128,3),bytes(pixels.flatten().tolist())),step=0)
                artifacts.append(dict(kind='image',path=path,label=f'{model}: {name}'))
        dash.record('full-environment/evidence',dict(cohort_sha256=result['sha256'],status=cohort['status'],
            elapsed_seconds=cohort['elapsed_seconds'],images=cohort['images'],
            completed_tiles={k:len(v) for k,v in cohort['tiles'].items()},wheel_sha256=WHEEL_HASH,
            manifest_sha256=MANIFEST_HASH,
            execution_segments=cohort.get('segments'),
            supervision={model:row['supervision'] for model,row in result['images'].items()}),step=0)
        artifacts.append(dict(kind='record',path='full-environment/evidence',label='Pixel errors, work and immutable identities'))
    send('demo',page='shading/environment-full-demo',title='Full constant-environment image evidence',artifacts=artifacts)
    send('finish',exit_code=0)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--sha256',required=True)
    p.add_argument('--corpus',type=Path,required=True);p.add_argument('--address',required=True);p.add_argument('--run',required=True)
    a=p.parse_args();bundle=load_bundle(a.corpus,path=MANIFEST,sha256=MANIFEST_HASH)
    result=load(a.root,a.sha256,bundle['buffers']['covered'],require_complete=False)
    publish(a.address,a.run,result)


if __name__=='__main__':main()
