"""Finite verified image checkpoints; uncomputed pixels are visibly gray."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from tools import affine_images as full
from tools.corpus_registry import _read
from .image_reference_demo import display_rgb


def load(root,parent,corpus):
    raw=_read(root/'cohort.json',65536);record=json.loads(raw)
    digest=hashlib.sha256(raw).hexdigest()
    if record['status']==full.COMPLETE:
        result=full.replay(root,digest,parent,corpus)
        for row in result['images'].values():row['known']=torch.ones((128,128),dtype=torch.bool)
        return result
    if record['status'] not in ('working','stopped') or record['profile']!=full.PROFILE or record['sources']!=full.sources():
        raise ValueError('unqualified progress identity')
    verified=full.parent(parent,corpus)
    bundle=full.fixed.load_bundle(corpus,path=full.fixed.MANIFEST,sha256=full.fixed.MANIFEST_HASH)
    images={m:dict(actual=torch.zeros((128,128,3),dtype=torch.float64),reference=torch.zeros((128,128,3),dtype=torch.float64),
                   known=torch.zeros((128,128),dtype=torch.bool),pixels=0) for m in ('classic','principled')}
    def add(tile):
        row=images[tile['model']]
        for pixel in tile['rows']:
            i=pixel['flat_index']
            if row['known'].flatten()[i]:raise ValueError('duplicate progress pixel')
            row['known'].flatten()[i]=True;row['pixels']+=1
            row['reference'].reshape(-1,3)[i]=torch.tensor(pixel['reference']['radiance'],dtype=torch.float64)
            row['actual'].reshape(-1,3)[i]=torch.tensor(pixel['actual']['estimates'],dtype=torch.float64).mean(0)
    for tile in verified['tiles']:add(tile)
    for index,d in enumerate(record['batches']):
        if d['path']!=f'batch-{index:02d}.json':raise ValueError('batch path/order mismatch')
        raw=_read(root/d['path'],65536)
        if hashlib.sha256(raw).hexdigest()!=d['sha256']:raise ValueError('batch identity mismatch')
        batch=json.loads(raw)
        if batch['status']=='stopped':break
        full.validate_batch(batch,index)
        for (model,i),td in zip(full.batches()[index],batch['tiles']):
            tile=full.read_tile(root/td['path'],td['sha256'],bundle['buffers'])
            if tile['model']!=model or tile['index']!=i:raise ValueError('tile ownership mismatch')
            add(tile)
    return dict(record=record,sha256=digest,images=images)


def publish(result,address,run):
    from gffx.dashboard import ArrayView,connect
    from .reporter import Reporter
    reporter=Reporter(address,run)
    def send(event,**values):
        if not reporter.send(event,**values):raise RuntimeError(reporter.error)
    send('start',kind='demo',provenance=dict(artifact_sha256=full.fixed.WHEEL_HASH));artifacts=[]
    record=result['record'];complete=record['status']==full.COMPLETE
    with connect(address,run=run,wait_acks=True) as dash:
        dash.text('affine-images/scope',
            f"Status: {record['status']}. "+
            'Each material needs 6536 checked surface pixels. Gray means not computed in this snapshot; '
            'completed images have exact black background. Reference, actual public result and error/tolerance '
            'are separate. The room light varies with world direction; material colour and roughness are fixed. '
            'Finite saved evidence, not a live renderer, arbitrary HDR-map support, CUDA or M1 acceptance.',step=0)
        artifacts.append(dict(kind='text',path='affine-images/scope',label='Verified checkpoint coverage and limitations'))
        for model,row in result['images'].items():
            error=((row['actual']-row['reference']).abs()/(1e-10+1e-8*row['reference'].abs())).clamp(0,1)
            for name,pixels in (('reference',display_rgb(row['reference'])),('public',display_rgb(row['actual'])),('error-ratio',(255*error).round().byte())):
                pixels=pixels.clone();pixels[~row['known']]=96
                path=f'affine-images/{model}/{name}'
                dash.image(path,ArrayView('uint8',(128,128,3),bytes(pixels.flatten().tolist())),step=0)
                artifacts.append(dict(kind='image',path=path,label=f"{model}: {name}, {row['pixels']}/6536 pixels"))
        dash.record('affine-images/evidence',dict(sha256=result['sha256'],cohort=record,
            complete=complete,pixels={m:r['pixels'] for m,r in result['images'].items()}),step=0)
        artifacts.append(dict(kind='record',path='affine-images/evidence',label='Checkpoint identities, progress and measured errors'))
    send('demo',page='shading/affine-images-demo',title='Coloured environment image evidence',artifacts=artifacts)
    send('finish',exit_code=0)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--corpus',type=Path,required=True);p.add_argument('--address',required=True);p.add_argument('--run',required=True)
    a=p.parse_args();publish(load(a.root,a.parent,a.corpus),a.address,a.run)


if __name__=='__main__':main()
