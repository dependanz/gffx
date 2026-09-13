"""Finite partial-image qualification, with explicit uncomputed-pixel display."""
import argparse
from pathlib import Path

import torch

from tools.affine_tiles import replay
from tools.environment_full_images import WHEEL_HASH


def coverage_map(tiles):
    """Green means measured here; gray means not computed in this qualification."""
    pixels=torch.full((128,128,3),96,dtype=torch.uint8);seen=set()
    for tile in tiles:
        for row in tile['rows']:
            index=row['flat_index']
            if type(index) is not int or not 0<=index<128*128 or index in seen:
                raise ValueError('invalid or duplicate pixel ownership')
            seen.add(index);pixels.reshape(-1,3)[index]=torch.tensor([40,170,100],dtype=torch.uint8)
    return pixels


def publish(result,address,run):
    from gffx.dashboard import ArrayView,connect
    from .reporter import Reporter
    reporter=Reporter(address,run)
    def send(event,**values):
        if not reporter.send(event,**values):raise RuntimeError(reporter.error)
    send('start',kind='demo',provenance=dict(artifact_sha256=WHEEL_HASH))
    artifacts=[]
    with connect(address,run=run,wait_acks=True) as dash:
        dash.text('affine-tiles/scope',
            'This is a test-cost and accuracy qualification, NOT a completed shaded image. '
            'Green: the 264 surface pixels checked per material. Gray: not computed here '
            '(including background). We selected tiles 0, 26 and 51 before execution. '
            'Each pixel has its own converged reference and TWO actual public lighting estimates. '
            'Only fixed material reference setup is reused within each worker. '
            'Timing is measured on this laptop; memory is sampled working set, not allocator peak. '
            'These six tiles do not prove a full-image runtime bound, HDR-map support, gradients or M1 acceptance.',step=0)
        artifacts.append(dict(kind='text',path='affine-tiles/scope',label='What this partial-image experiment means'))
        for model in ('classic','principled'):
            tiles=[x for x in result['tiles'] if x['model']==model]
            pixels=coverage_map(tiles);path='affine-tiles/'+model
            dash.image(path+'/coverage',ArrayView('uint8',(128,128,3),bytes(pixels.flatten().tolist())),step=0)
            artifacts.append(dict(kind='image',path=path+'/coverage',label=model+' — green: 264 tested pixels; gray: uncomputed'))
            for tile in tiles:
                index=tile['index']
                for key in ('setup_seconds','reference_seconds','candidate_seconds'):
                    dash.scalar(path+'/time/'+key,tile[key],step=index)
                dash.scalar(path+'/time/worker_seconds',tile['supervisor']['elapsed_seconds'],step=index)
                dash.scalar('affine-tiles/memory/'+model,tile['supervisor']['sampled_peak_working_set_bytes']/1024**2,step=index)
                ratio=max(e/t for p in tile['rows'] for error in p['actual']['errors'] for e,t in zip(error,p['actual']['limits']))
                dash.scalar('affine-tiles/error-ratio/'+model,ratio,step=index)
            artifacts.append(dict(kind='plot',path=path+'/time',label=model+' — setup, reference, public sampling and total worker seconds'))
        artifacts.append(dict(kind='plot',path='affine-tiles/memory',label='Sampled worker memory in MiB by tile index'))
        artifacts.append(dict(kind='plot',path='affine-tiles/error-ratio',label='Worst channel error / allowed error — must stay below 1'))
        tiles=[{k:row[k] for k in ('model','index','counts','setup_seconds','reference_seconds','candidate_seconds','supervisor')} for row in result['tiles']]
        dash.record('affine-tiles/evidence',dict(cohort=result['record'],sha256=result['sha256'],tiles=tiles),step=0)
        artifacts.append(dict(kind='record',path='affine-tiles/evidence',label='Exact tile identities, scope and measured costs'))
    send('demo',page='shading/affine-tiles-demo',title='Coloured lighting: selected image tiles',artifacts=artifacts)
    send('finish',exit_code=0)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--sha256',required=True)
    p.add_argument('--corpus',type=Path,required=True);p.add_argument('--address',required=True);p.add_argument('--run',required=True)
    a=p.parse_args();publish(replay(a.root,a.sha256,a.corpus),a.address,a.run)


if __name__=='__main__':main()
