"""Finite saved affine-lighting comparisons; no image-wide shader execution."""
import argparse
import math
from pathlib import Path

import torch

from tools.affine_environment import BANDS, replay
from tools.environment_full_images import WHEEL_HASH
from .image_reference_demo import display_rgb


def environment_map():
    """Incoming-light diagram: world +Z at top, +X at horizontal centre."""
    theta=(torch.arange(192,dtype=torch.float64)+.5)*math.pi/192
    phi=(torch.arange(384,dtype=torch.float64)+.5)*2*math.pi/384-math.pi
    directions=torch.stack((theta.sin()[:,None]*phi.cos()[None],
                            theta.sin()[:,None]*phi.sin()[None],
                            theta.cos()[:,None].expand(-1,384)),dim=-1)
    return 1+directions@torch.tensor(BANDS,dtype=torch.float64).T


def publish(result,address,run):
    from gffx.dashboard import ArrayView,connect
    from .reporter import Reporter
    reporter=Reporter(address,run)
    def send(event,**values):
        if not reporter.send(event,**values):raise RuntimeError(reporter.error)
    send('start',kind='demo',provenance=dict(artifact_sha256=WHEEL_HASH))
    artifacts=[]
    with connect(address,run=run,wait_acks=True) as dash:
        dash.text('affine/scope',
            'Coloured light arrives more strongly from different WORLD directions. '
            'The panorama is an analytic diagram of incoming light, not a shaded head: '
            '+Z at top, +X in the horizontal centre, azimuth -pi to pi. Display exposure '
            'is one-half, followed by sRGB; numerical evidence stays linear and unscaled. '
            'Plots show eight predeclared head points per material, ordered from grazing '
            'to frontal view. Dots are separate surface locations, not animation frames. '
            'Each has two converged reference transitions and two complete public integrals. '
            'Finite saved replay only: not full-image, arbitrary HDR maps, gradients, GPU or M1 acceptance.',step=0)
        artifacts.append(dict(kind='text',path='affine/scope',label='What these diagrams do and do not prove'))
        pixels=display_rgb(environment_map()/2)
        dash.image('affine/incoming-light',ArrayView('uint8',(192,384,3),bytes(pixels.flatten().tolist())),step=0)
        artifacts.append(dict(kind='image',path='affine/incoming-light',label='Incoming coloured world light — diagram, half exposure'))
        for model in ('classic','principled'):
            path='affine/'+model
            for row in (x for x in result['rows'] if x['model']==model):
                index=row['index'];actual=row['actual'];ref=row['reference']
                for k,channel in enumerate(('red','green','blue')):
                    dash.scalar(path+'/reference-'+channel,ref['radiance'][k],step=index)
                    dash.scalar(path+'/public-'+channel,sum(e[k] for e in actual['estimates'])/2,step=index)
                ratio=max(e/t for error in actual['errors'] for e,t in zip(error,actual['limits']))
                dash.scalar('affine/error-ratio/'+model,ratio,step=index)
            artifacts.append(dict(kind='plot',path=path,label=model+' — reference and public linear RGB by point rank'))
        artifacts.append(dict(kind='plot',path='affine/error-ratio',label='Worst channel error / allowed error — must stay below 1'))
        details=[dict(model=row['model'],index=row['index'],point=row['point'],
                      reference=row['reference'],actual=row['actual'],supervisor=row['supervisor']) for row in result['rows']]
        dash.record('affine/evidence',dict(cohort_sha256=result['sha256'],profile=result['record']['profile'],
            elapsed_seconds=result['record']['elapsed_seconds'],sources=result['record']['sources'],points=details),step=0)
        artifacts.append(dict(kind='record',path='affine/evidence',label='Exact points, convergence, errors, work and memory'))
    send('demo',page='shading/affine-environment-demo',title='Coloured world-lighting point comparisons',artifacts=artifacts)
    send('finish',exit_code=0)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--sha256',required=True)
    p.add_argument('--corpus',type=Path,required=True);p.add_argument('--address',required=True);p.add_argument('--run',required=True)
    a=p.parse_args();publish(replay(a.root,a.sha256,a.corpus),a.address,a.run)


if __name__=='__main__':main()
