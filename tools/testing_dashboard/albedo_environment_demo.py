"""Finite saved surface-colour comparisons; no live shader loop."""
import argparse
from pathlib import Path
import torch
from tools.albedo_environment import replay
from tools.environment_full_images import WHEEL_HASH
from .image_reference_demo import display_rgb


def swatches(worker):
    """Nine rows: albedo/reference/public for vertex, gradient, then checker."""
    rows=worker['rows']
    if len(rows)!=24:raise ValueError('24 ordered comparisons required')
    blocks=[]
    for offset in (0,8,16):
        group=rows[offset:offset+8]
        a=torch.tensor([r['albedo'] for r in group],dtype=torch.float64)
        ref=torch.tensor([r['reference']['radiance'] for r in group],dtype=torch.float64)
        actual=torch.tensor([r['actual']['estimates'] for r in group],dtype=torch.float64).mean(1)
        blocks.append(torch.stack((a,ref,actual)))
    values=torch.cat(blocks)
    if not bool(torch.isfinite(values).all()) or bool((values<0).any()):raise ValueError('invalid swatch values')
    return display_rgb(values/2).repeat_interleave(24,0).repeat_interleave(48,1)


def publish(result,address,run):
    from gffx.dashboard import ArrayView,connect
    from .reporter import Reporter
    reporter=Reporter(address,run)
    def send(event,**values):
        if not reporter.send(event,**values):raise RuntimeError(reporter.error)
    send('start',kind='demo',provenance=dict(artifact_sha256=WHEEL_HASH))
    artifacts=[]
    with connect(address,run=run,wait_acks=True) as dash:
        dash.text('albedo/scope',
            '96 MEASURED SURFACE SAMPLES, not full rendered images. Each image has three groups '
            'from top to bottom: vertex colour, gradient texture, checker texture. Each group '
            'has three rows: INPUT ALBEDO, INDEPENDENT REFERENCE, PUBLIC GFFX OUTPUT. '
            'Eight columns are fixed pixel locations ordered from grazing to frontal view. '
            'All rows use half exposure then sRGB for display; measurements remain linear and unscaled. '
            'Reference and public rows should look identical; the error plots test differences '
            'too small to see. A ratio below one passes. Dark diffuse colour may still reflect '
            'a glossy highlight. Uniform lighting and coloured world lighting are separate cases. '
            'This finite saved replay does not prove full images, image gradients, arbitrary HDR maps or M1 completion.',step=0)
        artifacts.append(dict(kind='text',path='albedo/scope',label='How to read the colour comparisons'))
        for worker in result['workers']:
            name=worker['model']+'-'+worker['environment'];path='albedo/'+name
            pixels=swatches(worker)
            dash.image(path,ArrayView('uint8',tuple(pixels.shape),bytes(pixels.flatten().tolist())),step=0)
            artifacts.append(dict(kind='image',path=path,label=name+' — vertex / gradient / checker; each input / reference / public'))
            for index,row in enumerate(worker['rows']):
                actual=row['actual'];ratio=max(e/t for error in actual['errors'] for e,t in zip(error,actual['limits']))
                dash.scalar('albedo/error-ratio/'+name,ratio,step=index)
        artifacts.append(dict(kind='plot',path='albedo/error-ratio',label='Worst RGB error / allowed error — each point must stay below 1'))
        dash.record('albedo/evidence',dict(cohort_sha256=result['sha256'],profile=result['record']['profile'],
            elapsed_seconds=result['record']['elapsed_seconds'],samples=result['samples'],
            sources=result['record']['sources'],workers=result['workers']),step=0)
        artifacts.append(dict(kind='record',path='albedo/evidence',label='Exact colours, pixel IDs, convergence, work and memory'))
    send('demo',page='shading/albedo-environment-demo',title='Surface colours under environment lighting',artifacts=artifacts)
    send('finish',exit_code=0)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--sha256',required=True)
    p.add_argument('--corpus',type=Path,required=True);p.add_argument('--address',required=True);p.add_argument('--run',required=True)
    a=p.parse_args();publish(replay(a.root,a.sha256,a.corpus),a.address,a.run)


if __name__=='__main__':main()
