"""Finite environment-adjoint preflight, never a fabricated complete image."""
import argparse
import json
from pathlib import Path
import torch
from tools import environment_loss as fixture
from .image_reference_demo import display_rgb


def panels(result):
    columns=[torch.tensor(result['inputs']['target'],dtype=torch.float64)]
    for p in result['passes']:
        columns.append(torch.tensor([r['values'] for r in p['offsets']],dtype=torch.float64).mean(0))
    if any(x.shape!=(2,3) or not bool(torch.isfinite(x).all()) for x in columns): raise ValueError('two measured RGB points required')
    swatches=display_rgb(torch.stack(columns,dim=1)).repeat_interleave(64,0).repeat_interleave(64,1)
    gradient=torch.tensor([r['gradient'] for r in result['passes'][0]['offsets']],dtype=torch.float64).mean(0)
    scale=float(gradient.abs().max())
    if gradient.shape!=(32,32,3) or not bool(torch.isfinite(gradient).all()) or scale<=0: raise ValueError('finite nonzero source gradient required')
    return dict(swatches=swatches,gradient=(gradient.abs()/scale*255).round().to(torch.uint8),gradient_scale=scale)


def publish(reports,sha256,address,run):
    from gffx.dashboard import ArrayView,connect
    from .reporter import Reporter
    reporter=Reporter(address,run)
    def send(event,**kw):
        if not reporter.send(event,**kw):raise RuntimeError(reporter.error)
    send('start',kind='demo',provenance=dict(artifact_sha256=fixture.direct.WHEEL_HASH));artifacts=[]
    with connect(address,run=run,wait_acks=True) as dash:
        dash.text('environment-loss/scope','SELECTED-PIXEL PREFLIGHT, NOT A FULL IMAGE. Each row is one measured surface point: '
                  'top is the most grazing selected view, bottom is the middle selected rank. Columns are TARGET / BEFORE / AFTER one texture step. '
                  'Each colour uses complete lighting from the hemisphere, not a partial angular chunk. Two quadrature offsets are checked separately; swatches display their mean. '
                  'The implementation first adds all light without retaining a graph, then recomputes small chunks with the complete pixel-error cotangent. '
                  'A gradient map shows absolute RGB source-texel derivatives normalized only for display. No continuous animation, full-image, geometry/light derivative or M1 claim. '
                  'sRGB conversion is display only; acceptance uses unscaled float64 RGB and gradients.',step=0)
        artifacts.append(dict(kind='text',path='environment-loss/scope',label='What these two rows measure, and what they do not'))
        for report in reports:
            key=report['model']+'-'+report['environment'];base='environment-loss/'+key;result=report['result'];p=panels(result)
            for name,label in (('swatches','Two selected points: target / before / after'),('gradient','Absolute source derivatives, normalized for display')):
                img=p[name];dash.image(base+'/'+name,ArrayView('uint8',tuple(img.shape),bytes(img.flatten().tolist())),step=0)
                artifacts.append(dict(kind='image',path=base+'/'+name,label=key+': '+label))
            for i,iteration in enumerate(result['passes']):
                for row in iteration['offsets']:dash.scalar(base+'/loss/offset-'+str(row['split']),row['loss'],step=i)
            artifacts.append(dict(kind='plot',path=base+'/loss',label=key+': complete selected-pixel objective, both offsets'))
            dash.record(base+'/evidence',dict(indices=report['indices'],calls=result['calls'],directions=result['directions'],
                gradient_display_scale=p['gradient_scale'],supervisor=report['supervisor'],
                comparisons=[[dict(split=row['split'],loss=row['loss'],metrics=row['metrics']) for row in p['offsets']] for p in result['passes']]),step=0)
            artifacts.append(dict(kind='record',path=base+'/evidence',label=key+': measured errors, pixel ownership and worker limits'))
        dash.record('environment-loss/identity',dict(cohort_sha256=sha256,wheel_sha256=fixture.direct.WHEEL_HASH,
            scope='Four profiles, two selected pixels each; not complete images',coverage='Deferred until after M1 by user'),step=0)
        artifacts.append(dict(kind='record',path='environment-loss/identity',label='Evidence identity and scope'))
    send('demo',page='shading/environment-loss-demo',title='Environment gradients: selected-pixel preflight',artifacts=artifacts)
    send('finish',exit_code=0)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    for k in ('sha256','address','run'):p.add_argument('--'+k,required=True)
    a=p.parse_args();torch.set_num_threads(1)
    cohort=fixture.replay(a.root,a.sha256)
    reports=[json.loads((a.root/r['path']).read_text()) for r in cohort['workers']]
    publish(reports,a.sha256,a.address,a.run)


if __name__=='__main__':main()
