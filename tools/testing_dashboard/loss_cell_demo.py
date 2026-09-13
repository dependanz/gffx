"""Finite magnified-loss comparison; local interpolation, not a material render."""
import argparse
import hashlib
import math
from pathlib import Path
import subprocess
import torch
from gffx.dashboard import connect
from tools import material_loss_cell as cell
from .reporter import Reporter


def measure():
    t=lambda x:torch.tensor(x,dtype=torch.float64)
    width=.005;r=.001;c=.43
    approximation=cell.generate(t([r*math.exp(-width),r*math.exp(width)]),t([c*math.exp(-width),c*math.exp(width)]))
    rows=[]
    for offset in torch.linspace(-.0045,.0045,17,dtype=torch.float64):
        roughness=(t(r)*offset.exp()).requires_grad_()
        expected=cell.reference.directional(t(c),roughness,t(1.),method='log',resolution=512)
        result=approximation.evaluate(t(c),roughness,t(1.))
        slope=torch.autograd.grad(result['loss'],roughness)[0]
        actual=torch.autograd.grad(expected,roughness)[0]
        rows.append(dict(roughness=roughness.item(),reference=expected.item(),loss=result['loss'].item(),
                         energy=result['energy'].item(),slope=slope.item(),reference_slope=actual.item()))
    return dict(rows=rows,scope='One local cell and 17 samples. No full angular resource, mean, furnace, public material or measured frame-rate acceptance.')


def publish(address,run):
    old=torch.get_num_threads();torch.set_num_threads(1)
    try:data=measure()
    finally:torch.set_num_threads(old)
    reporter=Reporter(address,run)
    def send(event,**values):
        if not reporter.send(event,**values):raise RuntimeError(reporter.error)
    digest=hashlib.sha256(b''.join(Path(p).read_bytes() for p in (__file__,cell.__file__,cell.reference.__file__))).hexdigest()
    send('start',kind='demo',provenance=dict(revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
         dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True)),artifact_sha256=digest))
    with connect(address,run=run,wait_acks=True) as dash:
        dash.text('loss-cell/read-me','Almost all light is reflected in this polished unit-Fresnel example. '
                  'The first plot magnifies the remaining loss by one trillion so the signal is visible. '
                  'The second compares changes with roughness, magnified by one billion. '
                  'Samples move across one tiny local cell; these are not animation frames. '
                  'Log-space interpolation preserves positive corner bounds, but passing this local experiment '
                  'does not admit a practical full table or energy-conserving material.',step=0)
        for step,row in enumerate(data['rows']):
            for key in ('reference','loss'):dash.scalar('loss-cell/magnified/'+key,row[key]*1e12,step=step)
            for key in ('slope','reference_slope'):dash.scalar('loss-cell/derivative/'+key,row[key]*1e9,step=step)
            dash.scalar('loss-cell/error/relative_percent',abs(row['loss']/row['reference']-1)*100,step=step)
        dash.record('loss-cell/measurements',data,step=0)
    send('demo',page='shading/loss-cell-demo',title='A trillion-times closer look at lost light',artifacts=[
        dict(kind=kind,path='loss-cell/'+path,label=label) for kind,path,label in (
            ('text','read-me','What the magnification means'),('plot','magnified','Loss multiplied by one trillion'),
            ('plot','derivative','Roughness derivative multiplied by one billion'),('plot','error','Relative loss error, percent'),
            ('record','measurements','Exact values and experimental scope'))])
    send('finish',exit_code=0)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--address',required=True);parser.add_argument('--run',required=True)
    args=parser.parse_args();publish(args.address,args.run)
