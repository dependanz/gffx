"""Finite comparison of old and refined angular integration, not table acceptance."""
import argparse
import hashlib
from pathlib import Path
import subprocess

import torch
from gffx.dashboard import connect
from tools import material_budget_reference as reference
from tools import shading_energy_probe as oracle
from .reporter import Reporter


def examples():
    rows=[]
    for resolution in (32,64,128,256,512):
        row=dict(resolution=resolution)
        for grid in ('midpoint','gauss'):
            args=tuple(torch.tensor(x,dtype=torch.float64,requires_grad=True)
                       for x in (.025,.213,.04))
            value=reference.directional(*args,resolution=resolution,azimuth=grid)
            gradients=torch.autograd.grad(value,args)
            row.update({grid:value.item(),grid+'_dc':gradients[0].item(),
                        grid+'_dr':gradients[1].item()})
        rows.append(row)
    return dict(rows=rows,cosine=.025,roughness=.213,f0=.04,
        independent_vndf=oracle.directional_albedo(.025,.213,f0=.04,resolution=512),
        scope='Finite reference convergence illustration; not table acceptance or a runtime speed claim.')


def publish(address,run):
    previous=torch.get_num_threads();torch.set_num_threads(1)
    try:data=examples()
    finally:torch.set_num_threads(previous)
    reporter=Reporter(address,run)
    source=b''.join(Path(p).read_bytes() for p in (__file__,reference.__file__,oracle.__file__))
    provenance=dict(revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True)),
        artifact_sha256=hashlib.sha256(source).hexdigest())
    def send(event,**kwargs):
        if not reporter.send(event,**kwargs):raise RuntimeError(reporter.error)
    send('start',kind='demo',provenance=provenance)
    with connect(address,run=run,wait_acks=True) as dash:
        dash.text('reference/read-me','View cosine 0.025, roughness 0.213, F0=0.04. '
            'The horizontal step is quadrature resolution: 32, 64, 128, 256, 512. '
            'The old midpoint grid misses a narrow reflection peak at shallow viewing angles. '
            'The Gauss grid uses reflection symmetry and fixed endpoint-clustered samples. '
            'Energy is a reflected fraction; derivatives show its changes per unit cosine or roughness. '
            'An independent visible-normal integration supplies the energy comparison line, not gradient truth. '
            'Only the separate tests set acceptance: a stable-looking curve alone is not enough. '
            'The new rule uses n*n samples; the old uses 2*n*n, with unchanged work caps. '
            'No lookup table was modified. Its separate accuracy failure stays visible.',step=0)
        for row in data['rows']:
            step=row['resolution']
            for grid in ('midpoint','gauss'):
                dash.scalar('reference/energy/'+grid,row[grid],step=step)
                dash.scalar('reference/view-gradient/'+grid,row[grid+'_dc'],step=step)
                dash.scalar('reference/roughness-gradient/'+grid,row[grid+'_dr'],step=step)
            dash.scalar('reference/energy/independent-vndf',data['independent_vndf'],step=step)
        dash.record('reference/samples',data,step=0)
    send('demo',page='shading/reference-demo',title='Resolving a grazing reflection',artifacts=[
        dict(kind=kind,path='reference/'+path,label=label) for kind,path,label in (
            ('text','read-me','Why the old sampling rule struggles'),
            ('plot','energy','Reflected energy versus resolution'),
            ('plot','view-gradient','Viewing-angle gradient versus resolution'),
            ('plot','roughness-gradient','Roughness gradient versus resolution'),
            ('record','samples','Exact coordinates, estimates and scope'))])
    send('finish',exit_code=0)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--address',required=True);parser.add_argument('--run',required=True)
    args=parser.parse_args();publish(args.address,args.run)
