"""Finite full-material versus Lambert-shortcut environment diagnostic."""
import argparse
import hashlib
import math
from pathlib import Path
import subprocess

import torch
from gffx.dashboard import connect
from tools import material_contract_reference as material
from tools import light_contract_reference as light
from tools import shade_contract_reference as shade
from tools import shading_energy_probe as oracle
from .reporter import Reporter


@torch.no_grad()
def examples():
    mu, weights=oracle._gauss_unit(64)
    directions=torch.stack((torch.sqrt(1-mu.square()),torch.zeros_like(mu),mu),-1)
    coefficients=torch.zeros(9,3,dtype=torch.float64)
    coefficients[0]=math.sqrt(4*math.pi)
    coefficients[2]=.5/math.sqrt(3/(4*math.pi))
    radiance=material.environment_radiance(directions,coefficients)
    cosines=torch.linspace(.1,.9,33,dtype=torch.float64)
    ev=1-cosines*torch.log((1+cosines)/cosines)
    el=1-mu*torch.log((1+mu)/mu)
    mean=4*(1-math.log(2))/3
    spec=1/(2*math.pi*(cosines[:,None]+mu[None,:]))
    albedo=torch.tensor([0.,.4,1.],dtype=torch.float64)
    brdf=spec[...,None]+albedo*(1-ev[:,None,None])*(1-el[None,:,None])/(math.pi*(1-mean))
    n=torch.tensor([0.,0.,1.],dtype=torch.float64).expand(33,3)
    actual=material.integrate_environment(n,directions,radiance,brdf,2*math.pi*weights)
    return dict(cosines=cosines.tolist(),curves={
        'full_black':actual[:,0].tolist(),'full_grey':actual[:,1].tolist(),
        'full_white':actual[:,2].tolist(),'shortcut_white':[4/3]*33},
        scope='Exact-budget roughness one, F0=1 control; not general material or scene rendering.',
        environment='Incident radiance 1+0.5*z; unoccluded flat +z surface, changing view cosine.')


def publish(address,run):
    previous=torch.get_num_threads(); torch.set_num_threads(1)
    try: data=examples()
    finally: torch.set_num_threads(previous)
    reporter=Reporter(address,run)
    paths=(__file__,material.__file__,light.__file__,shade.__file__,oracle.__file__)
    provenance=dict(revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                    dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True)),
                    artifact_sha256=hashlib.sha256(b''.join(Path(p).read_bytes() for p in paths)).hexdigest())
    def send(event,**kwargs):
        if not reporter.send(event,**kwargs): raise RuntimeError(reporter.error)
    send('start',kind='demo',provenance=provenance)
    with connect(address,run=run,wait_acks=True) as dash:
        dash.text('environment/read-me','The surrounding light is brighter overhead: radiance=1+0.5*z. '
                  'A flat +z surface is viewed from cosine 0.1 to 0.9 in 33 steps. '
                  'Three curves integrate the complete reciprocal material for black, grey (0.4), and white '
                  'matte albedo. A fourth deliberately incorrect curve treats the white material as Lambert. '
                  'These use exact budgets only for roughness one and F0=1; they do not implement general '
                  'principled materials or a metal workflow. The response above one is not light creation: '
                  'the incoming environment itself reaches 1.5. No shadows, image exposure, or display gamma. '
                  'Read the per-test analytic comparison before treating a numerical curve as acceptance.',step=0)
        for name,values in data['curves'].items():
            for index,value in enumerate(values): dash.scalar('environment/response/'+name,value,step=index)
        dash.record('environment/samples',data,step=0)
    send('demo',page='shading/environment-demo',title='Why the whole material must see the light',artifacts=[
        dict(kind='text',path='environment/read-me',label='Scene, assumptions and limits'),
        dict(kind='plot',path='environment/response',label='Full material versus the Lambert shortcut'),
        dict(kind='record',path='environment/samples',label='All 33 view cosines and responses')])
    send('finish',exit_code=0)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--address',required=True);parser.add_argument('--run',required=True)
    args=parser.parse_args();publish(args.address,args.run)
