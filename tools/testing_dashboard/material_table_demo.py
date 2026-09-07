"""Finite table/reference value, derivative and local timing comparison."""
import argparse
import hashlib
from pathlib import Path
import statistics
import subprocess
import time

import torch
from gffx.dashboard import connect
from tools import material_budget_table as tables
from tools import material_budget_reference as reference
from tools import shading_energy_probe as oracle
from tools import shade_contract_reference as shade
from .reporter import Reporter


def examples(table):
    rows=[]
    for index in range(41):
        r=torch.tensor(.3+.6*index/40,dtype=torch.float64,requires_grad=True)
        c=r.new_tensor(.43); f=r.new_tensor(.29)
        approx=table.evaluate(c,r,f)['directional']
        exact=reference.directional(c,r,f,resolution=256)
        rows.append(dict(roughness=r.item(),table=approx.item(),reference=exact.item(),
                         table_gradient=torch.autograd.grad(approx,r)[0].item(),
                         reference_gradient=torch.autograd.grad(exact,r)[0].item()))
    grazing=[]
    for cosine in (.013,.019,.025,.031,.047,.073):
        c=torch.tensor(cosine,dtype=torch.float64,requires_grad=True)
        r=c.new_tensor(.213);f=c.new_tensor(1.)
        approx=table.evaluate(c,r,f)['directional']
        exact=reference.directional(c,r,f,resolution=512)
        grazing.append(dict(cosine=cosine,table=approx.item(),reference=exact.item(),
            table_gradient=torch.autograd.grad(approx,c)[0].item(),
            reference_gradient=torch.autograd.grad(exact,c)[0].item()))
    return dict(rows=rows,grazing=grazing,table_bytes=sum(x.numel()*x.element_size() for x in
        (table.roughness,table.cosines,table.energy,table.mean)),metadata=table.metadata,
        scope='Candidate piecewise-quadratic budget and its slopes, not general material or speed acceptance.')


def timings(table):
    c=torch.linspace(.1,.9,16384,dtype=torch.float64,requires_grad=True)
    r=torch.tensor(.63,dtype=torch.float64,requires_grad=True)
    f=torch.tensor(.29,dtype=torch.float64,requires_grad=True)
    forward=[];backward=[];scalar=[]
    for _ in range(2):
        with torch.no_grad(): table.evaluate(c,r,f)
    for _ in range(7):
        start=time.perf_counter()
        with torch.no_grad(): table.evaluate(c,r,f)
        forward.append((time.perf_counter()-start)*1000)
        c.grad=r.grad=f.grad=None
        start=time.perf_counter(); value=table.evaluate(c,r,f)
        (value['directional'].sum()+value['average'].sum()).backward()
        backward.append((time.perf_counter()-start)*1000)
    for _ in range(3):
        start=time.perf_counter()
        with torch.no_grad():reference.directional(c.new_tensor(.43),r,f,resolution=256)
        scalar.append((time.perf_counter()-start)*1000)
    return dict(query_count=16384,dtype='float64',device='CPU',threads=torch.get_num_threads(),
        torch_version=torch.__version__,forward_ms=forward,forward_backward_ms=backward,
        median_forward_ms=statistics.median(forward),median_forward_backward_ms=statistics.median(backward),
        scalar_reference_ms=scalar,scope='Seven local batch timings after two forward warmups; three scalar oracle timings. Not render FPS, edge performance, peak memory or a cross-device benchmark.')


def publish(address,run):
    previous=torch.get_num_threads();torch.set_num_threads(1)
    try:
        start=time.perf_counter();table=tables.generate();generation=time.perf_counter()-start
        data=examples(table);speed=timings(table);speed['generation_seconds']=generation
    finally:torch.set_num_threads(previous)
    reporter=Reporter(address,run)
    source=b''.join(Path(p).read_bytes() for p in (__file__,tables.__file__,reference.__file__,oracle.__file__,shade.__file__))
    provenance=dict(revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True)),artifact_sha256=hashlib.sha256(source).hexdigest())
    def send(event,**kwargs):
        if not reporter.send(event,**kwargs):raise RuntimeError(reporter.error)
    send('start',kind='demo',provenance=provenance)
    with connect(address,run=run,wait_acks=True) as dash:
        dash.text('table/read-me','Step 0..40 sweeps roughness 0.3..0.9 with view cosine 0.43 and F0=0.29. '
                  'Compare the sampled reference values to a 33 by 257 table generated from our own equations. '
                  'Local quadratic interpolation keeps piecewise slopes; matching brightness alone is not enough. '
                  'The grazing panels use six increasing view cosines (0.013, 0.019, 0.025, 0.031, 0.047, 0.073), '
                  'roughness 0.213 and F0=1. Their derivatives are with respect to view cosine, not roughness. '
                  'This region exposed the earlier tables errors; all exact coordinates are in the sample record. '
                  'This table only covers roughness 0.2..1. Sharper surfaces are rejected, not rounded up. '
                  'This does not establish a public gffx roughness minimum. The separate test page checks energy. '
                  'The timing record measures one CPU thread; table construction is offline, not a per-frame job. '
                  'This is an experimental evaluator, not public material, rendering FPS, GPU or edge acceptance.',step=0)
        for index,row in enumerate(data['rows']):
            for name in ('table','reference'):dash.scalar('table/value/'+name,row[name],step=index)
            for name in ('table_gradient','reference_gradient'):dash.scalar('table/derivative/'+name,row[name],step=index)
        for index,row in enumerate(data['grazing']):
            for name in ('table','reference'):dash.scalar('table/grazing-value/'+name,row[name],step=index)
            for name in ('table_gradient','reference_gradient'):dash.scalar('table/grazing-derivative/'+name,row[name],step=index)
        dash.record('table/samples',data,step=0);dash.record('table/timing',speed,step=0)
    send('demo',page='shading/table-demo',title='Can a small table replace expensive integration?',artifacts=[
        dict(kind=kind,path='table/'+path,label=label) for kind,path,label in (
            ('text','read-me','How to read the approximation'),('plot','value','Reflected-energy values'),
            ('plot','derivative','Roughness derivatives: inspect the slopes'),
            ('plot','grazing-value','Grazing-angle brightness: six ordered samples'),
            ('plot','grazing-derivative','Grazing view-cosine derivatives'),
            ('record','samples','41 roughness samples, six grazing samples and table size'),
            ('record','timing','Bounded local CPU measurements'))])
    send('finish',exit_code=0)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--address',required=True);parser.add_argument('--run',required=True)
    args=parser.parse_args();publish(args.address,args.run)
