"""Bounded actual public-material environment execution, one point per worker."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
import torch
from tools.constant_environment_reference import radial,azimuth
from tools.material_loss_cost import check_deadline,supervise,write_report
from tools.principled_direct_images import RESOURCE,WHEEL_HASH,MANIFEST_HASH,producer_identity


def sample(cosine,*,model,order,deadline,budget=None,progress=None,chunk_size=10):
    from gffx.torch import material,light
    check_deadline(deadline)
    if type(chunk_size) is not int or not 1<=chunk_size<=1024:raise ValueError('unqualified light chunk size')
    if order not in (8,16,32,64,128):raise ValueError('unqualified quadrature order')
    if model not in ('classic','principled'):raise ValueError('unknown material')
    if not 0<cosine<=1:raise ValueError('invalid cosine')
    if model=='principled' and budget is None:raise ValueError('explicit accepted budget required')
    n=torch.tensor([[0.,0.,1.]],dtype=torch.float64)
    v=torch.tensor([[math.sqrt((1-cosine)*(1+cosine)),0.,cosine]],dtype=torch.float64)
    albedo=torch.tensor([.2,.4,.6],dtype=torch.float64)
    nodes,weights=radial(order,'square');estimates=[];calls=0;count=0
    started=time.perf_counter()
    with torch.no_grad():
        for split in (.5,.25):
            angles,aw=azimuth(order,split)
            points=[];omega=[]
            for mu,w in zip(nodes,weights):
                radius=math.sqrt((1-mu)*(1+mu))
                for phi,weight in zip(angles,aw):
                    points.append([radius*math.cos(phi),radius*math.sin(phi),mu])
                    omega.append(2*w*weight)
            total=torch.zeros(3,dtype=torch.float64)
            for start in range(0,len(points),chunk_size):
                check_deadline(deadline)
                directions=torch.tensor(points[start:start+chunk_size],dtype=torch.float64)
                size=len(directions)
                if model=='classic':
                    brdf=material.classic(n,v,directions,albedo=albedo,
                        specular=torch.tensor(.1,dtype=torch.float64),exponent=torch.tensor(2.,dtype=torch.float64))
                else:
                    brdf=material.principled(n,v,directions,albedo=albedo,
                        roughness=torch.tensor(1.,dtype=torch.float64),f0=torch.tensor(.04,dtype=torch.float64),budget=budget)
                packet=dict(measure='solid_angle',direction=directions[None],
                    radiance=torch.ones((1,size,3),dtype=torch.float64),valid=torch.ones((1,size),dtype=torch.bool),
                    solid_angle=torch.tensor([omega[start:start+chunk_size]],dtype=torch.float64))
                total+=light.integrate(n,brdf,packet)[0]
                calls+=1;count+=size
                if progress and calls%128==0:progress(dict(calls=calls,directions=count,elapsed_seconds=time.perf_counter()-started))
            estimates.append(total.tolist())
    check_deadline(deadline)
    return dict(estimates=estimates,calls=calls,directions=count,chunk_size=chunk_size,elapsed_seconds=time.perf_counter()-started,
                mean=[a/2+b/2 for a,b in zip(*estimates)])


def worker(output,wheel,reference_path,reference_hash):
    from gffx.torch.material import PrincipledBudget
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    deadline=time.perf_counter()+120
    report=dict(status='working',worker_pid=os.getpid(),reference_sha256=reference_hash)
    try:
        blob=reference_path.read_bytes()
        if len(blob)>262144 or hashlib.sha256(blob).hexdigest()!=reference_hash:raise ValueError('reference identity mismatch')
        ref=json.loads(blob)
        if (ref['reference']['status']!='converged' or ref['reference']['profile']!='constant-env-separated-v1'
                or ref['manifest']['sha256']!=MANIFEST_HASH or ref['supervisor']['exit_code']!=0
                or ref['supervisor']['failure'] is not None):raise ValueError('unqualified reference')
        producer=producer_identity(wheel)
        if producer['wheel']['sha256']!=WHEEL_HASH:raise ValueError('unqualified wheel')
        report.update(producer=producer,model=ref['model'],point=ref['point'],
            reference_radiance=ref['reference']['radiance'],order=ref['reference']['history'][-1]['n_mu'],
            source=dict(sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),text=Path(__file__).read_text(encoding='utf-8')))
        budget=PrincipledBudget.bundled(RESOURCE,device='cpu',dtype=torch.float64) if ref['model']=='principled' else None
        def progress(value):report['progress']=value;write_report(output,report)
        write_report(output,report)
        result=sample(ref['point']['cosine'],model=ref['model'],order=report['order'],deadline=deadline,budget=budget,progress=progress)
        limits=[1e-10+1e-8*abs(x) for x in ref['reference']['radiance']]
        errors=[[abs(x-y) for x,y in zip(row,ref['reference']['radiance'])] for row in result['estimates']]
        report.update(result=result,absolute_errors=errors,limits=limits)
        if any(x>limit for row in errors for x,limit in zip(row,limits)):raise ValueError('public environment reference mismatch')
        report['status']='passed public point comparison; no image acceptance'
    except Exception as error:
        report.update(status='stopped',error=f'{type(error).__name__}: {error}')
        write_report(output,report);raise
    write_report(output,report)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--wheel',type=Path,required=True);parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--sha256',required=True);parser.add_argument('--worker',action='store_true')
    args=parser.parse_args()
    if args.worker:worker(args.output,args.wheel,args.reference,args.sha256);return 0
    return supervise(args.fixtures,args.output,worker_module='tools.environment_candidate_point',
        worker_args=('--wheel',str(args.wheel),'--reference',str(args.reference),'--sha256',args.sha256))


if __name__=='__main__':raise SystemExit(main())
