"""Bounded complete-environment point references before image qualification."""
import argparse
import hashlib
import math
import os
from pathlib import Path
import time

import torch

from tools import shading_environment_reference as reference
from tools import shading_independent_evaluator as scalar
from tools.material_loss_cost import check_deadline, supervise, write_report
from tools.principled_direct_images import (
    MANIFEST, MANIFEST_HASH, RESOURCE, SCALAR_HASH, WHEEL_HASH, PreparedCurve,
    load_bundle, producer_identity)

PROFILE = dict(name='complete-environment-preflight-v1', materials=['classic','principled'],
    points=8, albedo=[.2,.4,.6], roughness=1., f0=.04, specular=.1, exponent=2.,
    environment='constant unit radiance', dtype='float64', device='cpu', threads=1,
    reference_profile='env-sh2-v1', reference_seconds=30, reference_evaluations=87296,
    cooperative_seconds=120, watchdog_seconds=180, sampled_working_set_limit_bytes=1024**3)


def select_pixels(buffers):
    mask = buffers['covered']
    n, v = buffers['normal'], buffers['view']
    if (mask.dtype != torch.bool or mask.numel()>16384 or n.shape != (*mask.shape,3)
            or v.shape != n.shape or n.dtype != torch.float64 or v.dtype != n.dtype
            or any(x.device.type != 'cpu' for x in (mask,n,v))):
        raise ValueError('invalid bounded reference buffers')
    rows=[]
    for index in mask.reshape(-1).nonzero().flatten().tolist():
        nn=scalar._normal(n.reshape(-1,3)[index].tolist())
        vv=scalar._normal(v.reshape(-1,3)[index].tolist())
        cosine=scalar._dot(nn,vv)
        if cosine>0: rows.append(dict(flat_index=index,cosine=cosine,normal=nn,view=vv))
    rows.sort(key=lambda row:(row['cosine'],row['flat_index']))
    if len(rows)<8:raise ValueError('eight positive-view points required')
    return [rows[i*(len(rows)-1)//7] for i in range(8)]


def reference_point(normal, view, *, model, curve=None, rule='uniform'):
    if model not in PROFILE['materials']:raise ValueError('unknown complete material')
    if model=='principled' and curve is None:raise ValueError('explicit prepared reference curve required')
    def brdf(n,v,l):
        if model=='classic':
            return scalar.classic(n,v,l,albedo=[.2,.4,.6],specular=.1,exponent=2.)
        return scalar.principled(n,v,l,albedo=[.2,.4,.6],roughness=1.,f0=.04,budget=curve)
    coefficients=[[math.sqrt(4*math.pi)]*3]+[[0.,0.,0.] for _ in range(8)]
    if rule=='uniform':
        result=reference.environment_reference(normal,view,coefficients,brdf=brdf)
    elif rule in ('split','split-square','separated'):
        from tools.constant_environment_reference import reference_point as refined
        cosine=scalar._dot(scalar._normal(normal),scalar._normal(view))
        result=refined(cosine,model=model,curve=curve,radial_rule='linear' if rule=='split' else 'square',
                       separated=rule=='separated')
    else:raise ValueError('unknown reference rule')
    if model=='classic':
        cosine=scalar._dot(scalar._normal(normal),scalar._normal(view))
        expected=[a+.1*(5*cosine/8+5/12) for a in (.2,.4,.6)]
        if any(abs(x-y)>1e-9+1e-9*abs(y) for x,y in zip(result['radiance'],expected)):
            raise ValueError('classic independent closed-form mismatch')
        result['closed_form']=expected
    return result


def worker(root,output,wheel,model,index,rule):
    from gffx.torch.material import PrincipledBudget
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    deadline=time.perf_counter()+120
    profile=dict(PROFILE,reference_profile={'uniform':'env-sh2-v1','split':'constant-env-split-v1',
                 'split-square':'constant-env-split-square-v1','separated':'constant-env-separated-v1'}[rule])
    report=dict(status='working',worker_pid=os.getpid(),profile=profile,model=model,point_rank=index,
                manifest=dict(path=MANIFEST,sha256=MANIFEST_HASH))
    try:
        producer=producer_identity(wheel)
        if producer['wheel']['sha256']!=WHEEL_HASH:raise ValueError('unqualified candidate wheel identity')
        report['producer']=producer
        if hashlib.sha256(Path(scalar.__file__).read_bytes()).hexdigest()!=SCALAR_HASH:
            raise ValueError('frozen scalar evaluator changed')
        files=[Path(__file__),Path(reference.__file__),Path(scalar.__file__),
               Path(__file__).with_name('principled_direct_images.py')]
        if rule!='uniform':files.append(Path(__file__).with_name('constant_environment_reference.py'))
        report['sources']={p.name:dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                                      text=p.read_text(encoding='utf-8')) for p in files}
        bundle=load_bundle(root,path=MANIFEST,sha256=MANIFEST_HASH)
        report['selected_points']=select_pixels(bundle['buffers'])
        point=report['selected_points'][index];report['point']=point
        curve=None
        if model=='principled':
            budget=PrincipledBudget.bundled(RESOURCE,device='cpu',dtype=torch.float64)
            report['budget']=budget.metadata
            curve=PreparedCurve(scalar.LossCurve(budget.to_envelope()['payload']),roughness=1.)
        check_deadline(deadline);write_report(output,report)
        result=reference_point(point['normal'],point['view'],model=model,curve=curve,rule=rule)
        report['reference']=result
        check_deadline(deadline)
        report['status']='converged reference point; not image or public execution acceptance'
    except reference.ReferenceConvergenceError as error:
        report.update(status='stopped',error=str(error),reference=error.report)
        write_report(output,report);raise
    except Exception as error:
        report.update(status='stopped',error=f'{type(error).__name__}: {error}')
        write_report(output,report);raise
    write_report(output,report)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--wheel',type=Path,required=True)
    parser.add_argument('--model',choices=PROFILE['materials'],required=True)
    parser.add_argument('--index',type=int,choices=range(8),required=True)
    parser.add_argument('--rule',choices=('uniform','split','split-square','separated'),default='uniform')
    parser.add_argument('--worker',action='store_true')
    args=parser.parse_args()
    if args.worker:worker(args.fixtures,args.output,args.wheel,args.model,args.index,args.rule);return 0
    return supervise(args.fixtures,args.output,worker_module='tools.environment_image_preflight',
        worker_args=('--wheel',str(args.wheel),'--model',args.model,'--index',str(args.index),'--rule',args.rule))


if __name__=='__main__':raise SystemExit(main())
