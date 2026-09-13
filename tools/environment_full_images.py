"""Bounded, resumable constant-environment image evidence, not a renderer API.

Reference tensor arithmetic is independently derived at roughness=1. Candidate
pixels always execute public material and light integration. No runtime imports
from this development tool, no resource generation and no persistent query cache.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import torch

from tools.constant_environment_reference import radial, azimuth, matte_integral
from tools.environment_candidate_point import sample
from tools.material_loss_cost import check_deadline, supervise, write_report
from tools.principled_direct_images import (
    MANIFEST, MANIFEST_HASH, SCALAR_HASH, RESOURCE, PreparedCurve, scalar,
    load_bundle, producer_identity, validate_buffers, tensor_bytes, tensor_from_bytes)
from tools.shading_environment_reference import ATOL, RTOL, ReferenceConvergenceError
from tools.shading_image_reference import compare_images

WHEEL_HASH = '1c97f8dfa1f565078131a5e1192492680dbecd8d4ed8620c56d2ab4fa825c4c9'
PROFILE = dict(name='constant-environment-images-v1', materials=['classic','principled'],
    width=128, height=128, covered_pixels=6536, points_per_tile=128, tiles=52,
    albedo=[.2,.4,.6], roughness=1., f0=.04, specular=.1, exponent=2.,
    environment='constant unit radiance', dtype='float64', device='cpu', threads=1,
    reference_evaluations_per_pixel=87296, candidate_directions_per_pixel=65536,
    reference_atol=ATOL, reference_rtol=RTOL, channel_atol=1e-10, channel_rtol=1e-8,
    aggregate_seconds=1800, cooperative_seconds=120, watchdog_seconds=180,
    sampled_working_set_limit_bytes=1024**3)


def tile_range(index):
    if type(index) is not int or not 0 <= index < PROFILE['tiles']:
        raise ValueError('invalid tile index')
    return index*128, min((index+1)*128,6536)


def quadrature(order, split):
    if order not in (8,16,32,64,128): raise ValueError('unqualified order')
    nodes, weights = radial(order,'square')
    angles, aw = azimuth(order,split)
    points, omega = [], []
    for mu,w in zip(nodes,weights):
        radius=math.sqrt((1-mu)*(1+mu))
        for phi,weight in zip(angles,aw):
            points.append([radius*math.cos(phi),radius*math.sin(phi),mu])
            omega.append(2*w*weight)
    return torch.tensor(points,dtype=torch.float64),torch.tensor(omega,dtype=torch.float64)


def reference_brdf(c, directions, *, model):
    """Fixed-profile independent formulas; principled returns glossy term only."""
    if not math.isfinite(c) or not 0<c<=1: raise ValueError('invalid view cosine')
    mu=directions[:,2]
    dot=math.sqrt((1-c)*(1+c))*directions[:,0]+c*mu
    if model=='classic':
        nh2=(c+mu).square()/(2*(1+dot))
        return torch.tensor([.2,.4,.6],dtype=torch.float64)[None]/math.pi + (.1*10/(8*math.pi)*nh2)[:,None]
    if model!='principled': raise ValueError('unknown material')
    vh=((1+dot)/2).sqrt()
    glossy=(.04+.96*(1-vh).pow(5))/(2*math.pi*(c+mu))
    return glossy[:,None].expand(-1,3)


def reference(c, *, model, curve, deadline, max_evaluations=87296):
    if not math.isfinite(c) or not 0<c<=1: raise ValueError('invalid view cosine')
    if model not in PROFILE['materials']: raise ValueError('unknown material')
    if type(max_evaluations) is not int or not 0<max_evaluations<=87296:
        raise ValueError('reference work cap')
    if model=='principled' and curve is None: raise ValueError('explicit independent curve required')
    deadline=min(deadline,time.perf_counter()+30)
    check_deadline(deadline)
    matte=torch.tensor(matte_integral(curve,c) if model=='principled' else [0.,0.,0.],dtype=torch.float64)
    history=[]; evaluations=0; previous=None; streak=0
    def fail(reason):
        raise ReferenceConvergenceError(dict(profile='constant-env-separated-tensor-v1',
            status='unresolved',reason=reason,evaluations=evaluations,history=history))
    with torch.no_grad():
        for order in (8,16,32,64,128):
            check_deadline(deadline)
            if evaluations+4*order*order>max_evaluations: fail('work_limit')
            estimates=[]
            for split in (.5,.25):
                directions,weights=quadrature(order,split)
                brdf=reference_brdf(c,directions,model=model)
                estimate=(brdf*(directions[:,2]*weights)[:,None]).sum(0)+matte
                if not bool(torch.isfinite(estimate).all()) or bool((estimate<0).any()): fail('invalid_sample')
                estimates.append(estimate)
                evaluations+=len(directions)
                check_deadline(deadline)
            estimates=torch.stack(estimates)
            mean=estimates.mean(0); thresholds=ATOL+RTOL*mean.abs()
            delta=(estimates[0]-estimates[1]).abs(); change=None
            if previous is not None:
                change=(estimates-previous).abs().amax(0)
                streak=streak+1 if bool(((change<=thresholds)&(delta<=thresholds)).all()) else 0
            history.append(dict(n_mu=order,estimates=estimates.tolist(),radiance=mean.tolist(),
                threshold=thresholds.tolist(),offset_change=delta.tolist(),
                change=None if change is None else change.tolist(),streak=streak,evaluations=evaluations))
            if streak==2:
                if model=='classic':
                    known=torch.tensor([a+.1*(5*c/8+5/12) for a in (.2,.4,.6)],dtype=torch.float64)
                    if bool(((mean-known).abs()>1e-9+1e-9*known.abs()).any()): fail('closed_form_mismatch')
                return dict(status='converged',radiance=mean.tolist(),evaluations=evaluations,history=history)
            previous=estimates
    fail('level_limit')


def candidate(c, *, model, budget, ref, deadline):
    if ref['status']!='converged' or ref['history'][-1]['streak']!=2:
        raise ValueError('unresolved reference')
    row=sample(c,model=model,budget=budget,order=ref['history'][-1]['n_mu'],
               deadline=deadline,chunk_size=1024)
    limits=[1e-10+1e-8*abs(x) for x in ref['radiance']]
    errors=[[abs(a-b) for a,b in zip(value,ref['radiance'])] for value in row['estimates']]
    row.update(errors=errors,limits=limits)
    if any(not math.isfinite(x) or x>limit for error in errors for x,limit in zip(error,limits)):
        raise ValueError('public pixel tolerance failed')
    return row


def pixel_layout(buffers):
    mask=validate_buffers(buffers,'constant')
    if int(mask.sum())!=6536: raise ValueError('immutable coverage count mismatch')
    n,v=buffers['normal'][mask],buffers['view'][mask]
    if not torch.equal(buffers['albedo/constant'][mask],torch.tensor([.2,.4,.6],dtype=torch.float64).expand(6536,3)):
        raise ValueError('constant albedo mismatch')
    c=((n/n.norm(dim=-1,keepdim=True))*(v/v.norm(dim=-1,keepdim=True))).sum(-1)
    if not bool(torch.isfinite(c).all()) or bool(((c<=0)|(c>1)).any()): raise ValueError('invalid covered view')
    return mask,mask.flatten().nonzero().flatten().tolist(),c.tolist()


def source_identities():
    names=('environment_full_images.py','environment_candidate_point.py',
           'constant_environment_reference.py','principled_direct_images.py',
           'shading_independent_evaluator.py','shading_environment_reference.py')
    return {name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in names}


def worker(root, output, wheel, model, index):
    from gffx.torch.material import PrincipledBudget
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    deadline=time.perf_counter()+120
    report=dict(status='working',worker_pid=os.getpid(),profile=PROFILE,model=model,index=index,
                manifest_sha256=MANIFEST_HASH,rows=[])
    try:
        start,end=tile_range(index)
        report.update(producer=producer_identity(wheel),sources=source_identities(),start=start,end=end)
        if report['producer']['wheel']['sha256']!=WHEEL_HASH: raise ValueError('wheel identity mismatch')
        if report['sources']['shading_independent_evaluator.py']!=SCALAR_HASH: raise ValueError('frozen scalar changed')
        bundle=load_bundle(root,path=MANIFEST,sha256=MANIFEST_HASH)
        mask,indices,cosines=pixel_layout(bundle['buffers'])
        budget=PrincipledBudget.bundled(RESOURCE,device='cpu',dtype=torch.float64)
        curve=PreparedCurve(scalar.LossCurve(budget.to_envelope()['payload']),roughness=1.)
        report['budget']=budget.metadata
        write_report(output,report)
        for offset in range(start,end):
            check_deadline(deadline)
            ref=reference(cosines[offset],model=model,curve=curve,deadline=deadline)
            actual=candidate(cosines[offset],model=model,budget=budget,ref=ref,deadline=deadline)
            report['rows'].append(dict(flat_index=indices[offset],cosine=cosines[offset],reference=ref,actual=actual))
            if len(report['rows'])%16==0: write_report(output,report)
        check_deadline(deadline)
        report['status']='passed tile; not a complete image'
    except Exception as error:
        report.update(status='stopped',error=f'{type(error).__name__}: {error}')
        if isinstance(error,ReferenceConvergenceError):report['unresolved']=error.report
        write_report(output,report);raise
    write_report(output,report)


def read_tile(path, digest):
    from tools.corpus_registry import _read
    blob=_read(Path(path),2*1024**2)
    if hashlib.sha256(blob).hexdigest()!=digest: raise ValueError('tile hash mismatch')
    row=json.loads(blob)
    s=row['supervisor']
    if (row['status']!='passed tile; not a complete image' or row['profile']!=PROFILE
        or row['manifest_sha256']!=MANIFEST_HASH or row['producer']['wheel']['sha256']!=WHEEL_HASH
        or row['sources']!=source_identities() or s['exit_code']!=0 or s['failure'] is not None
        or s['monitored_pid']!=row['worker_pid'] or not 0<s['sampled_peak_working_set_bytes']<=1024**3
        or not 0<s['elapsed_seconds']<=180): raise ValueError('tile qualification mismatch')
    return row


def assemble(records, mask):
    if len(records)!=52: raise ValueError('complete tile set required')
    if mask.shape!=(128,128) or mask.dtype!=torch.bool or int(mask.sum())!=6536:
        raise ValueError('image coverage mismatch')
    ids=mask.flatten().nonzero().flatten().tolist();seen=[];models=set()
    actual=torch.zeros((128,128,3),dtype=torch.float64);expected=torch.zeros_like(actual)
    counts=dict(reference_evaluations=0,candidate_directions=0,calls=0)
    for i,record in enumerate(records):
        start,end=tile_range(i)
        if (record['index']!=i or record['status']!='passed tile; not a complete image'
            or record['start']!=start or record['end']!=end or len(record['rows'])!=end-start):
            raise ValueError('invalid ordered tile coverage')
        models.add(record['model'])
        for offset,row in enumerate(record['rows'],start):
            index=row['flat_index'];seen.append(index)
            if index!=ids[offset]: raise ValueError('pixel ownership mismatch')
            ref,value=row['reference'],row['actual']
            if ref['status']!='converged' or ref['history'][-1]['streak']!=2:
                raise ValueError('unresolved pixel reference')
            limit=torch.tensor([1e-10+1e-8*abs(x) for x in ref['radiance']],dtype=torch.float64)
            estimates=torch.tensor(value['estimates'],dtype=torch.float64)
            reference_rgb=torch.tensor(ref['radiance'],dtype=torch.float64)
            if (estimates.shape!=(2,3) or not bool(torch.isfinite(estimates).all())
                or bool(((estimates-reference_rgb).abs()>limit).any())): raise ValueError('saved pixel tolerance failed')
            actual.view(-1,3)[index]=estimates.mean(0)
            expected.view(-1,3)[index]=reference_rgb
            counts['reference_evaluations']+=ref['evaluations']
            counts['candidate_directions']+=value['directions'];counts['calls']+=value['calls']
    if len(set(seen))!=6536 or len(models)!=1 or not models.issubset(PROFILE['materials']):
        raise ValueError('mixed or incomplete image')
    metrics=compare_images(actual,expected,mask)
    if not metrics['passed']: raise ValueError('image tolerance failed')
    return dict(actual=actual,reference=expected,metrics=metrics,counts=counts)


def run_cohort(root,output,wheel):
    """Fresh finite run only; never retries, overwrites or silently resumes tiles."""
    if output.exists(): raise FileExistsError('fresh cohort output required')
    output.mkdir(parents=True)
    started=time.perf_counter(); manifest=dict(profile=PROFILE,status='working',tiles={},images={})
    write_report(output/'cohort.json',manifest)
    try:
        for model in PROFILE['materials']:
            manifest['tiles'][model]=[]
            for index in range(52):
                # Reserve a complete watchdog period; do not overrun aggregate cap.
                check_deadline(started+PROFILE['aggregate_seconds']-180)
                path=output/f'{model}-{index:02d}.json'
                code=supervise(root,path,worker_module='tools.environment_full_images',
                    worker_args=('--wheel',str(wheel),'--model',model,'--index',str(index)))
                digest=hashlib.sha256(path.read_bytes()).hexdigest()
                manifest['tiles'][model].append(dict(path=path.name,sha256=digest))
                write_report(output/'cohort.json',manifest)
                if code: raise RuntimeError('tile failed; cohort stopped without retry')
            bundle=load_bundle(root,path=MANIFEST,sha256=MANIFEST_HASH)
            records=[read_tile(output/r['path'],r['sha256']) for r in manifest['tiles'][model]]
            result=assemble(records,bundle['buffers']['covered'])
            image=dict(metrics=result['metrics'],counts=result['counts'],arrays={})
            for name in ('actual','reference'):
                blob=tensor_bytes(result[name]);path=output/f'{model}-{name}.bin'
                with path.open('xb') as stream:stream.write(blob)
                image['arrays'][name]=dict(path=path.name,sha256=hashlib.sha256(blob).hexdigest(),bytes=len(blob))
            manifest['images'][model]=image
            write_report(output/'cohort.json',manifest)
        manifest['status']='passed two constant-environment images; not general environment or M1 acceptance'
    except Exception as error:
        manifest.update(status='stopped',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        manifest['elapsed_seconds']=time.perf_counter()-started
        write_report(output/'cohort.json',manifest)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixtures',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--wheel',type=Path,required=True);p.add_argument('--worker',action='store_true')
    p.add_argument('--model',choices=PROFILE['materials']);p.add_argument('--index',type=int)
    args=p.parse_args()
    if args.worker:worker(args.fixtures,args.output,args.wheel,args.model,args.index)
    else:run_cohort(args.fixtures,args.output,args.wheel)


if __name__=='__main__':main()
