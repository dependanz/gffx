"""Prepare bounded full-image work from hash-verified independent forward records."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

PROGRESS_HASH = '1752ce8ebcd57f17b67b6279edb5984bd2bbc7b4e02446871af44b631328dc80'
PASSED = 'prepared complete environment adjoint schedule; no new image evaluation'


def partition(orders):
    if (not isinstance(orders, list) or not 1 <= len(orders) <= 6536
            or any(type(k) is not int or k not in (8,16,32,64,128) for k in orders)):
        raise ValueError('bounded exact pixel orders required')
    result = []; start = 0; calls = 0; directions = 0
    for index, order in enumerate(orders):
        add = 2*math.ceil(4*order*order/1024)
        if calls+add > 4096 or index-start == 128:
            result.append(dict(start=start, end=index, calls=calls, directions=directions))
            start = index; calls = 0; directions = 0
        calls += add; directions += 8*order*order
    result.append(dict(start=start, end=len(orders), calls=calls, directions=directions))
    return result


def progress(path, expected=PROGRESS_HASH):
    raw = Path(path).read_bytes()
    if len(raw) > 2*1024**2 or hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('progress hash/size mismatch')
    return json.loads(raw)


def rebase(radiance, original_colour, slopes, target_colour):
    if any(len(x) != 3 or any(type(v) not in (float,int) or not math.isfinite(v) for v in x)
           for x in (radiance, original_colour, slopes, target_colour)):
        raise ValueError('finite independent RGB fields required')
    intercept = [r-a*s for r,a,s in zip(radiance, original_colour, slopes)]
    return intercept, [b+s*a for b,s,a in zip(intercept, slopes, target_colour)]


def sources():
    from tools import environment_adjoint_preflight as qualified
    from tools import albedo_image_assembly as assembly
    result = qualified.sources(); result.update(assembly.sources())
    result[Path(__file__).name] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return result


def prepare(corpus, output, root, parent, model, environment):
    import torch
    from tools import albedo_image_assembly as assembly
    from tools import environment_loss as old
    from gffx.torch.material import PrincipledBudget
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    start=time.perf_counter(); deadline=start+120
    row=dict(status='working', worker_pid=os.getpid(), model=model, environment=environment, sources=sources())
    try:
        if (model,environment) not in old.schedule(): raise ValueError('profile identity')
        history=progress(root/'progress.json')
        buffers=old.direct.load_bundle(corpus,path=old.direct.MANIFEST,sha256=old.direct.MANIFEST_HASH)['buffers']
        descriptors=assembly.descriptors(root,parent,history['batch_hashes'],(model,environment,'gradient'))
        budget=PrincipledBudget.bundled(old.direct.RESOURCE,device='cpu',dtype=torch.float64)
        curve=old.base.old.PreparedCurve(old.base.old.LossCurve(budget.to_envelope()['payload']),roughness=1.) if model=='principled' else None
        prepared=old.base.Prepared(model,curve,environment,deadline=deadline)
        target_texture=(.85*old.direct.texture()).tolist()
        data=[]
        for tile in assembly.records(descriptors,root,parent,buffers):
            for entry in tile['rows']:
                old.check_deadline(deadline)
                p=entry['point']; index=p['flat_index']; uv=buffers['uv'].reshape(-1,2)[index].tolist()
                weights=old.ref.stencil(32,32,uv)
                colour=[math.fsum(target_texture[y][x][c]*w for y,x,w in weights) for c in range(3)]
                slopes=old.slope(p['normal'],p['view'],model,environment,prepared)
                intercept,target=rebase(entry['reference']['radiance'],entry['albedo'],slopes,colour)
                data.append(dict(index=index,normal=p['normal'],view=p['view'],uv=uv,
                    order=entry['reference']['history'][-1]['n_mu'],intercept=intercept,slopes=slopes,target=target))
        ids=buffers['covered'].flatten().nonzero().flatten().tolist()
        if len(data)!=6536 or [p['index'] for p in data]!=ids: raise ValueError('complete image ownership')
        groups=partition([p['order'] for p in data])
        row.update(status=PASSED,manifest_sha256=old.direct.MANIFEST_HASH,progress_sha256=PROGRESS_HASH,
            tiles=descriptors,points=data,groups=groups,counts=dict(
                calls=2*sum(g['calls'] for g in groups),directions=2*sum(g['directions'] for g in groups),
                colour_backwards=6*sum(g['calls'] for g in groups),texture_backwards=4*6536,
                workers=2*len(groups)),elapsed_seconds=time.perf_counter()-start)
        old.check_deadline(deadline)
    except Exception as error:
        row.update(status='stopped',error=type(error).__name__+': '+str(error))
        old.write_report(output,row); raise
    old.write_report(output,row)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('fixtures','output','root','parent'): parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--model',required=True); parser.add_argument('--environment',required=True)
    parser.add_argument('--worker',action='store_true')
    args=parser.parse_args()
    if args.worker: prepare(args.fixtures,args.output,args.root,args.parent,args.model,args.environment)
    else:
        from tools.material_loss_cost import supervise
        return supervise(args.fixtures,args.output,worker_module='tools.environment_image_schedule',
            worker_args=('--root',str(args.root),'--parent',str(args.parent),'--model',args.model,'--environment',args.environment))
    return 0


if __name__=='__main__': raise SystemExit(main())
