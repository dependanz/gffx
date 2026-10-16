"""Bounded diagnostic first-VJP chain rule, with one sample VJP per RGB channel.

Shading microbatches see independent sample leaves. Store their disjoint sample
adjoints, sum direct parameter terms, then contract the real sample graph once.
No second-derivative claim; this is a probe helper, not a production renderer.
"""
import math

def evaluate(samples,parameter,field,*,chunk=4,checkpoint=lambda:None):
    import torch
    if type(chunk)!=int or chunk not in (1,2,4) or samples.ndim!=2 or samples.shape[1]!=4 or not 1<=len(samples)<=800 or parameter.numel()!=1 or not parameter.requires_grad:raise ValueError('bounded first-VJP probe profile')
    values=[];direct=[[],[],[]];adjoints=[[],[],[]]
    for i in range(0,len(samples),chunk):
        checkpoint();p=samples[i:i+chunk].detach().requires_grad_();rgb=field(p)
        if rgb.shape!=(len(p),3) or not bool(torch.isfinite(rgb).all()):raise ValueError('finite per-sample weighted RGB required')
        from gffx.torch._reference_channels import _sum_last
        value=_sum_last(rgb.transpose(0,1));values.append(value.detach().cpu().tolist())
        for channel in range(3):
            gp,gt=torch.autograd.grad(value[channel],(p,parameter),retain_graph=True,allow_unused=True)
            adjoints[channel].append(torch.zeros_like(p) if gp is None else gp.detach());direct[channel].append(0. if gt is None else gt.item())
    result=[];contracts=[];calls=0
    for channel in range(3):
        checkpoint();g=None
        if samples.requires_grad:
            g=torch.autograd.grad(samples,parameter,torch.cat(adjoints[channel]),retain_graph=True,allow_unused=True)[0];calls+=1
        contracts.append(0. if g is None else g.item());result.append(math.fsum([*direct[channel],contracts[-1]]))
    import os,json,hashlib
    capture=os.environ.get('GFFX_STREAM_CAPTURE')
    if capture:
        from pathlib import Path
        p=Path(capture);assert not p.exists();p.write_text(json.dumps(dict(direct=direct,contracts=contracts,adjoints_sha256=[hashlib.sha256(torch.cat(v).cpu().numpy().tobytes()).hexdigest() for v in adjoints],values=values),indent=2))
    return dict(value=[math.fsum(v[c] for v in values) for c in range(3)],derivative=result,sample_vjp_calls=calls)


def sum_faces(results,*,expected_faces):
    """Combine every admitted face of a selected pixel without tiny cutoffs."""
    try:
        if not isinstance(expected_faces,(list,tuple)) or not 1<=len(expected_faces)<=32 or any(type(f)!=int for f in expected_faces) or len(set(expected_faces))!=len(expected_faces):raise ValueError('unique bounded expected faces required')
        if not isinstance(results,list) or len(results)!=len(expected_faces) or any(type(r['face'])!=int for r in results) or {r['face'] for r in results}!=set(expected_faces):raise ValueError('every expected face exactly once required')
        for row in results:
            if type(row['sample_vjp_calls'])!=int or not 0<=row['sample_vjp_calls']<=3:raise ValueError('bounded first sample-VJP count required')
            for key in ('value','derivative'):
                if not isinstance(row[key],(list,tuple)) or len(row[key])!=3 or any(type(v) not in (int,float) or not math.isfinite(v) for v in row[key]):raise ValueError('three finite RGB values and derivatives required')
        return dict(value=[math.fsum(r['value'][c] for r in results) for c in range(3)],derivative=[math.fsum(r['derivative'][c] for r in results) for c in range(3)],sample_vjp_calls=sum(r['sample_vjp_calls'] for r in results))
    except (KeyError,TypeError) as error:raise ValueError('complete bounded face results required') from error
