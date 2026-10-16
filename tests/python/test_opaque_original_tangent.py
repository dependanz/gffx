"""Installed original-piece normal derivative against independent endpoints."""
import os,json,hashlib,math,struct
from pathlib import Path
from fractions import Fraction as Q
import torch


def test_first_original_normal_gradient_meets_independent_endpoints(record_property):
    from gffx.torch.render import opaque_quadrature
    from gffx.torch.material import PrincipledBudget,principled
    from gffx.torch._reference_channels import sum3,_sum_last
    device=os.environ.get('GFFX_KNOT_DEVICE','cpu');t=lambda v:torch.tensor(v,dtype=torch.float64,device=device)
    p=Path(os.environ['GFFX_FULL_INPUT']);assert hashlib.sha256(p.read_bytes()).hexdigest()=='a15bea40cb5cdead7994d828b57f5dcfac9f44fab0292a06e7d0a7b08719f73a';data=json.loads(p.read_bytes());ch=data['selected']
    refpath=Path(os.environ['GFFX_NORMAL_REFERENCE']);digest=hashlib.sha256(refpath.read_bytes()).hexdigest();assert digest==os.environ['GFFX_NORMAL_REFERENCE_SHA'];report=json.loads(refpath.read_bytes());assert report['status']=='qualified';ref=report['result'];assert ref['piece']==0 and ref['face']==5739
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    v=t(ch['screen_vertices']);f=torch.tensor(ch['faces'],dtype=torch.int32,device=device);curves=[]
    for row in data['faces']:
        nn=t(row['normal']);vv=t(row['view']);iw=1/t(row['clip_w']);dot=lambda a,b:sum3(a*b)
        co=[dot(nn[i],vv[i])*iw[i]*iw[i] for i in range(3)];co.extend((dot(nn[i],vv[j])+dot(nn[j],vv[i]))*iw[i]*iw[j] for i,j in ((0,1),(0,2),(1,2)));curves.append(torch.stack(co))
    ids,q=opaque_quadrature(v,f,v.new_empty((len(f),3,0)),1,1,curves=torch.stack(curves));fi=ch['candidate_faces'].index(5739);nodes=q[0][ids[0]==fi];row=data['faces'][fi]
    nn=t(row['normal']).requires_grad_();vv=t(row['view']);iw=1/t(row['clip_w']);direction=.01*torch.stack((nn[:,2].detach(),torch.zeros_like(nn[:,1]),-nn[:,0].detach()),-1);lamp=t([[1,1,2]])
    budget=PrincipledBudget.bundled('ggx-smith-loss-'+device+'-f64-v1',device=device,dtype=torch.float64);values=[];derivatives=[]
    # The independent moving-front certificate proves this selected visible
    # piece has no horizon-domain contribution locally. Other pieces stay open.
    for start in range(0,len(nodes),4):
        part=nodes[start:start+4];beta=part[:,:3]*iw;n=torch.stack([sum3(beta*nn[:,k]) for k in range(3)],-1);view=torch.stack([sum3(beta*vv[:,k]) for k in range(3)],-1)
        rgb=principled(n.contiguous(),view.contiguous(),lamp,albedo=t([.2,.4,.6]),roughness=t(1),f0=t([.04]*3),budget=budget)[:,0,:];nl=sum3(n*lamp[0])/(sum3(n*n).sqrt()*sum3(lamp[0]*lamp[0]).sqrt());integral=torch.stack([_sum_last(rgb[:,c]*nl*part[:,3]) for c in range(3)])
        d=torch.stack([_sum_last((torch.autograd.grad(integral[c],nn,retain_graph=c<2)[0]*direction).reshape(-1)) for c in range(3)]);values.append(integral.detach().cpu().tolist());derivatives.append(d.detach().cpu().tolist())
    actual=[[math.fsum(row[c] for row in rows) for c in range(3)] for rows in [values,derivatives]];checks=[]
    for name,vals in zip(['value','derivative'],actual):
        for chn,(a,pair) in enumerate(zip(vals,ref[name])):
            lo,hi=[Q(z.strip()) for z in pair.strip('[]').split(',')];assert hi-lo<=Q('2e-12')
            for endpoint in [lo,hi]:
                error=abs(Q(a)-endpoint);limit=Q('1e-10')+Q('1e-9')*abs(endpoint);checks.append(dict(quantity=name,channel=chn,error_over_limit=float(error/limit),passed=error<=limit))
    result=dict(device=device,piece=0,face=5739,nodes=len(nodes),value=actual[0],derivative=actual[1],bytes=struct.pack('<6d',*actual[0],*actual[1]).hex(),checks=checks,reference_sha256=digest,scope='One strictly-front visible original piece and fixed-domain normal tilt only; other pieces, moving geometry and horizons remain open.')
    if device=='cuda':result['gpu_peak_bytes']=torch.cuda.max_memory_allocated();assert result['gpu_peak_bytes']<=67108864
    dest=Path(os.environ['GFFX_FULL_NATIVE']);assert not dest.exists();dest.write_text(json.dumps(result,indent=2),encoding='utf8');record_property('original_normal_accuracy',json.dumps(result));assert all(q['passed'] for q in checks),checks
