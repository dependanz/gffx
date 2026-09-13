"""Reference-only public sampler for new pixels; frozen assessment remains unchanged."""
import hashlib
import math
from pathlib import Path
import time
import torch
from tools import albedo_sampling as assessed
from tools import albedo_environment as base
from tools.material_loss_cost import check_deadline

ASSESSMENT_SOURCE='d93a3c90b5dd916a639eaba1ef45920bea82c377af759a54294d43516987021f'
SCREEN_SHA256='6416cb6f53e6d841d80690baaa83f98b250f3afd978363dbee6e3254c2ea6a21'
groups=assessed.groups

def route(model,environment):
    if (model,environment) not in base.schedule():raise ValueError('unsupported image profile')
    return 1 if (model,environment)==('principled','affine') else 8

def validate_input(row,model,environment):
    route(model,environment)
    ref=row['reference'];a=base.colour(row['albedo']);point=row['point']
    if (ref['model']!=model or ref['environment']!=environment or ref['albedo']!=list(a)
        or ref['normal']!=list(base.old.scalar._normal(point['normal']))
        or ref['view']!=list(base.old.scalar._normal(point['view']))):raise ValueError('reference input identity')
    base.validate_reference(ref)

def sources():
    result=assessed.sources()
    if result['albedo_sampling.py']!=ASSESSMENT_SOURCE:raise ValueError('frozen assessment changed')
    result[Path(__file__).name]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return result

def sample(rows,*,model,environment,budget,deadline):
    from gffx.torch import material,light
    check_deadline(deadline);width=route(model,environment);plan=groups(rows,width);b=base.bands(environment)
    for row in rows:validate_input(row,model,environment)
    started=time.perf_counter();estimates=[None]*len(rows);calls=0;directions=0;max_pairs=0;max_points=0
    with torch.no_grad():
        for group in plan:
            ids=group['indices'];size=len(ids);chunk=group['chunk']
            selected=[rows[i] for i in ids]
            n=torch.tensor([r['point']['normal'] for r in selected],dtype=torch.float64)
            v=torch.tensor([r['point']['view'] for r in selected],dtype=torch.float64)
            a=torch.tensor([r['albedo'] for r in selected],dtype=torch.float64)
            bases=[base.old.frame(r['point']['normal'],r['point']['view'])[1] for r in selected];values=[]
            for split in (.5,.25):
                q,w=base.old.quadrature(group['order'],split);total=torch.zeros((size,3),dtype=torch.float64)
                for start in range(0,len(q),chunk):
                    check_deadline(deadline)
                    d=torch.stack([q[start:start+chunk]@basis for basis in bases]);length=d.shape[1]
                    args=dict(albedo=a)
                    if model=='classic':
                        brdf=material.classic(n,v,d,specular=torch.tensor(.1,dtype=torch.float64),exponent=torch.tensor(2.,dtype=torch.float64),**args)
                    else:
                        brdf=material.principled(n,v,d,roughness=torch.tensor(1.,dtype=torch.float64),f0=torch.tensor(.04,dtype=torch.float64),budget=budget,**args)
                    packet=dict(measure='solid_angle',direction=d,radiance=base.old.field(d.reshape(-1,3),b).reshape(size,length,3),
                        valid=torch.ones((size,length),dtype=torch.bool),solid_angle=w[start:start+chunk][None].expand(size,-1))
                    total+=light.integrate(n,brdf,packet);calls+=1;directions+=size*length
                    max_pairs=max(max_pairs,size*length);max_points=max(max_points,size)
                values.append(total)
            for j,i in enumerate(ids):estimates[i]=[x[j].tolist() for x in values]
    check_deadline(deadline)
    result=dict(estimates=estimates,groups=plan,width=width,calls=calls,directions=directions,
        max_pairs=max_pairs,max_points=max_points,seconds=time.perf_counter()-started)
    validate(result,rows,model,environment);return result

def validate(result,rows,model,environment):
    if result['width']!=route(model,environment):raise ValueError('unqualified sampling route')
    for row in rows:validate_input(row,model,environment)
    plan=groups(rows,result['width']);calls=0;directions=0;max_pairs=0;max_points=0
    for group in plan:
        k=len(group['indices']);q=4*group['order']**2;chunk=group['chunk']
        calls+=2*math.ceil(q/chunk);directions+=2*k*q
        max_pairs=max(max_pairs,k*min(q,chunk));max_points=max(max_points,k)
    if (result['groups']!=plan or result['calls']!=calls or result['directions']!=directions
        or result['max_pairs']!=max_pairs or result['max_points']!=max_points
        or len(result['estimates'])!=len(rows) or not 0<result['seconds']<120):raise ValueError('sampling work identity')
    for values,row in zip(result['estimates'],rows):
        if len(values)!=2 or any(len(v)!=3 for v in values):raise ValueError('sampling RGB shape')
        for estimate in values:
            for actual,reference in zip(estimate,row['reference']['radiance']):
                if not math.isfinite(actual) or abs(actual-reference)>1e-10+1e-8*abs(reference):raise ValueError('sampling reference tolerance')
