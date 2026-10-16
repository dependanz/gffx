"""Original visible patch and a complete production curved-horizon pixel."""
import json,os,math,hashlib,time
from pathlib import Path
from decimal import Decimal,localcontext
import torch

def fixture(size=1,device='cpu'):
    from gffx.torch import scene,material,light
    from gffx.torch.camera import Camera
    t=lambda x:torch.tensor(x,dtype=torch.float64,device=device)
    v=t([[-.008,.008,-2],[.008,.008,-2],[-.008,-.008,-2]])
    n=t([[1,0,-.005],[1.001,0,.007],[.999,0,.001]]).requires_grad_()
    f=torch.tensor([[0,1,2]],dtype=torch.int32,device=device)
    budget=material.PrincipledBudget.bundled('ggx-smith-loss-'+device+'-f64-v1',device=device,dtype=torch.float64)
    mat=scene.Material('m',material.principled,dict(albedo=t([0,0,0]),f0=t([1,1,1]),roughness=t(1),budget=budget))
    lamp=scene.Light('l',light.directional,dict(directions=t([[1,0,0]]),irradiance=t([[1,1,1]])))
    model=scene.Scene([scene.Mesh('mesh',v,f,attributes={'normals':scene.Attribute(n,'vertex')},material='m'),mat,lamp])
    camera=Camera.from_intrinsics(t(100.*size),t(100.*size),t(.5*size),t(.5*size),size,size,.1,10)
    return model,camera,n

def bounds(got,want):
    if isinstance(want,str):lo,hi=map(Decimal,want.strip('[]').split(','))
    else:lo,hi=want.lo,want.hi
    with localcontext() as ctx:
        ctx.prec=90;g=Decimal.from_float(got);assert max(lo-g,g-hi,Decimal(0))<=Decimal('1e-10')+Decimal('1e-9')*abs((lo+hi)/2),(got,str(want))

def test_original_complete_visibility_and_curved_patch_matches_saved_reference(record_property):
    from gffx.torch.render import opaque_quadrature
    from gffx.torch.material import PrincipledBudget,principled
    from gffx.torch._reference_channels import sum3,multiply,_sum_last
    dev=os.environ.get('GFFX_CURVED_DEVICE','cpu');t=lambda x:torch.tensor(x,dtype=torch.float64,device=dev)
    path=Path(os.environ['GFFX_CURVED_INPUT']);assert hashlib.sha256(path.read_bytes()).hexdigest()==os.environ['GFFX_CURVED_INPUT_SHA']
    saved=json.loads(path.read_bytes());ch=saved['selected'];row=next(x for x in saved['faces'] if x['face']==ch['face']);fi=ch['candidate_faces'].index(ch['face'])
    reference_path=Path(os.environ['GFFX_CURVED_REFERENCE']);assert hashlib.sha256(reference_path.read_bytes()).hexdigest()==os.environ['GFFX_CURVED_REFERENCE_SHA'];ref=json.loads(reference_path.read_bytes())
    v=t(ch['screen_vertices']);f=torch.tensor(ch['faces'],dtype=torch.int32,device=dev);normal=t(row['normal']).requires_grad_();view=t(row['view']);iw=1/t(row['clip_w'])
    dot=lambda a,b:sum3(a*b)
    co=[dot(normal[i],view[i])*iw[i]*iw[i] for i in range(3)]
    co += [(dot(normal[i],view[j])+dot(normal[j],view[i]))*iw[i]*iw[j] for i,j in ((0,1),(0,2),(1,2))]
    curves=v.new_zeros((len(f),6));curves[fi]=torch.stack(co)
    ids,q=opaque_quadrature(v,f,v.new_empty((len(f),3,0)),1,1,curves=curves);nodes=q[0][ids[0]==fi]
    budget=PrincipledBudget.bundled('ggx-smith-loss-'+dev+'-f64-v1',device=dev,dtype=torch.float64)
    values=[];flux=[];front=[];front_flux=[]
    for start in range(0,len(nodes),4):
        piece=nodes[start:start+4];beta=piece[:,:3]*iw;nn=torch.stack([sum3(beta*normal[:,k]) for k in range(3)],-1);vv=torch.stack([sum3(beta*view[:,k]) for k in range(3)],-1);lamp=t([[1,1,2]])
        rgb=principled(nn.contiguous(),vv.contiguous(),lamp,albedo=t([0,0,0]),roughness=t(1),f0=t([1,1,1]),budget=budget)[:,0,0]
        nl=dot(nn,lamp[0])/(dot(nn,nn).sqrt()*dot(lamp[0],lamp[0]).sqrt())
        value=_sum_last((rgb*nl)*piece[:,3]);area=_sum_last(piece[:,3]*(dot(nn,vv)>0))
        gn,=torch.autograd.grad(value,normal,retain_graph=True);ga,=torch.autograd.grad(area,normal,retain_graph=True)
        values.append(value.item());flux.append(math.fsum(gn[:,2].tolist()));front.append(area.item());front_flux.append(math.fsum(ga[:,2].tolist()))
    actual={'value':[math.fsum(values),math.fsum(front)],'derivative':[math.fsum(flux),math.fsum(front_flux)]}
    for key in actual:
        for got,want in zip(actual[key],ref['result'][key]):bounds(got,want)
    from fractions import Fraction as Q
    area=Q(0)
    for part in ch['visible_parts']:
        a,b,c=[[Q(z) for z in p] for p in part['triangle']];area+=abs((b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]))/2
    assert abs(q[...,3].sum().item()-float(area))<1e-12
    record_property('original_curved',json.dumps(dict(actual=actual,device=dev,samples=int((ids>=0).sum()),face_samples=len(nodes),complete_mask=float(area),reference=str(reference_path),scope='Complete depth visibility/mask,original face5743 radiance contribution with declared F0=roughness1 control; other-face RGB and original material remain unqualified.')))

def test_complete_production_curved_pixel(record_property):
    from gffx.torch import render
    dev=os.environ.get('GFFX_CURVED_DEVICE','cpu');s,c,n=fixture(device=dev)
    p=Path(os.environ['GFFX_CURVED_PIXEL_REFERENCE']);ref=json.loads(p.read_bytes())
    out=render.render(s,c,coverage='opaque',cull_mode=render.CULL_NONE)
    gn,=torch.autograd.grad(out['colour'][0,0,0,0],n);motion=math.fsum(gn[:,2].tolist())
    for value in out['colour'].reshape(-1).tolist():bounds(value,ref['result']['value'][0])
    bounds(motion,ref['result']['derivative'][0]);assert abs(out['mask'].item()-.32)<1e-12
    record_property('complete_curved_pixel',json.dumps(dict(device=dev,colour=out['colour'].tolist(),mask=out['mask'].item(),normal_z_derivative=motion,normal_vjp=gn.tolist(),reference_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),scope='Complete production single-triangle curved-horizon Principled pixel,roughness=F0=1,albedo0.')))
