"""Full original-material pixel inputs and public installed-wheel evaluation."""
import json,os,hashlib,time
from pathlib import Path
from dataclasses import replace
import torch


def original(device='cpu'):
    from tools.corpus_registry import load_entry
    from tools.deferred_flame import create_scene
    from gffx.torch.scene import Scene
    blob=load_entry(Path('D:/devspace/gffx/corpora'),'synthetic_flame_m1',allow_development=True)['files']['scene.json']
    assert hashlib.sha256(blob).hexdigest()=='293d0f876b91be535fa09b78423b2f5bde56c36d2462241cec3d4acc20967b39'
    s,c=create_scene(json.loads(blob),'principled',device=device)
    m=s.prims[1];m=replace(m,parameters=dict(m.parameters,f0=m.parameters['f0'].repeat(3)))
    return Scene([s.prims[0],m,s.prims[2]]),c,hashlib.sha256(blob).hexdigest()


def test_extract_complete_original_material(record_property):
    from gffx.torch._opaque_scene import _project,_normals
    from gffx.torch import render
    s,c,sha=original();packed,c,pixels,keep,clip=_project(s,c,render.CULL_BACK)
    p=Path(os.environ['GFFX_CURVED_INPUT']);assert hashlib.sha256(p.read_bytes()).hexdigest()==os.environ['GFFX_CURVED_INPUT_SHA']
    previous=json.loads(p.read_bytes());chosen=previous['selected'];ids=chosen['candidate_faces'];used=chosen['original_vertex_ids'];ox,oy=chosen['origin']
    local=pixels[used].clone();local[:,0]-=ox;local[:,1]-=oy
    assert local.tolist()==chosen['screen_vertices']
    n=_normals(packed);world=packed.vertices;rows=[]
    for face in ids:
        indices=packed.faces[face].long()
        rows.append(dict(face=face,vertices=indices.tolist(),corners=pixels[indices,:2].tolist(),clip_w=clip[indices,3].tolist(),normal=n[face].tolist(),view=(c.pose[0,:3,3]-world[indices]).tolist()))
    b=s.prims[1].parameters['budget']
    result=dict(scene_sha256=sha,selected=chosen,faces=rows,budget={k:v.tolist() for k,v in b._data.items()},material=dict(roughness=1.,f0=[.04]*3,albedo=[.2,.4,.6],direction=[1.,1.,2.]),normalization='Actual packed binary64 per-corner normal samples; subsequent interval integration certifies this explicit packed-input model.',parent_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    dest=Path(os.environ['GFFX_FULL_INPUT']);assert not dest.exists();dest.write_text(json.dumps(result,indent=2),encoding='utf-8')
    record_property('full_original_input',json.dumps(dict(path=str(dest),sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),candidate_faces=len(ids),visible_pieces=len(chosen['visible_parts']))))


def test_complete_original_material_pixel(record_property):
    from gffx.torch import render
    dev=os.environ.get('GFFX_CURVED_DEVICE','cpu');s,c,sha=original(dev);a=s.prims[1].parameters['albedo'];a.requires_grad_(os.environ.get('GFFX_FULL_FORWARD_ONLY')!='1')
    if dev=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter();out=render.opaque_window(s,c,(78,101,1,1),cull_mode=render.CULL_BACK)
    grad=torch.stack([torch.autograd.grad(out['colour'][0,0,0,i],a,retain_graph=i<2)[0] for i in range(3)]) if a.requires_grad else a.new_empty((0,3))
    if dev=='cuda':torch.cuda.synchronize();assert torch.cuda.max_memory_allocated()<=67108864
    result=dict(scene_sha256=sha,device=dev,seconds=time.perf_counter()-start,image={k:v.detach().cpu().tolist() for k,v in out.items()},albedo_jacobian=grad.cpu().tolist(),bytes={k:v.detach().cpu().contiguous().numpy().tobytes().hex() for k,v in {**out,'albedo_jacobian':grad}.items()},cuda_peak_bytes=torch.cuda.max_memory_allocated() if dev=='cuda' else None)
    dest=Path(os.environ['GFFX_FULL_NATIVE']);assert not dest.exists();dest.write_text(json.dumps(result,indent=2),encoding='utf-8')
    record_property('full_original_native',json.dumps(result))


def test_constant_material_reservation_retains_unknown_cut_fallback(record_property):
    from gffx.torch._opaque_scene import _extra_geometry_bytes
    from gffx.torch._deferred import _plan_workload
    from gffx.torch import render,scene
    from copy import copy
    s,c,_=original();s.prims[1].parameters['albedo'].requires_grad_()
    reserve=_extra_geometry_bytes(s,c);assert reserve==32640000
    proxy=copy(c);proxy.image_height=1;proxy.image_width=800
    plan=_plan_workload(s,proxy,1,67108864-1048576-65536-reserve-96-48,('colour','mask'))
    assert plan['backward'] and plan['estimated_bytes']<=67108864-1048576-65536-reserve-96-48
    m=s.prims[1];m=replace(m,parameters=dict(m.parameters,albedo=render.AttributeSource('normals')))
    assert _extra_geometry_bytes(scene.Scene([s.prims[0],m,s.prims[2]]),c)==46080000
    record_property('original_reservation',json.dumps(dict(geometry_reserve_bytes=reserve,unknown_cut_reserve_bytes=46080000,plan=plan)))


def test_first_original_material_piece_matches_independent_certificate(record_property):
    """One complete visible piece only; never aggregate a failed pixel as passed."""
    from fractions import Fraction as Q
    from decimal import Decimal,localcontext
    import math,struct
    from gffx.torch.render import opaque_quadrature
    from gffx.torch.material import PrincipledBudget,principled
    from gffx.torch._reference_channels import sum3,_sum_last
    dev=os.environ.get('GFFX_CURVED_DEVICE','cpu');t=lambda x:torch.tensor(x,dtype=torch.float64,device=dev)
    source=Path(os.environ['GFFX_FULL_INPUT']);assert hashlib.sha256(source.read_bytes()).hexdigest()=='a15bea40cb5cdead7994d828b57f5dcfac9f44fab0292a06e7d0a7b08719f73a'
    data=json.loads(source.read_bytes());ch=data['selected'];refs=Path(os.environ['GFFX_FIRST_PIECE_REFERENCE']);assert hashlib.sha256(refs.read_bytes()).hexdigest()==os.environ['GFFX_FIRST_PIECE_REFERENCE_SHA'];r=json.loads(refs.read_bytes())[0]
    assert r['piece']==0 and r['face']==5739 and not r['curved']
    fi=ch['candidate_faces'].index(5739);assert sum(p['face']==fi for p in ch['visible_parts'])==1
    v=t(ch['screen_vertices']);f=torch.tensor(ch['faces'],dtype=torch.int32,device=dev);curves=[]
    for row in data['faces']:
        nn=t(row['normal']);vv=t(row['view']);iw=1/t(row['clip_w']);dot=lambda a,b:sum3(a*b)
        co=[dot(nn[i],vv[i])*iw[i]*iw[i] for i in range(3)]
        co.extend((dot(nn[i],vv[j])+dot(nn[j],vv[i]))*iw[i]*iw[j] for i,j in ((0,1),(0,2),(1,2)))
        curves.append(torch.stack(co))
    ids,q=opaque_quadrature(v,f,v.new_empty((len(f),3,0)),1,1,curves=torch.stack(curves));nodes=q[0][ids[0]==fi]
    row=data['faces'][fi];nn=t(row['normal']);vv=t(row['view']);iw=1/t(row['clip_w']);albedo=t([.2,.4,.6]).requires_grad_();lamp=t([[1,1,2]])
    budget=PrincipledBudget.bundled('ggx-smith-loss-'+dev+'-f64-v1',device=dev,dtype=torch.float64);pieces=[];gradients=[]
    for start in range(0,len(nodes),4):
        part=nodes[start:start+4];beta=part[:,:3]*iw;n=torch.stack([sum3(beta*nn[:,k]) for k in range(3)],-1);view=torch.stack([sum3(beta*vv[:,k]) for k in range(3)],-1)
        rgb=principled(n.contiguous(),view.contiguous(),lamp,albedo=albedo,roughness=t(1),f0=t([.04]*3),budget=budget)[:,0,:]
        nl=sum3(n*lamp[0])/(sum3(n*n).sqrt()*sum3(lamp[0]*lamp[0]).sqrt())
        value=torch.stack([_sum_last(rgb[:,c]*nl*part[:,3]) for c in range(3)])
        gradients.append(torch.stack([torch.autograd.grad(value[c],albedo,retain_graph=c<2)[0] for c in range(3)]).cpu().tolist());pieces.append(value.cpu().tolist())
    values=[math.fsum(row[c] for row in pieces) for c in range(3)];jac=[[math.fsum(row[i][j] for row in gradients) for j in range(3)] for i in range(3)]
    with localcontext() as context:
        context.prec=90
        spec,matte=[tuple(map(Decimal,z.strip('[]').split(','))) for z in r['value']]
        assert max(b-a for a,b in (spec,matte))<=Decimal('1e-12')*2
        bounds=[(spec[0]+Decimal.from_float(a)*matte[0],spec[1]+Decimal.from_float(a)*matte[1]) for a in (.2,.4,.6)]
        errors=[]
        for actual,(lo,hi) in zip(values+[jac[i][i] for i in range(3)],bounds+[matte]*3):
            error=max(lo-Decimal.from_float(actual),Decimal.from_float(actual)-hi,Decimal(0));limit=Decimal('1e-10')+Decimal('1e-9')*abs((lo+hi)/2);errors.append(str(error));assert error<=limit,(actual,str(lo),str(hi),str(error),str(limit))
    assert all(jac[i][j]==0 for i in range(3) for j in range(3) if i!=j)
    record_property('first_original_piece',json.dumps(dict(device=dev,face=5739,nodes=len(nodes),colour=values,albedo_jacobian=jac,distances=errors,reference_sha256=hashlib.sha256(refs.read_bytes()).hexdigest(),bytes=struct.pack('<12d',*values,*(z for row in jac for z in row)).hex(),scope='Only first exact visible piece of failed complete-pixel attempt;all remaining pieces and complete-frame/showcase gates remain open.')))
