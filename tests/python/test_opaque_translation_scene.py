"""Public scene/camera transport, full original material, independent edge bounds."""
import os,json,time,hashlib
from pathlib import Path
from fractions import Fraction as Q
from decimal import Decimal as D,localcontext
import pytest,torch

@pytest.mark.parametrize('axis',['x','y'])
@pytest.mark.parametrize('domain',['constant','vertex'])
def test_public_scene_camera_translation_has_independent_boundary_derivative(domain,axis,record_property):
    from gffx.torch import scene,material,light,render
    from gffx.torch.camera import Camera
    from tools.opaque_shading_intervals import I,Jet
    from tools.opaque_principled_reference import PreparedPrincipled
    device=os.environ.get('GFFX_KNOT_DEVICE','cpu');start=time.perf_counter();t=lambda x:torch.tensor(x,dtype=torch.float64,device=device)
    p=Path(os.environ['GFFX_FULL_INPUT']);assert hashlib.sha256(p.read_bytes()).hexdigest()=='a15bea40cb5cdead7994d828b57f5dcfac9f44fab0292a06e7d0a7b08719f73a';data=json.loads(p.read_bytes())['budget'];curve=PreparedPrincipled(data,1.)
    # Principal-point motion translates the entire screen field. Its derivative
    # is the left-edge integral minus the right-edge integral. On the right edge
    # N.V<0, so it is zero. The left edge uses an outward enclosure of ALL y,
    # not a centre sample: fy=2^40 makes that enclosure narrower than 2e-12.
    n=[Jet(1),Jet(0),Jet(0)];v=[Jet(Q(1,100)),Jet(I(I(-Q(1,2**40)).lo,I(Q(1,2**40)).hi)),Jet(2)];l=[Jet(1),Jet(1),Jet(2)]
    if axis=='y':n=[Jet(0),Jet(-1),Jet(0)];v=[v[1],Jet(-Q(1,100)),Jet(2)];l=[Jet(1),Jet(-1),Jet(2)]
    ref=[a[0,0] for a in curve.radiance(n,v,l,[Jet(a) for a in (.2,.4,.6)],[Jet(.04)]*3)]
    assert all(a.width<D('2e-12') for a in ref)
    theta=t(0).requires_grad_();vertices=t([[-.02,2**-39,-2],[.02,2**-39,-2],[.02,-2**-39,-2],[-.02,-2**-39,-2]]);faces=torch.tensor([[0,1,2],[0,2,3]],device=device,dtype=torch.int32);normal=t([1,0,0])
    if axis=='y':vertices=t([[-2**-39,.02,-2],[2**-39,.02,-2],[2**-39,-.02,-2],[-2**-39,-.02,-2]]);normal=t([0,-1,0])
    attr=scene.Attribute(normal.repeat(4,1) if domain=='vertex' else normal,domain)
    budget=material.PrincipledBudget.bundled('ggx-smith-loss-'+device+'-f64-v1',device=device,dtype=torch.float64)
    mat=scene.Material('m',material.principled,dict(albedo=t([.2,.4,.6]),f0=t([.04]*3),roughness=t(1),budget=budget));lamp=scene.Light('l',light.directional,dict(directions=t([[1,1,2]] if axis=='x' else [[1,-1,2]]),irradiance=t([[1,1,1]])))
    s=scene.Scene([scene.Mesh('mesh',vertices,faces,attributes={'normals':attr},material='m'),mat,lamp]);c=Camera.from_intrinsics(t(100),t(2**40),t(.5)+theta,t(.5),1,1,.1,10)
    if axis=='y':c=Camera.from_intrinsics(t(2**40),t(100),t(.5),t(.5)+theta,1,1,.1,10)
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    out=render.render(s,c,coverage='opaque',cull_mode=render.CULL_NONE);rgb=out['colour'][0,0,0];grad=[torch.autograd.grad(rgb[i],theta,retain_graph=True)[0].item() for i in range(3)];checks=[]
    with localcontext() as ctx:
        ctx.prec=90
        for got,bound in zip(grad,ref):
            for endpoint in (bound.lo,bound.hi):
                error=abs(D.from_float(got)-endpoint);limit=D('1e-10')+D('1e-9')*abs(endpoint);checks.append(dict(error_over_limit=float(error/limit),passed=error<=limit))
    row=dict(domain=domain,axis=axis,device=device,value=rgb.detach().cpu().tolist(),derivative=grad,reference=ref,checks=checks,seconds=time.perf_counter()-start,scope='Full public renderer on a controlled plane; exact camera-translation boundary identity. This is not arbitrary original mesh motion.')
    if device=='cuda':row['gpu_peak_bytes']=torch.cuda.max_memory_allocated();assert row['gpu_peak_bytes']<67108864
    with torch.no_grad():plain=render.render(s,c,coverage='opaque',cull_mode=render.CULL_NONE)
    assert all(torch.equal(out[k].detach().contiguous().view(torch.uint8),plain[k].contiguous().view(torch.uint8)) for k in out)
    path=Path(os.environ['GFFX_FULL_NATIVE']);rows=json.loads(path.read_bytes()) if path.exists() else [];rows.append(row);path.write_text(json.dumps(rows,indent=2,default=str),encoding='utf8');record_property('scene_translation',json.dumps(row,default=str));assert abs(out['mask'].item()-1)<1e-12
    assert all(x['passed'] for x in checks),row
