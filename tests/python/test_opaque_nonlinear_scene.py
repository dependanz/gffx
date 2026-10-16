"""Installed nonlinear opaque scene integration with independent pixel evidence."""
import json,torch,pytest
from fractions import Fraction as Q
from decimal import Decimal as D,localcontext
from gffx.torch import scene,render,material,light
from gffx.torch.camera import Camera

def fixture(*,texture=False,rough=False,diagonal=False,device="cpu"):
    t=lambda x:torch.tensor(x,dtype=torch.float64,device=device)
    budget=material.PrincipledBudget.bundled('ggx-smith-loss-'+device+'-f64-v1',device=device,dtype=torch.float64)
    v=t([[-1/128,1/128,-2.],[1/128,1/128,-2.],[1/128,-1/128,-2.],[-1/128,-1/128,-2.]])
    f=torch.tensor([[0,1,3],[1,2,3]] if diagonal else [[0,1,2],[0,2,3]],dtype=torch.int32,device=device)
    normals=t([[.01,.01,1.]]*4);uv=t([[0.,0.],[1.,0.],[1.,1.],[0.,1.]])
    attrs={'normals':scene.Attribute(normals,'vertex'),'uv':scene.Attribute(uv,'vertex')};albedo=t([.2,.4,.6]);r=t(.827)
    tex=t([[[.2,.1,.3],[.8,.3,.2]],[[.4,.7,.5],[.6,.5,.9]]])
    if texture:
        pyramid,offsets=render.texture_pyramid(tex,levels=1);albedo=render.TextureSource(pyramid,offsets,2,2,3,wrap_u=render.WRAP_CLAMP,wrap_v=render.WRAP_CLAMP)
    if rough:
        attrs['roughness']=scene.Attribute(t([[.826],[.828],[.828],[.826]]),'vertex');r=render.AttributeSource('roughness',scalar=True)
    mat=scene.Material('m',material.principled,dict(albedo=albedo,roughness=r,f0=t([.04,.05,.08]),budget=budget))
    lamp=scene.Light('l',light.directional,dict(directions=t([[.1,.1,1.]]),irradiance=t([[1.,1.,1.]])))
    model=scene.Scene([scene.Mesh('q',v,f,attributes=attrs,material='m'),mat,lamp]);camera=Camera.from_intrinsics(t(128.),t(128.),t(.5),t(.5),1,1,.1,10.)
    return model,camera,v,f,budget

def test_opaque_scene_integrates_nonlinear_principled_against_independent_pixel(record_property):
    from tools.opaque_principled_reference import PreparedPrincipled,perspective_field,integrate_pixel
    model,camera,v,f,b=fixture();out=render.render(model,camera,coverage='opaque',cull_mode=render.CULL_NONE)
    data={k:a.tolist() for k,a in b._data.items()};curve=PreparedPrincipled(data,.827);pixels=[(0,0,0),(1,0,0),(1,1,0),(0,1,0)];fields=[]
    for face in f.tolist():
        fields.append(perspective_field(curve,[pixels[i] for i in face],inverse_w=[.5]*3,normals=[[.01,.01,1.]]*3,views=[(-v[i]).tolist() for i in face],albedo=[[.2,.4,.6]]*3,f0=[.04,.05,.08],direction=[.1,.1,1.]))
    reference=integrate_pixel(pixels,f.tolist(),fields)
    for a,w in zip(reference['value'][0][0],out['colour'][0,0,0].tolist()):
        with localcontext() as c:
            c.prec=90;w=D.from_float(w);assert max(a.lo-w,w-a.hi,D(0))<=D('1e-10')+D('1e-9')*abs((a.lo+a.hi)/2)
    assert abs(out['mask'].item()-1)<1e-12
    record_property('scene_reference',json.dumps(dict(reference=reference,installed={k:q.tolist() for k,q in out.items()}),default=str))

@pytest.mark.parametrize('texture,rough',[(False,True),(True,False),(True,True)])
def test_opaque_scene_no_interior_transparency_across_diagonals(texture,rough):
    a,c,*_=fixture(texture=texture,rough=rough);b,d,*_=fixture(texture=texture,rough=rough,diagonal=True)
    aa=render.render(a,c,coverage='opaque',cull_mode=render.CULL_NONE);bb=render.render(b,d,coverage='opaque',cull_mode=render.CULL_NONE)
    torch.testing.assert_close(aa['colour'],bb['colour'],atol=1e-10,rtol=1e-9)
    assert abs(aa['mask'].item()-1)<1e-12 and abs(bb['mask'].item()-1)<1e-12


def test_opaque_scene_camera_material_and_texture_gradients_are_not_detached():
    model,camera,v,f,b=fixture();theta=torch.tensor(0.,dtype=torch.float64,requires_grad=True)
    def evaluate(theta):
        t=v.new_tensor;vv=v*torch.stack([1+theta*.1,theta*0+1,theta*0+1]);uv=t([[0,0],[1,0],[1,1],[0,1]])+theta*.03
        tex=t([[[.2,.1,.3],[.8,.3,.2]],[[.4,.7,.5],[.6,.5,.9]]])+theta*.02;pyr,off=render.texture_pyramid(tex,levels=1)
        m=scene.Material('m',material.principled,dict(albedo=render.TextureSource(pyr,off,2,2,3,wrap_u=render.WRAP_CLAMP,wrap_v=render.WRAP_CLAMP),roughness=t(.827)+theta*.001,f0=t([.04,.05,.08])+theta*.002,budget=b))
        lamp=scene.Light('l',light.directional,dict(directions=t([[.1,.1,1.]])+theta*.001,irradiance=t([[1,1,1]])+theta*.01))
        s=scene.Scene([scene.Mesh('q',vv,f,material='m',attributes={'uv':scene.Attribute(uv,'vertex'),'normals':scene.Attribute(t([[.01,.01,1.]]*4)+theta*.001,'vertex')}),m,lamp])
        cam=Camera.from_intrinsics(t(128.)+theta*.2,t(128.),t(.48),t(.49),1,1,.1,10.)
        out=render.render(s,cam,coverage='opaque',cull_mode=render.CULL_NONE,background=t([.1,.1,.1]));return (out['colour']*t([.3,-.2,.7])).sum()
    actual=torch.autograd.grad(evaluate(theta),theta)[0]
    assert abs(actual.item())>1e-6
    for epsilon in (1e-4,5e-5):
        expected=(evaluate(theta.detach()+epsilon)-evaluate(theta.detach()-epsilon))/(2*epsilon)
        torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-4)

@pytest.mark.parametrize('index',range(8))
def test_opaque_scene_independent_parameter_directions(index,record_property):
    model,camera,v,f,b=fixture();theta=torch.zeros(8,dtype=torch.float64,requires_grad=True)
    def evaluate(theta):
        t=v.new_tensor;vv=v*torch.stack([1+theta[0]*.1,theta[0]*0+1,theta[0]*0+1]);uv=t([[0,0],[1,0],[1,1],[0,1]])+theta[1]*.03
        tex=t([[[.2,.1,.3],[.8,.3,.2]],[[.4,.7,.5],[.6,.5,.9]]])+theta[2]*.02;pyr,off=render.texture_pyramid(tex,levels=1)
        m=scene.Material('m',material.principled,dict(albedo=render.TextureSource(pyr,off,2,2,3,wrap_u=render.WRAP_CLAMP,wrap_v=render.WRAP_CLAMP),roughness=t(.827)+theta[3]*.001,f0=t([.04,.05,.08])+theta[4]*.002,budget=b))
        lamp=scene.Light('l',light.directional,dict(directions=t([[.1,.1,1.]])+theta[5]*.001,irradiance=t([[1,1,1]])+theta[5]*.01))
        s=scene.Scene([scene.Mesh('q',vv,f,material='m',attributes={'uv':scene.Attribute(uv,'vertex'),'normals':scene.Attribute(t([[.01,.01,1.]]*4)+theta[6]*.001,'vertex')}),m,lamp])
        cam=Camera.from_intrinsics(t(128.)+theta[7]*.2,t(128.),t(.48),t(.49),1,1,.1,10.)
        out=render.render(s,cam,coverage='opaque',cull_mode=render.CULL_NONE,background=t([.1,.1,.1]));return (out['colour']*t([.3,-.2,.7])).sum()
    actual=torch.autograd.grad(evaluate(theta),theta)[0][index]
    assert abs(actual.item())>1e-9
    for epsilon in (1e-4,5e-5):
        direction=torch.zeros_like(theta);direction[index]=epsilon
        expected=(evaluate(theta.detach()+direction)-evaluate(theta.detach()-direction))/(2*epsilon)
        torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-4)

    record_property('partial_gradient',json.dumps(dict(parameter=index,gradient=actual.item())))


def test_opaque_scene_saves_texture_roughness_demonstration(record_property):
    import time
    model,old,*_=fixture(texture=True,rough=True)
    t=old.fx.new_tensor;camera=Camera.from_intrinsics(t(512.),t(512.),t(2.),t(2.),4,4,.1,10.)
    start=time.perf_counter()
    out=render.render(model,camera,coverage='opaque',cull_mode=render.CULL_NONE)
    assert out['colour'].shape==(1,4,4,3) and torch.isfinite(out['colour']).all()
    torch.testing.assert_close(out['mask'],torch.ones_like(out['mask']),atol=1e-12,rtol=0)
    record_property('scene_demo',json.dumps(dict(image=out['colour'][0].tolist(),mask=out['mask'][0].tolist(),seconds=time.perf_counter()-start,description='Actual 4x4 complete scene pixels: bilinear2x2 texture,roughness0.826..0.828,full Principled,directional light. Magnify nearest-neighbour only; not final showcase.')))
