"""Independent stdlib reference controls, not public gffx acceptance."""
import importlib
import math

import pytest


def module():
    return importlib.import_module('tools.shading_independent_evaluator')


def payload():
    return dict(roughness_nodes=[.001,1.],cosine_nodes=[0.,1.],losses=[[[.9,0.],[.7,.4]]]*2)


def test_independent_loss_curve_linear_value_mean_and_domain():
    m=module(); curve=m.LossCurve(payload())
    assert curve.loss(.4,.2,.25) == pytest.approx(.75*(.9-.2*.4)+.25*.4*.4,abs=1e-13)
    assert curve.mean(.2,.25) == pytest.approx(.75*(.9-.4/3)+.25*.8/3,abs=1e-13)
    for args in [(-.1,.2,.3),(.2,0.,.3),(.2,.2,2.)]:
        with pytest.raises(ValueError):curve.loss(*args)


def test_independent_log_curve_knots_and_integral():
    m=module(); cs=[0.,.1,.3,.6,1.]
    pairs=[[.9,0.]]+[[.7+.02*math.log(c),.2+.01*math.log(c)] for c in cs[1:]]
    curve=m.LossCurve(dict(roughness_nodes=[.001,1.],cosine_nodes=cs,losses=[pairs]*2))
    for c in [.1,.17,.3,.45,.6,.81,1.]:
        assert curve.loss(c,.2,.4) == pytest.approx(.6*(.7+.02*math.log(c))+.4*(.2+.01*math.log(c)),abs=1e-12)
    # Exact integral: linear horizon segment, then A+B log(c).
    a,b=.6*.7+.4*.2,.016
    at=.1; y0=.6*.9; y1=a+b*math.log(at)
    first=at*at*(y0+2*y1)/3
    primitive=lambda c:a*c*c+b*c*c*(math.log(c)-.5)
    assert curve.mean(.2,.4) == pytest.approx(first+primitive(1)-primitive(at),abs=1e-11)


def test_independent_material_closed_forms_reciprocity_and_mask():
    m=module(); curve=m.LossCurve(payload())
    args=dict(albedo=[.2,.4,.6],roughness=1.,f0=[.1,.3,.7],budget=curve)
    n=[0.,0.,1.]; v=[.6,0.,.8]; l=[0.,.8,.6]
    result=m.principled(n,v,l,**args)
    assert result == pytest.approx(m.principled(n,l,v,**args),abs=1e-13)
    vh=math.sqrt(.74)
    for i,f in enumerate(args['f0']):
        loss=lambda c:(1-f)*(.9-.2*c)+f*.4*c
        mean=(1-f)*(.9-.4/3)+f*.8/3
        expected=(f+(1-f)*(1-vh)**5)/(2*math.pi*1.4)+args['albedo'][i]*loss(.8)*loss(.6)/(math.pi*mean)
        assert result[i] == pytest.approx(expected,abs=1e-13)
    assert m.principled(n,n,[1.,0.,0.],**args) == [0.,0.,0.]
    assert m.classic(n,n,n,albedo=[.2,.4,.6],specular=.1,exponent=2.) == pytest.approx([a/math.pi+.1*10/(8*math.pi) for a in [.2,.4,.6]])


def test_independent_bilinear_sampler_preserves_orientation_and_repeat_seam():
    m=module(); texture=[[[.1]*3,[.5]*3],[[.3]*3,[.7]*3]]
    assert m.bilinear(texture,[.25,.25]) == pytest.approx([.1]*3)
    assert m.bilinear(texture,[.25,.75]) == pytest.approx([.3]*3)
    assert m.bilinear(texture,[.5,.5]) == pytest.approx([.4]*3)
    assert m.bilinear(texture,[0.,.25]) == m.bilinear(texture,[1.,.25])


def test_reference_imports_only_standard_library_and_limits_work():
    import ast
    from pathlib import Path
    m=module()
    tree=ast.parse(Path(m.__file__).read_text())
    names=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
    names += [a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names]
    assert set(names) <= {'math','bisect'}
    with pytest.raises(ValueError):m.LossCurve(dict(roughness_nodes=[.001,1.],cosine_nodes=[0.,1.],losses=[[[.1,0.],[.2,.4]]]*2))


def test_reference_direct_light_buffer_masks_background():
    m=module(); n=[0.,0.,1.]
    result=m.direct_buffer([n,n],[n,n],[True,False],n,[2.,3.,4.],
                           model='classic',albedo=[.2,.4,.6],specular=0.,exponent=2.)
    assert result[0] == pytest.approx([.4/math.pi,1.2/math.pi,2.4/math.pi])
    assert result[1] == [0.,0.,0.]
    with pytest.raises(ValueError):m.direct_buffer([n],[n],[False],n,[1.,1.,1.],model='classic',albedo=[.2]*3,specular=0.,exponent=2.)


def test_reference_nonconstant_roughness_matches_analytic_power_field():
    m=module(); rs=[.01,.02,.04,.08]; cs=[0.,.1,.2,.5,1.]
    rows=[]
    for r in rs:
        row=[[.4*r**.2,0.]]
        for c in cs[1:]:
            q=math.log(c)
            row.append([(.5+.01*q+.001*q*q)*r**.2,(.1+.02*q+.002*q*q)*r**.2])
        rows.append(row)
    curve=m.LossCurve(dict(roughness_nodes=rs,cosine_nodes=cs,losses=rows))
    r,c,f=.031,.37,.4; q=math.log(c)
    expected=r**.2*((1-f)*(.5+.01*q+.001*q*q)+f*(.1+.02*q+.002*q*q))
    assert curve.loss(c,r,f) == pytest.approx(expected,abs=1e-13)
    step=1e-6
    derivative=(curve.loss(c,r+step,f)-curve.loss(c,r-step,f))/(2*step)
    assert derivative == pytest.approx(.2*expected/r,rel=1e-8)


@pytest.mark.parametrize('bad', ['bool','nan','duplicate_axis','oversized_axis','wrong_pair'])
def test_reference_rejects_malformed_payloads(bad):
    m=module(); p=payload()
    if bad=='bool':p['losses'][0][0][0]=True
    elif bad=='nan':p['losses'][0][0][0]=float('nan')
    elif bad=='duplicate_axis':p['roughness_nodes']=[.1,.1]
    elif bad=='oversized_axis':p['cosine_nodes']=[0.]*514
    else:p['losses'][0][0]=[.9]
    with pytest.raises(ValueError):m.LossCurve(p)


def test_reference_payload_copy_and_nonconvergence_are_explicit():
    m=module(); p=payload(); curve=m.LossCurve(p)
    before=curve.loss(.4,.2,.3)
    p['losses'][0][0][0]=.1
    assert curve.loss(.4,.2,.3)==before
    with pytest.raises(ValueError,match='converge'):
        m._integrate(lambda x:math.exp(x),0.,1.,tolerance=1e-16,depth=0)


def test_native_sampler_matches_independent_reference_on_fixture_patterns():
    # Deliberately use the installed adapter, not a source-native fallback.
    import torch
    from gffx.torch import render
    from tools.synthetic_face_fixture import generate
    m=module(); scene=generate()
    uv=[[.25,.25],[.25,.75],[.31,.43],[0.,.2],[1.,.2]]
    for pixels in scene['textures'].values():
        texture=torch.tensor(pixels,dtype=torch.float64)
        pyramid,offsets=render.texture_pyramid(texture,levels=1)
        actual=render.texture(pyramid,offsets,32,32,torch.tensor(uv,dtype=torch.float64),channel_count=3,
                              filter=render.FILTER_BILINEAR,mip_filter=render.MIP_NEAREST,
                              wrap_u=render.WRAP_REPEAT,wrap_v=render.WRAP_REPEAT)
        expected=torch.tensor([m.bilinear(pixels,p) for p in uv],dtype=torch.float64)
        torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)
