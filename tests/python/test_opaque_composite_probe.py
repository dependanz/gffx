"""Sampling diagnosis only: preserve failed endpoint checks; no production admission."""
import pytest,torch
@pytest.mark.parametrize('panels,order,power',[(1,16,2),(2,8,2),(4,4,2),(8,2,2),(1,16,3),(2,8,3),(4,4,3),(4,4,4)])
def test_composite_scene_radial_diagnostic(panels,order,power,monkeypatch,record_property):
    from gffx.torch import render
    from tools.opaque_composite_probe import remap
    from tests.python.test_opaque_translation_scene import test_public_scene_camera_translation_has_independent_boundary_derivative as check
    original=render.opaque_quadrature
    def sample(*args,**kwargs):
        ids,q=original(*args,**kwargs);valid=ids[0]>=0;assert int(valid.sum())==768
        values=remap(q[0,valid],panels,order,power);result=q.clone();result[0,valid]=values
        return ids,result
    monkeypatch.setattr(render,'opaque_quadrature',sample)
    record_property('diagnostic_rule',f'{panels} panels x {order}, power{power}')
    check('vertex',record_property)

@pytest.mark.parametrize('panels,order,power',[(3,9,2),(1,27,2),(2,13,2),(5,5,2),(4,6,2),(3,9,3)])
def test_asymmetric_scene_radial_diagnostic(panels,order,power,monkeypatch,record_property):
    from gffx.torch import render
    from tools.opaque_composite_probe import remap
    from tests.python.test_opaque_translation_scene import test_public_scene_camera_translation_has_independent_boundary_derivative as check
    original=render.opaque_quadrature
    def sample(*args,**kwargs):
        ids,q=original(*args,**kwargs);valid=ids[0]>=0;assert int(valid.sum())==768
        beta=q[0,valid].reshape(2,2,12,16,4)[:,0,0,0,:3];c=kwargs['curves'];a,b,d=beta.unbind(-1)
        sign=a*a*c[:,0]+b*b*c[:,1]+d*d*c[:,2]+a*b*c[:,3]+a*d*c[:,4]+b*d*c[:,5]
        positive=[0 if x>0 else 1 for x in sign.tolist()]
        result=q.clone();result[0,valid]=remap(q[0,valid],panels,order,power,positive)
        return ids,result
    monkeypatch.setattr(render,'opaque_quadrature',sample)
    record_property('diagnostic_rule',f'positive:{panels}x{order}, negative:{32-panels*order}, power{power}')
    check('vertex',record_property)

@pytest.mark.parametrize('panels,order,power,grade',[(3,9,2,[0,.01,.1,1]),(3,9,2,[0,.001,.03,1]),(3,9,2,[0,.0001,.01,1]),(4,6,2,[0,.001,.01,.1,1]),(4,6,2,[0,.0001,.01,.1,1]),(3,9,1,[0,.000001,.001,1])])
def test_graded_scene_radial_diagnostic(panels,order,power,grade,monkeypatch,record_property):
    from gffx.torch import render
    from tools.opaque_composite_probe import remap
    from tests.python.test_opaque_translation_scene import test_public_scene_camera_translation_has_independent_boundary_derivative as check
    original=render.opaque_quadrature
    def sample(*args,**kwargs):
        ids,q=original(*args,**kwargs);valid=ids[0]>=0;assert int(valid.sum())==768
        beta=q[0,valid].reshape(2,2,12,16,4)[:,0,0,0,:3];c=kwargs['curves'];a,b,d=beta.unbind(-1)
        sign=a*a*c[:,0]+b*b*c[:,1]+d*d*c[:,2]+a*b*c[:,3]+a*d*c[:,4]+b*d*c[:,5]
        positive=[0 if x>0 else 1 for x in sign.tolist()]
        result=q.clone();result[0,valid]=remap(q[0,valid],panels,order,power,positive,grade)
        return ids,result
    monkeypatch.setattr(render,'opaque_quadrature',sample)
    record_property('diagnostic_rule',f'positive:{panels}x{order}, negative:{32-panels*order}, power{power}, grade{grade}')
    check('vertex',record_property)

# Retired candidate diagnostic, pinned so default qualification cannot use it.
import os
pytestmark=pytest.mark.skipif(os.environ.get("GFFX_OPAQUE_WHEEL_SHA")!="61b05ea6562fe6a2f625f266f47a8a4c4610611c24b88faf5ed74660ad674313",reason="requires retired opaque124 candidate1 installed wheel")
