"""Skip only structurally zero angular corrections; retain all live derivatives."""
import pytest
from tools.opaque_shading_intervals import I,Jet
from tools.opaque_peano_material import MaterialCorrection,segmented
from tests.packaging.test_opaque_localized_peano import material_fixture

@pytest.mark.parametrize('order',[0,3])
@pytest.mark.parametrize('view_live,light_live',[(False,False),(False,True),(True,False),(True,True)])
def test_only_needed_smooth_branches_are_evaluated_with_exact_coefficients(order,view_live,light_live,monkeypatch):
    e,_,_=material_fixture();c=MaterialCorrection(e,None,curved=False)
    nv=Jet.variable(I('.6' if view_live else '.4'),order=order) if order else Jet(I('.6' if view_live else '.4'))
    nl=Jet.variable(I('.4' if light_live else '.6'),order=order) if order else Jet(I('.4' if light_live else '.6'))
    u=Jet(1,order=order);jac=u.lift(2);av=c.smooth.at(nv,e.indices[0]);bl=c.smooth.at(nl,e.indices[1]);dv=segmented(e,nv,e.indices[0]);dl=segmented(e,nl,e.indices[1])
    expected=nl*(c.mix(dv)*c.mix(bl)+c.mix(av)*c.mix(dl)+c.mix(dv)*c.mix(dl))*jac/(e.pi*c.mean)
    original=c.smooth.at;calls=[]
    def observed(*args):calls.append(args[1]);return original(*args)
    monkeypatch.setattr(c.smooth,'at',observed);got=c._ordinary_inputs(u,(nv,nl,u,jac))[1]
    assert got.c.keys()==expected.c.keys()
    assert all((got[k].lo,got[k].hi)==(expected[k].lo,expected[k].hi) for k in got.c)
    assert len(calls)==int(view_live)+int(light_live)


def test_zero_correction_keeps_invalid_denominator_refusal():
    e,_,_=material_fixture();e.mean=[I(0),I(0)];c=MaterialCorrection(e,None,curved=False);u=Jet(1)
    with pytest.raises(ValueError,match='denominator'):c._ordinary_inputs(u,(Jet('.4'),Jet('.6'),u,u))


def test_tiny_high_order_correction_is_not_treated_as_zero(monkeypatch):
    from tools import opaque_peano_material as module
    e,_,_=material_fixture();c=MaterialCorrection(e,None,curved=False);u=Jet(1,order=3);nv=u.lift('.4');nl=u.lift('.6')
    def sparse(ext,q,base,**kw):return [Jet(order=3,coeff={(3,0):I('1e-80')}) if q is nv else q.lift(0),q.lift(0)]
    monkeypatch.setattr(module,'segmented',sparse);got=c._ordinary_inputs(u,(nv,nl,u,u))[1]
    assert got[3,0].lo>0 and got[0,0].zero
