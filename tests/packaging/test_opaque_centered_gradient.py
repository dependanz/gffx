"""Independent smooth-geometry derivative enclosures and fail-closed integration."""
from fractions import Fraction as Q
from types import SimpleNamespace
import pytest
from tools.opaque_shading_intervals import I,Jet,Unresolved

@pytest.mark.parametrize('axis',[0,1])
@pytest.mark.parametrize('sign',[1,-1])
def test_centered_gradient_recovers_sign_lost_to_dependency(axis,sign):
    from tools.opaque_centered_gradient import CenteredInputs
    # Algebraically f=sign*(.001*y + x*y - x*y): derivative is exactly .001.
    def f(x,y):
        a,b=(x,y) if axis==1 else (y,x)
        return [sign*(b*Q(1,1000)+a*b-a*b)]
    box=[I(1,2),I(1,2)];args=[Jet(v,order=3) for v in box];args[axis]=Jet.variable(box[axis],order=3)
    raw=f(*args)[0];assert raw[1,0].lo<0<raw[1,0].hi
    refined=CenteredInputs(f)(*args)[0]
    target=Q(sign,1000);assert Q(refined[1,0].lo)<=target<=Q(refined[1,0].hi)
    assert refined[1,0].lo>0 if sign==1 else refined[1,0].hi<0
    assert all((refined[k].lo,refined[k].hi)==(raw[k].lo,raw[k].hi) for k in raw.c if k!=(1,0))

@pytest.mark.parametrize('axis',[0,1])
def test_centered_gradient_keeps_a_real_turn_uncertain(axis):
    from tools.opaque_centered_gradient import CenteredInputs
    f=lambda x,y:[(x if axis==0 else y)**2]
    args=[Jet(I(-1,1),order=2),Jet(I(-1,1),order=2)];args[axis]=Jet.variable(I(-1,1),order=2)
    q=CenteredInputs(f)(*args)[0];assert q[1,0].lo<=-2 and q[1,0].hi>=2

@pytest.mark.parametrize('axis',[0,1])
def test_centered_gradient_bounds_nonlinear_polynomial_at_exact_points(axis):
    from tools.opaque_centered_gradient import CenteredInputs
    f=lambda x,y:[x*x*y+y*y*y-x*Q(1,3)]
    box=[I('-0.1','0.2'),I('-0.3','0.4')];args=[Jet(v,order=4) for v in box];args[axis]=Jet.variable(box[axis],order=4)
    q=CenteredInputs(f)(*args)[0]
    for x in (Q(-1,10),Q(0),Q(1,5)):
        for y in (Q(-3,10),Q(0),Q(2,5)):
            d=2*x*y-Q(1,3) if axis==0 else x*x+3*y*y
            assert Q(q[1,0].lo)<=d<=Q(q[1,0].hi)

@pytest.mark.parametrize('kind',['tangent','scaled','both','curved','constant'])
def test_centered_gradient_does_not_reinterpret_composed_jets(kind):
    from tools.opaque_centered_gradient import CenteredInputs
    x=Jet.variable(I(0,1),order=3);y=Jet(I(0,1),order=3)
    if kind=='tangent':
        y=Jet.variable(I(0,1),axis=1,order=3);y.c[1,1]=I(1)
    if kind=='scaled':x.c[1,0]=I(2)
    if kind=='both':y=Jet.variable(I(0,1),order=3)
    if kind=='curved':x.c[2,0]=I(1)
    if kind=='constant':x=Jet(I(0,1),order=3)
    calls=[]
    def f(a,b):calls.append(1);return [a*b-a*b]
    q=CenteredInputs(f)(x,y);assert len(calls)==1


def test_centered_gradient_propagates_cancellation():
    from tools.opaque_centered_gradient import CenteredInputs
    def stop():raise Unresolved('cancel')
    with pytest.raises(Unresolved):CenteredInputs(lambda x,y:[x],checkpoint=stop)(Jet.variable(I(0,1)),Jet(0,order=6))


def test_centered_gradient_rejects_disjoint_producer_evidence():
    from tools.opaque_centered_gradient import CenteredInputs
    calls=[0]
    def false(x,y):
        calls[0]+=1
        if calls[0]==1:return [Jet(order=x.order,coeff={(1,0):I(-1,1)})]
        return [Jet(order=x.order,coeff={(1,0):I(4)})]
    with pytest.raises(ValueError,match='disjoint'):CenteredInputs(false)(Jet.variable(I(0,1),order=3),Jet(0,order=3))

@pytest.mark.parametrize('axis',[0,1])
@pytest.mark.parametrize('sx,sy',[(1,1),(-1,1),(1,-1),(-1,-1)])
def test_centered_boundary_keeps_independent_closed_log_integral(monkeypatch,axis,sx,sy):
    from tools.opaque_centered_gradient import CenteredBoundaryMaterial
    from tools import opaque_boundary_material as boundary
    from tests.packaging import test_opaque_boundary_material as f
    from tools.opaque_decimal_convolution import decimal_convolution
    monkeypatch.setattr(boundary,'BoundaryMaterial',CenteredBoundaryMaterial)
    with decimal_convolution():f.test_transport_keeps_independent_log_squared_rectangle(axis,sx,sy,2)

@pytest.mark.parametrize('failure',[False,True])
def test_centered_boundary_restores_geometry_hooks(monkeypatch,failure):
    from tools.opaque_centered_gradient import CenteredBoundaryMaterial,CenteredInputs
    from tools.opaque_boundary_material import BoundaryMaterial
    original=lambda x,y:[x,y,None,x.lift(1)];role=lambda role,x,y:(x,y)[role]
    c=SimpleNamespace(inputs=original,role_inputs=role,checkpoint=lambda:None)
    def call(self,*args,**kwargs):
        assert isinstance(self.c.inputs,CenteredInputs)
        if failure:raise Unresolved('stop')
        return [I(0),I(1)]
    monkeypatch.setattr(BoundaryMaterial,'__call__',call);p=object.__new__(CenteredBoundaryMaterial);p.c=c
    if failure:
        with pytest.raises(Unresolved):p((I(0,1),I(0,1)))
    else:assert p((I(0,1),I(0,1)))[1].lo==1
    assert c.inputs is original and c.role_inputs is role

@pytest.mark.parametrize('order',[0,3])
@pytest.mark.parametrize('sign',[1,-1])
def test_centered_gradient_refines_only_unit_auxiliary_partial(order,sign):
    from tools.opaque_centered_gradient import CenteredInputs
    def f(x,y):return [sign*(y*Q(1,1000)+x*y-x*y)]
    x=Jet(I(1,2),order=order);y=Jet(I(1,2),order=order)
    if order:x.c[1,0]=I(1);y.c[1,0]=I(2);y.c[2,0]=I(3)
    y.c[0,1]=I(1);raw=f(x,y)[0];assert raw[0,1].lo<0<raw[0,1].hi
    refined=CenteredInputs(f)(x,y)[0];target=Q(sign,1000)
    assert Q(refined[0,1].lo)<=target<=Q(refined[0,1].hi)
    assert refined[0,1].lo>0 if sign==1 else refined[0,1].hi<0
    assert all((refined[k].lo,refined[k].hi)==(raw[k].lo,raw[k].hi) for k in raw.c if k!=(0,1))
