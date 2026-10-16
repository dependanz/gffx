"""Closed integrals and refusal controls for finite-table signed transport."""
from fractions import Fraction as Q
import pytest

@pytest.mark.parametrize('parts',[1,2,4])
@pytest.mark.parametrize('axis',[0,1])
@pytest.mark.parametrize('sx,sy',[(1,1),(-1,1),(1,-1),(-1,-1)])
def test_transport_keeps_independent_log_squared_rectangle(axis,sx,sy,parts):
    from tools.opaque_boundary_material import BoundaryMaterial as TransportMaterial
    from tests.packaging.test_opaque_aggregate_transport import fixture
    from tools.opaque_shading_intervals import I
    from decimal import Decimal as D,localcontext
    got=TransportMaterial(fixture(sx,sy),order=6,parts=parts)((I(0,1),I(0,1)),integrated_axis=axis)[1]
    with localcontext() as c:
        c.prec=90;k=D('.5');a=D('.3');b=D('.01');d=D('.4')
        def f(q):
            if q<=k:return D(0)
            z=(q/k).ln();return q*q/2*(z*z-3*z+D('3.5'))-2*k*q+k*k/4
        expected=(f(a+b+d)-f(a+b)-f(a+d)+f(a))/(b*d)*(1+D(2).ln()**2)
        assert got.lo<=expected<=got.hi

def test_transport_rejects_disjoint_integrated_evidence():
    from tools.opaque_high_peano import high_error
    from tools.opaque_shading_intervals import I
    row=dict(order=3,derivative=I(0),jumps=[dict(position=I('.5'),values=[I(0)]*3,required=True,transverse_integral=I(1))])
    with pytest.raises(ValueError,match='disjoint'):high_error(row,I(0,1),I(1),integrated=True)

@pytest.mark.parametrize('case',['steps','simultaneous','refusal'])
def test_transport_preserves_discontinuous_and_refusal_controls(case,monkeypatch):
    from tools.opaque_boundary_material import BoundaryMaterial as TransportMaterial
    from tools import opaque_aggregate_peano as module
    from tests.packaging import test_opaque_aggregate_peano as controls
    from tools import opaque_high_material as hm
    from tests.packaging.test_opaque_high_material import high_steps
    monkeypatch.setattr(module,'AggregateMaterial',TransportMaterial)
    if case=='steps':
        monkeypatch.setattr(hm,'HighMaterial',TransportMaterial);high_steps()
    elif case=='simultaneous':controls.test_simultaneous_view_and_light_steps_preserve_product_jumps()
    else:controls.test_unknown_monotonicity_and_cancellation_refuse()

@pytest.mark.parametrize('base',[0,1])
def test_curved_boundary_coefficient_and_other_role_residual_match_closed_cubic(base):
    from tools.opaque_boundary_material import BoundaryMaterial
    from tests.packaging.test_opaque_aggregate_transport import fixture
    from tools.opaque_shading_intervals import I
    from decimal import Decimal as D,localcontext
    c=fixture();c.extension.indices[1]=base
    def inputs(u,v):return u*u*I('.2')+v*I('.4')+u.lift('.3'),u.lift(1),u.lift(1),u.lift(1)
    c.inputs=inputs;p=BoundaryMaterial(c);q,other=p.coefficient((I('.2','.8'),I('.18','.48')),0,0,I('.5'),I('.4'),I('.42'),I(8),I(1),1)
    with localcontext() as ctx:
        ctx.prec=90;loss=1+D(2).ln()**2;factor=D('1.28')*(1 if base==0 else loss)
        for j,v in enumerate([D('.4')**3,3*D('.4')**2,3*D('.4'),D(1)]):assert q[j,0].lo<=factor*v<=q[j,0].hi
        exact=D('1.28')*D('.4')**3*(loss-1 if base==0 else 0)
        assert other.lo<=exact<=other.hi
        total=q[0,0]+other;actual=D('1.28')*D('.4')**3*loss
        assert total.lo<=actual<=total.hi

def test_conditioned_root_refinement_encloses_exact_quadratic_root():
    from tools.opaque_high_material import HighMaterial
    from tests.packaging.test_opaque_aggregate_transport import fixture
    from tools.opaque_shading_intervals import I
    c=fixture()
    c.inputs=lambda x,y:(x.lift('.3')+x*I('.2')+x*x*I('.1'),x.lift(1),x.lift(1),x.lift(1))
    p=HighMaterial(c,order=6,refine_roots=True);box=(I(0,1),I('.5'))
    old=p.axis_bounds(box,0)[1]['jumps'][0]['position'];new=p.axis_bounds(box,0,root_refinements=3)[1]['jumps'][0]['position'];exact=I(3).sqrt()-1
    assert new.lo<=exact.lo<=exact.hi<=new.hi and new.width<old.width/100
    with pytest.raises(ValueError):p.axis_bounds(box,0,root_refinements=4)
