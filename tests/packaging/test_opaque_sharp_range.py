"""Per-branch bounds checked against independent derivatives of log polynomials."""
import math,pytest
from fractions import Fraction as Q
from decimal import Decimal as D,localcontext

@pytest.mark.parametrize('order',[0,1,3,6,10])
@pytest.mark.parametrize('base',[0,3,6])
def test_individual_segments_retain_independent_high_derivatives(order,base):
    from tools.opaque_sharp_range import SharpRangeIndex
    from tools.opaque_residual_range import ResidualRangeIndex
    from tools.opaque_shading_intervals import I,Jet
    from tests.packaging.test_opaque_residual_range import curve
    c=curve(7);sharp=SharpRangeIndex(c);old=ResidualRangeIndex(c);q=Jet.variable(I('.14','.99'),order=order) if order else Jet(I('.14','.99'));got=sharp.high_delta_jet(q,base);prior=old.high_delta_jet(q,base)
    for a,b in zip(got,prior):
        for k in range(order+1):assert b[k,0].lo<=a[k,0].lo<=a[k,0].hi<=b[k,0].hi
    with localcontext() as ctx:
        ctx.prec=90
        for branch in range(7):
            t=(D(branch+1)+D('.5'))/8
            for ch in [0,1]:
                co=[D((a-b).numerator)/D((a-b).denominator) for a,b in zip(c.global_power(branch,ch),c.global_power(base,ch))]
                for k in range(order+1):
                    z=t.ln();value=sum((a*(z**j if j else D(1)) for j,a in enumerate(co)),D(0))/(t**k*math.factorial(k))
                    assert got[ch][k,0].lo<=value<=got[ch][k,0].hi
                    # d(q^-k P(log q))/dq = q^(-k-1)(P'-kP).
                    co=[(j+1)*co[j+1]-k*co[j] if j<3 else -k*co[j] for j in range(4)]

@pytest.mark.parametrize('order',[0,3,10])
def test_exact_same_branch_zero_and_horizon_are_retained(order):
    from tools.opaque_sharp_range import SharpRangeIndex
    from tools.opaque_shading_intervals import I,Jet
    from tests.packaging.test_opaque_residual_range import curve
    p=SharpRangeIndex(curve(7));q=Jet.variable(I('.01','.05'),order=order) if order else Jet(I('.01','.05'))
    assert all(all(v.zero for v in z.c.values()) for z in p.high_delta_jet(q,-1))
    q=Jet.variable(I('.3','.32'),order=order) if order else Jet(I('.3','.32'))
    assert all(all(v.zero for v in z.c.values()) for z in p.high_delta_jet(q,1))

def test_sharp_ranges_keep_snapshot_cancel_and_domain_refusal():
    from tools.opaque_sharp_range import SharpRangeIndex
    from tools.opaque_shading_intervals import I,Jet
    from tests.packaging.test_opaque_residual_range import curve
    c=curve(7);p=SharpRangeIndex(c);before=p.delta(I('.2','.9'),1);c.x[0]=I(100)
    assert [(v.lo,v.hi) for v in before]==[(v.lo,v.hi) for v in p.delta(I('.2','.9'),1)]
    for v in [I(-1,1),I(0,2)]:
        with pytest.raises(ValueError):p.delta(v,1)
    def stop():raise RuntimeError('cancel branch proof')
    p.checkpoint=stop
    with pytest.raises(RuntimeError):p.delta(I('.3','.4'),1)


@pytest.mark.parametrize('order',[0,3,10])
def test_fixed_geometry_sharp_bounds_refuse_parameter_tangents(order):
    from tools.opaque_sharp_range import SharpRangeIndex
    from tools.opaque_shading_intervals import I,Jet
    from tests.packaging.test_opaque_residual_range import curve
    p=SharpRangeIndex(curve(7));q=Jet(I('.3'),order=order,tangent=1)
    with pytest.raises(ValueError):p.high_delta_jet(q,1)


@pytest.mark.parametrize('sx,sy',[(1,1),(-1,1),(1,-1),(-1,-1)])
def test_sharp_full_material_keeps_independent_log_integral(monkeypatch,sx,sy):
    from tools import opaque_aggregate_peano as module
    from tools.opaque_sharp_range import SharpRangeIndex
    from tests.packaging.test_opaque_aggregate_peano import test_aggregate_material_bound_contains_closed_log_integral
    monkeypatch.setattr(module,'ResidualRangeIndex',SharpRangeIndex)
    test_aggregate_material_bound_contains_closed_log_integral(sx,sy)

def test_sharp_full_material_keeps_simultaneous_product_jumps(monkeypatch):
    from tools import opaque_aggregate_peano as module
    from tools.opaque_sharp_range import SharpRangeIndex
    from tests.packaging.test_opaque_aggregate_peano import test_simultaneous_view_and_light_steps_preserve_product_jumps
    monkeypatch.setattr(module,'ResidualRangeIndex',SharpRangeIndex)
    test_simultaneous_view_and_light_steps_preserve_product_jumps()
