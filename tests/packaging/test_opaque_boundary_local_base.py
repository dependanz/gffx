"""A local smooth extension must retain the complete other-role residual."""
import pytest
from fractions import Fraction as Q

@pytest.mark.parametrize('axis',[0,1])
@pytest.mark.parametrize('sx,sy',[(1,1),(-1,1),(1,-1),(-1,-1)])
def test_local_extension_keeps_closed_integral(monkeypatch,axis,sx,sy):
    from tools import opaque_boundary_material as m
    from tests.packaging.test_opaque_boundary_material import test_transport_keeps_independent_log_squared_rectangle
    original=m.BoundaryMaterial
    monkeypatch.setattr(m,'BoundaryMaterial',lambda *a,**k:original(*a,**k,boundary_order=4,local_other_base=True))
    test_transport_keeps_independent_log_squared_rectangle(axis,sx,sy,4)

def test_local_base_is_explicit_and_domain_checked():
    from tools.opaque_boundary_material import BoundaryMaterial
    from tests.packaging.test_opaque_aggregate_transport import fixture
    assert BoundaryMaterial(fixture()).local_other_base is False
    with pytest.raises(ValueError):BoundaryMaterial(fixture(),local_other_base=1)

@pytest.mark.parametrize('base',[0,1])
def test_explicit_local_base_preserves_complete_value_and_residual(base):
    from tools.opaque_boundary_material import BoundaryMaterial
    from tests.packaging.test_opaque_aggregate_transport import fixture
    from tools.opaque_shading_intervals import I
    from decimal import Decimal as D,localcontext
    c=fixture();c.extension.indices[1]=0
    c.inputs=lambda u,v:(u*u*I('.2')+v*I('.4')+u.lift('.3'),u.lift(1),u.lift(1),u.lift(1))
    p=BoundaryMaterial(c,boundary_order=4,local_other_base=True)
    q,other=p.coefficient((I('.2','.8'),I('.18','.48')),0,0,I('.5'),I('.4'),I('.42'),I(8),I(1),1,other_base=base)
    with localcontext() as ctx:
        ctx.prec=90;expected=D('1.28')*D('.4')**3*(1+D(2).ln()**2)
        actual=q[0,0]+other;assert actual.lo<=expected<=actual.hi
    if base==1:assert other.zero
    else:assert not other.zero
