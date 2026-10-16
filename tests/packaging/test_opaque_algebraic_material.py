"""Independent full integrals and strict fallback for algebraic curve transport."""
import pytest
from tools.opaque_shading_intervals import I

@pytest.mark.parametrize('axis',[0,1])
@pytest.mark.parametrize('sx,sy',[(1,1),(1,-1),(-1,1),(-1,-1)])
@pytest.mark.parametrize('parts',[2,4])
def test_algebraic_boundary_keeps_independent_material_integral(monkeypatch,axis,sx,sy,parts):
    from tools import opaque_boundary_material as m
    from tests.packaging import test_opaque_aggregate_transport as f
    from tests.packaging.test_opaque_boundary_jump_localization import test_localized_higher_jumps_keep_independent_closed_integral
    old=f.fixture;constructor=m.BoundaryMaterial;instances=[]
    def fixture(*a):
        c=old(*a);c.role_inputs=lambda role,u,v:c.inputs(u,v)[role]
        def implicit(role,k,u,v):
            q=c.inputs(u,v)[role];assert q[0,0].lo>0 and I(k).lo>0
            return q*q-I(k)*I(k)
        c.implicit_inputs=implicit;return c
    def make(*a,**kw):
        z=constructor(*a,**kw,algebraic=True);instances.append(z);return z
    monkeypatch.setattr(f,'fixture',fixture);monkeypatch.setattr(m,'BoundaryMaterial',make)
    test_localized_higher_jumps_keep_independent_closed_integral(monkeypatch,axis,sx,sy,parts)
    assert instances and sum(z.algebraic_uses for z in instances)>0

@pytest.mark.parametrize('kind',['fallback','cancel','invalid'])
def test_algebraic_boundary_fallback_is_specific_and_cancellation_propagates(monkeypatch,kind):
    from tools import opaque_boundary_material as m
    from tools.opaque_shading_intervals import Unresolved
    from tests.packaging import test_opaque_aggregate_transport as f
    from tests.packaging.test_opaque_boundary_material import test_transport_keeps_independent_log_squared_rectangle
    old=f.fixture;constructor=m.BoundaryMaterial;calls=[]
    def fixture(*a):
        c=old(*a)
        def implicit(*args):
            calls.append(1)
            if kind=='cancel':raise Unresolved('stop algebraic')
            raise ValueError('positive algebraic boundary branch required' if kind=='fallback' else 'corrupt algebraic equation')
        c.implicit_inputs=implicit;return c
    monkeypatch.setattr(f,'fixture',fixture);monkeypatch.setattr(m,'BoundaryMaterial',lambda *a,**k:constructor(*a,**k,algebraic=True))
    if kind=='fallback':test_transport_keeps_independent_log_squared_rectangle(0,1,1,2)
    else:
        with pytest.raises(Unresolved if kind=='cancel' else ValueError,match='stop algebraic' if kind=='cancel' else 'corrupt algebraic'):test_transport_keeps_independent_log_squared_rectangle(0,1,1,2)
    assert calls

def test_algebraic_boundary_is_opt_in_and_validated():
    from tools.opaque_boundary_material import BoundaryMaterial
    from tests.packaging.test_opaque_aggregate_transport import fixture
    assert BoundaryMaterial(fixture()).algebraic is False
    with pytest.raises(ValueError):BoundaryMaterial(fixture(),algebraic=1)

@pytest.mark.parametrize('axis',[0,1])
@pytest.mark.parametrize('parts',[1,2,4])
def test_boundary_does_not_require_unused_specular_half_vector(monkeypatch,axis,parts):
    from tests.packaging import test_opaque_aggregate_transport as f
    from tests.packaging.test_opaque_boundary_material import test_transport_keeps_independent_log_squared_rectangle
    old=f.fixture
    def fixture(*a):
        c=old(*a);original=c.inputs
        def material(u,v):
            x=list(original(u,v));x[2]=None;return x
        c.inputs=material;return c
    monkeypatch.setattr(f,'fixture',fixture)
    test_transport_keeps_independent_log_squared_rectangle(axis,1,1,parts)
