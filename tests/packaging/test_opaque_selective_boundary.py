"""Selective complete-proof intersection never treats scheduling as acceptance."""
import pytest
from tools.opaque_shading_intervals import I,Unresolved
from types import SimpleNamespace
@pytest.mark.parametrize('case',['skip','intersect','refuse','disjoint','cancel'])
def test_selective_boundary_preserves_both_proofs_and_stopping(monkeypatch,case):
    from tools import opaque_selective_boundary as m
    c=SimpleNamespace(checkpoint=lambda:None,preferred_split_axis=1,envelope_integral=lambda box:[I(0),I(-2,2)])
    class Proof:
        def __init__(self,*a,**k):pass
        def __call__(self,*a,**k):
            if case=='cancel':raise Unresolved('stop stronger')
            if case=='refuse':raise ValueError('strict monotonicity required')
            if case=='disjoint':return [I(0),I(4,5)]
            return [I(0),I(-1,1)]
    monkeypatch.setattr(m,'BoundaryMaterial',Proof);p=m.SelectiveBoundary(c)
    box=(I(0,1),I(0,1)) if case=='skip' else (I('.5','.53125'),I('.5',1))
    if case in ('cancel','disjoint'):
        with pytest.raises(Unresolved if case=='cancel' else ValueError):p(box)
    else:
        v=p(box)[1];expected=2 if case in ('skip','refuse') else 1
        assert v.lo==-expected and v.hi==expected
        assert p.error_breakdown is None
@pytest.mark.parametrize('axis',[0,1])
@pytest.mark.parametrize('sx,sy',[(1,1),(-1,1),(1,-1),(-1,-1)])
def test_selective_boundary_encloses_independent_complete_log_integral(monkeypatch,axis,sx,sy):
    from tools import opaque_selective_boundary as m
    from tests.packaging import test_opaque_boundary_material as f
    from tools import opaque_boundary_material as boundary
    constructor=m.SelectiveBoundary
    monkeypatch.setattr(m,'eligible',lambda box:True)
    class Adapter(constructor):
        def __init__(self,c,**kw):super().__init__(c)
    monkeypatch.setattr(boundary,'BoundaryMaterial',Adapter)
    f.test_transport_keeps_independent_log_squared_rectangle(axis,sx,sy,2)

@pytest.fixture(autouse=True,params=[False,True])
def compiled_identity_profile(request):
    from contextlib import nullcontext
    from tools.opaque_decimal_convolution import decimal_convolution
    with decimal_convolution() if request.param else nullcontext():yield
