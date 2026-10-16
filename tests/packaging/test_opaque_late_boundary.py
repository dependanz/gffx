"""Late scheduling must retain complete proofs and the original global acceptance."""
from fractions import Fraction as Q
from types import SimpleNamespace
import pytest
from tools.opaque_shading_intervals import I,Unresolved

@pytest.mark.parametrize('kind',['broad','near_horizon','already_tight','refine'])
def test_late_rule_spends_only_on_small_unresolved_corrections(monkeypatch,kind):
    from tools.opaque_late_boundary import LateBoundary
    from tools import opaque_selective_boundary as base
    box=(I('.5','.53125'),I('.5','.625'))
    if kind=='broad':box=(I('.5','.53125'),I('.5',1))
    if kind=='near_horizon':box=(I(0,'.03125'),I(0,'.125'))
    old=[I(0),I('-1e-15','1e-15')] if kind!='already_tight' else [I(0),I(0)]
    calls=[];c=SimpleNamespace(checkpoint=lambda:None,preferred_split_axis=0,envelope_integral=lambda box:old)
    class Proof:
        def __init__(self,*a,**k):pass
        def __call__(self,b,*,integrated_axis):calls.append(integrated_axis);return [I(0),I('-1e-20','1e-20')]
    monkeypatch.setattr(base,'BoundaryMaterial',Proof)
    out=LateBoundary(c,tolerance='1e-14')(box)
    assert calls==([1] if kind=='refine' else [])
    assert out[1].width==(I('2e-20').lo if kind=='refine' else old[1].width)

@pytest.mark.parametrize('kind',['cancel','disjoint','fallback'])
def test_late_rule_keeps_failure_and_complete_intersection(monkeypatch,kind):
    from tools.opaque_late_boundary import LateBoundary
    from tools import opaque_selective_boundary as base
    c=SimpleNamespace(checkpoint=lambda:None,preferred_split_axis=0,envelope_integral=lambda box:[I(0),I(-1,1)])
    class Proof:
        def __init__(self,*a,**k):pass
        def __call__(self,*a,**k):
            if kind=='cancel':raise Unresolved('cancel')
            if kind=='fallback':raise ValueError('strict monotonicity required')
            return [I(0),I(2,3)]
    monkeypatch.setattr(base,'BoundaryMaterial',Proof);p=LateBoundary(c,tolerance='1e-14');box=(I('.5','.53125'),I('.5','.625'))
    if kind=='fallback':assert p(box)[1].width==2 and c.preferred_split_axis==0
    else:
        with pytest.raises(Unresolved if kind=='cancel' else ValueError):p(box)

@pytest.mark.parametrize('value',['0','-1','NaN','1e-5'])
def test_late_schedule_refuses_invalid_target(value):
    from tools.opaque_late_boundary import LateBoundary
    with pytest.raises(ValueError):LateBoundary(None,tolerance=value)

@pytest.mark.parametrize('axis',[0,1])
@pytest.mark.parametrize('sx,sy',[(1,1),(-1,1),(1,-1),(-1,-1)])
def test_late_proof_encloses_independent_complete_log_integral(monkeypatch,axis,sx,sy):
    from tools.opaque_late_boundary import LateBoundary
    from tools import opaque_boundary_material as boundary
    from tests.packaging import test_opaque_boundary_material as f
    from tools.opaque_decimal_convolution import decimal_convolution
    # Exercise the proof on a closed fixture independently of work scheduling.
    monkeypatch.setattr(LateBoundary,'should_refine',lambda self,b,old:True)
    class Adapter(LateBoundary):
        def __init__(self,c,**kw):super().__init__(c,tolerance='1e-14')
    monkeypatch.setattr(boundary,'BoundaryMaterial',Adapter)
    with decimal_convolution():f.test_transport_keeps_independent_log_squared_rectangle(axis,sx,sy,2)
