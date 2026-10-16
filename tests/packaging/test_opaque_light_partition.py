"""Exact light-side partition; darkness is proved, never thresholded."""
from fractions import Fraction as Q
import pytest

def source(values):return dict(corners=[[0,0],[1,0],[0,1]],clip_w=[1,1,1],normal=[[v,0,1] for v in values])
def test_light_partition_retains_tiny_positive_area_and_exact_complement():
    from tools.opaque_light_partition import partition
    p=dict(face=0,triangle=[[0,0],[1,0],[0,1]])
    for epsilon in (Q(1),Q(1,10**50)):
        parts,cert=partition(p,source([-1,epsilon,-1]),[0,0],[1,0,0])
        expected=epsilon**2/(2*(1+epsilon)**2)
        assert Q(cert['lit_area'])==expected>0 and Q(cert['dark_area'])+expected==Q(1,2)
        assert len(parts)==1

def test_light_partition_dark_lit_crossing_and_orientation():
    from tools.opaque_light_partition import partition
    for reverse in (False,True):
        tri=[[0,0],[1,0],[0,1]];tri=tri[::-1] if reverse else tri;p=dict(face=0,triangle=tri)
        for values,lit,count in [([-1,-1,-1],Q(0),0),([0,0,0],Q(0),0),([0,1,1],Q(1,2),1),([-1,1,1],Q(3,8),2)]:
            parts,c=partition(p,source(values),[0,0],[1,0,0]);assert Q(c['lit_area'])==lit and len(parts)==count
            assert Q(c['lit_area'])+Q(c['dark_area'])==Q(1,2)

def test_light_partition_perspective_sign_and_cancel():
    from tools.opaque_light_partition import partition
    p=dict(face=0,triangle=[[0,0],[1,0],[0,1]]);s=source([-1,1,-1]);s['clip_w']=[1,2,1]
    parts,c=partition(p,s,[0,0],[1,0,0]);assert Q(c['lit_area'])==Q(1,18)
    s['clip_w'][0]=0
    with pytest.raises(ValueError):partition(p,s,[0,0],[1,0,0])
    def stop():raise TimeoutError('cancel')
    with pytest.raises(TimeoutError):partition(p,source([1,1,1]),[0,0],[1,0,0],checkpoint=stop)


def test_light_partition_nonlinear_integral_has_independent_log_answer():
    from tools.opaque_light_partition import partition
    from tools.opaque_square_reference import integrate_square
    from tools.opaque_shading_intervals import I
    src=source([-1,1,-1]);src['corners']=[[-1,0],[1,0],[-1,1]]
    parts,c=partition(dict(face=0,triangle=src['corners']),src,[0,0],[1,0,0]);total=I(0)
    for part in parts:
        a,b,c=part['triangle'];det=abs((b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]))
        def field(u,v):
            x=a[0]+u*((1-v)*b[0]+v*c[0]-a[0]);return [x/(1+x)*u*det]
        value=integrate_square(field)['value'][0];total+=value
    expected=I('0.75')-I(2).ln()
    assert max(total.lo,expected.lo)<=min(total.hi,expected.hi) and total.width<=I('2e-12').lo


def test_remaining_light_partition_charges_all_components_and_certifies_darkness(monkeypatch):
    import json
    from pathlib import Path
    from tools import opaque_full_material_reference as m
    from tools.opaque_remaining_reference_contract import allocation
    from tools.opaque_shading_intervals import I
    d=json.loads(Path('D:/devspace/gffx/qualification-20260913/m1-continuation-20260913/opaque169-20261002/pixel2-input2.json').read_bytes());name='remaining28-offline-two-hour-v1';calls=[]
    def integrate(field,**kw):calls.append(kw);return dict(value=[I(0),I(0)],cells=1)
    monkeypatch.setattr(m,'integrate_square',integrate)
    r=m.full_pixel(d,reference_profile=name,streamed_piece=9)
    assert len(calls)==2 and r['cells']==2 and [c['max_cells'] for c in calls]==[8192,8191]
    assert sum((I(c['tolerance']) for c in calls),I(0)).hi<=I(allocation(d,reference_profile=name)[9]).lo
    calls.clear();r=m.full_pixel(d,reference_profile=name,streamed_piece=18)
    assert not calls and r['cells']==0 and all(q.lo==q.hi==0 for q in r['basis'])
    assert Q(r['light_partitions'][0]['dark_area'])>0
