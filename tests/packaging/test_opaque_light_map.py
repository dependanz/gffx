"""Exact endpoint-flattening coordinate change; no removed support or new roots."""
from fractions import Fraction as Q
import pytest
from tools.opaque_shading_intervals import I,Jet


def test_light_map_preserves_area_and_polynomial_moments():
    from tools.opaque_light_map import coordinates
    from tools.opaque_square_reference import integrate_square
    def field(u,v):
        x,y,j=coordinates(u,v,4);return [j,x*j,y*j]
    r=integrate_square(field,tolerance='1e-12',max_cells=256)
    for z,expected in zip(r['value'],(Q(1),Q(1,2),Q(1,2))):assert Q(z.lo)<=expected<=Q(z.hi)


def test_light_map_complete_nonlinear_integral_matches_log_two():
    from tools.opaque_light_map import coordinates
    from tools.opaque_square_reference import integrate_square
    def field(u,v):
        x,y,j=coordinates(u,v,4);return [j/(1+x)]
    r=integrate_square(field,tolerance='1e-12',max_cells=256,remainder_partition=(4,4));expected=I(2).ln()
    assert r['value'][0].lo<=expected.lo<=expected.hi<=r['value'][0].hi


def test_light_map_keeps_tiny_support_and_rejects_undeclared_power():
    from tools.opaque_light_map import coordinates
    x,y,j=coordinates(Jet(I('1e-40')),Jet(I('1e-40')),4)
    assert x[0,0].lo>0 and y[0,0].lo>0 and j[0,0].lo>0
    for p in (True,0,2,8):
        with pytest.raises(ValueError):coordinates(Jet(0),Jet(1),p)
    a,b,j=coordinates(Jet.variable(I('.2'),order=3),Jet(I('.3'),order=3),1)
    assert a[1,0].lo==1 and b[0,0].lo==I('.3').lo and j[0,0].lo==1


def test_full_checker_map_is_explicit_and_matches_change_of_variables(monkeypatch):
    import json
    from pathlib import Path
    from tools import opaque_full_material_reference as m
    d=json.loads(Path('D:/devspace/gffx/qualification-20260913/m1-continuation-20260913/opaque169-20261002/pixel2-input2.json').read_bytes())
    for power in (True,2,8):
        with pytest.raises(ValueError):m.full_pixel(d,reference_profile='remaining28-offline-two-hour-v1',streamed_piece=9,light_map_power=power)
    class Captured(Exception):pass
    charts=[]
    def capture(field,**kw):charts.append(dict(field=field,**kw));raise Captured
    monkeypatch.setattr(m,'integrate_square',capture)
    for power in (1,4):
        with pytest.raises(Captured):m.full_pixel(d,reference_profile='remaining28-offline-two-hour-v1',streamed_piece=9,light_map_power=power,material_only_inputs=True,aggregate_residual=True,tight_cosine_slopes=True)
    for a,b in [(Q(1,2),Q(1,4)),(Q(1,16),Q(1,16)),(Q(3,4),Q(1,2))]:
        u=Jet.variable(I(a),order=3);v=Jet(I(b),order=3);x,y=u**4,v**4;j=16*(u**3)*(v**3)
        charts[0]['prepare'](x,y,(x[0,0],y[0,0]));charts[1]['prepare'](u,v,(u[0,0],v[0,0]))
        expected=[z*j for z in charts[0]['field'](x,y)];actual=charts[1]['field'](u,v)
        for aa,bb in zip(actual,expected):
            for k in range(4):assert max(aa[k,0].lo,bb[k,0].lo)<=min(aa[k,0].hi,bb[k,0].hi)


def test_light_map_leaves_fully_lit_pieces_on_original_coordinates(monkeypatch):
    import json
    from pathlib import Path
    from tools import opaque_full_material_reference as m
    data=json.loads(Path('D:/devspace/gffx/qualification-20260913/m1-continuation-20260913/opaque169-20261002/pixel1-input2.json').read_bytes());charts=[]
    class Capture(Exception):pass
    def capture(field,**kw):charts.append(dict(field=field,**kw));raise Capture
    monkeypatch.setattr(m,'integrate_square',capture)
    for power in (1,4):
        with pytest.raises(Capture):m.full_pixel(data,reference_profile='remaining33-offline-two-hour-v1',streamed_piece=0,light_map_power=power,material_only_inputs=True,aggregate_residual=True)
    u=Jet.variable(I('.5'),order=3);v=Jet(I('.25'),order=3);rows=[]
    for ch in charts:
        ch['prepare'](u,v,(u[0,0],v[0,0]));rows.append(ch['field'](u,v))
    assert [[(q[k,0].lo,q[k,0].hi) for k in range(4)] for q in rows[0]]==[[(q[k,0].lo,q[k,0].hi) for k in range(4)] for q in rows[1]]
