"""Fail early only when immutable depth-limited intervals prevent acceptance."""
import pytest
from tools.opaque_shading_intervals import I,Unresolved
from tools.opaque_square_reference import integrate_square


def test_depth_fixed_width_stops_before_refining_other_charts():
    calls=[]
    def correction(box):calls.append(1);return [I(0,'1e-5')]
    charts=[dict(field=lambda x,y:[x.lift(0)],depth=2,correction_integrator=correction),dict(field=lambda x,y:[x.lift(0)],depth=2),dict(field=lambda x,y:[x.lift(0)],depth=1)]
    with pytest.raises(Unresolved,match='irreducible depth-limited width') as exc:integrate_square(None,chart_cover=charts,max_depth=2,max_cells=32,tolerance='1e-12')
    d=exc.value.details;assert d['cells']==5 and d['last_complete_enclosure']['complete_cover'] and len(calls)==1


def test_opposite_signed_fixed_values_cannot_cancel_interval_width():
    charts=[dict(field=lambda x,y:[x.lift(0)],depth=2,correction_integrator=lambda b:[I(-1,'-.999999')]),dict(field=lambda x,y:[x.lift(0)],depth=2,correction_integrator=lambda b:[I('.999999',1)]),dict(field=lambda x,y:[x.lift(0)],depth=1)]
    with pytest.raises(Unresolved,match='irreducible depth-limited width') as exc:integrate_square(None,chart_cover=charts,max_depth=2,max_cells=32,tolerance='1e-12')
    assert exc.value.details['cells']==5


def test_depth_fixed_uncertainty_below_whole_allowance_still_passes():
    charts=[dict(field=lambda x,y:[x.lift(0)],depth=2,correction_integrator=lambda b:[I(0,'1e-13')]),dict(field=lambda x,y:[x.lift(0)],depth=2),dict(field=lambda x,y:[x.lift(0)],depth=1)]
    r=integrate_square(None,chart_cover=charts,max_depth=2,max_cells=32,tolerance='1e-12');assert r['value'][0].lo<=0<=r['value'][0].hi and r['cells']==5
