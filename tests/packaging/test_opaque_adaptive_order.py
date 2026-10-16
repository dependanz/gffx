"""Per-box tensor-order selection is work guidance; full bounds own acceptance."""
from fractions import Fraction as Q
from functools import partial
import pytest
from tools.opaque_shading_intervals import I,Jet,Unresolved

@pytest.mark.parametrize('case',range(15))
@pytest.mark.parametrize('full_mode',[False,True])
def test_adaptive_material_keeps_independent_analytical_integrals(case,full_mode,monkeypatch):
    from tools import opaque_peano_material as m
    from tests.packaging import test_opaque_transverse_peano as controls
    monkeypatch.setattr(m,'MaterialCorrection',partial(m.MaterialCorrection,adaptive_order=True,smooth_partition=full_mode,segmented_conditioned=full_mode))
    record=lambda *args:None
    if case<12:
        sx,sy=[(-1,-1),(-1,1),(1,-1),(1,1)][case//3]
        controls.test_material_transverse_producer_encloses_clipped_log_rectangle(sx,sy,['.4988','.4992','.4998'][case%3],record)
    elif case==12:controls.test_material_transverse_curved_map_preserves_jacobian_and_exact_integral(record)
    else:controls.test_material_transverse_nonunit_box_keeps_physical_scale([-1,1][case-13])

@pytest.mark.parametrize('preferred',[0,1])
@pytest.mark.parametrize('winner',[0,1])
def test_selector_charges_both_candidates_reuses_chosen_rows_and_contains_exact_polynomial(preferred,winner):
    from tools.opaque_transverse_peano import adaptive_conditioned_integral,conditioned_integral
    calls=[];lines=[];choices=[];box=(I(2,4),I(-3,0))
    def rows(b,axis):
        calls.append(axis)
        return [dict(third=I('1e-20' if axis==winner else '1e-10'),jumps=[I(0)]*3)]
    def line(b,axis):lines.append(axis);return [dict(third=I(0),jumps=[I(0)]*3)]
    f=lambda x,y:[x*x+2*y*y]
    got=adaptive_conditioned_integral(f,box,line,rows,preferred_axis=preferred,selection=choices.append)[0]
    assert calls==[preferred,1-preferred] and lines==[1-winner]*5 and choices==[winner]
    exact=Q(4**3-2**3,3)*3+2*Q(0-(-3)**3,3)*2
    assert Q(got.lo)<=exact<=Q(got.hi)
    fixed=conditioned_integral(f,box,line,rows,integrated_axis=winner)[0]
    assert (got.lo,got.hi)==(fixed.lo,fixed.hi)

@pytest.mark.parametrize('preferred',[0,1])
def test_tie_keeps_preferred_axis_and_failure_of_alternate_keeps_valid_bound(preferred):
    from tools.opaque_transverse_peano import adaptive_conditioned_integral
    choices=[];zero=lambda *a:[dict(third=I(0),jumps=[I(0)]*3)]
    for fail in (False,True):
        def rows(b,axis):
            if fail and axis!=preferred:raise ValueError('unsupported alternative')
            return zero()
        got=adaptive_conditioned_integral(lambda x,y:[x+y],(I(0,1),I(0,1)),zero,rows,preferred_axis=preferred,selection=choices.append)[0]
        assert got.lo<=1<=got.hi
    assert choices==[preferred,preferred]

def test_cancellation_and_two_unavailable_orders_cannot_be_accepted():
    from tools.opaque_transverse_peano import adaptive_conditioned_integral
    zero=lambda *a:[dict(third=I(0),jumps=[I(0)]*3)]
    def cancel(*a):raise Unresolved('cancelled')
    with pytest.raises(Unresolved):adaptive_conditioned_integral(lambda *a:[],(I(0,1),I(0,1)),zero,cancel)
    def invalid(*a):raise ValueError('unsupported')
    with pytest.raises(ValueError):adaptive_conditioned_integral(lambda *a:[],(I(0,1),I(0,1)),zero,invalid)

@pytest.mark.parametrize('axis',[None,True,-1,2,'0'])
def test_invalid_selector_is_refused_before_callbacks(axis):
    from tools.opaque_transverse_peano import adaptive_conditioned_integral
    calls=[]
    with pytest.raises(ValueError):adaptive_conditioned_integral(lambda *a:calls.append(a),(I(0,1),I(0,1)),None,None,preferred_axis=axis)
    assert calls==[]

@pytest.mark.parametrize('piece',range(13))
def test_only_fourth_original_piece_enables_adaptive_order(piece,monkeypatch):
    import os,json
    from pathlib import Path
    from tools import opaque_full_material_reference as m
    class Stop(Exception):pass
    def capture(*a,**kw):
        assert kw.get('adaptive_order',False)==(piece==3)
        raise Stop
    monkeypatch.setattr(m,'MaterialCorrection',capture)
    with pytest.raises(Stop):m.full_pixel(json.loads(Path(os.environ['GFFX_FULL_INPUT']).read_bytes()),selected_piece=piece)

def test_same_saved_regions_adaptive_order_bounds_and_cost(monkeypatch):
    import os,json,time
    from pathlib import Path
    from tools import opaque_full_material_reference as m
    source=Path(os.environ['GFFX_FULL_INPUT']);out=Path(os.environ['GFFX_FULL_REFERENCE']);prior=json.loads((out.parent/'diagnostic1-reference.json').read_bytes());assert not out.exists()
    report=dict(status='working',rows=[]);started=time.perf_counter();calls=0
    class Stop(Exception):pass
    def capture(field,**kwargs):
        nonlocal calls
        c=kwargs['correction_integrator'];original=c.inputs;before_check=c.checkpoint
        def check():
            before_check();assert prior['seconds']+time.perf_counter()-started<20,'diagnostic20seconds'
            assert prior['input_calls']+calls<=2048,'diagnostic2048inputs'
        def inputs(*args):
            nonlocal calls
            check();assert prior['input_calls']+calls<2048;calls+=1;return original(*args)
        c.inputs=inputs;c.checkpoint=check
        try:
            for row in prior['boxes']:
                box=tuple(I(*v) for v in row['box']);bounds=[];times=[];selected=[]
                for adaptive in (False,True):
                    check();t=time.perf_counter();kwargs['prepare'](0,0,box);c.adaptive_order=adaptive;v=c(box)[1];times.append(time.perf_counter()-t);bounds.append(v);selected.append(c.last_integrated_axis)
                old,new=bounds;assert max(old.lo,new.lo)<=min(old.hi,new.hi)
                report['rows'].append(dict(box=row['box'],old=[str(old.lo),str(old.hi)],new=[str(new.lo),str(new.hi)],width_ratio=float(new.width/old.width),seconds=times,selected=selected))
            assert [r['selected'][1] for r in report['rows']]==[0,1,1]
            assert all(r['width_ratio']<=1 for r in report['rows'])
            report['status']='passed'
        finally:c.inputs=original;c.checkpoint=before_check
        raise Stop
    monkeypatch.setattr(m,'integrate_square',capture)
    try:
        with pytest.raises(Stop):m.full_pixel(json.loads(source.read_bytes()),selected_piece=3)
    finally:
        report.update(seconds=time.perf_counter()-started,input_calls=calls,aggregate_seconds=prior['seconds']+time.perf_counter()-started,aggregate_calls=prior['input_calls']+calls)
        out.write_text(json.dumps(report,indent=2))


def test_small_selection_score_never_replaces_full_conditioned_error():
    from tools.opaque_transverse_peano import adaptive_conditioned_integral,conditioned_integral
    def integrated(box,axis):return [dict(third=I(0 if axis==0 else 1),jumps=[I(0)]*3)]
    def conditioned(box,axis):return [dict(third=I(1000),jumps=[I(0)]*3)]
    f=lambda x,y:[x+y];box=(I(0,1),I(0,1));choices=[]
    got=adaptive_conditioned_integral(f,box,conditioned,integrated,selection=choices.append)[0]
    fixed=conditioned_integral(f,box,conditioned,integrated,integrated_axis=0)[0]
    assert choices==[0] and (got.lo,got.hi)==(fixed.lo,fixed.hi)
    assert got.width>I('1e-2').hi
