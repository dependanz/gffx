"""Both tensor identities retain exact analytical integrals and physical axes."""
from functools import partial
from fractions import Fraction as Q
import pytest
from tools.opaque_shading_intervals import I,Jet

@pytest.mark.parametrize('axis',[0,1])
@pytest.mark.parametrize('case',range(21))
def test_both_orders_enclose_independent_material_integrals(axis,case,monkeypatch):
    from tools import opaque_peano_material as module
    from tests.packaging import test_opaque_transverse_peano as controls
    real=module.MaterialCorrection
    monkeypatch.setattr(module,'MaterialCorrection',partial(real,integrated_axis=axis))
    record=lambda *args:None
    if case<12:
        sx,sy=[(-1,-1),(-1,1),(1,-1),(1,1)][case//3]
        controls.test_material_transverse_producer_encloses_clipped_log_rectangle(sx,sy,['.4988','.4992','.4998'][case%3],record)
    elif case==12:controls.test_material_transverse_curved_map_preserves_jacobian_and_exact_integral(record)
    elif case<15:controls.test_material_transverse_nonunit_box_keeps_physical_scale([-1,1][case-13])
    else:
        from tools import opaque_transverse_peano as tensor
        fn=tensor.conditioned_integral
        monkeypatch.setattr(tensor,'conditioned_integral',partial(fn,integrated_axis=axis))
        controls.test_conditioned_error_recovers_diagonal_clipped_polynomial((case-15)//2,[-1,1][case%2],record)

@pytest.mark.parametrize('axis',[0,1])
def test_physical_axis_callbacks_and_nonunit_polynomial(axis):
    from tools.opaque_transverse_peano import conditioned_integral
    calls=[];parts=[];box=(I(2,4),I(-3,0))
    def bounds(b,a):calls.append(('line',a,b));return [dict(third=I(0),jumps=[I(0)]*3)]
    def integrated(b,a):calls.append(('integrated',a,b));return [dict(third=I(0),jumps=[I(0)]*3)]
    got=conditioned_integral(lambda x,y:[x*x+2*y*y],box,bounds,integrated,integrated_axis=axis,error_components=parts.append)[0]
    exact=Q(4**3-2**3,3)*3+2*Q(0-(-3)**3,3)*2
    assert Q(got.lo)<=exact<=Q(got.hi) and got.width<I('1e-50').lo
    assert [a for kind,a,b in calls if kind=='line']==[1-axis]*5
    assert [a for kind,a,b in calls if kind=='integrated']==[axis]
    assert all(b[1-axis].width==box[1-axis].width for kind,a,b in calls if kind=='line')
    assert len(parts)==1 and len(parts[0])==2

@pytest.mark.parametrize('axis',[-1,2,True,None,'0'])
def test_invalid_order_refused_before_field_evaluation(axis):
    from tools.opaque_transverse_peano import conditioned_integral
    calls=[]
    with pytest.raises(ValueError,match='axis'):conditioned_integral(lambda *a:calls.append(a),(I(0,1),I(0,1)),None,None,integrated_axis=axis)
    assert not calls

@pytest.mark.parametrize('piece',range(13))
def test_original_profile_uses_qualified_fourth_piece_order(piece,monkeypatch):
    import os,json,hashlib
    from pathlib import Path
    from tools import opaque_full_material_reference as module
    source=Path(os.environ['GFFX_FULL_INPUT']);assert hashlib.sha256(source.read_bytes()).hexdigest()=='a15bea40cb5cdead7994d828b57f5dcfac9f44fab0292a06e7d0a7b08719f73a'
    class Captured(Exception):pass
    def capture(*args,**kwargs):
        assert kwargs.get('integrated_axis')==(0 if piece==3 else 1)
        raise Captured
    monkeypatch.setattr(module,'MaterialCorrection',capture)
    with pytest.raises(Captured):module.full_pixel(json.loads(source.read_bytes()),selected_piece=piece)


def test_saved_fourth_piece_order_enclosures_and_cost(monkeypatch):
    import os,json,time
    from pathlib import Path
    from tools import opaque_full_material_reference as module
    source=Path(os.environ['GFFX_FULL_INPUT']);out=Path(os.environ['GFFX_FULL_REFERENCE']);prior=json.loads((out.parent/'diagnostic1-reference.json').read_bytes())
    assert not out.exists();rows=[];calls=0;started=time.perf_counter()
    report=dict(status='working',prior_seconds=prior['seconds'],prior_calls=prior['input_calls'],rows=rows)
    class Captured(Exception):pass
    def capture(field,**kwargs):
        nonlocal calls
        c=kwargs['correction_integrator'];assert c.integrated_axis==0
        original=c.inputs;check=c.checkpoint
        def bounded():
            check()
            if prior['seconds']+time.perf_counter()-started>=20:raise RuntimeError('aggregate diagnostic20seconds')
        def inputs(*args):
            nonlocal calls
            bounded()
            if prior['input_calls']+calls>=2048:raise RuntimeError('aggregate diagnostic2048inputs')
            calls+=1;return original(*args)
        c.inputs=inputs;c.checkpoint=bounded
        for box in prior['boxes']:
            b=tuple(I(*v) for v in box['box']);bounds=[];times=[]
            for axis in (1,0):
                bounded();t=time.perf_counter();kwargs['prepare'](0,0,b);c.integrated_axis=axis;v=c(b)[1];bounds.append(v);times.append(time.perf_counter()-t)
            old,new=bounds;assert max(old.lo,new.lo)<=min(old.hi,new.hi)
            assert new.width<old.width
            rows.append(dict(box=box['box'],old=[str(old.lo),str(old.hi)],new=[str(new.lo),str(new.hi)],width_ratio=float(new.width/old.width),seconds=times))
        report['status']='passed';raise Captured
    monkeypatch.setattr(module,'integrate_square',capture)
    try:
        with pytest.raises(Captured):module.full_pixel(json.loads(source.read_bytes()),selected_piece=3)
    finally:
        report.update(seconds=time.perf_counter()-started,input_calls=calls,aggregate_seconds=prior['seconds']+time.perf_counter()-started,aggregate_calls=prior['input_calls']+calls)
        out.write_text(json.dumps(report,indent=2))
