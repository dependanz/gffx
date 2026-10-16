"""Refinement heuristics never become numerical acceptance evidence."""
from decimal import Decimal as D,localcontext
import pytest
from tools.opaque_shading_intervals import I,Unresolved

@pytest.mark.parametrize('axis',[0,1])
def test_guided_nonlinear_integral_and_tangent_still_match_closed_form(axis):
    from tools.opaque_square_reference import integrate_square
    calls=[]
    def guide(u,v):
        calls.append((u.order,v.order));q=(u,v)[axis]
        return [1/(q+q.lift(2,tangent=1))]
    r=integrate_square(lambda u,v:[1/(u+v+u.lift(2,tangent=1))],refinement_field=guide)
    with localcontext() as c:
        c.prec=90;value=4*D(4).ln()-6*D(3).ln()+2*D(2).ln();tangent=(D(8)/9).ln()
    for k,want in [('value',value),('derivative',tangent)]:
        assert r[k][0].lo<=want<=r[k][0].hi and r[k][0].width<=D('2e-12')
    assert len(calls)==2*r['cells'] and all(x==(10,10) for x in calls)

@pytest.mark.parametrize('guide',[lambda u,v:[u.lift(0)],lambda u,v:[u.lift('1e-80')]])
def test_zero_or_tiny_guide_never_admits_unresolved_bound(guide):
    from tools.opaque_square_reference import integrate_square
    with pytest.raises(Unresolved):integrate_square(lambda u,v:[1/(u+v+u.lift(2,tangent=1))],refinement_field=guide,max_cells=1)

def test_invalid_guidance_falls_back_but_stop_propagates():
    from tools.opaque_square_reference import integrate_square
    field=lambda u,v:[u*v]
    def bad(*args):raise ValueError('guidance unavailable')
    r=integrate_square(field,refinement_field=bad);assert r['value'][0].lo<=D('.25')<=r['value'][0].hi
    def stop(*args):raise InterruptedError('stop')
    with pytest.raises(InterruptedError):integrate_square(field,refinement_field=stop)

def test_guidance_does_not_change_exact_one_cell_evidence():
    from tools.opaque_square_reference import integrate_square
    field=lambda u,v:[u*v]
    a=integrate_square(field);b=integrate_square(field,refinement_field=lambda u,v:[u**10*I('1e40')])
    assert a['cells']==b['cells']==1
    assert str(a['value'])==str(b['value']) and str(a['derivative'])==str(b['derivative'])


def test_original_checker_supplies_separate_refinement_field(monkeypatch):
    import os,json
    from pathlib import Path
    from tools import opaque_full_material_reference as m
    class Stop(Exception):pass
    def capture(field,**kwargs):
        assert callable(kwargs.get('refinement_field')) and kwargs['refinement_field'] is not field
        raise Stop
    monkeypatch.setattr(m,'integrate_square',capture)
    with pytest.raises(Stop):m.full_pixel(json.loads(Path(os.environ['GFFX_FULL_INPUT']).read_bytes()),selected_piece=1)


def test_guidance_cost_on_same_saved_regions(monkeypatch,record_property):
    import os,json,time,math
    from pathlib import Path
    from fractions import Fraction as Q
    from tools import opaque_full_material_reference as m
    from tools.opaque_shading_intervals import Jet
    source=Path(os.environ['GFFX_FULL_INPUT']);dest=Path(os.environ['GFFX_FULL_REFERENCE']);prior=json.loads((dest.parent/'signed-diagnostic1-reference.json').read_bytes())
    class Stop(Exception):pass
    def capture(field,**kwargs):
        start=time.perf_counter();calls=0;rows=[];complete=False
        def check():
            kwargs['checkpoint']();assert time.perf_counter()-start+prior['aggregate_seconds']<20;assert calls+prior['aggregate_calls']<=2048
        try:
            for saved in prior['rows']:
                a,b,c,d=map(Q,saved['box']);box=(I(I(a).lo,I(b).hi),I(I(c).lo,I(d).hi));check();kwargs['prepare'](0,0,box)
                errors=[];t=time.perf_counter()
                for axis in (0,1):
                    args=[Jet(v,order=10) for v in box];args[axis]=Jet.variable(box[axis],order=10)
                    check();calls+=1;vals=kwargs['refinement_field'](*args)
                    h=I(b-a) if axis==0 else I(d-c);outer=I(d-c) if axis==0 else I(b-a)
                    factor=I(Q(math.factorial(5)**4,11*math.factorial(10)**2))
                    errors.append(max((outer*h**11*q[10,j].bound*factor).hi for q in vals for j in (0,1)))
                rows.append(dict(box=saved['box'],seconds=time.perf_counter()-t,derivative_calls=2,bernstein_error_axis=list(map(str,errors)),base_priority=str((2*(I(errors[0])+I(errors[1]))).hi),current_correction_width=saved['correction_width']))
            complete=True
        finally:
            report=dict(status='diagnostic_only' if complete else 'stopped_partial',seconds=time.perf_counter()-start,aggregate_seconds=time.perf_counter()-start+prior['aggregate_seconds'],field_calls=calls,aggregate_calls=prior['aggregate_calls']+calls,max_seconds=20,max_inputs=2048,rows=rows,scope='Six extra order10 guidance-field evaluations on the same three parent boxes; preparation timed, no full-piece prediction.')
            dest.write_text(json.dumps(report,indent=2));record_property('guidance_cost',json.dumps(report))
        check();raise Stop
    monkeypatch.setattr(m,'integrate_square',capture)
    with pytest.raises(Stop):m.full_pixel(json.loads(source.read_bytes()),selected_piece=1)
