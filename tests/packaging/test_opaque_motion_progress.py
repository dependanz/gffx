"""Bounded next-piece diagnostics; no sampled convergence acceptance."""
from fractions import Fraction as Q
import json,os,time
from pathlib import Path

def test_remaining_front_piece_proof_directions(record_property,monkeypatch):
    import tools.opaque_fourth_tangent as f
    from tools.opaque_original_motion_reference import reference
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    boxes=[(Q(5,8),Q(11,16),Q(3,4),Q(1)),(Q(5,8),Q(3,4),Q(1,4),Q(3,8)),(Q(5,8),Q(11,16),Q(1,2),Q(3,4))]
    original=f.integral;rows=[]
    for axis in (0,1):
        details=[]
        def observe(*a,**kw):
            callback=kw.get('error_components')
            def capture(d):
                details.append(dict(box=list(map(str,a[1])),directions=d))
                if callback:callback(d)
            kw['error_components']=capture
            return original(*a,**kw)
        monkeypatch.setattr(f,'integral',observe)
        with backend(profile='ball',fast_conversion=True):
            start=time.monotonic();r=reference(frozen(),piece=2,motion='normal_tilt',diagnostic_boxes=boxes,correction_order=4,conditioned_correction=True,mixed_curve=True,transverse_correction=True,correction_axis=axis,material_inputs=True)
            rows.append(dict(axis=axis,seconds=time.monotonic()-start,result=r,details=details))
    p=Path(os.environ['GFFX_FULL_REFERENCE']);assert not p.exists();p.write_text(json.dumps(dict(rows=rows,scope='Three saved regions; direction identities, uncertainty and cost only.'),default=str,indent=2),encoding='utf8');record_property('front_directions',str(p))


def test_crossing_contraction_contains_all_independent_nonlinear_roots():
    import math
    from tools.opaque_tangent_transverse import contract_crossing
    from tools.opaque_shading_intervals import I
    for flip in (False,True):
        def inputs(x,y):return ((1-x) if flip else x)**2+y,x.lift(1),x.lift(1),x.lift(1)
        box=(I('.1','.9'),I('.001','.002'));got=contract_crossing(inputs,box,0,0,I('.3'),box[0])
        assert got.width<I('.002').lo
        for y in (Q(1,1000),Q(3,2000),Q(1,500)):
            value=Q(3,10)-y;scale=10**80;n=math.isqrt(value.numerator*scale*scale//value.denominator)
            lo,hi=Q(n,scale),Q(n+1,scale)
            if flip:lo,hi=1-hi,1-lo
            assert Q(got.lo)<=lo<=hi<=Q(got.hi)


def test_crossing_contraction_keeps_tiny_roots_uncertainty_and_cancellation():
    import pytest
    from tools.opaque_tangent_transverse import contract_crossing
    from tools.opaque_shading_intervals import I,Unresolved
    tiny=Q(1,2**70)
    def inputs(x,y):return x+y,x.lift(1),x.lift(1),x.lift(1)
    result=contract_crossing(inputs,(I(0,1),I(0)),0,0,I(tiny),I(0,1))
    assert Q(result.lo)<=tiny<=Q(result.hi) and result.lo>0
    box=(I(-1,1),I(0))
    stationary=lambda x,y:(x*x,x.lift(1),x.lift(1),x.lift(1))
    result=contract_crossing(stationary,box,0,0,I('.5'),box[0]);assert result.lo==-1 and result.hi==1
    def stop():raise Unresolved('contraction cancelled')
    with pytest.raises(Unresolved,match='contraction cancelled'):contract_crossing(inputs,(I(0,1),I(0)),0,0,I('.5'),I(0,1),checkpoint=stop)
    assert contract_crossing(inputs,(I(0,1),I(0)),0,0,I(2),I(0,1)) is None


def test_crossing_contraction_selected_backend():
    from tools.opaque_arb_jets import backend
    with backend(profile='ball',fast_conversion=True):
        test_crossing_contraction_contains_all_independent_nonlinear_roots()
        test_crossing_contraction_keeps_tiny_roots_uncertainty_and_cancellation()


def test_discrete_line_proof_contracts_before_forming_jumps(monkeypatch):
    from tools.opaque_tangent_peano import TangentCorrection
    from tools.opaque_full_material_reference import HorizonExtension
    from tests.packaging.test_opaque_motion_tangent import curve
    from tools.opaque_shading_intervals import I
    import tools.opaque_tangent_transverse as m
    e=HorizonExtension(curve());e.indices=[-1,0]
    def inputs(x,y):return x/2+y/8+x.lift(Q(1,4),tangent=1),x.lift(1),x.lift(1),x.lift(1)
    calls=[];old=m.contract_crossing
    def seen(*a,**kw):calls.append(a);return old(*a,**kw)
    monkeypatch.setattr(m,'contract_crossing',seen)
    t=TangentCorrection(e,inputs,order=4,horizon=True,transverse=True)
    t.axis_bounds((I(0,1),I('.5')),0,transverse=False)
    assert calls and all(c[2]==0 for c in calls)


def test_continuous_proof_contracts_only_certified_interior_edge_roots(monkeypatch):
    from tools.opaque_shading_intervals import I
    import tools.opaque_tangent_transverse as m
    def inputs(x,y):return x*x+y,x.lift(1),x.lift(1),x.lift(1)
    calls=[];old=m.contract_crossing
    def seen(*a,**kw):calls.append(a);return old(*a,**kw)
    monkeypatch.setattr(m,'contract_crossing',seen)
    def rows():return [dict(jumps=[dict(position=I('.1','.9'),values=[I(1),I(0),I(0),I(0)],required=True,role=0,knot=0)])]
    r=rows();m.refine(r,inputs,(I('.1','.9'),I('.1','.2')),0,[I('.5')])
    assert len(calls)==2 and all(not c[1][1].width for c in calls)
    calls.clear();m.refine(rows(),inputs,(I('.1','.9'),I('.6','.7')),0,[I('.5')])
    assert not calls


def test_exact_slope_ratio_cancels_shared_norm_and_retains_full_box():
    from tools.opaque_tangent_transverse import CosinePartials
    from tools.opaque_moving_reference import P
    from tools.opaque_shading_intervals import I
    from tools.opaque_arb_jets import backend
    coefficients={'n':[[P(0),P(1),P(1)],[P(0)]*3,[P(1),P(0),P(0)]],'v':[[P(1),P(0),P(0)],[P(0)]*3,[P(0)]*3]}
    with backend(profile='ball',fast_conversion=True):
        c=CosinePartials(coefficients);box=(I('.2','.3'),I('.4','.6'))
        for axis in (0,1):
            r=c.ratio(0,box,axis);raw=c.bound(0,box,axis)/c.bound(0,box,1-axis)
            assert r.width<raw.width
            for x in (Q(1,5),Q(1,4),Q(3,10)):
                for y in (Q(2,5),Q(1,2),Q(3,5)):
                    exact=(1+y)/x if axis==0 else x/(1+y)
                    assert Q(r.lo)<=exact<=Q(r.hi)


def test_transverse_proof_uses_independent_exact_ratio(monkeypatch):
    import tools.opaque_tangent_transverse as m
    from tools.opaque_shading_intervals import I
    calls=[]
    def inputs(x,y):return x+y,x.lift(1),x.lift(1),x.lift(1)
    def ratio(role,box,axis):calls.append((role,box,axis));return I(1)
    rows=[dict(jumps=[dict(position=I(0,1),values=[I(1),I(0),I(0),I(0)],required=False,role=0,knot=0)])]
    m.refine(rows,inputs,(I(0,1),I(0,1)),0,[I('.5')],partial_ratio=ratio)
    assert calls and 'transverse_integral' in rows[0]['jumps'][0]


def test_remaining_front_piece_bound_components(record_property,monkeypatch):
    import tools.opaque_tangent_peano as t
    from tools.opaque_high_peano import _mass,kernel_range
    from tools.opaque_peano import gauss5,kernel_local_range
    from tools.opaque_original_motion_reference import reference
    from tools.opaque_arb_jets import backend
    from tools.opaque_shading_intervals import I
    from tests.packaging.test_opaque_motion_tangent import frozen
    old=t.TangentCorrection.axis_bounds;rows=[]
    def observe(self,box,axis,**kw):
        r=old(self,box,axis,**kw);q=r[1];h=I(box[axis].hi)-I(box[axis].lo);other=I(box[1-axis].hi)-I(box[1-axis].lo);nodes,weights=gauss5();data=tuple((x.lo,x.hi,w.lo,w.hi) for x,w in zip(nodes,weights));d=q['derivative_range'];anchor=I(I((Q(d.lo)+Q(d.hi))/2).lo);ordinary=I((d-anchor).bound)*h**5*I(_mass(4,data));jumps=[]
        for w in q['jumps']:
            z=(w['position']-I(box[axis].lo))/h;z=I(max(I(0).lo,z.lo),min(I(1).hi,z.hi));value=I(0)
            for j,v in enumerate(w['values']):value=value+v*h**(j+1)*(kernel_local_range(z,j+1) if j<3 else kernel_range(z,4))
            if not w['required']:value=I(min(I(0).lo,value.lo),max(I(0).hi,value.hi))
            refined=w.get('transverse_integral');jumps.append(dict(role=w['role'],knot=w['knot'],uniform_width=str(value.width),refined_width=None if refined is None else str(refined.width),root_width=str(w['position'].width)))
        rows.append(dict(axis=axis,box=list(map(str,box)),transverse=str(other),ordinary_radius=str(ordinary),jumps=jumps));return r
    monkeypatch.setattr(t.TangentCorrection,'axis_bounds',observe)
    with backend(profile='ball',fast_conversion=True):r=reference(frozen(),piece=2,motion='normal_tilt',diagnostic_boxes=[(Q(5,8),Q(3,4),Q(1,4),Q(3,8))],correction_order=4,conditioned_correction=True,mixed_curve=True,transverse_correction=True,correction_axis=0,material_inputs=True)
    p=Path(os.environ['GFFX_FULL_REFERENCE']);assert not p.exists();p.write_text(json.dumps(dict(rows=rows,result=r,scope='Dominant saved-region proof terms only.'),default=str,indent=2),encoding='utf8');record_property('front_components',str(p))


def test_remaining_front_cyclic_maps_preserve_physical_fields_and_motion():
    from tools.opaque_original_motion_reference import prepare,mapped_inputs
    from tools.opaque_shading_intervals import Jet
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    with backend(profile='ball',fast_conversion=True):
        for piece in (2,3,10):
            for motion in ('normal_tilt','view_x','screen_x'):
                prepared=[prepare(frozen(),piece=piece,motion=motion,anchor=a) for a in range(3)]
                assert all(not p['curved'] and all(p['det'][k]==prepared[0]['det'][k] for k in (0,1)) for p in prepared)
                for weights in [(Q(1,3),)*3,(Q(1,8),Q(3,8),Q(1,2)),(Q(3,4),Q(1,8),Q(1,8))]:
                    rows=[]
                    for anchor,p in enumerate(prepared):
                        w=weights[anchor:]+weights[:anchor];u=w[1]+w[2];v=w[2]/u;r=mapped_inputs(p,Jet(u),Jet(v));rows.append([*r[:3],r[3]/u])
                    for c in range(4):
                        for derivative in (0,1):assert max(r[c][0,derivative].lo for r in rows)<=min(r[c][0,derivative].hi for r in rows)


def test_remaining_front_coordinate_feasibility(record_property):
    from tools.opaque_original_motion_reference import prepare
    from tools.opaque_tangent_transverse import CosinePartials
    from tools.opaque_shading_intervals import I
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    rows=[]
    with backend(profile='ball',fast_conversion=True):
        for anchor in range(3):
            p=prepare(frozen(),piece=2,motion='normal_tilt',anchor=anchor);proof=CosinePartials(p['fields'].coefficients);counts=[[0,0],[0,0]]
            for i in range(4):
                for j in range(4):
                    box=(I(I(Q(i,4)).lo,I(Q(i+1,4)).hi),I(I(Q(j,4)).lo,I(Q(j+1,4)).hi))
                    for role in (0,1):
                        for axis in (0,1):
                            v=proof.bound(role,box,axis);counts[role][axis]+=int(v.lo>0 or v.hi<0)
            rows.append(dict(anchor=anchor,strict_partial_boxes=counts,total_boxes=16))
    p=Path(os.environ['GFFX_FULL_REFERENCE']);assert not p.exists();p.write_text(json.dumps(dict(rows=rows,scope='Coordinate feasibility only; no numerical integral acceptance.'),indent=2),encoding='utf8');record_property('coordinate_feasibility',str(p))


def test_remaining_front_subdivision_response(record_property):
    from tools.opaque_original_motion_reference import reference
    from tools.opaque_arb_jets import backend
    from tools.opaque_shading_intervals import I
    from tests.packaging.test_opaque_motion_tangent import frozen
    rows=[];profile=dict(correction_order=4,conditioned_correction=True,mixed_curve=True,transverse_correction=True,correction_axis=0,material_inputs=True)
    for box in [(Q(7,8),Q(15,16),Q(1,2),Q(3,4)),(Q(15,32),Q(1,2),Q(0),Q(1)),(Q(3,8),Q(13,32),Q(0),Q(1))]:
        a,b,c,d=box
        for axis in (0,1):
            children=[(a,(a+b)/2,c,d),((a+b)/2,b,c,d)] if axis==0 else [(a,b,c,(c+d)/2),(a,b,(c+d)/2,d)]
            with backend(profile='ball',fast_conversion=True):
                start=time.monotonic();r=reference(frozen(),piece=2,motion='normal_tilt',anchor=0,diagnostic_boxes=[box,*children],**profile);v=r['rows'];combined=[x+y for x,y in zip(v[1]['ordinary'],v[2]['ordinary'])]
                # Each child chooses its own smooth extension. Residual centers
                # cannot be compared; widths guide work only, never acceptance.
                rows.append(dict(box=list(map(str,box)),axis=axis,seconds=time.monotonic()-start,parent_widths=[str(v.width) for v in v[0]['ordinary']],children_widths=[str(v.width) for v in combined],result=r))
    p=Path(os.environ['GFFX_FULL_REFERENCE']);assert not p.exists();p.write_text(json.dumps(dict(rows=rows,scope='Residual uncertainty for parent versus children with their own smooth extensions; centers are not comparable. Work selection only.'),default=str,indent=2),encoding='utf8');record_property('subdivision_response',str(p))


def test_contraction_reuses_independent_slope_without_rebuilding_field_jets():
    from tools.opaque_tangent_transverse import contract_crossing
    from tools.opaque_shading_intervals import I
    orders=[]
    def inputs(x,y):orders.append(x.order);return x*x+y,x.lift(1),x.lift(1),x.lift(1)
    def partial(role,box,axis):assert role==0 and axis==0;return 2*box[0]
    got=contract_crossing(inputs,(I('.2','.8'),I('.25')),0,0,I('.5'),I('.2','.8'),partial_bound=partial)
    assert got.lo<=I('.5').lo<=got.hi and got.width<I('1e-7').lo
    assert orders and set(orders)=={0}


def test_independent_slope_reuse_keeps_uncertain_roots_and_full_bracket():
    import math
    from tools.opaque_tangent_transverse import contract_crossing
    from tools.opaque_shading_intervals import I
    from tools.opaque_arb_jets import backend
    with backend(profile='ball',fast_conversion=True):
        for flip in (False,True):
            boxes=[]
            def inputs(x,y):return ((1-x) if flip else x)**2+y,x.lift(1),x.lift(1),x.lift(1)
            def partial(role,box,axis):
                assert role==axis==0;boxes.append(box)
                return -2*(1-box[0]) if flip else 2*box[0]
            box=(I('.1','.9'),I('.001','.002'));got=contract_crossing(inputs,box,0,0,I('.3'),box[0],partial_bound=partial)
            assert boxes[0][0].lo==box[0].lo and boxes[0][0].hi==box[0].hi
            assert all(b[1].lo==box[1].lo and b[1].hi==box[1].hi for b in boxes)
            for y in (Q(1,1000),Q(3,2000),Q(1,500)):
                value=Q(3,10)-y;scale=10**80;n=math.isqrt(value.numerator*scale*scale//value.denominator);lo,hi=Q(n,scale),Q(n+1,scale)
                if flip:lo,hi=1-hi,1-lo
                assert Q(got.lo)<=lo<=hi<=Q(got.hi)
        tiny=Q(1,2**70)
        def linear(x,y):return x+y,x.lift(1),x.lift(1),x.lift(1)
        got=contract_crossing(linear,(I(0,1),I(0)),0,0,I(tiny),I(0,1),partial_bound=lambda *a:I(1))
        assert Q(got.lo)<=tiny<=Q(got.hi) and got.lo>0
        box=(I(-1,1),I(0));got=contract_crossing(linear,box,0,0,I('.5'),box[0],partial_bound=lambda *a:I(-1,1))
        assert got.lo==-1 and got.hi==1


def test_next_piece_saved_transport_and_contraction_cost(record_property,monkeypatch):
    import importlib.util
    import tools.opaque_original_motion_reference as m
    import tools.opaque_tangent_transverse as t
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    oldpath=Path('D:/devspace/gffx/qualification-20260913/m1-continuation-20260913/opaque152-20260930/piece3-normal1-candidate/tools/opaque_tangent_transverse.py')
    spec=importlib.util.spec_from_file_location('frozen_before_root_cost',oldpath);oldmodule=importlib.util.module_from_spec(spec);spec.loader.exec_module(oldmodule)
    old=t.contract_crossing;original=m.material_transport;rows=[]
    boxes=[(Q(7,16),Q(1,2),Q(1,2),Q(1)),(Q(5,8),Q(11,16),Q(1,2),Q(1)),(Q(1,2),Q(9,16),Q(1,2),Q(1))]
    for selected,call in [('prior',oldmodule.contract_crossing),('reuse',old)]:
        flux=[]
        def capture(*args,**kw):
            start=time.monotonic();r=original(*args,**kw);flux.append(dict(box=list(map(str,args[2])),seconds=time.monotonic()-start,result=r));return r
        monkeypatch.setattr(m,'material_transport',capture);monkeypatch.setattr(t,'contract_crossing',call)
        with backend(profile='ball',fast_conversion=True):
            start=time.monotonic();r=m.reference(frozen(),piece=3,motion='normal_tilt',anchor=0,diagnostic_boxes=boxes,correction_order=4,conditioned_correction=True,mixed_curve=True,transverse_correction=True,correction_axis=0,material_inputs=True)
            rows.append(dict(selected=selected,seconds=time.monotonic()-start,flux=flux,result=r))
    # Both methods retain all terms. Interval endpoint equality is not required:
    # independently proved derivatives can change enclosure widths.
    for a,b in zip(rows[0]['result']['rows'],rows[1]['result']['rows']):
        for key in ('ordinary','combined_tangent'):
            if key in a:
                for x,y in zip(a[key],b[key]):assert (x.lo,x.hi)==(y.lo,y.hi)
    path=Path(os.environ['GFFX_FULL_REFERENCE']);assert not path.exists();path.write_text(json.dumps(dict(rows=rows,scope='Three saved piece3 boxes; complete correction and conservative flux, local timings only.'),default=str,indent=2),encoding='utf8');record_property('next_piece_transport',str(path))


def test_next_piece_subdivision_response(record_property):
    from tools.opaque_original_motion_reference import reference
    from tools.opaque_arb_jets import backend
    from tools.opaque_shading_intervals import I
    from tests.packaging.test_opaque_motion_tangent import frozen
    rows=[];profile=dict(correction_order=4,conditioned_correction=True,mixed_curve=True,transverse_correction=True,correction_axis=0,material_inputs=True)
    for box in [(Q(7,16),Q(1,2),Q(1,2),Q(1)),(Q(5,8),Q(11,16),Q(1,2),Q(1)),(Q(1,2),Q(9,16),Q(1,2),Q(1))]:
        a,b,c,d=box
        for axis in (0,1):
            children=[(a,(a+b)/2,c,d),((a+b)/2,b,c,d)] if axis==0 else [(a,b,c,(c+d)/2),(a,b,(c+d)/2,d)]
            with backend(profile='ball',fast_conversion=True):
                start=time.monotonic();r=reference(frozen(),piece=3,motion='normal_tilt',anchor=0,diagnostic_boxes=[box,*children],**profile);v=r['rows'];combined=[x+y for x,y in zip(v[1]['ordinary'],v[2]['ordinary'])]
                # Each child chooses its own smooth extension. Residual centers
                # cannot be compared; widths guide work only, never acceptance.
                rows.append(dict(box=list(map(str,box)),axis=axis,seconds=time.monotonic()-start,parent_widths=[str(v.width) for v in v[0]['ordinary']],children_widths=[str(v.width) for v in combined],result=r))
    p=Path(os.environ['GFFX_FULL_REFERENCE']);assert not p.exists();p.write_text(json.dumps(dict(rows=rows,scope='Residual uncertainty for parent versus children with their own smooth extensions; centers are not comparable. Work selection only.'),default=str,indent=2),encoding='utf8');record_property('subdivision_response',str(p))



def test_piece10_saved_correction_directions_and_refinement(record_property):
    from tools.opaque_original_motion_reference import reference
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    seed=Path(os.environ['GFFX_FRONT_DIAGNOSTIC']);spec=json.loads(seed.read_bytes());source=Path(spec['path'])
    import hashlib
    assert hashlib.sha256(source.read_bytes()).hexdigest()==spec['sha256']
    failed=json.loads(source.read_bytes());assert failed['status']=='unresolved' and failed['case']==dict(piece=10,motion='normal_tilt',anchor=0)
    boxes=[tuple(map(Q,row)) for row in spec['boxes']];assert len(boxes)==3
    rows=[]
    for box in boxes:
        a,b,c,d=box
        children=[(a,(a+b)/2,c,d),((a+b)/2,b,c,d)]
        for direction in (0,1):
            with backend(profile='ball',fast_conversion=True):
                start=time.monotonic();result=reference(frozen(),piece=10,motion='normal_tilt',anchor=0,diagnostic_boxes=[box,*children],correction_order=4,conditioned_correction=True,mixed_curve=True,transverse_correction=True,correction_axis=direction,material_inputs=True)
                v=result['rows'];combined=[x+y for x,y in zip(v[1]['combined_tangent'],v[2]['combined_tangent'])]
                rows.append(dict(box=list(map(str,box)),integrated_axis=direction,seconds=time.monotonic()-start,parent_tangent_widths=[str(q.width) for q in v[0]['combined_tangent']],radial_children_tangent_widths=[str(q.width) for q in combined],result=result))
    p=Path(os.environ['GFFX_FULL_REFERENCE']);assert not p.exists();p.write_text(json.dumps(dict(rows=rows,source_sha256=spec['sha256'],scope='Three saved piece10 regions, two integration directions and fully recomputed radial children; local bounds/cost only, no complete-piece acceptance.'),default=str,indent=2),encoding='utf8');record_property('piece10_direction_diagnostic',str(p))
