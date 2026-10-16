"""Motion identity and complete moving-map obligations for shared references."""
from fractions import Fraction as Q
from types import SimpleNamespace
import pytest
from tools.opaque_shading_intervals import I,Unresolved

@pytest.mark.parametrize('motion',['normal_tilt','view_x','screen_x'])
def test_requested_motion_reaches_geometry_before_construction(monkeypatch,motion):
    import tools.opaque_original_motion_reference as m
    called=[]
    def stop(data,**kw):called.append(kw);raise Unresolved('test construction stop')
    monkeypatch.setattr(m,'prepare',stop)
    with pytest.raises(Unresolved,match='construction stop'):
        m.angular_material_cover({},piece=9,motion=motion,anchor=2)
    assert called[0]['piece']==9 and called[0]['motion']==motion and called[0]['anchor']==2

@pytest.mark.parametrize('case',[dict(piece=True),dict(piece=-1),dict(piece=13),dict(motion='world_camera'),dict(anchor=True),dict(anchor=3)])
def test_angular_selection_refuses_undeclared_inputs_before_geometry(case):
    import tools.opaque_original_motion_reference as m
    for method in [m.angular_material_cover,m.angular_piece_reference]:
        with pytest.raises(ValueError,match='declared angular piece/motion/anchor'):method({},**case)

@pytest.mark.parametrize('motion',['view_x','screen_x'])
def test_complete_and_failed_reports_keep_requested_identity(monkeypatch,motion):
    import tools.opaque_original_motion_reference as m
    calls=[]
    def cover(data,**kw):calls.append(kw);return dict(face=5743),None,SimpleNamespace(cells=1,depth=0,evaluations=0),[]
    monkeypatch.setattr(m,'angular_material_cover',cover)
    monkeypatch.setattr(m,'integrate_square',lambda *a,**kw:dict(value=[I(0),I(0)],derivative=[I(0),I(0)]))
    r=m.angular_piece_reference({'material':{'albedo':[.2,.4,.6]}},piece=9,motion=motion,anchor=2)
    assert (r['piece'],r['face'],r['motion'],r['anchor'])==(9,5743,motion,2)
    assert calls[0]['piece']==9 and calls[0]['motion']==motion and calls[0]['anchor']==2
    def fail(*a,**kw):raise ValueError('retained failure')
    monkeypatch.setattr(m,'integrate_square',fail)
    with pytest.raises(Unresolved) as caught:m.angular_piece_reference({},piece=9,motion=motion,anchor=2)
    assert all(caught.value.details[k]==v for k,v in [('piece',9),('motion',motion),('anchor',2)])


PROFILE=dict(tight_discriminant=True,algebraic_inverse=True,mapped_transport=True,exit_angle=True,unsquared_inverse=True,physical_angle=True,tensor_rule='gauss3x8-local4',compiled_series=True,correction_order=4,conditioned_correction=True,contraction_stop=True,mixed_curve=True,material_inputs=True)

@pytest.mark.parametrize('motion',['normal_tilt','view_x','screen_x'])
def test_declared_motion_builds_fresh_complete_cover_and_keeps_horizon(motion):
    from tools.opaque_original_motion_reference import angular_material_cover
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen,test_original_motion_fields_match_independent_vector_chain,test_original_curved_map_keeps_moving_horizon_zero
    with backend(profile='ball',fast_conversion=True):
        p,curve,cover,charts=angular_material_cover(frozen(),piece=8,motion=motion,anchor=0,**PROFILE)
        assert p['piece']==8 and p['motion']==motion and p['face']==5742
        assert len(charts)==72 and cover.cells==143
        assert sum(Q(1,2**r['depth']) for r in charts)==1
        assert cover.radial.lo>0 and cover.edge_slope.hi<0
        test_original_motion_fields_match_independent_vector_chain(motion)
        test_original_curved_map_keeps_moving_horizon_zero(motion,8)


def test_selected_arithmetic_retains_independent_roots_measure_and_tiny_motion(monkeypatch):
    import tools.opaque_arb_jets as a
    from tests.packaging.test_opaque_tight_transport import test_certified_inverse_contraction_stop_keeps_closed_controls,test_direct_quarter_conditioned_proof_keeps_independent_integrals
    old=a.backend
    def selected(*args,**kw):kw['fast_conversion']=True;return old(*args,**kw)
    monkeypatch.setattr(a,'backend',selected)
    test_certified_inverse_contraction_stop_keeps_closed_controls(monkeypatch)
    test_direct_quarter_conditioned_proof_keeps_independent_integrals(monkeypatch)


def test_selected_profile_refuses_a_noncurved_piece():
    from tools.opaque_original_motion_reference import angular_material_cover
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    with backend(profile='ball',fast_conversion=True):
        with pytest.raises(ValueError,match='outside curved'):angular_material_cover(frozen(),piece=0,**PROFILE)


def test_selected_motion_piece_fresh_reference(record_property):
    import os,json,hashlib
    from pathlib import Path
    from tools.opaque_original_motion_reference import angular_piece_reference
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    case=json.loads(os.environ['GFFX_MOTION_CASE']);assert set(case)=={'piece','motion','anchor'}
    path=Path(os.environ['GFFX_FULL_REFERENCE']);assert not path.exists();rows=[]
    seed=Path(os.environ['GFFX_MOTION_SCHEDULE']);assert hashlib.sha256(seed.read_bytes()).hexdigest()==os.environ['GFFX_MOTION_SCHEDULE_SHA']
    leaves=json.loads(seed.read_bytes())['leaves']
    try:
        with backend(profile='ball',fast_conversion=True):result=angular_piece_reference(frozen(),progress=rows.append,initial_leaves=leaves,**case,**PROFILE)
    except (ValueError,Unresolved) as e:
        path.write_text(json.dumps(dict(status='unresolved',case=case,profile=PROFILE,details=getattr(e,'details',str(e)),progress=rows),default=str,indent=2),encoding='utf8');raise
    path.write_text(json.dumps(dict(status='qualified',case=case,profile=PROFILE,result=result,progress=rows),default=str,indent=2),encoding='utf8');record_property('fresh_motion_reference',str(path))


@pytest.mark.parametrize('case',[dict(correction_order=True),dict(correction_order=5),dict(conditioned_correction=True),dict(mixed_curve=True)])
def test_front_correction_selection_refuses_invalid_profiles(case):
    from tools.opaque_original_motion_reference import reference
    with pytest.raises(ValueError,match='explicit original correction profile'):reference({},piece=1,motion='normal_tilt',**case)


def test_front_selected_correction_keeps_all_constructor_obligations(monkeypatch):
    import tools.opaque_original_motion_reference as m
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    calls=[]
    def stop(*a,**kw):calls.append(kw);raise Unresolved('constructor control')
    monkeypatch.setattr(m,'TangentCorrection',stop)
    with backend(profile='ball',fast_conversion=True):
        with pytest.raises(Unresolved,match='constructor control'):m.reference(frozen(),piece=1,motion='normal_tilt',correction_order=4,conditioned_correction=True,mixed_curve=True)
    r=calls[0];assert r['order']==4 and r['conditioned'] and r['mixed_curve'] and r['centered'] and r['horizon'] and not r['curved']


def test_front_saved_correction_comparison(record_property):
    import os,json,time
    from pathlib import Path
    from tools.opaque_original_motion_reference import reference
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    boxes=[(Q(7,16),Q(15,32),Q(1,2),Q(1)),(Q(5,8),Q(41,64),Q(1,2),Q(1)),(Q(31,32),Q(1),Q(3,4),Q(1))];rows=[]
    for selected in [False,True]:
        start=time.monotonic()
        with backend(profile='ball',fast_conversion=True):r=reference(frozen(),piece=1,motion='normal_tilt',diagnostic_boxes=boxes,**(dict(correction_order=4,conditioned_correction=True,mixed_curve=True) if selected else {}))
        rows.append(dict(selected=selected,seconds=time.monotonic()-start,result=r))
    for old,new in zip(rows[0]['result']['rows'],rows[1]['result']['rows']):
        for a,b in zip(old['ordinary'],new['ordinary']):assert max(a.lo,b.lo)<=min(a.hi,b.hi)
    path=Path(os.environ['GFFX_FULL_REFERENCE']);assert not path.exists();path.write_text(json.dumps(dict(rows=rows,scope='Three saved front-piece correction regions only; interval overlap is consistency, not accuracy acceptance.'),default=str,indent=2),encoding='utf8');record_property('front_comparison',str(path))



def test_front_schedule_forwards_complete_single_chart_and_charges(monkeypatch):
    import tools.opaque_original_motion_reference as m
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    seen=[];schedule=[dict(chart=0,box=['0','1','0','1'],depth=0)]
    def integrate(field,**kw):seen.append((field,kw));return dict(value=[I(0)]*2,derivative=[I(0)]*2)
    monkeypatch.setattr(m,'integrate_square',integrate)
    with backend(profile='ball',fast_conversion=True):m.reference(frozen(),piece=1,motion='normal_tilt',correction_order=4,conditioned_correction=True,mixed_curve=True,initial_leaves=schedule)
    field,r=seen[0];assert field is None and r['initial_leaves']==schedule and r['max_cells']==256 and r['max_depth']==10
    assert len(r['chart_cover'])==1 and r['chart_cover'][0]['depth']==0
    assert all(callable(r['chart_cover'][0][k]) for k in ['field','prepare','correction_integrator','tangent_correction_integrator'])


def test_front_selected_piece_reference(record_property):
    import os,json,hashlib
    from pathlib import Path
    from tools.opaque_original_motion_reference import reference
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    path=Path(os.environ['GFFX_FULL_REFERENCE']);assert not path.exists();rows=[];case=json.loads(os.environ['GFFX_MOTION_CASE']);seed=Path(os.environ['GFFX_MOTION_SCHEDULE']);assert hashlib.sha256(seed.read_bytes()).hexdigest()==os.environ['GFFX_MOTION_SCHEDULE_SHA'];leaves=json.loads(seed.read_bytes())['leaves']
    try:
        with backend(profile='ball',fast_conversion=True):result=reference(frozen(),progress=rows.append,initial_leaves=leaves,**case,correction_order=4,conditioned_correction=True,mixed_curve=True)
    except (ValueError,Unresolved) as error:
        path.write_text(json.dumps(dict(status='unresolved',case=case,details=getattr(error,'details',str(error)),progress=rows),default=str,indent=2),encoding='utf8');raise
    path.write_text(json.dumps(dict(status='qualified',case=case,result=result,progress=rows),default=str,indent=2),encoding='utf8');record_property('front_piece_reference',str(path))



def test_front_transverse_profile_requires_the_qualified_correction():
    from tools.opaque_original_motion_reference import reference
    for value in [True,1]:
        with pytest.raises(ValueError,match='explicit original correction profile'):reference({},piece=1,motion='normal_tilt',transverse_correction=value)


def test_front_transverse_constructor_preserves_selected_profile(monkeypatch):
    import tools.opaque_original_motion_reference as m
    old=m.reference
    def selected(*args,**kw):kw['transverse_correction']=True;return old(*args,**kw)
    monkeypatch.setattr(m,'reference',selected)
    test_front_selected_correction_keeps_all_constructor_obligations(monkeypatch)


def test_front_transverse_saved_comparison(record_property,monkeypatch):
    import tools.opaque_original_motion_reference as m
    old=m.reference
    def selected(*args,**kw):
        strong=kw.get('conditioned_correction',False)
        kw.update(correction_order=4,conditioned_correction=True,mixed_curve=True,transverse_correction=strong)
        return old(*args,**kw)
    monkeypatch.setattr(m,'reference',selected)
    test_front_saved_correction_comparison(record_property)



def test_front_remaining_proof_components(record_property,monkeypatch):
    import os,json
    from pathlib import Path
    import tools.opaque_tangent_peano as tp
    from tools.opaque_high_peano import _mass,kernel_range
    from tools.opaque_peano import gauss5
    from tools.opaque_original_motion_reference import reference
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    old=tp.TangentCorrection.axis_bounds;rows=[]
    def audit(self,box,axis,**kw):
        result=old(self,box,axis,**kw);h=I(box[axis].hi)-I(box[axis].lo);nodes,weights=gauss5();data=tuple((x.lo,x.hi,w.lo,w.hi) for x,w in zip(nodes,weights));r=result[1];d=r['derivative_range'];anchor=I(I((Q(d.lo)+Q(d.hi))/2).lo);ordinary=I((d-anchor).bound)*h**5*I(_mass(4,data));jump=I(0)
        for w in r['jumps']:
            t=(w['position']-I(box[axis].lo))/h;t=I(max(I(0).lo,t.lo),min(I(1).hi,t.hi))
            for j,v in enumerate(w['values']):jump=jump+I(v.bound)*h**(j+1)*I(kernel_range(t,j+1).bound)
        rows.append(dict(axis=axis,box=list(map(str,box)),ordinary_radius=str(ordinary),absolute_jump_bound=str(jump),jumps=len(r['jumps']),transverse_proofs=sum('transverse_integral' in w for w in r['jumps'])))
        return result
    monkeypatch.setattr(tp.TangentCorrection,'axis_bounds',audit)
    with backend(profile='ball',fast_conversion=True):result=reference(frozen(),piece=1,motion='normal_tilt',correction_order=4,conditioned_correction=True,mixed_curve=True,transverse_correction=True,diagnostic_boxes=[(Q(7,16),Q(15,32),Q(1,2),Q(1))])
    path=Path(os.environ['GFFX_FULL_REFERENCE']);assert not path.exists();path.write_text(json.dumps(dict(rows=rows,result=result,scope='Bound-component diagnostic only; no new acceptance.'),default=str,indent=2),encoding='utf8');record_property('front_proof_components',str(path))


def test_front_conditioned_directions_saved_comparison(record_property,monkeypatch):
    import os,json,time
    from pathlib import Path
    import tools.opaque_fourth_tangent as ft
    from tools.opaque_original_motion_reference import reference
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    boxes=[(Q(7,16),Q(15,32),Q(1,2),Q(1)),(Q(5,8),Q(41,64),Q(1,2),Q(1)),(Q(31,32),Q(1),Q(3,4),Q(1))];rows=[];old=ft.integral
    for axis in (1,0):
        for transverse in (False,True):
            def selected(*a,**kw):kw['integrated_axis']=axis;return old(*a,**kw)
            monkeypatch.setattr(ft,'integral',selected);start=time.monotonic()
            with backend(profile='ball',fast_conversion=True):r=reference(frozen(),piece=1,motion='normal_tilt',diagnostic_boxes=boxes,correction_order=4,conditioned_correction=True,mixed_curve=True,transverse_correction=transverse)
            rows.append(dict(axis=axis,transverse=transverse,seconds=time.monotonic()-start,result=r))
    for row in rows[1:]:
        for old,new in zip(rows[0]['result']['rows'],row['result']['rows']):
            for a,b in zip(old['ordinary'],new['ordinary']):assert max(a.lo,b.lo)<=min(a.hi,b.hi)
    path=Path(os.environ['GFFX_FULL_REFERENCE']);assert not path.exists();path.write_text(json.dumps(dict(rows=rows,scope='Three saved boxes, two independently controlled tensor identities; consistency and cost only.'),default=str,indent=2),encoding='utf8');record_property('front_direction_comparison',str(path))


def test_front_selected_gradient_guidance_reaches_shared_integrator(monkeypatch):
    import tools.opaque_original_motion_reference as m
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    class Correction:
        def __init__(self,*a,**kw):self.range_fallbacks=0;self.preferred_split_axis=1
        def __call__(self,box):return [I(0),I('-1e-10','1e-10')]
    monkeypatch.setattr(m,'TangentCorrection',Correction);monkeypatch.setattr(m,'material_transport',lambda *a,**kw:dict(bound=[I(0)]))
    def integrate(field,**kw):
        box=(I('.4','.5'),I('.5','1'));kw['prepare'](Q(9,20),Q(3,4),box);kw['tangent_correction_integrator'](box)
        assert kw['tangent_correction_integrator'].preferred_split_axis==1
        return dict(value=[I(0)]*2,derivative=[I(0)]*2)
    monkeypatch.setattr(m,'integrate_square',integrate)
    with backend(profile='ball',fast_conversion=True):m.reference(frozen(),piece=1,motion='normal_tilt',correction_order=4,conditioned_correction=True,mixed_curve=True)


def test_front_axis_selection_requires_conditioned_profile(monkeypatch):
    import tools.opaque_original_motion_reference as m
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    for axis in (0,False,2):
        with pytest.raises(ValueError,match='explicit original correction profile'):m.reference({},piece=1,motion='normal_tilt',correction_axis=axis)
    calls=[]
    def stop(*a,**kw):calls.append(kw);raise Unresolved('selected axis control')
    monkeypatch.setattr(m,'TangentCorrection',stop)
    with backend(profile='ball',fast_conversion=True):
        with pytest.raises(Unresolved,match='selected axis control'):m.reference(frozen(),piece=1,motion='normal_tilt',correction_order=4,conditioned_correction=True,mixed_curve=True,transverse_correction=True,correction_axis=0)
    assert calls[0]['integrated_axis']==0 and callable(calls[0]['transverse_partial'])


def _selected_front_tolerance(value):
    from decimal import Decimal,InvalidOperation
    if value is None:return '5e-13'
    if not isinstance(value,str):raise ValueError('explicit stronger reference tolerance required')
    try:q=Decimal(value)
    except InvalidOperation as e:raise ValueError('finite positive stronger reference tolerance required') from e
    if not q.is_finite() or not 0<q<=Decimal('5e-13'):raise ValueError('finite positive reference target at most5e-13required')
    return value

def test_front_guided_piece_reference(record_property):
    import os,json
    from pathlib import Path
    from tools.opaque_original_motion_reference import reference
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    path=Path(os.environ['GFFX_FULL_REFERENCE']);assert not path.exists();rows=[];case=json.loads(os.environ['GFFX_MOTION_CASE'])
    profile=dict(correction_order=4,conditioned_correction=True,mixed_curve=True,transverse_correction=True,correction_axis=0,material_inputs=True)
    if os.environ.get('GFFX_FRONT_REFERENCE_TOLERANCE') is not None:profile['tolerance']=_selected_front_tolerance(os.environ['GFFX_FRONT_REFERENCE_TOLERANCE'])
    if os.environ.get('GFFX_FRONT_REFINEMENT_AXIS') is not None:
        selected=os.environ['GFFX_FRONT_REFINEMENT_AXIS']
        if selected not in ('0','1'):raise ValueError('explicit fixed refinement axis0or1required')
        profile['refinement_axis']=int(selected)
    if os.environ.get('GFFX_FRONT_POLYNOMIAL_INPUTS') is not None:
        if os.environ['GFFX_FRONT_POLYNOMIAL_INPUTS']!='1':raise ValueError('explicit polynomial-input selection required')
        profile['polynomial_inputs']=True
    schedule={}
    if os.environ.get('GFFX_FRONT_FRESH_SCHEDULE'):
        import hashlib
        seed=Path(os.environ['GFFX_FRONT_FRESH_SCHEDULE']);assert hashlib.sha256(seed.read_bytes()).hexdigest()==os.environ['GFFX_FRONT_FRESH_SCHEDULE_SHA']
        schedule['initial_leaves']=json.loads(seed.read_bytes())['leaves']
    try:
        with backend(profile='ball',fast_conversion=True):result=reference(frozen(),progress=rows.append,**case,**profile,**schedule)
    except (ValueError,Unresolved) as error:
        path.write_text(json.dumps(dict(status='unresolved',case=case,profile=profile,details=getattr(error,'details',str(error)),progress=rows),default=str,indent=2),encoding='utf8');raise
    path.write_text(json.dumps(dict(status='qualified',case=case,profile=profile,result=result,progress=rows),default=str,indent=2),encoding='utf8');record_property('front_piece_reference',str(path))


def test_front_exact_partials_saved_boxes(record_property):
    import os,json
    from pathlib import Path
    from tools.opaque_original_motion_reference import prepare,mapped_inputs
    from tools.opaque_tangent_transverse import CosinePartials
    from tools.opaque_shading_intervals import Jet
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    rows=[]
    with backend(profile='ball',fast_conversion=True):
        p=prepare(frozen(),piece=1,motion='normal_tilt');proof=CosinePartials(p['fields'].coefficients)
        for a,b,c,d in [(Q(7,16),Q(15,32),Q(1,2),Q(1)),(Q(5,8),Q(41,64),Q(1,2),Q(1)),(Q(31,32),Q(1),Q(3,4),Q(1))]:
            box=(I(I(a).lo,I(b).hi),I(I(c).lo,I(d).hi));raw=mapped_inputs(p,Jet(box[0]),Jet(box[1]));roles=[]
            for role in (0,1):
                old=raw[role][0,1];new=proof.bound(role,box,2);assert max(old.lo,new.lo)<=min(old.hi,new.hi)
                roles.append(dict(raw=str(old),exact_polynomial=str(new),raw_width=str(old.width),polynomial_width=str(new.width)))
            rows.append(dict(box=list(map(str,box)),roles=roles))
    path=Path(os.environ['GFFX_FULL_REFERENCE']);assert not path.exists();path.write_text(json.dumps(dict(rows=rows,scope='Derivative bound diagnostic only; no complete-piece or runtime acceptance.'),indent=2),encoding='utf8');record_property('front_partial_diagnostic',str(path))


@pytest.mark.parametrize('motion',['normal_tilt','view_x','screen_x'])
@pytest.mark.parametrize('order',[0,1,4,10])
def test_front_material_only_keeps_all_consumed_endpoints(motion,order):
    from tools.opaque_original_motion_reference import prepare,mapped_inputs
    from tools.opaque_shading_intervals import Jet
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    with backend(profile='ball',fast_conversion=True):
        p=prepare(frozen(),piece=1,motion=motion);u=Jet.variable(I('.4','.45'),order=order) if order else Jet(I('.4','.45'));s=Jet(I('.5','.6'),order=order)
        full=mapped_inputs(p,u,s);reduced=mapped_inputs(p,u,s,material_only=True)
        for channel in (0,1,3):
            for i in range(order+1):
                for j in (0,1):assert (full[channel][i,j].lo,full[channel][i,j].hi)==(reduced[channel][i,j].lo,reduced[channel][i,j].hi)
        assert not reduced[2].c


def test_front_material_only_actual_cost(record_property,monkeypatch):
    import tools.opaque_original_motion_reference as m
    old=m.reference
    def selected(*args,**kw):
        reduced=kw.get('conditioned_correction',False)
        kw.update(correction_order=4,conditioned_correction=True,mixed_curve=True,transverse_correction=True,correction_axis=0,material_inputs=reduced)
        return old(*args,**kw)
    monkeypatch.setattr(m,'reference',selected);test_front_saved_correction_comparison(record_property)


def test_front_exact_motion_actual_cost(record_property,monkeypatch):
    import tools.opaque_original_motion_reference as m
    old=m.reference
    def selected(*args,**kw):
        strong=kw.get('conditioned_correction',False)
        kw.update(correction_order=4,conditioned_correction=True,mixed_curve=True,transverse_correction=True,correction_axis=0,material_inputs=True,exact_motion=strong)
        return old(*args,**kw)
    monkeypatch.setattr(m,'reference',selected);test_front_saved_correction_comparison(record_property)


def test_remaining_outside_curved_fresh_reference(record_property):
    import os,json
    from pathlib import Path
    from tools.opaque_original_motion_reference import angular_piece_reference
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    case=json.loads(os.environ['GFFX_MOTION_CASE']);assert case['piece'] in (8,9) and case['anchor']==0
    path=Path(os.environ['GFFX_FULL_REFERENCE']);assert not path.exists();rows=[];profile=dict(PROFILE)
    if os.environ.get('GFFX_CURVED_TRANSVERSE')=='1':profile['curved_transverse']=True
    if os.environ.get('GFFX_POLYNOMIAL_LIGHT')=='1':profile['polynomial_light']=True
    if os.environ.get('GFFX_CURVED_REFERENCE_TOLERANCE') is not None:
        from tools.opaque_reference_profile import stronger_target
        profile['tolerance']=stronger_target(os.environ['GFFX_CURVED_REFERENCE_TOLERANCE'])
    schedule={}
    if os.environ.get('GFFX_FRONT_FRESH_SCHEDULE'):
        import hashlib
        seed=Path(os.environ['GFFX_FRONT_FRESH_SCHEDULE']);assert hashlib.sha256(seed.read_bytes()).hexdigest()==os.environ['GFFX_FRONT_FRESH_SCHEDULE_SHA']
        schedule['initial_leaves']=json.loads(seed.read_bytes())['leaves']
    try:
        with backend(profile='ball',fast_conversion=True):result=angular_piece_reference(frozen(),progress=rows.append,**case,**profile,**schedule)
    except (ValueError,Unresolved) as e:
        path.write_text(json.dumps(dict(status='unresolved',case=case,profile=profile,details=getattr(e,'details',str(e)),progress=rows),default=str,indent=2),encoding='utf8');raise
    path.write_text(json.dumps(dict(status='qualified',case=case,profile=profile,result=result,progress=rows),default=str,indent=2),encoding='utf8');record_property('outside_curved_reference',str(path))


def test_piece9_saved_transverse_correction(record_property,monkeypatch):
    import os,json,time,hashlib
    from pathlib import Path
    from tools.opaque_original_motion_reference import angular_material_cover,angular_material_chart,mapped_inputs
    from tools.opaque_tangent_peano import TangentCorrection
    from tools.opaque_loss_extension import bounded_cosines
    from tools.opaque_shading_intervals import I,Jet
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    seed=Path(os.environ['GFFX_CURVED_DIAGNOSTIC']);spec=json.loads(seed.read_bytes());source=Path(spec['path']);assert hashlib.sha256(source.read_bytes()).hexdigest()==spec['sha256']
    failed=json.loads(source.read_bytes());assert failed['status']=='unresolved' and failed['case']==dict(piece=9,motion='normal_tilt',anchor=0)
    start=time.monotonic()
    def check():
        if time.monotonic()-start>=90:raise Unresolved('piece9 local diagnostic deadline')
    rows=[];index=spec['chart'];old=TangentCorrection.__init__
    with backend(profile='ball',fast_conversion=True):
        p,curve,cover,charts=angular_material_cover(frozen(),piece=9,motion='normal_tilt',anchor=0,checkpoint=check,**PROFILE)
        bounds=bounded_cosines(lambda u,s:list(mapped_inputs(p,u,s,flatten=False)[:2]),I(0,1),I(0,1),monotone=True,partial=True)
        def selected(self,*args,**kw):kw.update(transverse=True,curved_transverse=True);return old(self,*args,**kw)
        monkeypatch.setattr(TangentCorrection,'__init__',selected)
        candidate=angular_material_chart(p,curve,cover,index,bounds,checkpoint=check)
        assert len(spec['boxes'])==3
        for k,words in enumerate(spec['boxes']):
            a,b,c,d=map(Q,words);box=(I(I(a).lo,I(b).hi),I(I(c).lo,I(d).hi))
            cases=[]
            for label,row in ([('baseline',charts[index]),('transverse',candidate)] if k%2==0 else [('transverse',candidate),('baseline',charts[index])]):
                before=time.monotonic();row['prepare']((a+b)/2,(c+d)/2,box);ordinary=row['correction_integrator'](box);combined=row['tangent_correction_integrator'](box)
                cases.append(dict(selected=label,seconds=time.monotonic()-before,ordinary=list(map(str,ordinary)),combined_tangent=list(map(str,combined)),ordinary_widths=[str(x.width) for x in ordinary],combined_tangent_widths=[str(x.width) for x in combined]))
            rows.append(dict(chart=index,box=words,cases=cases))
    p=Path(os.environ['GFFX_FULL_REFERENCE']);assert not p.exists();p.write_text(json.dumps(dict(rows=rows,seconds=time.monotonic()-start,source_sha256=spec['sha256'],scope='Three saved piece9 boundary regions; generic certified transverse jump integration, every term retained. Alternating local timing only; no complete-piece or runtime admission.'),indent=2),encoding='utf8');record_property('piece9_transverse_diagnostic',str(p))


@pytest.mark.parametrize('bad',[dict(curved_transverse=1),dict(curved_transverse=True),dict(curved_transverse=True,physical_angle=True,unsquared_inverse=True,exit_angle=True,mapped_transport=True,algebraic_inverse=True,tight_discriminant=True)])
def test_curved_transverse_profile_refuses_unqualified_cover_before_geometry(monkeypatch,bad):
    import tools.opaque_original_motion_reference as m
    def forbidden(*a,**kw):pytest.fail('invalid profile reached geometry')
    monkeypatch.setattr(m,'prepare',forbidden)
    for method in (m.angular_material_cover,m.angular_piece_reference):
        with pytest.raises(ValueError,match='curved transverse'):method({},**bad)


def test_curved_transverse_whole_reference_keeps_profile_identity(monkeypatch):
    import tools.opaque_original_motion_reference as m
    calls=[]
    def cover(data,**kw):calls.append(kw);return dict(face=5742),None,SimpleNamespace(cells=1,depth=0,evaluations=0),[]
    monkeypatch.setattr(m,'angular_material_cover',cover)
    monkeypatch.setattr(m,'integrate_square',lambda *a,**kw:dict(value=[I(0),I(0)],derivative=[I(0),I(0)]))
    r=m.angular_piece_reference({'material':{'albedo':[.2,.4,.6]}},piece=9,motion='normal_tilt',anchor=0,**PROFILE,curved_transverse=True)
    assert calls[0]['curved_transverse'] is True and r['curved_transverse'] is True
    assert (r['piece'],r['motion'],r['anchor'])==(9,'normal_tilt',0)


@pytest.mark.parametrize('piece',[8,9])
def test_curved_transverse_real_cover_forwards_each_chart_and_keeps_motion(monkeypatch,piece):
    import tools.opaque_original_motion_reference as m
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    seen=[];old=m.TangentCorrection
    def selected(*a,**kw):seen.append(kw);return old(*a,**kw)
    monkeypatch.setattr(m,'TangentCorrection',selected)
    with backend(profile='ball',fast_conversion=True):
        p,curve,cover,charts=m.angular_material_cover(frozen(),piece=piece,motion='normal_tilt',anchor=0,**PROFILE,curved_transverse=True)
        assert p['curved_transverse'] is True and p['motion']=='normal_tilt'
        assert len(seen)==len(charts)==len(cover.charts)>0
        assert all(k['curved_transverse'] is True and k['transverse'] is True and k['curved'] is True and k['order']==4 for k in seen)


def test_remaining_inside_curved_fresh_reference(record_property):
    import os,json
    from pathlib import Path
    from tools.opaque_original_motion_reference import prepare,reference
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    case=json.loads(os.environ['GFFX_MOTION_CASE']);assert case['piece'] in (11,12) and case['anchor']==0
    path=Path(os.environ['GFFX_FULL_REFERENCE']);assert not path.exists();rows=[]
    with backend(profile='ball',fast_conversion=True):
        p=prepare(frozen(),**case);assert p['curved'] and p['patch'].inside
    try:
        with backend(profile='ball',fast_conversion=True):result=reference(frozen(),progress=rows.append,**case)
    except (ValueError,Unresolved) as e:
        path.write_text(json.dumps(dict(status='unresolved',case=case,profile={},details=getattr(e,'details',str(e)),progress=rows),default=str,indent=2),encoding='utf8');raise
    assert result['curved'] and result['curved_certificate']['region']=='radial_inside'
    path.write_text(json.dumps(dict(status='qualified',case=case,profile={},result=result,progress=rows),default=str,indent=2),encoding='utf8');record_property('inside_curved_reference',str(path))


def test_inside_angular_piece_fresh_reference(record_property):
    import os,json,hashlib
    from pathlib import Path
    from tools.opaque_inside_angle import inside_piece_reference
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    case=json.loads(os.environ['GFFX_MOTION_CASE']);assert case['piece'] in (11,12) and case['anchor']==0
    path=Path(os.environ['GFFX_FULL_REFERENCE']);assert not path.exists();rows=[];leaves=None;profile={'inside_angle':True};target={}
    if os.environ.get('GFFX_POLYNOMIAL_LIGHT')=='1':profile['polynomial_light']=True;target['polynomial_light']=True
    if os.environ.get('GFFX_CURVED_REFERENCE_TOLERANCE') is not None:
        from tools.opaque_reference_profile import stronger_target
        profile['tolerance']=stronger_target(os.environ['GFFX_CURVED_REFERENCE_TOLERANCE']);target['tolerance']=profile['tolerance']
    if os.environ.get('GFFX_FRONT_FRESH_SCHEDULE'):
        seed=Path(os.environ['GFFX_FRONT_FRESH_SCHEDULE']);assert hashlib.sha256(seed.read_bytes()).hexdigest()==os.environ['GFFX_FRONT_FRESH_SCHEDULE_SHA'];leaves=json.loads(seed.read_bytes())['leaves']
    try:
        with backend(profile='ball',fast_conversion=True):result=inside_piece_reference(frozen(),progress=rows.append,initial_leaves=leaves,**case,**target)
    except (ValueError,Unresolved) as e:
        path.write_text(json.dumps(dict(status='unresolved',case=case,profile=profile,details=getattr(e,'details',str(e)),progress=rows),default=str,indent=2),encoding='utf8');raise
    path.write_text(json.dumps(dict(status='qualified',case=case,profile=profile,result=result,progress=rows),default=str,indent=2),encoding='utf8');record_property('inside_angular_reference',str(path))

@pytest.mark.parametrize('value',['0','-1','1e-12','NaN','Infinity','bad',True,1])
def test_stronger_front_reference_tolerance_rejects_looser_or_invalid_values(value):
    with pytest.raises(ValueError):_selected_front_tolerance(value)

@pytest.mark.parametrize('value',[None,'5e-13','1e-13'])
def test_stronger_front_reference_tolerance_is_forwarded_without_changing_limits(monkeypatch,value):
    import tools.opaque_original_motion_reference as m
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    selected=_selected_front_tolerance(value);seen=[]
    def integrate(field,**kw):seen.append(kw);return dict(value=[I(0)]*2,derivative=[I(0)]*2,tolerance=kw['tolerance'])
    monkeypatch.setattr(m,'integrate_square',integrate)
    with backend(profile='ball',fast_conversion=True):r=m.reference(frozen(),piece=6,motion='normal_tilt',anchor=0,tolerance=selected,correction_order=4,conditioned_correction=True,mixed_curve=True,transverse_correction=True,correction_axis=0,material_inputs=True)
    assert seen[0]['tolerance']==selected and seen[0]['max_cells']==256 and seen[0]['max_depth']==10 and r['basis']['tolerance']==selected


def test_failed_normal_saved_refinement_directions(record_property):
    import os,json,hashlib,time
    from pathlib import Path
    from tools.opaque_original_motion_reference import reference
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    from decimal import Decimal as D
    failed=Path('D:/devspace/gffx/qualification-20260913/m1-continuation-20260913/opaque154-20260930/piece1-normal-tight4-reference.json')
    assert hashlib.sha256(failed.read_bytes()).hexdigest()=='6316d316bfd32855562dceb6a482d9c50e8af5e72053d44fefe779b3950af857'
    j=json.loads(failed.read_bytes());assert j['status']=='unresolved' and j['case']==dict(piece=1,motion='normal_tilt',anchor=0)
    saved=j['details']['cause']['last_complete_enclosure']['error_accounting']['largest_leaves'][:3]
    assert len(saved)==3 and all(r['depth']<10 for r in saved)
    boxes=[tuple(map(Q,r['box'])) for r in saved];start=time.monotonic();profile=dict(j['profile']);rows=[]
    def evaluate(qs):
        if time.monotonic()-start>=90:raise Unresolved('saved-region diagnostic deadline')
        r=reference(frozen(),piece=1,motion='normal_tilt',anchor=0,diagnostic_boxes=qs,**profile)
        if time.monotonic()-start>=90:raise Unresolved('saved-region diagnostic deadline')
        assert r['status']=='diagnostic_only' and r['piece']==1 and r['motion']=='normal_tilt'
        return r
    parse=lambda s:I(*map(D,s.strip('[]').split(','))) if isinstance(s,str) else I(s.lo,s.hi)
    with backend(profile='ball',fast_conversion=True):
        parent=evaluate(boxes)
        for axis in (0,1):
            children=[]
            for a,b,c,d in boxes:
                if axis==0:
                    m=(a+b)/2;pair=[(a,m,c,d),(m,b,c,d)]
                else:
                    m=(c+d)/2;pair=[(a,b,c,m),(a,b,m,d)]
                assert sum((z-x)*(v-y) for x,z,y,v in pair)==(b-a)*(d-c)
                children.extend(pair)
            results=[row for q in (children[:3],children[3:]) for row in evaluate(q)['rows']]
            for i,box in enumerate(boxes):
                summaries={}
                for term in ('ordinary','combined_tangent'):
                    old=[parse(v) for v in parent['rows'][i][term]];new=[parse(a)+parse(b) for a,b in zip(results[2*i][term],results[2*i+1][term])]
                    summaries[term]=dict(parent=list(map(str,old)),children_sum=list(map(str,new)),parent_widths=[str(v.width) for v in old],child_widths=[str(v.width) for v in new])
                rows.append(dict(box=list(map(str,box)),axis=axis,depth=saved[i]['depth'],children=[list(map(str,q)) for q in children[2*i:2*i+2]],terms=summaries))
    p=Path(os.environ['GFFX_FULL_REFERENCE']);assert not p.exists();p.write_text(json.dumps(dict(status='diagnostic_complete',failed_sha256='6316d316bfd32855562dceb6a482d9c50e8af5e72053d44fefe779b3950af857',profile=profile,case=j['case'],seconds=time.monotonic()-start,rows=rows,scope='Three saved region contribution/refinement bounds only. Complete smooth/gradient error and whole-piece checking runtime are unqualified; no reference or renderer admission.'),indent=2),encoding='utf8');record_property('failed_normal_refinement',str(p))


def test_remaining_front_saved_refinement_directions(record_property):
    """Fresh local diagnostics; saved certificates select geometry only."""
    import os,json,hashlib,time
    from pathlib import Path
    from decimal import Decimal as D
    from tools.opaque_original_motion_reference import reference
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    pinned=Path(os.environ['GFFX_FRONT_DIAGNOSTIC']);assert hashlib.sha256(pinned.read_bytes()).hexdigest()==os.environ['GFFX_FRONT_DIAGNOSTIC_SHA']
    spec=json.loads(pinned.read_bytes());assert len(spec['cases'])==3
    start=time.monotonic();rows=[];profile=dict(correction_order=4,conditioned_correction=True,mixed_curve=True,transverse_correction=True,correction_axis=0,material_inputs=True)
    def check():
        if time.monotonic()-start>=90:raise Unresolved('remaining front diagnostic deadline')
    parse=lambda s:I(*map(D,s.strip('[]').split(','))) if isinstance(s,str) else I(s.lo,s.hi)
    with backend(profile='ball',fast_conversion=True):
        for selected in spec['cases']:
            src=Path(selected['path']);assert hashlib.sha256(src.read_bytes()).hexdigest()==selected['sha256']
            old=json.loads(src.read_bytes());r=old['result'];piece=selected['piece'];assert old['status']=='qualified' and r['piece']==piece and r['motion']=='normal_tilt' and r['anchor']==0 and not r['curved']
            saved=r['basis']['error_accounting']['largest_leaves'][:3];assert len(saved)==3 and all(v['depth']<10 for v in saved)
            boxes=[tuple(map(Q,v['box'])) for v in saved]
            def evaluate(qs):
                check();v=reference(frozen(),piece=piece,motion='normal_tilt',anchor=0,diagnostic_boxes=qs,**profile);check()
                assert v['status']=='diagnostic_only' and v['piece']==piece and v['motion']=='normal_tilt';return v
            parents=evaluate(boxes)
            for axis in (0,1):
                children=[]
                for a,b,c,d in boxes:
                    m=((a+b) if axis==0 else (c+d))/2
                    pair=[(a,m,c,d),(m,b,c,d)] if axis==0 else [(a,b,c,m),(a,b,m,d)]
                    assert sum((z-x)*(v-y) for x,z,y,v in pair)==(b-a)*(d-c)
                    children.extend(pair)
                evaluated=[v for qs in (children[:3],children[3:]) for v in evaluate(qs)['rows']]
                for k,box in enumerate(boxes):
                    terms={}
                    for term in ('ordinary','combined_tangent'):
                        parent=[parse(v) for v in parents['rows'][k][term]];child=[parse(a)+parse(b) for a,b in zip(evaluated[2*k][term],evaluated[2*k+1][term])]
                        terms[term]=dict(parent=list(map(str,parent)),children_sum=list(map(str,child)),parent_widths=[str(v.width) for v in parent],child_widths=[str(v.width) for v in child])
                    rows.append(dict(piece=piece,box=list(map(str,box)),depth=saved[k]['depth'],axis=axis,terms=terms))
    dst=Path(os.environ['GFFX_FULL_REFERENCE']);assert not dst.exists();dst.write_text(json.dumps(dict(status='diagnostic_complete',selection_sha256=os.environ['GFFX_FRONT_DIAGNOSTIC_SHA'],seconds=time.monotonic()-start,profile=profile,rows=rows,scope='Three saved regions per front piece2/3/10. Directional contribution bounds only; no complete-piece, runtime or renderer admission.'),indent=2),encoding='utf8');record_property('remaining_front_refinement',str(dst))


@pytest.mark.parametrize('bad',[True,False,2,-1,'0'])
def test_front_fixed_refinement_axis_refuses_invalid_or_unqualified_selection(bad):
    from tools.opaque_original_motion_reference import reference
    with pytest.raises(ValueError,match='explicit front refinement axis'):reference({},piece=2,motion='normal_tilt',refinement_axis=bad)

@pytest.mark.parametrize('axis',[0,1])
def test_front_fixed_refinement_axis_is_geometry_only(monkeypatch,axis):
    import tools.opaque_original_motion_reference as m
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    seen=[]
    class Correction:
        def __init__(self,*a,**kw):self.range_fallbacks=0;self.preferred_split_axis=1-axis
        def __call__(self,box):return [I(0),I('-1e-10','1e-10')]
    monkeypatch.setattr(m,'TangentCorrection',Correction);monkeypatch.setattr(m,'material_transport',lambda *a,**kw:dict(bound=[I(0)]))
    def integrate(field,**kw):
        box=(I('.4','.5'),I('.5','1'));kw['prepare'](Q(9,20),Q(3,4),box)
        value=kw['correction_integrator'](box);tangent=kw['tangent_correction_integrator'](box)
        assert value[0].width==value[1].width==0 and tangent[1].lo==I('-1e-10').lo and tangent[1].hi==I('1e-10').hi
        assert kw['correction_integrator'].preferred_split_axis==axis and kw['tangent_correction_integrator'].preferred_split_axis==axis
        assert kw['max_cells']==256 and kw['max_depth']==10 and kw['tolerance']=='5e-13';seen.append(True)
        return dict(value=[I(0)]*2,derivative=[I(0)]*2)
    monkeypatch.setattr(m,'integrate_square',integrate)
    with backend(profile='ball',fast_conversion=True):r=m.reference(frozen(),piece=2,motion='normal_tilt',refinement_axis=axis,correction_order=4,conditioned_correction=True,mixed_curve=True,transverse_correction=True,correction_axis=0,material_inputs=True)
    assert seen==[True] and r['refinement_axis']==axis

def test_front_fixed_refinement_axis_refuses_a_curved_chart():
    from tools.opaque_original_motion_reference import reference
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    with backend(profile='ball',fast_conversion=True):
        with pytest.raises(ValueError,match='front refinement axis requires noncurved'):reference(frozen(),piece=9,motion='normal_tilt',refinement_axis=0,correction_order=4,conditioned_correction=True,mixed_curve=True,transverse_correction=True,correction_axis=0,material_inputs=True)
