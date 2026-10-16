"""Independent controls and bounded diagnostics for moving shading-boundary flux."""
from fractions import Fraction as Q
import pytest
from tools.opaque_shading_intervals import I,Jet

BOXES=[(Q(19,32),Q(5,8),Q(0),Q(1,4)),(Q(21,32),Q(11,16),Q(0),Q(1,4)),(Q(15,16),Q(1),Q(1,2),Q(5,8))]
PROFILE=dict(correction_order=4,conditioned_correction=True,mixed_curve=True,transverse_correction=True,correction_axis=0,material_inputs=True)

def test_saved_boundary_components(record_property,monkeypatch):
    import os,json,time
    from pathlib import Path
    import tools.opaque_original_motion_reference as m
    from tools.opaque_tangent_transverse import CosinePartials
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    rows=[];old=m.material_transport
    with backend(profile='ball',fast_conversion=True):
        p=m.prepare(frozen(),piece=1,motion='normal_tilt');partial=CosinePartials(p['fields'].coefficients)
        def audit(curve,field,box,*a,**kw):
            started=time.monotonic();r=old(curve,field,box,*a,**kw)
            row=dict(box=list(map(str,box)),bound=r['bound'],roles=[v for v in r['roles'] if v['channel']==0],slopes=[],seconds=time.monotonic()-started)
            for role in (0,1):
                slopes=[]
                for axis in (0,1):
                    args=[Jet(v,order=1) for v in box];args[axis]=Jet.variable(box[axis],order=1)
                    slopes.append(dict(axis=axis,raw=field(*args)[role][1,0],exact=partial.bound(role,box,axis),outer_width=box[1-axis].width))
                row['slopes'].append(dict(role=role,axes=slopes))
            rows.append(row);return r
        monkeypatch.setattr(m,'material_transport',audit)
        result=m.reference(frozen(),piece=1,motion='normal_tilt',diagnostic_boxes=BOXES,**PROFILE)
    path=Path(os.environ['GFFX_FULL_REFERENCE']);assert not path.exists()
    path.write_text(json.dumps(dict(rows=rows,result=result),default=str,indent=2),encoding='utf8');record_property('boundary_components',str(path))


def test_saved_weighted_jump_components(record_property,monkeypatch):
    import os,json,time
    from pathlib import Path
    import tools.opaque_tangent_transverse as tt
    import tools.opaque_fourth_tangent as ft
    from tools.opaque_high_transport import kernel_integral
    from tools.opaque_high_peano import kernel_range
    from tools.opaque_original_motion_reference import reference
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    rows=[];directions=[];old=tt.weighted;integration=ft.integral
    def audit(coefficient,a,b,order):
        r=old(coefficient,a,b,order)
        rows.append(dict(order=order,coefficient=coefficient,roots=[a,b],kernel=kernel_range(I(a.lo,b.hi),order),signed_mass=kernel_integral(a,b,order),result=r,width=str(r.width)))
        return r
    def integrate(*a,**kw):
        callback=kw.get('error_components')
        def observe(v):
            directions.append(dict(box=list(map(str,a[1])),directions=v))
            if callback:callback(v)
        kw['error_components']=observe;return integration(*a,**kw)
    monkeypatch.setattr(tt,'weighted',audit);monkeypatch.setattr(ft,'integral',integrate)
    with backend(profile='ball',fast_conversion=True):result=reference(frozen(),piece=1,motion='normal_tilt',diagnostic_boxes=BOXES,**PROFILE)
    path=Path(os.environ['GFFX_FULL_REFERENCE']);assert not path.exists();path.write_text(json.dumps(dict(rows=rows,directions=directions,result=result),default=str,indent=2),encoding='utf8');record_property('weighted_jump_components',str(path))


@pytest.mark.parametrize('order',[1,2,3,4])
def test_weighted_mass_uses_exact_first_segment_and_retains_tiny_terms(order):
    import math
    from tools.opaque_high_transport import weighted_mass
    for sign in (-1,1):
        scale=Q(sign,2**80);mass=Q(1,100)**(order+1)/math.factorial(order+1)
        result=weighted_mass(I(-2,2)*scale,I(0),I('.01'),order)
        exact=2*abs(scale)*mass
        assert Q(result.lo)<=-exact and Q(result.hi)>=exact
        assert Q(result.width)<Q(401,100)*abs(scale)*mass
        assert not result.zero

@pytest.mark.parametrize('order',[1,2,3,4])
@pytest.mark.parametrize('sign',[-1,1])
@pytest.mark.parametrize('domain',[(Q(0),Q(1)),(Q(1,7),Q(4,5)),(Q(49,100),Q(51,100))])
def test_weighted_mass_contains_independent_closed_polynomial(order,sign,domain):
    import math
    from decimal import Decimal as D,localcontext
    from tools.opaque_high_transport import weighted_mass
    a,b=domain;scale=Q(sign,2**60)
    result=weighted_mass(I(I(1+a).lo,I(1+b).hi)*scale,I(a),I(b),order)
    with localcontext() as ctx:
        ctx.prec=90;a,b=[D(v.numerator)/D(v.denominator) for v in domain]
        root=(D(10)/7).sqrt();x=(5+2*root).sqrt()/6;y=(5-2*root).sqrt()/6
        ns=[D('.5')-x,D('.5')-y,D('.5'),D('.5')+y,D('.5')+x];u=(322-13*D(70).sqrt())/1800;v=(322+13*D(70).sqrt())/1800;ws=[u,v,D(64)/225,v,u]
        expected=D(0);edges=sorted({a,b,*[x for x in ns if a<x<b]})
        for lo,hi in zip(edges,edges[1:]):
            co=[D(0)]*(order+1)
            for n,w in zip(ns,ws):
                if n>(lo+hi)/2:
                    for j in range(order):co[j]+=w*math.comb(order-1,j)*n**(order-1-j)*(-1)**j/D(math.factorial(order-1))
            for j in range(order+1):co[j]-=D(math.comb(order,j)*(-1)**j)/math.factorial(order)
            for j,z in enumerate(co):expected+=z*((hi**(j+1)-lo**(j+1))/(j+1)+(hi**(j+2)-lo**(j+2))/(j+2))
        expected*=D(scale.numerator)/D(scale.denominator)
        assert result.lo<=expected<=result.hi

@pytest.mark.parametrize('order',[1,2,3,4])
def test_weighted_mass_keeps_uncertain_roots_and_refuses_invalid_support(order):
    from tools.opaque_high_transport import weighted_mass,weighted
    a,b=I('.1','.2'),I('.7','.9');c=I(-3,2)
    got=weighted_mass(c,a,b,order);old=weighted(c,a,b,order)
    assert old.lo<=got.lo<=got.hi<=old.hi
    for x in ('.1','.15','.2'):
        for y in ('.7','.8','.9'):
            point=weighted_mass(I(2),I(x),I(y),order)
            assert got.lo<=point.lo<=point.hi<=got.hi
    assert weighted_mass(I(-1,2),I('.5'),I('.5'),order).zero
    for x,y in [(I(-1),I(1)),(I('.7'),I('.2'))]:
        with pytest.raises(ValueError):weighted_mass(c,x,y,order)


def test_weighted_mass_selected_arithmetic_and_bounded_cache():
    from tools.opaque_arb_jets import backend
    from tools.opaque_high_transport import weighted_mass,_absolute_mass
    with backend(profile='ball',fast_conversion=True):
        for order in (1,2,3,4):
            test_weighted_mass_uses_exact_first_segment_and_retains_tiny_terms(order)
            test_weighted_mass_keeps_uncertain_roots_and_refuses_invalid_support(order)
            for sign in (-1,1):
                for domain in [(Q(0),Q(1)),(Q(1,7),Q(4,5)),(Q(49,100),Q(51,100))]:test_weighted_mass_contains_independent_closed_polynomial(order,sign,domain)
        first=weighted_mass(I(-2,2),I(0),I('.01'),4);endpoint=first.hi;first.hi=I(999).hi
        assert weighted_mass(I(-2,2),I(0),I('.01'),4).hi==endpoint
        for n in range(130):weighted_mass(I(-1,1),I(Q(n,256)),I(Q(n+1,256)),4)
        assert _absolute_mass.cache_info().currsize<=128


def test_saved_weighted_mass_comparison(record_property,monkeypatch):
    import os,json,time
    from pathlib import Path
    import tools.opaque_tangent_transverse as tt
    from tools.opaque_high_transport import weighted,weighted_mass
    from tools.opaque_original_motion_reference import reference
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    rows=[]
    for selected in (False,True):
        monkeypatch.setattr(tt,'weighted',weighted_mass if selected else weighted);started=time.monotonic()
        with backend(profile='ball',fast_conversion=True):result=reference(frozen(),piece=1,motion='normal_tilt',diagnostic_boxes=BOXES,**PROFILE)
        rows.append(dict(selected=selected,seconds=time.monotonic()-started,result=result))
    metrics=[]
    for old,new in zip(rows[0]['result']['rows'],rows[1]['result']['rows']):
        a,b=old['combined_tangent'][1],new['combined_tangent'][1]
        assert a.lo<=b.lo<=b.hi<=a.hi
        metrics.append(dict(box=old['box'],old_width=str(a.width),new_width=str(b.width),ratio=float(b.width/a.width)))
    path=Path(os.environ['GFFX_FULL_REFERENCE']);assert not path.exists();path.write_text(json.dumps(dict(rows=rows,metrics=metrics,scope='Complete correction plus retained distributional flux on three saved regions. Local diagnostic, not complete-piece admission.'),default=str,indent=2),encoding='utf8');record_property('saved_mass_comparison',str(path))


def test_saved_boundary_cost_profile(record_property,monkeypatch):
    import os,json,time,cProfile,pstats
    from pathlib import Path
    import tools.opaque_tangent_transverse as tt
    from tools.opaque_high_transport import weighted,weighted_mass
    from tools.opaque_original_motion_reference import reference
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    rows=[]
    for selected in (False,True):
        monkeypatch.setattr(tt,'weighted',weighted_mass if selected else weighted);profile=cProfile.Profile();wall=time.monotonic();cpu=time.process_time()
        with backend(profile='ball',fast_conversion=True):
            profile.enable();r=reference(frozen(),piece=1,motion='normal_tilt',diagnostic_boxes=BOXES,**PROFILE);profile.disable()
        stats=pstats.Stats(profile).stats
        top=sorted([dict(file=k[0],line=k[1],function=k[2],calls=v[1],own_seconds=v[2],cumulative_seconds=v[3]) for k,v in stats.items()],key=lambda r:r['cumulative_seconds'],reverse=True)[:30]
        rows.append(dict(selected=selected,wall_seconds=time.monotonic()-wall,cpu_seconds=time.process_time()-cpu,top=top,complete_widths=[str(v['combined_tangent'][1].width) for v in r['rows']]))
    path=Path(os.environ['GFFX_FULL_REFERENCE']);assert not path.exists();path.write_text(json.dumps(dict(rows=rows,scope='Instrumented three-region diagnostic. Profiling time is not uninstrumented performance qualification.'),indent=2),encoding='utf8');record_property('boundary_cost_profile',str(path))


@pytest.mark.parametrize('axis',[0,1])
def test_quadrature_lines_skip_unused_continuous_proofs(axis):
    from tools.opaque_fourth_tangent import integral
    from tools.opaque_arb_jets import backend
    with backend(profile='ball',fast_conversion=True):
        calls=[]
        def bounds(box,active):
            d=360*box[0]**2*box[1]**3 if active==0 else I(0)
            return [dict(order=4,derivative=I(d.bound),derivative_range=d,jumps=[])]
        def continuous(box,active):calls.append('strip');assert box[1-active].width>I('1e-30').hi;return bounds(box,active)
        def line(box,active):calls.append('line');assert box[1-active].width<I('1e-30').lo;return bounds(box,active)
        field=lambda x,y:[x**6*y**3];kw=dict(conditioned=True,direct_conditioned=True,condition_parts=4,integrated_axis=axis)
        old=integral(field,(I(0,1),I(0,1)),bounds,**kw)
        got=integral(field,(I(0,1),I(0,1)),continuous,line_bounds=line,**kw)
        assert calls.count('line')==5 and calls.count('strip')==4
        assert got[0].lo==old[0].lo and got[0].hi==old[0].hi
        assert got[0].lo<=I(Q(1,28)).lo<=I(Q(1,28)).hi<=got[0].hi


def test_actual_front_builds_continuous_proofs_only_on_consumed_strips(monkeypatch):
    import tools.opaque_tangent_transverse as tt
    from tools.opaque_original_motion_reference import reference
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    old=tt.refine;calls=[]
    def count(rows,inputs,box,axis,knots,**kw):calls.append((box,axis));return old(rows,inputs,box,axis,knots,**kw)
    monkeypatch.setattr(tt,'refine',count)
    with backend(profile='ball',fast_conversion=True):reference(frozen(),piece=1,motion='normal_tilt',diagnostic_boxes=BOXES[:1],**PROFILE)
    assert len(calls)==4
    assert all(box[1-axis].width>I('1e-30').hi for box,axis in calls)


def test_saved_line_proof_endpoints_and_cost(record_property,monkeypatch):
    import os,json,time
    from pathlib import Path
    import tools.opaque_fourth_tangent as ft
    from tools.opaque_original_motion_reference import reference
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    old=ft.integral;rows=[]
    for selected in (False,True):
        def choice(*a,**kw):
            if not selected:kw.pop('line_bounds',None)
            return old(*a,**kw)
        monkeypatch.setattr(ft,'integral',choice);wall=time.monotonic();cpu=time.process_time()
        with backend(profile='ball',fast_conversion=True):result=reference(frozen(),piece=1,motion='normal_tilt',diagnostic_boxes=BOXES,**PROFILE)
        rows.append(dict(selected=selected,seconds=time.monotonic()-wall,cpu_seconds=time.process_time()-cpu,result=result))
    for a,b in zip(rows[0]['result']['rows'],rows[1]['result']['rows']):
        for key in ('ordinary','combined_tangent'):
            assert [(v.lo,v.hi) for v in a[key]]==[(v.lo,v.hi) for v in b[key]]
    path=Path(os.environ['GFFX_FULL_REFERENCE']);assert not path.exists();path.write_text(json.dumps(dict(rows=rows,all_endpoints_identical=True),default=str,indent=2),encoding='utf8');record_property('line_proof_cost',str(path))
