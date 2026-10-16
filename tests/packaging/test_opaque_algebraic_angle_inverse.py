from fractions import Fraction as Q
import pytest
from tools.opaque_shading_intervals import I,Jet

@pytest.mark.parametrize('axis',[0,1])
def test_residual_inverse_retains_closed_high_and_motion_coefficients(axis):
    from tools.opaque_knot_partition import MonotoneAngularCover
    from tools.opaque_arb_jets import backend
    field=lambda u,s:(u+u*u)/(1+s)+u*u.lift(0,tangent=Q(1,8))
    radial=lambda u,s:(1+2*u)/(1+s)+u.lift(0,tangent=Q(1,8))
    calls=[]
    def residual(u,s,k):
        calls.append(u.order);return u+u*u+(u*u.lift(0,tangent=Q(1,8))-k)*(1+s)
    with backend(profile='ball'):
        cover=MonotoneAngularCover(field,[I(Q(1,8)),I(Q(1,4))],inverse_residual=residual)
        args=[Jet(Q(1,4),order=10),Jet(Q(1,4),order=10)];args[axis]=Jet.variable(Q(1,4),order=10)
        t,y=args;row=cover.map_angle(1,t,y,radial);k=(1+t)/8;b=1+(1+y)*t.lift(0,tangent=Q(1,8))
        exact=(-b+(b*b+4*k*(1+y)).sqrt())/2;jac=(1+y)/(8*(b*b+4*k*(1+y)).sqrt())
        for actual,want in [(row['u'],exact),(row['jacobian'],jac)]:
            for i in range(11):
                for j in (0,1):
                    assert actual[i,j].lo<=want[i,j].hi and want[i,j].lo<=actual[i,j].hi
                    assert actual[i,j].width<I('1e-30').lo
        # The radial-edge map uses the same residual recurrence, preserving motion.
        row=cover.map(1,*args)
        for key in ('u','s','jacobian'):assert all(v.width<I('1e-28').lo for v in row[key].c.values())
        assert 10 in calls


def test_residual_inverse_refuses_uncertified_slope_and_bad_contract():
    from tools.opaque_knot_partition import MonotoneAngularCover
    field=lambda u,s:u/(1+s)
    with pytest.raises(ValueError):MonotoneAngularCover(field,[I('.125')],inverse_residual=42)
    cover=MonotoneAngularCover(field,[I('.125')],inverse_residual=lambda u,s,k:0*u)
    with pytest.raises(ValueError,match='residual radial derivative'):cover.map_angle(0,Jet('.5'),Jet('.5'),lambda u,s:1/(1+s))


@pytest.mark.parametrize('motion',['normal_tilt','view_x','screen_x'])
def test_original_quartic_residual_encloses_direct_vector_identity(motion):
    from tools.opaque_algebraic_angle_inverse import QuarticAngleResidual
    from tools.opaque_prepared_angle import PreparedAngle
    from tools.opaque_original_motion_reference import prepare
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    with backend(profile='ball'):
        p=prepare(frozen(),piece=8,motion=motion,anchor=0,tight_discriminant=True);residual=QuarticAngleResidual(PreparedAngle(p))
        def direct(u,s,k):
            t,_=p['patch'].radial_map(u,s);fields=p['fields'](t,s);n,v=fields['n'],fields['v']
            dot=lambda a,b:sum((x*y for x,y in zip(a,b)),u.lift(0))
            a=dot(n,v);return a*a-k*k*dot(n,n)*dot(v,v)
        for axis in (0,1):
            args=[Jet(I('.25','.75'),order=10),Jet(I('.25','.75'),order=10)]
            args[axis]=Jet.variable(args[axis][0,0],order=10);args[0]+=args[0].lift(0,tangent=Q(1,17));args[1]+=args[1].lift(0,tangent=Q(1,19))
            box=residual(*args,Jet(Q(1,16),order=10))
            for x in (Q(1,4),Q(1,2),Q(3,4)):
                pts=[Jet(x,order=10),Jet(x,order=10)];pts[axis]=Jet.variable(x,order=10);pts[0]+=pts[0].lift(0,tangent=Q(1,17));pts[1]+=pts[1].lift(0,tangent=Q(1,19))
                k=Jet(Q(1,16),order=10);a=residual(*pts,k);b=direct(*pts,k)
                for i in range(11):
                    for j in (0,1):
                        assert a[i,j].lo<=b[i,j].hi and b[i,j].lo<=a[i,j].hi,(axis,i,j)
                        assert box[i,j].lo<=b[i,j].hi and b[i,j].lo<=box[i,j].hi,(axis,i,j)
                        assert a[i,j].width<I('1e-25').lo
        assert residual.report()['entries']<=64
        with pytest.raises(ValueError):residual(Jet('.5'),Jet('.5'),Jet(0))



def test_algebraic_shared_original_chart27_integrates(record_property):
    import json,os,time
    from pathlib import Path
    from tools.opaque_original_motion_reference import angular_material_cover
    from tools.opaque_square_reference import integrate_square
    from tools.opaque_shading_intervals import Unresolved
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    start=time.monotonic();failure=None
    def check():
        if time.monotonic()-start>=90:raise Unresolved('algebraic local construction inclusive deadline')
    with backend(profile='ball'):
        try:
            p,curve,cover,charts=angular_material_cover(frozen(),checkpoint=check,tight_discriminant=True,algebraic_inverse=True);row=charts[27]
            result=integrate_square(row['field'],prepare=row['prepare'],correction_integrator=row['correction_integrator'],tangent_correction_integrator=row['tangent_correction_integrator'],max_cells=256-cover.cells+1,max_depth=10-row['depth'],tolerance='5e-13',checkpoint=check)
            result=dict(status='local_chart_passed',basis=result,seconds=time.monotonic()-start,evaluations=cover.evaluations,residual=p['inverse_residual'].report())
        except ValueError as error:failure=str(error);result=dict(status='unresolved',failure=getattr(error,'details',str(error)),seconds=time.monotonic()-start)
    path=Path(os.environ['GFFX_FULL_REFERENCE']);assert not path.exists();path.write_text(json.dumps(result,default=str,indent=2),encoding='utf8');record_property('algebraic_local',path.read_text());assert failure is None,result


def test_algebraic_shared_original_complete_piece(record_property):
    import json,os
    from pathlib import Path
    from tools.opaque_original_motion_reference import angular_piece_reference
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    path=Path(os.environ['GFFX_FULL_REFERENCE']);failure=None;rows=[]
    with backend(profile='ball'):
        try:result=angular_piece_reference(frozen(),tight_discriminant=True,algebraic_inverse=True,progress=rows.append)
        except ValueError as error:failure=str(error);result=dict(status='unresolved',failure=getattr(error,'details',str(error)))
    result['progress']=rows
    assert not path.exists();path.write_text(json.dumps(result,default=str,indent=2),encoding='utf8');record_property('algebraic_shared_piece',path.read_text())
    assert failure is None,result
    assert result['algebraic_inverse'] and result['basis']['chart_count']==72 and result['basis']['cells']<=256
    assert all(v.width<=I('2e-12').lo for v in result['value']+result['derivative'])


@pytest.mark.parametrize('scale',[Q(1),Q(1,2**60)])
def test_residual_cover_keeps_complete_moving_measure_and_tiny_terms(scale):
    from tools.opaque_knot_partition import MonotoneAngularCover
    from tools.opaque_square_reference import integrate_square
    from tools.opaque_arb_jets import backend
    field=lambda u,s:u*(1+u.lift(0,tangent=Q(1,7)))/(1+s)
    radial=lambda u,s:(1+u.lift(0,tangent=Q(1,7)))/(1+s)
    residual=lambda u,s,k:u*(1+u.lift(0,tangent=Q(1,7)))-k*(1+s)
    with backend(profile='ball'):
        cover=MonotoneAngularCover(field,[I(Q(1,4)),I(Q(3,4))],inverse_residual=residual)
        charts=[]
        for index,c in enumerate(cover.charts):
            def integrand(t,y,index=index):
                row=cover.map_angle_cover(index,t,y,radial);mass=scale*row['jacobian']*(1+row['s'])*(1+t.lift(0,tangent=Q(1,5)))
                return [mass,mass*row['u']]
            charts.append(dict(field=integrand,depth=c['depth']))
        result=integrate_square(None,chart_cover=charts,tolerance=str((I(scale)*I('1e-12')).lo))
        assert result['cells']<=256
        for value,tangent,want in zip(result['value'],result['derivative'],[scale*Q(3,2),scale*Q(3,4)]):
            expected=I(want);assert value.lo<=expected.lo<=expected.hi<=value.hi
            expected=I(want/5);assert tangent.lo<=expected.lo<=expected.hi<=tangent.hi
            assert value.lo>0 and tangent.lo>0


def test_algebraic_selector_is_explicit_and_keeps_construction_deadline(monkeypatch):
    import types
    import tools.opaque_original_motion_reference as m
    from tools.opaque_shading_intervals import Unresolved
    clock=[0.];calls=[]
    monkeypatch.setattr(m,'time',types.SimpleNamespace(monotonic=lambda:clock[0]))
    def construct(data,*,checkpoint,**kw):
        calls.append(kw);clock[0]+=91;checkpoint();raise AssertionError('deadline bypass')
    monkeypatch.setattr(m,'angular_material_cover',construct)
    with pytest.raises(ValueError):m.angular_piece_reference({},algebraic_inverse=True)
    with pytest.raises(ValueError):m.angular_piece_reference({},tight_discriminant=True,algebraic_inverse=1)
    with pytest.raises(Unresolved,match='deadline'):m.angular_piece_reference({},tight_discriminant=True,algebraic_inverse=True)
    assert calls==[dict(tight_discriminant=True,algebraic_inverse=True)]
    for kw in [dict(mapped_transport=True),dict(tight_discriminant=True,mapped_transport=1),dict(exit_angle=True),dict(tight_discriminant=True,algebraic_inverse=True,exit_angle=True),dict(tight_discriminant=True,algebraic_inverse=True,mapped_transport=True,exit_angle=1)]:
        with pytest.raises(ValueError):m.angular_piece_reference({},**kw)
    with pytest.raises(Unresolved,match='deadline'):m.angular_piece_reference({},tight_discriminant=True,algebraic_inverse=True,mapped_transport=True,exit_angle=True)
    assert calls[-1]==dict(tight_discriminant=True,algebraic_inverse=True,mapped_transport=True,exit_angle=True)


@pytest.mark.parametrize('motion',['normal_tilt','view_x','screen_x'])
def test_quartic_root_slope_identity_matches_direct_positive_derivative(motion):
    from tools.opaque_algebraic_angle_inverse import QuarticAngleResidual
    from tools.opaque_prepared_angle import PreparedAngle
    from tools.opaque_original_motion_reference import prepare
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    with backend(profile='ball'):
        p=prepare(frozen(),piece=8,motion=motion,anchor=0,tight_discriminant=True);field=PreparedAngle(p);residual=QuarticAngleResidual(field)
        for x in (Q(1,1000),Q(1,4),Q(3,4)):
            for y in (Q(1,4),Q(3,4)):
                u=Jet(x);s=Jet(y);k=field(u,s)[0,0];slope=field.radial(u,s)[0,0]
                exact=residual(Jet.variable(x,order=1),Jet(y,order=1),Jet(k,order=1))[1,0]
                result=residual.root_slope(I(x),I(y),k,slope)
                assert result.lo>0 and result.lo<=exact.hi and exact.lo<=result.hi
                assert result.width<I('1e-25').lo
        with pytest.raises(ValueError):residual.root_slope(I('.5'),I('.5'),I('.1'),I(-1,1))



def test_saved_exit_transport_diagnostic_compares_both_inverses(record_property,monkeypatch):
    import json,os,time
    from pathlib import Path
    from tools.opaque_original_motion_reference import angular_material_cover
    import tools.opaque_knot_transport as transport
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    from tools.opaque_shading_intervals import Unresolved
    path=Path(os.environ['GFFX_FULL_REFERENCE']);results=[];original=transport.transversal_flux
    for algebraic in (False,True):
        start=time.monotonic();failures=[]
        def check():
            if time.monotonic()-start>=45:raise Unresolved('bounded saved-exit diagnostic deadline')
        def observe(field,box,knots):
            try:return original(field,box,knots)
            except ValueError as error:
                x,y=box;q=field(Jet(x),Jet(y));dx=field(Jet.variable(x,order=1),Jet(y,order=1));dy=field(Jet(x,order=1),Jet.variable(y,order=1))
                failures.append(dict(role=field.__defaults__[0],reason=str(error),value=str(q[0,0]),motion=str(q[0,1]),dx=str(dx[1,0]),dy=str(dy[1,0]),knots=len(knots)));raise
        monkeypatch.setattr(transport,'transversal_flux',observe)
        with backend(profile='ball'):
            p,curve,cover,charts=angular_material_cover(frozen(),checkpoint=check,tight_discriminant=True,algebraic_inverse=algebraic);row=charts[66];box=(I(0,'.25'),I('.5',1));stages=[]
            for name,fn in [('prepare',lambda:row['prepare'](Q(1,8),Q(3,4),box)),('center',lambda:row['field'](Jet(Q(1,8)),Jet(Q(3,4)))),('transport',lambda:row['tangent_correction_integrator'](box))]:
                try:values=fn();stages.append(dict(stage=name,status='evaluated',value=str(values)))
                except ValueError as error:stages.append(dict(stage=name,status='refused',reason=str(error)))
            results.append(dict(algebraic=algebraic,chart=cover.charts[66],seconds=time.monotonic()-start,stages=stages,failures=failures))
    assert not path.exists();path.write_text(json.dumps(dict(scope='Saved exit-box diagnostic only, not an integral certificate.',results=results),indent=2),encoding='utf8');record_property('exit_diagnostic',path.read_text());assert len(results)==2


@pytest.mark.parametrize('scale',[Q(1),Q(1,2**60)])
def test_mapped_flux_keeps_collapsing_chart_jump_and_tiny_measure(scale):
    from tools.opaque_mapped_flux import mapped_monotone_flux
    from tools.opaque_knot_transport import transversal_flux
    # x=t*(1-y), jacobian=1-y. For q=x-theta and knot k=1/4,
    # integral delta(q-k)*jacobian dt dy = 1-k =3/4.
    field=lambda t,y:t*(1-y)+t.lift(0,tangent=-1)
    box=(I(0,1),I(0,1));knots=[(I('.25'),I(3))]
    with pytest.raises(ValueError,match='transversal'):transversal_flux(field,box,knots)
    row=mapped_monotone_flux(I(0,1),I(-1),knots,radial_slope=I(1),transverse_extent=I(1),physical_jacobian=I(scale))
    exact=I(scale*Q(9,4));assert 0<exact.lo<=exact.hi<=row['bound'].hi
    assert row['bound'].hi<=I(3*scale).hi
    assert mapped_monotone_flux(I(0,1),I(0),knots,radial_slope=I(1),transverse_extent=I(1),physical_jacobian=I(scale))['bound'].zero


def test_mapped_flux_refuses_missing_physical_monotonicity_or_measure():
    from tools.opaque_mapped_flux import mapped_monotone_flux
    for slope,extent,jac in [(I(0,1),I(1),I(1)),(I(1),I(-1,1),I(1)),(I(1),I(1),I(-1,1))]:
        with pytest.raises(ValueError):mapped_monotone_flux(I(0,1),I(1),[(I('.5'),I(1))],radial_slope=slope,transverse_extent=extent,physical_jacobian=jac)


def test_material_flux_override_preserves_both_roles_and_weights_measure_once():
    from tools.opaque_knot_transport import material_transport,transversal_flux
    from tests.packaging.test_opaque_loss_extension import Curve
    class Curve(Curve):
        pi=I('3.14159');mean=[I(1),I(1)]
    fields=lambda x,y:[x*Q(1,2)+x.lift(Q(1,4),tangent=1),y*Q(1,2)+y.lift(Q(1,4),tangent=-2)]
    box=(I(0,1),I(0,1));calls=[]
    base=material_transport(Curve(),fields,box,[.2,.4,.6],[.04]*3)
    def override(role,box,knots):
        calls.append(role);row=transversal_flux(lambda x,y:fields(x,y)[role],box,knots)
        row['bound']*=I(7);return row
    weighted=material_transport(Curve(),fields,box,[.2,.4,.6],[.04]*3,flux_override=override)
    assert calls==[0,0,0,1,1,1]
    for got,old in zip(weighted['bound'],base['bound']):
        expected=old*7;assert got.lo<=expected.hi and expected.lo<=got.hi and got.hi>0
        assert abs(got.hi-expected.hi)<I('1e-25').lo
    fallback=material_transport(Curve(),fields,box,[.2,.4,.6],[.04]*3,flux_override=lambda *a:None)
    assert [(v.lo,v.hi) for v in fallback['bound']]==[(v.lo,v.hi) for v in base['bound']]
    with pytest.raises(ValueError):material_transport(Curve(),fields,box,[.2,.4,.6],[.04]*3,flux_override=42)
    with pytest.raises(ValueError):material_transport(Curve(),fields,box,[.2,.4,.6],[.04]*3,flux_override=lambda *a:dict(bound=I(-1,1)))


def test_saved_exit_mapped_transport_retains_finite_complete_correction(record_property):
    import json,os,time
    from pathlib import Path
    from tools.opaque_original_motion_reference import angular_material_cover
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    from tools.opaque_shading_intervals import Unresolved
    rows=[]
    for algebraic in (False,True):
        start=time.monotonic()
        def check():
            if time.monotonic()-start>=45:raise Unresolved('saved mapped-exit diagnostic deadline')
        with backend(profile='ball'):
            p,curve,cover,charts=angular_material_cover(frozen(),checkpoint=check,tight_discriminant=True,algebraic_inverse=algebraic,mapped_transport=True)
            row=charts[66];box=(I(0,'.25'),I('.5',1));row['prepare'](Q(1,8),Q(3,4),box)
            center=row['field'](Jet(Q(1,8)),Jet(Q(3,4)))
            correction=row['tangent_correction_integrator'](box)
            assert all(v.lo.is_finite() and v.hi.is_finite() for v in correction)
            rows.append(dict(algebraic=algebraic,chart=cover.charts[66],seconds=time.monotonic()-start,center=[[str(v[0,j]) for j in (0,1)] for v in center],correction=list(map(str,correction))))
    path=Path(os.environ['GFFX_FULL_REFERENCE']);assert not path.exists();path.write_text(json.dumps(dict(scope='Saved mapped-exit box only; not complete integral acceptance.',rows=rows),indent=2),encoding='utf8');record_property('mapped_exit',path.read_text())


def test_algebraic_mapped_original_chart66_integrates(record_property):
    import json,os,time
    from pathlib import Path
    from tools.opaque_original_motion_reference import angular_material_cover
    from tools.opaque_square_reference import integrate_square
    from tools.opaque_shading_intervals import Unresolved
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    start=time.monotonic();failure=None
    def check():
        if time.monotonic()-start>=90:raise Unresolved('mapped local construction inclusive deadline')
    with backend(profile='ball'):
        try:
            p,curve,cover,charts=angular_material_cover(frozen(),checkpoint=check,tight_discriminant=True,algebraic_inverse=True,mapped_transport=True);row=charts[66]
            result=integrate_square(row['field'],prepare=row['prepare'],correction_integrator=row['correction_integrator'],tangent_correction_integrator=row['tangent_correction_integrator'],max_cells=256-cover.cells+1,max_depth=10-row['depth'],tolerance='5e-13',checkpoint=check)
            result=dict(status='local_chart_passed',basis=result,seconds=time.monotonic()-start,evaluations=cover.evaluations)
        except ValueError as error:failure=str(error);result=dict(status='unresolved',failure=getattr(error,'details',str(error)),seconds=time.monotonic()-start)
    path=Path(os.environ['GFFX_FULL_REFERENCE']);assert not path.exists();path.write_text(json.dumps(result,default=str,indent=2),encoding='utf8');record_property('mapped_local',path.read_text());assert failure is None,result


@pytest.mark.parametrize('axis',[0,1])
def test_positive_exit_angle_keeps_closed_inverse_and_complete_mixed_measure(axis):
    from tools.opaque_knot_partition import MonotoneAngularCover
    from tools.opaque_arb_jets import backend
    field=lambda u,s:u*(1+u.lift(0,tangent=Q(1,7)))/(1+s)
    radial=lambda u,s:(1+u.lift(0,tangent=Q(1,7)))/(1+s)
    residual=lambda u,s,k:u*(1+u.lift(0,tangent=Q(1,7)))-k*(1+s)
    with backend(profile='ball'):
        cover=MonotoneAngularCover(field,[I(Q(1,4)),I(Q(3,4))],inverse_residual=residual)
        for index,c in enumerate(cover.charts):
            if c['high'] is not None or c['low'] is None:continue
            args=[Jet(Q(1,4),order=10),Jet(Q(1,4),order=10)];args[axis]=Jet.variable(Q(1,4),order=10)
            t,y=args;row=cover.map_exit_angle(index,t,y,radial)
            a=t.lift(0) if c['start'] is None else (1+t.lift(0,tangent=Q(1,7)))/Q(3,4)-1
            b=t.lift(1) if c['end'] is None else (1+t.lift(0,tangent=Q(1,7)))/Q(3,4)-1
            span=b-a;s0=a+span*y;low=cover.levels[c['low']]
            delta=field(t.lift(1),s0)-low;target=t*delta+low
            exact=target*(1+s0)/(1+t.lift(0,tangent=Q(1,7)))
            jac=delta*span*(1+s0)/(1+t.lift(0,tangent=Q(1,7)))
            for actual,want in [(row['u'],exact),(row['s'],s0),(row['angle'],target),(row['jacobian'],jac)]:
                for i in range(11):
                    for j in (0,1):
                        assert actual[i,j].lo<=want[i,j].hi and want[i,j].lo<=actual[i,j].hi
                        assert actual[i,j].width<I('1e-28').lo
        with pytest.raises(ValueError):cover.map_exit_angle(0,Jet('.5'),Jet('.5'),radial)


@pytest.mark.parametrize('scale',[Q(1),Q(1,2**60)])
def test_positive_exit_angle_cover_integrates_all_moving_measure(scale):
    from tools.opaque_knot_partition import MonotoneAngularCover
    from tools.opaque_square_reference import integrate_square
    from tools.opaque_arb_jets import backend
    field=lambda u,s:u*(1+u.lift(0,tangent=Q(1,7)))/(1+s)
    radial=lambda u,s:(1+u.lift(0,tangent=Q(1,7)))/(1+s)
    residual=lambda u,s,k:u*(1+u.lift(0,tangent=Q(1,7)))-k*(1+s)
    with backend(profile='ball'):
        cover=MonotoneAngularCover(field,[I(Q(1,4)),I(Q(3,4))],inverse_residual=residual);charts=[]
        for index,c in enumerate(cover.charts):
            def integrand(t,y,index=index,c=c):
                method=cover.map_exit_angle if c['high'] is None and c['low'] is not None else cover.map_angle_cover
                row=method(index,t,y,radial);mass=scale*row['jacobian']*(1+row['s'])*(1+t.lift(0,tangent=Q(1,5)))
                return [mass,mass*row['u']]
            charts.append(dict(field=integrand,depth=c['depth']))
        result=integrate_square(None,chart_cover=charts,tolerance=str((I(scale)*I('1e-12')).lo))
        for value,tangent,want in zip(result['value'],result['derivative'],[scale*Q(3,2),scale*Q(3,4)]):
            expected=I(want);assert value.lo<=expected.lo<=expected.hi<=value.hi
            expected=I(want/5);assert tangent.lo<=expected.lo<=expected.hi<=tangent.hi
            assert value.lo>0 and tangent.lo>0

def test_positive_exit_angle_original_chart66_integrates(record_property):
    import json,os,time
    from pathlib import Path
    from tools.opaque_original_motion_reference import angular_material_cover
    from tools.opaque_square_reference import integrate_square
    from tools.opaque_shading_intervals import Unresolved
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    start=time.monotonic();failure=None
    def check():
        if time.monotonic()-start>=90:raise Unresolved('mapped local construction inclusive deadline')
    with backend(profile='ball'):
        try:
            p,curve,cover,charts=angular_material_cover(frozen(),checkpoint=check,tight_discriminant=True,algebraic_inverse=True,mapped_transport=True,exit_angle=True);row=charts[66]
            result=integrate_square(row['field'],prepare=row['prepare'],correction_integrator=row['correction_integrator'],tangent_correction_integrator=row['tangent_correction_integrator'],max_cells=256-cover.cells+1,max_depth=10-row['depth'],tolerance='5e-13',checkpoint=check)
            result=dict(status='local_chart_passed',basis=result,seconds=time.monotonic()-start,evaluations=cover.evaluations)
        except ValueError as error:failure=str(error);result=dict(status='unresolved',failure=getattr(error,'details',str(error)),seconds=time.monotonic()-start)
    path=Path(os.environ['GFFX_FULL_REFERENCE']);assert not path.exists();path.write_text(json.dumps(result,default=str,indent=2),encoding='utf8');record_property('mapped_local',path.read_text());assert failure is None,result




def test_positive_exit_largest_leaf_error_diagnostic(record_property,monkeypatch):
    import json,os,time
    from pathlib import Path
    from tools.opaque_original_motion_reference import angular_material_cover
    from tools.opaque_tangent_peano import TangentCorrection
    from tools.opaque_arb_jets import backend
    import tools.opaque_original_motion_reference as motion
    from tests.packaging.test_opaque_motion_tangent import frozen
    from tools.opaque_shading_intervals import Unresolved
    start=time.monotonic();axes=[];pairs=[];flux=[];old_axis=TangentCorrection.axis_bounds;old_pair=TangentCorrection.__call__;old_transport=motion.material_transport
    def check():
        if time.monotonic()-start>=45:raise Unresolved('exit residual diagnostic deadline')
    def axis(self,box,index):
        args=[Jet(v,order=3) for v in box];args[index]=Jet.variable(box[index],order=3);values=self.inputs(*args)
        detail=dict(axis=index,angles=[dict(value=str(q[0,0]),spatial=str(q[1,0]),motion=str(q[0,1])) for q in values[:2]])
        try:detail['result']=old_axis(self,box,index);return detail['result']
        except ValueError as e:detail['failure']=str(e);raise
        finally:axes.append(detail)
    def pair(self,box):
        value=old_pair(self,box);pairs.append(dict(value=list(map(str,value)),range_fallbacks=self.range_fallbacks));return value
    def transport(*a,**kw):
        value=old_transport(*a,**kw);flux.append(value);return value
    monkeypatch.setattr(TangentCorrection,'axis_bounds',axis);monkeypatch.setattr(TangentCorrection,'__call__',pair);monkeypatch.setattr(motion,'material_transport',transport)
    with backend(profile='ball'):
        p,curve,cover,charts=angular_material_cover(frozen(),checkpoint=check,tight_discriminant=True,algebraic_inverse=True,mapped_transport=True,exit_angle=True);row=charts[66];box=(I('.75',1),I(0,'.5'));row['prepare'](Q(7,8),Q(1,4),box);row['tangent_correction_integrator'](box)
    path=Path(os.environ['GFFX_FULL_REFERENCE']);assert not path.exists();path.write_text(json.dumps(dict(axes=axes,pairs=pairs,flux=flux,seconds=time.monotonic()-start,scope='Error diagnosis only, no integral acceptance.'),default=str,indent=2),encoding='utf8');record_property('exit_error_diagnostic',path.read_text())


@pytest.mark.parametrize('axis',[0,1])
def test_normal_light_spatial_bound_matches_independent_normalization(axis):
    from tools.opaque_normal_light import normal_light_spatial_bound
    from tools.opaque_arb_jets import backend
    from tools.opaque_moving_reference import P
    coefficients=[list(map(P,row)) for row in [(Q(1,2),Q(1,8),Q(1,16)),(Q(1,4),Q(-1,16),Q(1,32)),(1,Q(1,4),Q(-1,8))]]
    with backend(profile='ball'):
        def args(x,y):
            a=[Jet(x,order=1),Jet(y,order=1)];a[axis]=Jet.variable(a[axis][0,0],order=1);return a
        broad=normal_light_spatial_bound(coefficients,*args(I('.25','.75'),I('.25','.75')))
        for x in (Q(1,4),Q(1,2),Q(3,4)):
            for y in (Q(1,4),Q(1,2),Q(3,4)):
                t,v=args(x,y);n=[t.lift(a[0])+t*(t.lift(b[0])+v*t.lift(c[0])) for a,b,c in coefficients]
                nn=sum((z*z for z in n),t.lift(0));exact=(sum((z*k for z,k in zip(n,(1,1,2))),t.lift(0))/(6*nn).sqrt())[1,0]
                bound=normal_light_spatial_bound(coefficients,t,v)
                assert bound.lo<=exact.hi and exact.lo<=bound.hi
                assert broad.lo<=exact.hi and exact.lo<=broad.hi
                assert bound.width<I('1e-30').lo
        with pytest.raises(ValueError):normal_light_spatial_bound(coefficients,Jet('.5'),Jet('.5'))



@pytest.mark.parametrize('scale',[Q(1),Q(1,2**60)])
def test_normal_light_tangent_bound_keeps_intrinsic_and_coordinate_motion(scale):
    from tools.opaque_normal_light import normal_light_tangent_bound
    from tools.opaque_arb_jets import backend
    from tools.opaque_moving_reference import P
    coefficients=[[P([a,scale*Q(i+1,100)]) for i,a in enumerate(row)] for row in [(Q(1,2),Q(1,8),Q(1,16)),(Q(1,4),Q(-1,16),Q(1,32)),(1,Q(1,4),Q(-1,8))]]
    with backend(profile='ball'):
        broad=normal_light_tangent_bound(coefficients,Jet(I('.25','.75'),tangent=scale*Q(1,7)),Jet(I('.25','.75'),tangent=-scale*Q(1,11)))
        for x,y in [(Q(1,4),Q(3,4)),(Q(1,2),Q(1,2)),(Q(3,4),Q(1,4))]:
            t=Jet(x,tangent=scale*Q(1,7));v=Jet(y,tangent=-scale*Q(1,11));n=[t.lift(a[0],tangent=a[1])+t*(t.lift(b[0],tangent=b[1])+v*t.lift(c[0],tangent=c[1])) for a,b,c in coefficients]
            nn=sum((z*z for z in n),t.lift(0));exact=(sum((z*k for z,k in zip(n,(1,1,2))),t.lift(0))/(6*nn).sqrt())[0,1]
            bound=normal_light_tangent_bound(coefficients,t,v)
            assert bound.lo<=exact.hi and exact.lo<=bound.hi
            assert broad.lo<=exact.hi and exact.lo<=broad.hi
            assert not bound.zero and bound.width<(I(scale)*I('1e-30')).lo


def test_exit_chart_forwards_current_residual_split_hint(monkeypatch):
    import tools.opaque_original_motion_reference as m
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    class Correction:
        def __init__(self,*a,**kw):self.preferred_split_axis=None
        def __call__(self,box):
            self.preferred_split_axis=0 if box[0].lo==0 else 1
            return [I(1),I(2)]
    monkeypatch.setattr(m,'TangentCorrection',Correction)
    with backend(profile='ball'):
        p,curve,cover,charts=m.angular_material_cover(frozen(),tight_discriminant=True,algebraic_inverse=True,mapped_transport=True,exit_angle=True)
        correct=charts[66]['correction_integrator']
        for box,axis in [((I(0,'.5'),I(0,1)),0),((I('.5',1),I(0,1)),1),((I(0,'.5'),I(0,1)),0)]:
            result=correct(box);assert result[0].zero and result[1].lo==1 and result[1].hi==1
            assert correct.preferred_split_axis==axis


def test_exit_correction_error_components(record_property,monkeypatch):
    import json,os,time
    from pathlib import Path
    import tools.opaque_tangent_peano as tp
    from tools.opaque_original_motion_reference import angular_material_cover
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    from tools.opaque_shading_intervals import Unresolved
    original=tp.residual_integral;rows=[];start=time.monotonic()
    def check():
        if time.monotonic()-start>=45:raise Unresolved('correction component diagnostic deadline')
    def inspect(*args,**kw):
        kw['error_breakdown']=lambda v:rows.append(dict(box=args[1],components=v))
        return original(*args,**kw)
    monkeypatch.setattr(tp,'residual_integral',inspect)
    with backend(profile='ball'):
        p,curve,cover,charts=angular_material_cover(frozen(),checkpoint=check,tight_discriminant=True,algebraic_inverse=True,mapped_transport=True,exit_angle=True)
        for box in [(I('.75',1),I(0,'.5')),(I('.5','.75'),I(0,'.5'))]:
            row=charts[66];row['prepare']((box[0].lo+box[0].hi)/2,(box[1].lo+box[1].hi)/2,box)
            correction=row['tangent_correction_integrator'](box);rows[-1]['correction']=correction
    path=Path(os.environ['GFFX_FULL_REFERENCE']);assert not path.exists()
    path.write_text(json.dumps(dict(rows=rows,seconds=time.monotonic()-start,scope='Largest two saved leaves; diagnostic only.'),default=str,indent=2),encoding='utf8')
    record_property('correction_components',path.read_text())


@pytest.mark.parametrize('order',[0,3,10])
@pytest.mark.parametrize('scale',[Q(1),Q(1,2**60)])
def test_prepared_outer_edge_matches_direct_normalized_fields(order,scale):
    from tools.opaque_prepared_angle import PreparedAngle
    from tools.opaque_arb_jets import backend
    from types import SimpleNamespace
    # The outer physical coordinate is exactly one for every moving horizon.
    coeff={name:[[(Q(a),scale*Q(i+1,200)),(Q(b),-scale*Q(i+1,300)),(Q(c),scale*Q(i+1,500))] for i,(a,b,c) in enumerate(rows)] for name,rows in {'n':[(1,2,1),(2,1,-1),(3,-1,2)],'v':[(2,1,-1),(1,2,1),(4,-1,1)]}.items()}
    prepared=PreparedAngle(dict(curved=True,patch=SimpleNamespace(inside=False),fields=SimpleNamespace(coefficients=coeff)))
    with backend(profile='ball'):
        def value(x):
            s=Jet.variable(x,order=order) if order else Jet(x)
            s=s+s.lift(0,tangent=scale*Q(1,7))
            fields={name:[s.lift(a[0],tangent=a[1])+s.lift(b[0],tangent=b[1])+s*s.lift(c[0],tangent=c[1]) for a,b,c in rows] for name,rows in coeff.items()}
            n,v=fields['n'],fields['v'];dot=lambda a,b:sum((x*y for x,y in zip(a,b)),s.lift(0))
            return prepared.edge(s),dot(n,v)/(dot(n,n)*dot(v,v)).sqrt()
        broad,_=value(I('.2','.4'))
        for x in (Q(1,5),Q(3,10),Q(2,5)):
            got,want=value(x)
            for i in range(order+1):
                for j in (0,1):
                    assert got[i,j].lo<=want[i,j].hi and want[i,j].lo<=got[i,j].hi
                    assert broad[i,j].lo<=want[i,j].hi and want[i,j].lo<=broad[i,j].hi
                    assert got[i,j].width<I('1e-30').lo
            assert not got[0,1].zero
        with pytest.raises(ValueError):prepared.edge(Jet(I(-1,1)))


def test_repaired_exit_shared_original_complete_piece(record_property):
    import json,os
    from pathlib import Path
    from tools.opaque_original_motion_reference import angular_piece_reference
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    path=Path(os.environ['GFFX_FULL_REFERENCE']);failure=None;rows=[]
    with backend(profile='ball'):
        try:result=angular_piece_reference(frozen(),tight_discriminant=True,algebraic_inverse=True,mapped_transport=True,exit_angle=True,progress=rows.append)
        except ValueError as error:failure=str(error);result=dict(status='unresolved',failure=getattr(error,'details',str(error)))
    result['progress']=rows
    assert not path.exists();path.write_text(json.dumps(result,default=str,indent=2),encoding='utf8');record_property('algebraic_shared_piece',path.read_text())
    assert failure is None,result
    assert result['algebraic_inverse'] and result['basis']['chart_count']==72 and result['basis']['cells']<=256
    assert all(v.width<=I('2e-12').lo for v in result['value']+result['derivative'])



def test_repaired_piece_remaining_chart_diagnostic(record_property,monkeypatch):
    import json,os,time
    from pathlib import Path
    from tools.opaque_original_motion_reference import angular_material_cover
    from tools.opaque_arb_jets import backend
    from tools.opaque_shading_intervals import Unresolved
    from tests.packaging.test_opaque_motion_tangent import frozen
    from tools.opaque_tangent_peano import TangentCorrection
    start=time.monotonic();angles=[];original=TangentCorrection.__call__
    def check():
        if time.monotonic()-start>=45:raise Unresolved('remaining chart diagnostic deadline')
    def inspect(self,box):
        for axis in (0,1):
            args=[Jet(v,order=1) for v in box];args[axis]=Jet.variable(box[axis],order=1)
            angles.append(dict(axis=axis,fields=[{str(k):str(v) for k,v in q.c.items()} for q in self.inputs(*args)[:2]]))
        return original(self,box)
    monkeypatch.setattr(TangentCorrection,'__call__',inspect)
    with backend(profile='ball'):
        p,curve,cover,charts=angular_material_cover(frozen(),checkpoint=check,tight_discriminant=True,algebraic_inverse=True,mapped_transport=True,exit_angle=True)
        row=charts[68];box=(I(0,'.5'),I(0,1));row['prepare'](Q(1,4),Q(1,2),box)
        correction=row['correction_integrator'](box)
        try:transport=row['tangent_correction_integrator'](box);failure=None
        except ValueError as error:transport=None;failure=str(error)
    path=Path(os.environ['GFFX_FULL_REFERENCE']);assert not path.exists()
    path.write_text(json.dumps(dict(charts=cover.charts,angles=angles,correction=correction,transport=transport,failure=failure,seconds=time.monotonic()-start,scope='Saved chart68 refusal, no integral qualification.'),default=str,indent=2),encoding='utf8');record_property('remaining_chart',path.read_text())

def test_repaired_initial_cell_cost_diagnostic(record_property):
    import cProfile,pstats,time,json,os
    from pathlib import Path
    from tools.opaque_original_motion_reference import angular_material_cover
    from tools.opaque_square_reference import integrate_square
    from tools.opaque_arb_jets import backend
    from tools.opaque_shading_intervals import Unresolved
    from tests.packaging.test_opaque_motion_tangent import frozen
    start=time.monotonic()
    def check():
        if time.monotonic()-start>=45:raise Unresolved('repaired initial-cell diagnostic deadline')
    with backend(profile='ball'):
        p,curve,cover,charts=angular_material_cover(frozen(),tight_discriminant=True,algebraic_inverse=True,mapped_transport=True,exit_angle=True,checkpoint=check)
        row=charts[27];prof=cProfile.Profile();prof.enable()
        try:result=integrate_square(row['field'],prepare=row['prepare'],correction_integrator=row['correction_integrator'],tangent_correction_integrator=row['tangent_correction_integrator'],tolerance='5e-13',max_cells=1,max_depth=10,checkpoint=check)
        except Unresolved as error:result=error.details
        finally:prof.disable()
        rows=sorted([dict(file=k[0],line=k[1],function=k[2],calls=v[1],self_seconds=v[2],cumulative_seconds=v[3]) for k,v in pstats.Stats(prof).stats.items()],key=lambda r:r['self_seconds'],reverse=True)
    path=Path(os.environ['GFFX_FULL_REFERENCE']);assert not path.exists();path.write_text(json.dumps(dict(seconds=time.monotonic()-start,result=result,self_costs=rows[:35],cumulative_costs=sorted(rows,key=lambda r:r['cumulative_seconds'],reverse=True)[:35],scope='One initial cell with profiler overhead; no piece acceptance.'),default=str,indent=2),encoding='utf8');record_property('repaired_cell_cost',path.read_text())


def test_repaired_exit_light_flux_is_finite(record_property,monkeypatch):
    import json,os
    from pathlib import Path
    test_repaired_piece_remaining_chart_diagnostic(record_property,monkeypatch)
    result=json.loads(Path(os.environ['GFFX_FULL_REFERENCE']).read_bytes())
    assert result['failure'] is None,result['failure']
    assert result['transport'] is not None


@pytest.mark.parametrize('axis',[0,1])
@pytest.mark.parametrize('scale',[Q(1),Q(1,2**60)])
def test_incremental_quartic_inverse_matches_closed_mixed_coefficients(axis,scale):
    from tools.opaque_algebraic_angle_inverse import QuarticAngleResidual
    from tools.opaque_arb_jets import backend
    with backend(profile='ball'):
        residual=object.__new__(QuarticAngleResidual)
        residual.coefficients=lambda s:([s.lift(v) for v in (0,0,1,2,1)],[(1+s)**2,*[s.lift(0) for _ in range(4)]])
        def sample(x,y):
            args=[Jet(x,order=10),Jet(y,order=10)];args[axis]=Jet.variable(args[axis][0,0],order=10)
            t,s=args;t+=t.lift(0,tangent=scale*Q(1,17));s+=s.lift(0,tangent=-scale*Q(1,19));k=(1+t)/8
            exact=((1+4*k*(1+s)).sqrt()-1)/2;root=exact[0,0]
            slope=2*(root+root**2)*(1+2*root)
            result=residual.implicit_coefficients(root,s,k,slope)
            return Jet(order=10,coeff=result),exact
        broad,_=sample(I('.2','.4'),I('.2','.4'))
        for x,y in [(Q(1,5),Q(2,5)),(Q(3,10),Q(3,10)),(Q(2,5),Q(1,5))]:
            actual,exact=sample(x,y)
            for i in range(11):
                for j in (0,1):
                    assert actual[i,j].lo<=exact[i,j].hi and exact[i,j].lo<=actual[i,j].hi
                    assert broad[i,j].lo<=exact[i,j].hi and exact[i,j].lo<=broad[i,j].hi
                    assert actual[i,j].width<I('1e-28').lo
            assert not actual[0,1].zero
        with pytest.raises(ValueError):residual.implicit_coefficients(I(0,1),Jet('.5'),Jet('.1'),I(-1,1))


@pytest.mark.parametrize('axis',[0,1])
def test_original_incremental_inverse_matches_preserved_recurrence(axis,monkeypatch):
    from tools.opaque_original_motion_reference import angular_material_cover
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    with backend(profile='ball'):
        p,curve,cover,charts=angular_material_cover(frozen(),tight_discriminant=True,algebraic_inverse=True,mapped_transport=True,exit_angle=True)
        inverse=p['inverse_residual'];original=inverse.implicit_coefficients;calls=[]
        def solve(*a):calls.append(1);return original(*a)
        for value in (I('.25'),I('.2','.3')):
            args=[Jet(value,order=10),Jet(value,order=10)];args[axis]=Jet.variable(value,order=10)
            monkeypatch.setattr(inverse,'implicit_coefficients',solve);cover.cache.clear()
            got=cover.map_angle(27,*args,p['prepared_angle'].radial)
            monkeypatch.setattr(inverse,'implicit_coefficients',None);cover.cache.clear()
            old=cover.map_angle(27,*args,p['prepared_angle'].radial)
            for name in ('u','jacobian'):
                for i in range(11):
                    for j in (0,1):
                        a,b=got[name][i,j],old[name][i,j]
                        assert a.lo<=b.hi and b.lo<=a.hi,(name,i,j)
                        if value.width==0:assert a.width<I('1e-23').lo
        assert calls,'incremental solver was not connected'


def test_positive_exit_angle_original_chart68_integrates(record_property):
    import json,os,time
    from pathlib import Path
    from tools.opaque_original_motion_reference import angular_material_cover
    from tools.opaque_square_reference import integrate_square
    from tools.opaque_shading_intervals import Unresolved
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    start=time.monotonic();failure=None
    def check():
        if time.monotonic()-start>=90:raise Unresolved('mapped local construction inclusive deadline')
    with backend(profile='ball'):
        try:
            p,curve,cover,charts=angular_material_cover(frozen(),checkpoint=check,tight_discriminant=True,algebraic_inverse=True,mapped_transport=True,exit_angle=True);row=charts[68]
            result=integrate_square(row['field'],prepare=row['prepare'],correction_integrator=row['correction_integrator'],tangent_correction_integrator=row['tangent_correction_integrator'],max_cells=256-cover.cells+1,max_depth=10-row['depth'],tolerance='5e-13',checkpoint=check)
            result=dict(status='local_chart_passed',basis=result,seconds=time.monotonic()-start,evaluations=cover.evaluations)
        except ValueError as error:failure=str(error);result=dict(status='unresolved',failure=getattr(error,'details',str(error)),seconds=time.monotonic()-start)
    path=Path(os.environ['GFFX_FULL_REFERENCE']);assert not path.exists();path.write_text(json.dumps(result,default=str,indent=2),encoding='utf8');record_property('mapped_local',path.read_text());assert failure is None,result





def test_exit68_largest_leaf_monotonicity_diagnostic(record_property,monkeypatch):
    import json,os,time
    from pathlib import Path
    from tools.opaque_original_motion_reference import angular_material_cover
    from tools.opaque_arb_jets import backend
    from tools.opaque_shading_intervals import Unresolved
    from tests.packaging.test_opaque_motion_tangent import frozen
    from tools.opaque_tangent_peano import TangentCorrection
    start=time.monotonic();angles=[];original=TangentCorrection.__call__
    def check():
        if time.monotonic()-start>=45:raise Unresolved('remaining chart diagnostic deadline')
    def inspect(self,box):
        for axis in (0,1):
            args=[Jet(v,order=1) for v in box];args[axis]=Jet.variable(box[axis],order=1)
            angles.append(dict(axis=axis,fields=[{str(k):str(v) for k,v in q.c.items()} for q in self.inputs(*args)[:2]]))
        result=original(self,box);angles.append(dict(range_fallbacks=self.range_fallbacks))
        for x,y in [(Q(1,2),Q(0)),(Q(5,8),Q(1,4)),(Q(3,4),Q(1,2))]:
            q=self.inputs(Jet(x,order=1),Jet.variable(y,order=1))[1];angles.append(dict(point=[str(x),str(y)],light_y=str(q[1,0])))
        return result
    monkeypatch.setattr(TangentCorrection,'__call__',inspect)
    with backend(profile='ball'):
        p,curve,cover,charts=angular_material_cover(frozen(),checkpoint=check,tight_discriminant=True,algebraic_inverse=True,mapped_transport=True,exit_angle=True)
        row=charts[68];box=(I('.5','.75'),I(0,'.5'));row['prepare'](Q(5,8),Q(1,4),box)
        correction=row['correction_integrator'](box)
        try:transport=row['tangent_correction_integrator'](box);failure=None
        except ValueError as error:transport=None;failure=str(error)
    path=Path(os.environ['GFFX_FULL_REFERENCE']);assert not path.exists()
    path.write_text(json.dumps(dict(charts=cover.charts,angles=angles,correction=correction,transport=transport,failure=failure,seconds=time.monotonic()-start,scope='Saved chart68 refusal, no integral qualification.'),default=str,indent=2),encoding='utf8');record_property('remaining_chart',path.read_text())


@pytest.mark.parametrize('key',[(1,0),(0,1)])
@pytest.mark.parametrize('scale',[Q(1),Q(1,2**60)])
def test_angle_light_differential_retains_coordinate_and_intrinsic_motion(key,scale):
    from tools.opaque_angle_light import AngleLightDifferential
    from tools.opaque_moving_reference import P
    from tools.opaque_arb_jets import backend
    coeff={name:[[P([a,scale*Q(i+1,1000)]) for i,a in enumerate(row)] for row in rows] for name,rows in {'n':[(Q(1,5),Q(1,2),Q(1,10)),(Q(1,10),Q(1,20),0),(1,Q(1,50),0)],'v':[(1,0,0),(Q(1,5),0,0),(Q(1,10),0,Q(1,100))]}.items()}
    differential=AngleLightDifferential(coeff)
    with backend(profile='ball'):
        def sample(x):
            z=Jet.variable(x,order=1);t=z*z/5+z+z.lift(0,tangent=scale*Q(1,7));s=z/3+z.lift(Q(1,5),tangent=-scale*Q(1,11))
            fields={name:[t.lift(a[0],tangent=a[1])+t*(t.lift(b[0],tangent=b[1])+s*t.lift(c[0],tangent=c[1])) for a,b,c in rows] for name,rows in coeff.items()}
            n,v=fields['n'],fields['v'];dot=lambda a,b:sum((a*b for a,b in zip(a,b)),z.lift(0));angle=dot(n,v)/(dot(n,n)*dot(v,v)).sqrt();light=dot(n,[z.lift(i) for i in (1,1,2)])/(6*dot(n,n)).sqrt()
            return differential.bound(t,s,angle,key),light[key]
        broad,_=sample(I('.2','.4'))
        for x in (Q(1,5),Q(3,10),Q(2,5)):
            actual,exact=sample(x);assert actual.lo<=exact.hi and exact.lo<=actual.hi
            assert broad.lo<=exact.hi and exact.lo<=broad.hi
            assert actual.width<I('1e-28').lo and not actual.zero


def test_exit68_light_differential_proves_saved_monotonicity(record_property,monkeypatch):
    import json,os
    from pathlib import Path
    test_exit68_largest_leaf_monotonicity_diagnostic(record_property,monkeypatch)
    result=json.loads(Path(os.environ['GFFX_FULL_REFERENCE']).read_bytes())
    high=I(result['angles'][1]['fields'][1]['(1, 0)'].strip('[]').split(',')[1])
    assert high.hi<0
    assert result['angles'][2]['range_fallbacks']==0


def test_low_angle_inverse_bound_diagnostic(record_property):
    import json,os,time
    from pathlib import Path
    from tools.opaque_original_motion_reference import angular_material_cover
    from tools.opaque_square_reference import integrate_square
    from tools.opaque_arb_jets import backend
    from tools.opaque_shading_intervals import Unresolved
    from tests.packaging.test_opaque_motion_tangent import frozen
    start=time.monotonic();rows=[]
    def check():
        if time.monotonic()-start>=90:raise Unresolved('low-angle diagnostic deadline')
    with backend(profile='ball'):
        for algebraic in (True,False):
            p,curve,cover,charts=angular_material_cover(frozen(),checkpoint=check,tight_discriminant=True,algebraic_inverse=algebraic)
            for index in (1,7,27):
                row=charts[index];begin=time.monotonic()
                try:result=integrate_square(row['field'],prepare=row['prepare'],correction_integrator=row['correction_integrator'],tangent_correction_integrator=row['tangent_correction_integrator'],tolerance='5e-13',max_cells=1,max_depth=10-row['depth'],checkpoint=check)
                except Unresolved as error:result=error.details
                rows.append(dict(algebraic=algebraic,chart=index,result=result,seconds=time.monotonic()-begin))
    path=Path(os.environ['GFFX_FULL_REFERENCE']);assert not path.exists();path.write_text(json.dumps(dict(rows=rows,seconds=time.monotonic()-start,scope='One initial cell per formulation/chart; no integral acceptance.'),default=str,indent=2),encoding='utf8');record_property('low_angle_comparison',path.read_text())


@pytest.mark.parametrize('axis',[0,1])
@pytest.mark.parametrize('scale',[Q(1),Q(1,2**60)])
def test_unsquared_inverse_matches_closed_mixed_coefficients(axis,scale):
    from tools.opaque_algebraic_angle_inverse import UnsquaredAngleResidual
    from tools.opaque_arb_jets import backend
    with backend(profile='ball'):
        residual=object.__new__(UnsquaredAngleResidual)
        residual.coefficients=lambda s:([s.lift(v) for v in (0,1,1)],[(1+s)**2,*[s.lift(0) for _ in range(4)]])
        def sample(x,y):
            args=[Jet(x,order=10),Jet(y,order=10)];args[axis]=Jet.variable(args[axis][0,0],order=10)
            t,s=args;t+=t.lift(0,tangent=scale*Q(1,17));s+=s.lift(0,tangent=-scale*Q(1,19));k=(1+t)/8
            exact=((1+4*k*(1+s)).sqrt()-1)/2;root=exact[0,0]
            slope=1+2*root
            result=residual.implicit_coefficients(root,s,k,slope)
            return Jet(order=10,coeff=result),exact
        broad,_=sample(I('.2','.4'),I('.2','.4'))
        for x,y in [(Q(1,5),Q(2,5)),(Q(3,10),Q(3,10)),(Q(2,5),Q(1,5))]:
            actual,exact=sample(x,y)
            for i in range(11):
                for j in (0,1):
                    assert actual[i,j].lo<=exact[i,j].hi and exact[i,j].lo<=actual[i,j].hi
                    assert broad[i,j].lo<=exact[i,j].hi and exact[i,j].lo<=broad[i,j].hi
                    assert actual[i,j].width<I('1e-28').lo
            assert not actual[0,1].zero
        with pytest.raises(ValueError):residual.implicit_coefficients(I(0,1),Jet('.5'),Jet('.1'),I(-1,1))



def test_unsquared_low_angle_bound_diagnostic(record_property):
    import json,os,time
    from pathlib import Path
    from tools.opaque_original_motion_reference import angular_material_cover
    from tools.opaque_square_reference import integrate_square
    from tools.opaque_arb_jets import backend
    from tools.opaque_shading_intervals import Unresolved
    from tests.packaging.test_opaque_motion_tangent import frozen
    start=time.monotonic();rows=[]
    def check():
        if time.monotonic()-start>=90:raise Unresolved('low-angle diagnostic deadline')
    with backend(profile='ball'):
        for algebraic in (True,):
            p,curve,cover,charts=angular_material_cover(frozen(),checkpoint=check,tight_discriminant=True,algebraic_inverse=algebraic)
            from tools.opaque_algebraic_angle_inverse import UnsquaredAngleResidual
            cover.inverse_residual=UnsquaredAngleResidual(p['prepared_angle']);cover.cache.clear()
            for index in (1,7,27):
                row=charts[index];begin=time.monotonic()
                try:result=integrate_square(row['field'],prepare=row['prepare'],correction_integrator=row['correction_integrator'],tangent_correction_integrator=row['tangent_correction_integrator'],tolerance='5e-13',max_cells=1,max_depth=10-row['depth'],checkpoint=check)
                except Unresolved as error:result=error.details
                rows.append(dict(algebraic=algebraic,chart=index,result=result,seconds=time.monotonic()-begin))
    path=Path(os.environ['GFFX_FULL_REFERENCE']);assert not path.exists();path.write_text(json.dumps(dict(rows=rows,seconds=time.monotonic()-start,scope='One initial cell per formulation/chart; no integral acceptance.'),default=str,indent=2),encoding='utf8');record_property('low_angle_comparison',path.read_text())


@pytest.mark.parametrize('axis',[0,1])
def test_original_unsquared_inverse_matches_preserved_recurrence(axis,monkeypatch):
    from tools.opaque_original_motion_reference import angular_material_cover
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    with backend(profile='ball'):
        p,curve,cover,charts=angular_material_cover(frozen(),tight_discriminant=True,algebraic_inverse=True,mapped_transport=True,exit_angle=True)
        from tools.opaque_algebraic_angle_inverse import UnsquaredAngleResidual
        inverse=UnsquaredAngleResidual(p['prepared_angle']);cover.inverse_residual=inverse;cover.cache.clear();original=inverse.implicit_coefficients;calls=[]
        def solve(*a):calls.append(1);return original(*a)
        for value in (I('.25'),I('.2','.3')):
            args=[Jet(value,order=10),Jet(value,order=10)];args[axis]=Jet.variable(value,order=10)
            monkeypatch.setattr(inverse,'implicit_coefficients',solve);cover.cache.clear()
            got=cover.map_angle(27,*args,p['prepared_angle'].radial)
            monkeypatch.setattr(inverse,'implicit_coefficients',None);cover.cache.clear()
            old=cover.map_angle(27,*args,p['prepared_angle'].radial)
            for name in ('u','jacobian'):
                for i in range(11):
                    for j in (0,1):
                        a,b=got[name][i,j],old[name][i,j]
                        assert a.lo<=b.hi and b.lo<=a.hi,(name,i,j)
                        if value.width==0:assert a.width<I('1e-23').lo
        assert calls,'incremental solver was not connected'



@pytest.mark.parametrize('index',[27,66,68])
def test_unsquared_original_chart_integrates(index,record_property):
    import json,os,time
    from pathlib import Path
    from tools.opaque_original_motion_reference import angular_material_cover
    from tools.opaque_square_reference import integrate_square
    from tools.opaque_shading_intervals import Unresolved
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    start=time.monotonic();failure=None
    def check():
        if time.monotonic()-start>=90:raise Unresolved('mapped local construction inclusive deadline')
    with backend(profile='ball'):
        try:
            p,curve,cover,charts=angular_material_cover(frozen(),checkpoint=check,tight_discriminant=True,algebraic_inverse=True,mapped_transport=True,exit_angle=True,unsquared_inverse=True);row=charts[index]
            result=integrate_square(row['field'],prepare=row['prepare'],correction_integrator=row['correction_integrator'],tangent_correction_integrator=row['tangent_correction_integrator'],max_cells=256-cover.cells+1,max_depth=10-row['depth'],tolerance='5e-13',checkpoint=check)
            result=dict(status='local_chart_passed',basis=result,seconds=time.monotonic()-start,evaluations=cover.evaluations)
        except ValueError as error:failure=str(error);result=dict(status='unresolved',failure=getattr(error,'details',str(error)),seconds=time.monotonic()-start)
    path=Path(os.environ['GFFX_FULL_REFERENCE']);assert not path.exists();path.write_text(json.dumps(result,default=str,indent=2),encoding='utf8');record_property('mapped_local',path.read_text());assert failure is None,result





def test_unsquared_shared_original_complete_piece(record_property):
    import json,os
    from pathlib import Path
    from tools.opaque_original_motion_reference import angular_piece_reference
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    path=Path(os.environ['GFFX_FULL_REFERENCE']);failure=None;rows=[]
    with backend(profile='ball'):
        try:result=angular_piece_reference(frozen(),tight_discriminant=True,algebraic_inverse=True,mapped_transport=True,exit_angle=True,unsquared_inverse=True,progress=rows.append)
        except ValueError as error:failure=str(error);result=dict(status='unresolved',failure=getattr(error,'details',str(error)))
    result['progress']=rows
    assert not path.exists();path.write_text(json.dumps(result,default=str,indent=2),encoding='utf8');record_property('algebraic_shared_piece',path.read_text())
    assert failure is None,result
    assert result['algebraic_inverse'] and result['basis']['chart_count']==72 and result['basis']['cells']<=256
    assert all(v.width<=I('2e-12').lo for v in result['value']+result['derivative'])




@pytest.mark.parametrize('motion',['normal_tilt','view_x','screen_x'])
def test_original_unsquared_residual_encloses_direct_vector_identity(motion):
    from tools.opaque_algebraic_angle_inverse import UnsquaredAngleResidual
    from tools.opaque_prepared_angle import PreparedAngle
    from tools.opaque_original_motion_reference import prepare
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    with backend(profile='ball'):
        p=prepare(frozen(),piece=8,motion=motion,anchor=0,tight_discriminant=True);residual=UnsquaredAngleResidual(PreparedAngle(p))
        def direct(u,s,k):
            t,_=p['patch'].radial_map(u,s);fields=p['fields'](t,s);n,v=fields['n'],fields['v']
            dot=lambda a,b:sum((x*y for x,y in zip(a,b)),u.lift(0))
            a=dot(n,v);return a-k*(dot(n,n)*dot(v,v)).sqrt()
        for axis in (0,1):
            args=[Jet(I('.25','.75'),order=10),Jet(I('.25','.75'),order=10)]
            args[axis]=Jet.variable(args[axis][0,0],order=10);args[0]+=args[0].lift(0,tangent=Q(1,17));args[1]+=args[1].lift(0,tangent=Q(1,19))
            box=residual(*args,Jet(Q(1,16),order=10))
            for x in (Q(1,4),Q(1,2),Q(3,4)):
                pts=[Jet(x,order=10),Jet(x,order=10)];pts[axis]=Jet.variable(x,order=10);pts[0]+=pts[0].lift(0,tangent=Q(1,17));pts[1]+=pts[1].lift(0,tangent=Q(1,19))
                k=Jet(Q(1,16),order=10);a=residual(*pts,k);b=direct(*pts,k)
                for i in range(11):
                    for j in (0,1):
                        assert a[i,j].lo<=b[i,j].hi and b[i,j].lo<=a[i,j].hi,(axis,i,j)
                        assert box[i,j].lo<=b[i,j].hi and b[i,j].lo<=box[i,j].hi,(axis,i,j)
                        assert a[i,j].width<I('1e-25').lo
        assert residual.report()['entries']<=64
        with pytest.raises(ValueError):residual(Jet('.5'),Jet('.5'),Jet(0))




@pytest.mark.parametrize('motion',['normal_tilt','view_x','screen_x'])
def test_unsquared_root_slope_identity_matches_direct_positive_derivative(motion):
    from tools.opaque_algebraic_angle_inverse import UnsquaredAngleResidual
    from tools.opaque_prepared_angle import PreparedAngle
    from tools.opaque_original_motion_reference import prepare
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    with backend(profile='ball'):
        p=prepare(frozen(),piece=8,motion=motion,anchor=0,tight_discriminant=True);field=PreparedAngle(p);residual=UnsquaredAngleResidual(field)
        for x in (Q(1,1000),Q(1,4),Q(3,4)):
            for y in (Q(1,4),Q(3,4)):
                u=Jet(x);s=Jet(y);k=field(u,s)[0,0];slope=field.radial(u,s)[0,0]
                exact=residual(Jet.variable(x,order=1),Jet(y,order=1),Jet(k,order=1))[1,0]
                result=residual.root_slope(I(x),I(y),k,slope)
                assert result.lo>0 and result.lo<=exact.hi and exact.lo<=result.hi
                assert result.width<I('1e-25').lo
        with pytest.raises(ValueError):residual.root_slope(I('.5'),I('.5'),I('.1'),I(-1,1))




def test_unsquared_selector_requires_qualified_base_and_charges_construction(monkeypatch):
    import types
    import tools.opaque_original_motion_reference as m
    from tools.opaque_shading_intervals import Unresolved
    with pytest.raises(ValueError):m.angular_material_cover({},unsquared_inverse=True)
    with pytest.raises(ValueError):m.angular_piece_reference({},unsquared_inverse=1)
    clock=[0.];calls=[]
    monkeypatch.setattr(m,'time',types.SimpleNamespace(monotonic=lambda:clock[0]))
    def construct(data,*,checkpoint,**kw):
        calls.append(kw);clock[0]=91;checkpoint()
    monkeypatch.setattr(m,'angular_material_cover',construct)
    with pytest.raises(Unresolved,match='deadline'):m.angular_piece_reference({},tight_discriminant=True,algebraic_inverse=True,unsquared_inverse=True)
    assert calls==[dict(tight_discriminant=True,algebraic_inverse=True,unsquared_inverse=True)]


@pytest.mark.parametrize('axis',[0,1])
@pytest.mark.parametrize('scale',[Q(1),Q(1,2**60)])
def test_physical_angle_inverse_keeps_closed_moving_radius_and_jacobian(axis,scale):
    from tools.opaque_physical_angle import PhysicalAngleResidual
    from tools.opaque_moving_reference import P
    from tools.opaque_arb_jets import backend
    coeff={'n':[[P(0),P([1,scale*Q(1,7)]),P([1,scale*Q(1,7)])],[P(0)]*3,[P(1),P(0),P(0)]],'v':[[P(1),P(0),P(0)],[P(0)]*3,[P(0)]*3]}
    with backend(profile='ball'):
        field=PhysicalAngleResidual(coeff)
        for x in (I('.25'),I('.2','.3')):
            args=[Jet(x,order=10),Jet(x,order=10)];args[axis]=Jet.variable(x,order=10)
            t,s=args;t+=t.lift(0,tangent=scale*Q(1,17));s+=s.lift(0,tangent=-scale*Q(1,19));k=(1+t)/8
            expected=k/((1+s)*(1+s.lift(0,tangent=scale*Q(1,7)))*(1-k*k).sqrt())
            actual=field.inverse(k,s,lower=I(0));jac=(t.lift(Q(1,8))/field.radial(actual,s))
            exact_jac=1/(8*(1+s)*(1+s.lift(0,tangent=scale*Q(1,7)))*(1-k*k)**1*(1-k*k).sqrt())
            for a,b in [(actual,expected),(jac,exact_jac)]:
                for i in range(11):
                    for j in (0,1):
                        assert a[i,j].lo<=b[i,j].hi and b[i,j].lo<=a[i,j].hi,(i,j)
                        if x.width==0:assert a[i,j].width<I('1e-25').lo
            assert not actual[0,1].zero
        assert field.report()['entries']<=64
        with pytest.raises(ValueError):field.inverse(Jet(0),Jet('.5'),lower=I(0))



@pytest.mark.parametrize('scale',[Q(1),Q(1,2**60)])
def test_physical_angle_cover_retains_complete_moving_measure(scale):
    from tools.opaque_physical_angle import PhysicalAngleResidual,physical_angle_chart
    from tools.opaque_knot_partition import MonotoneAngularCover
    from tools.opaque_square_reference import integrate_square
    from tools.opaque_moving_reference import P
    from tools.opaque_arb_jets import backend
    coeff={'n':[[P(0),P([Q(1,4),Q(1,28)]),P(0)],[P(0)]*3,[P(1),P(0),P(Q(1,8))]],'v':[[P(1),P(0),P(0)],[P(0)]*3,[P(0)]*3]}
    with backend(profile='ball'):
        field=PhysicalAngleResidual(coeff)
        def angle(u,s):
            a,b=field.coefficients(s);return field.evaluate(a,u)/field.evaluate(b,u).sqrt()
        cover=MonotoneAngularCover(angle,[I(Q(1,8)),I(Q(15,64))]);charts=[];used=[]
        for index,c in enumerate(cover.charts):
            def integrand(t,y,index=index):
                row=physical_angle_chart(field,cover,index,t,y,lower=lambda s:I(0))
                if row is None:row=cover.map_angle_cover(index,t,y,field.radial);radius=row['u']
                else:radius=row['radius'];used.append(index)
                mass=scale*radius*row['jacobian']*(1+t.lift(0,tangent=Q(1,5)))
                return [mass,mass*radius]
            charts.append(dict(field=integrand,depth=c['depth']))
        result=integrate_square(None,chart_cover=charts,tolerance=str((I(scale)*I('1e-12')).lo))
        for val,grad,exact in zip(result['value'],result['derivative'],[scale*Q(1,2),scale*Q(1,3)]):
            a,b=I(exact),I(exact/5);assert val.lo<=a.lo<=a.hi<=val.hi;assert grad.lo<=b.lo<=b.hi<=grad.hi
            assert val.lo>0 and grad.lo>0
        assert used and any(cover.charts[i]['high'] is None for i in used)
        assert {i for i,c in enumerate(cover.charts) if c['low'] is not None}<=set(used)


@pytest.mark.parametrize('index',[27,57,63,65,68])
def test_physical_original_map_matches_complete_material_jet(index):
    from tools.opaque_original_motion_reference import angular_material_cover
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    with backend(profile='ball'):
        cases=[]
        for physical in (False,True):
            p,c,cover,charts=angular_material_cover(frozen(),tight_discriminant=True,algebraic_inverse=True,mapped_transport=True,exit_angle=True,unsquared_inverse=True,physical_angle=physical)
            if not physical and index in (63,65):
                # Independent normalized-root route at the same fixed-angle
                # target and event span, not the old linear-radius parameter.
                def entry(i,t,y,radial_derivative):
                    c=cover.charts[i];mapped=cover.map(i,t,y)
                    span=t.lift(cover.events[c['end']][0,0],tangent=cover.events[c['end']][0,1])
                    delta=cover.levels[c['high']]-cover.levels[c['low']];target=t*delta+cover.levels[c['low']]
                    u=cover._angle_inverse(target,mapped['s']);slope=p['prepared_angle'].radial(u,mapped['s'])
                    return dict(u=u,s=mapped['s'],angle=target,jacobian=span*delta/slope,band=c['band'],angular_range=mapped['angular_range'],coordinate_kind='angle')
                cover.map_angle_cover=entry
            row=charts[index];row['prepare'](None,None,(I('.5'),I('.5')))
            cases.append([row['field'](Jet.variable('.5',order=10),Jet('.5',order=10)),row['field'](Jet('.5',order=10),Jet.variable('.5',order=10))])
        for a,b in zip(cases[0],cases[1]):
            for x,y in zip(a,b):
                for i in range(11):
                    for j in (0,1):
                        assert x[i,j].lo<=y[i,j].hi and y[i,j].lo<=x[i,j].hi,(index,i,j)
                        assert y[i,j].width<I('1e-20').lo


def test_physical_selector_and_construction_deadline(monkeypatch):
    import types
    import tools.opaque_original_motion_reference as m
    from tools.opaque_shading_intervals import Unresolved
    with pytest.raises(ValueError):m.angular_material_cover({},physical_angle=True)
    with pytest.raises(ValueError):m.angular_piece_reference({},physical_angle=1)
    clock=[0.];calls=[]
    monkeypatch.setattr(m,'time',types.SimpleNamespace(monotonic=lambda:clock[0]))
    def construct(data,*,checkpoint,**kw):
        calls.append(kw);clock[0]=91;checkpoint()
    monkeypatch.setattr(m,'angular_material_cover',construct)
    with pytest.raises(Unresolved,match='deadline'):m.angular_piece_reference({},tight_discriminant=True,algebraic_inverse=True,mapped_transport=True,exit_angle=True,unsquared_inverse=True,physical_angle=True)
    assert calls==[dict(tight_discriminant=True,algebraic_inverse=True,mapped_transport=True,exit_angle=True,unsquared_inverse=True,physical_angle=True)]


@pytest.mark.parametrize('index',[27,57,63,65,66,68])
def test_physical_original_chart_integrates(index,record_property):
    import json,os,time
    from pathlib import Path
    from tools.opaque_original_motion_reference import angular_material_cover
    from tools.opaque_square_reference import integrate_square
    from tools.opaque_shading_intervals import Unresolved
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    start=time.monotonic();failure=None
    def check():
        if time.monotonic()-start>=90:raise Unresolved('mapped local construction inclusive deadline')
    with backend(profile='ball'):
        try:
            p,curve,cover,charts=angular_material_cover(frozen(),checkpoint=check,tight_discriminant=True,algebraic_inverse=True,mapped_transport=True,exit_angle=True,unsquared_inverse=True,physical_angle=True);row=charts[index]
            result=integrate_square(row['field'],prepare=row['prepare'],correction_integrator=row['correction_integrator'],tangent_correction_integrator=row['tangent_correction_integrator'],max_cells=256-cover.cells+1,max_depth=10-row['depth'],tolerance='5e-13',checkpoint=check)
            result=dict(status='local_chart_passed',basis=result,seconds=time.monotonic()-start,evaluations=cover.evaluations)
        except ValueError as error:failure=str(error);result=dict(status='unresolved',failure=getattr(error,'details',str(error)),seconds=time.monotonic()-start)
    path=Path(os.environ['GFFX_FULL_REFERENCE']);assert not path.exists();path.write_text(json.dumps(result,default=str,indent=2),encoding='utf8');record_property('mapped_local',path.read_text());assert failure is None,result






def test_physical_shared_original_complete_piece(record_property):
    import json,os
    from pathlib import Path
    from tools.opaque_original_motion_reference import angular_piece_reference
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    path=Path(os.environ['GFFX_FULL_REFERENCE']);failure=None;rows=[]
    with backend(profile='ball'):
        try:result=angular_piece_reference(frozen(),tight_discriminant=True,algebraic_inverse=True,mapped_transport=True,exit_angle=True,unsquared_inverse=True,physical_angle=True,progress=rows.append)
        except ValueError as error:failure=str(error);result=dict(status='unresolved',failure=getattr(error,'details',str(error)))
    result['progress']=rows
    assert not path.exists();path.write_text(json.dumps(result,default=str,indent=2),encoding='utf8');record_property('algebraic_shared_piece',path.read_text())
    assert failure is None,result
    assert result['algebraic_inverse'] and result['basis']['chart_count']==72 and result['basis']['cells']<=256
    assert all(v.width<=I('2e-12').lo for v in result['value']+result['derivative'])







def test_physical_entry_remaining_bound_diagnostic(record_property,monkeypatch):
    import json,os,time
    from pathlib import Path
    from tools.opaque_original_motion_reference import angular_material_cover
    from tools.opaque_arb_jets import backend
    from tools.opaque_shading_intervals import Unresolved
    from tools.opaque_tangent_peano import TangentCorrection
    from tests.packaging.test_opaque_motion_tangent import frozen
    start=time.monotonic();objects=[];original=TangentCorrection.__init__
    def capture(self,*args,**kwargs):original(self,*args,**kwargs);objects.append(self)
    monkeypatch.setattr(TangentCorrection,'__init__',capture)
    def check():
        if time.monotonic()-start>=45:raise Unresolved('entry bound diagnostic deadline')
    rows=[]
    with backend(profile='ball'):
        p,curve,cover,charts=angular_material_cover(frozen(),checkpoint=check,tight_discriminant=True,algebraic_inverse=True,mapped_transport=True,exit_angle=True,unsquared_inverse=True,physical_angle=True)
        row=charts[63];box=(I('.75',1),I('.5',1));row['prepare'](Q(7,8),Q(3,4),box)
        for axis in (0,1):
            args=[Jet(v,order=10) for v in box];args[axis]=Jet.variable(box[axis],order=10)
            fields=objects[63].inputs(*args);smooth=row['field'](*args)
            point=[Jet(Q(7,8),order=10),Jet(Q(3,4),order=10)];point[axis]=Jet.variable(point[axis][0,0],order=10)
            rows.append(dict(axis=axis,inputs=[{str(k):str(v) for k,v in q.c.items()} for q in fields],smooth=[{str(k):str(v) for k,v in q.c.items()} for q in smooth],point_smooth=[{str(k):str(v) for k,v in q.c.items()} for q in row['field'](*point)]))
        correction=row['correction_integrator'](box);tangent=row['tangent_correction_integrator'](box)
    result=dict(rows=rows,correction=correction,tangent=tangent,seconds=time.monotonic()-start,scope='Saved largest entry leaf; point coefficients are diagnostic only, not acceptance.')
    path=Path(os.environ['GFFX_FULL_REFERENCE']);assert not path.exists();path.write_text(json.dumps(result,default=str,indent=2),encoding='utf8');record_property('entry_bound',path.read_text())


def test_physical_entry_light_bound_removes_range_fallback(record_property,monkeypatch):
    import json,os,time
    from pathlib import Path
    from tools.opaque_original_motion_reference import angular_material_cover
    from tools.opaque_arb_jets import backend
    from tools.opaque_shading_intervals import Unresolved
    from tools.opaque_tangent_peano import TangentCorrection
    from tests.packaging.test_opaque_motion_tangent import frozen
    start=time.monotonic();objects=[];original=TangentCorrection.__init__
    def capture(self,*args,**kwargs):original(self,*args,**kwargs);objects.append(self)
    monkeypatch.setattr(TangentCorrection,'__init__',capture)
    def check():
        if time.monotonic()-start>=45:raise Unresolved('entry bound diagnostic deadline')
    rows=[]
    with backend(profile='ball'):
        p,curve,cover,charts=angular_material_cover(frozen(),checkpoint=check,tight_discriminant=True,algebraic_inverse=True,mapped_transport=True,exit_angle=True,unsquared_inverse=True,physical_angle=True)
        row=charts[63];box=(I('.75',1),I('.5',1));row['prepare'](Q(7,8),Q(3,4),box)
        for axis in (0,1):
            args=[Jet(v,order=10) for v in box];args[axis]=Jet.variable(box[axis],order=10)
            fields=objects[63].inputs(*args);smooth=row['field'](*args)
            point=[Jet(Q(7,8),order=10),Jet(Q(3,4),order=10)];point[axis]=Jet.variable(point[axis][0,0],order=10)
            rows.append(dict(axis=axis,inputs=[{str(k):str(v) for k,v in q.c.items()} for q in fields],smooth=[{str(k):str(v) for k,v in q.c.items()} for q in smooth],point_smooth=[{str(k):str(v) for k,v in q.c.items()} for q in row['field'](*point)]))
        correction=row['correction_integrator'](box);tangent=row['tangent_correction_integrator'](box)
    assert objects[63].range_fallbacks==0,objects[63].range_fallbacks
    result=dict(rows=rows,correction=correction,tangent=tangent,seconds=time.monotonic()-start,scope='Saved largest entry leaf; point coefficients are diagnostic only, not acceptance.')
    path=Path(os.environ['GFFX_FULL_REFERENCE']);assert not path.exists();path.write_text(json.dumps(result,default=str,indent=2),encoding='utf8');record_property('entry_bound',path.read_text())



def test_physical_prepared_polynomial_keeps_outward_endpoints_and_context():
    from tools.opaque_physical_angle import PreparedIntervalPolynomial
    from tools.opaque_angle_light import _evaluate
    from tools.opaque_arb_jets import backend
    import tools.opaque_shading_intervals as scalar
    p={(0,0,0):Q(1,3),(1,1,0):Q(-3,7),(2,0,0):Q(1,2**60),(4,2,0):Q(13,17),(2,1,1):Q(7,11)}
    actual=PreparedIntervalPolynomial(p)
    for precision in (60,75,60):
        with backend(profile='ball'):
            scalar.PRECISION=precision
            for t,s in [(I('.1','.9'),I('.2','.7')),(I(-1,1),I(-2,3)),(I(0),I(1)),(I(Q(1,2**60)),I(Q(1,3)))]:
                a,b=actual(t,s),_evaluate(p,t,s)
                assert (a.lo,a.hi)==(b.lo,b.hi)
    assert actual.rebuilds==3
    assert PreparedIntervalPolynomial({})(I(0),I(1)).zero



def test_physical_material_uses_direct_light_proof_without_repeated_cross_products(monkeypatch):
    from tools.opaque_original_motion_reference import angular_material_cover
    from tools.opaque_arb_jets import backend
    import tools.opaque_normal_light as old
    from tests.packaging.test_opaque_motion_tangent import frozen
    def unexpected(*a,**k):raise AssertionError('redundant cross-product proof reconstructed')
    with backend(profile='ball'):
        p,c,cover,charts=angular_material_cover(frozen(),tight_discriminant=True,algebraic_inverse=True,mapped_transport=True,exit_angle=True,unsquared_inverse=True,physical_angle=True)
        monkeypatch.setattr(old,'_normal_light_bound',unexpected)
        row=charts[63];box=(I('.75',1),I('.5',1));row['prepare'](Q(7,8),Q(3,4),box)
        assert all(q[0,0].lo>0 for q in row['field'](Jet.variable(box[0],order=10),Jet(box[1],order=10)))


def test_physical_remaining_cell_cost_diagnostic(record_property):
    import cProfile,pstats,time,json,os
    from pathlib import Path
    from tools.opaque_original_motion_reference import angular_material_cover
    from tools.opaque_square_reference import integrate_square
    from tools.opaque_arb_jets import backend
    from tools.opaque_shading_intervals import Unresolved
    from tests.packaging.test_opaque_motion_tangent import frozen
    start=time.monotonic()
    def check():
        if time.monotonic()-start>=45:raise Unresolved('repaired initial-cell diagnostic deadline')
    with backend(profile='ball'):
        p,curve,cover,charts=angular_material_cover(frozen(),tight_discriminant=True,algebraic_inverse=True,mapped_transport=True,exit_angle=True,unsquared_inverse=True,physical_angle=True,checkpoint=check)
        row=charts[46];prof=cProfile.Profile();prof.enable()
        try:result=integrate_square(row['field'],prepare=row['prepare'],correction_integrator=row['correction_integrator'],tangent_correction_integrator=row['tangent_correction_integrator'],tolerance='5e-13',max_cells=1,max_depth=10,checkpoint=check)
        except Unresolved as error:result=error.details
        finally:prof.disable()
        rows=sorted([dict(file=k[0],line=k[1],function=k[2],calls=v[1],self_seconds=v[2],cumulative_seconds=v[3]) for k,v in pstats.Stats(prof).stats.items()],key=lambda r:r['self_seconds'],reverse=True)
    path=Path(os.environ['GFFX_FULL_REFERENCE']);assert not path.exists();path.write_text(json.dumps(dict(seconds=time.monotonic()-start,result=result,self_costs=rows[:35],cumulative_costs=sorted(rows,key=lambda r:r['cumulative_seconds'],reverse=True)[:35],scope='One initial cell with profiler overhead; no piece acceptance.'),default=str,indent=2),encoding='utf8');record_property('repaired_cell_cost',path.read_text())
