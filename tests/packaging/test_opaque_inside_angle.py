"""Closed area/motion proofs and actual inside-horizon field obligations."""
from fractions import Fraction as Q
import pytest
from tools.opaque_shading_intervals import I,Jet

@pytest.mark.parametrize('scale',[Q(1),-Q(1,2**60)])
def test_constant_edge_cover_preserves_area_and_tiny_motion(scale):
    from tools.opaque_inside_angle import ConstantEdgeAngularCover
    from tools.opaque_square_reference import integrate_square
    from tools.opaque_arb_jets import backend
    with backend(profile='ball',fast_conversion=True):
        def field(u,s):return u*u.lift(Q(3,4),tangent=scale)
        def derivative(u,s):return u.lift(Q(3,4),tangent=scale)
        cover=ConstantEdgeAngularCover(field,[I(Q(1,4)),I(Q(1,2))]);charts=[]
        assert not cover.events and len(cover.charts)==3 and cover.cells==5
        for k,c in enumerate(cover.charts):
            def integrand(t,y,index=k):
                row=cover.map_exit_angle(index,t,y,derivative) if cover.charts[index]['high'] is None else cover.map_angle_cover(index,t,y,derivative)
                u=row['u'];j=row['jacobian'];return [scale*(1-u)*j,field(u,row['s'])*(1-u)*j]
            charts.append(dict(field=integrand,depth=c['depth']))
        result=integrate_square(None,chart_cover=charts,tolerance='1e-35')
        for value,expected in zip(result['value'],[I(scale/2),I(Q(1,8))]):assert value.lo<=expected.lo<=expected.hi<=value.hi
        for value,expected in zip(result['derivative'],[I(0),I(scale/6)]):assert value.lo<=expected.lo<=expected.hi<=value.hi
        assert sum(Q(1,2**c['depth']) for c in cover.charts)==1

@pytest.mark.parametrize('bad',['variable_edge','flat_radial','moving_horizon','outside_level'])
def test_constant_edge_cover_refuses_missing_proofs(bad):
    from tools.opaque_inside_angle import ConstantEdgeAngularCover
    def field(u,s):
        if bad=='variable_edge':return u*(1-s/4)
        if bad=='flat_radial':return u*0+Q(3,4)
        if bad=='moving_horizon':return u+u.lift(0,tangent=Q(1,2**60))
        return u*Q(3,4)
    with pytest.raises(ValueError):ConstantEdgeAngularCover(field,[I(1) if bad=='outside_level' else I(Q(1,4))])

@pytest.mark.parametrize('piece',[11,12])
@pytest.mark.parametrize('motion',['normal_tilt','view_x','screen_x'])
def test_inside_prepared_angle_retains_actual_horizon_and_motion(piece,motion):
    from tools.opaque_inside_angle import InsideAngle
    from tools.opaque_original_motion_reference import prepare,mapped_inputs
    from tests.packaging.test_opaque_motion_tangent import frozen
    from tools.opaque_arb_jets import backend
    with backend(profile='ball',fast_conversion=True):
        p=prepare(frozen(),piece=piece,motion=motion,anchor=0,tight_discriminant=True);field=InsideAngle(p)
        for u,s in [(Q(0),I(0,1)),(Q(1,4),Q(1,3)),(Q(3,4),Q(2,3)),(Q(1),I(0,1))]:
            a=field(Jet.variable(u,order=2),Jet(s,order=2));b=mapped_inputs(p,1-Jet.variable(u,order=2),Jet(s,order=2),flatten=False)[0]
            for k in set(a.c)|set(b.c):assert max(a[k].lo,b[k].lo)<=min(a[k].hi,b[k].hi)
        horizon=field(Jet(0),Jet(I(0,1)));assert horizon[0,0].zero and horizon[0,1].zero
        edge=field.edge(Jet.variable(I(0,1),order=2));assert all(edge[i,j].zero for i in (1,2) for j in (0,1))
        natural=field(Jet.variable(I(I(Q(1,4)).lo,I(Q(3,4)).hi),order=1),Jet(I(0,1),order=1))[1,0]
        analytic=field.radial(Jet(I(I(Q(1,4)).lo,I(Q(3,4)).hi)),Jet(I(0,1)))[0,0]
        assert max(natural.lo,analytic.lo)<=min(natural.hi,analytic.hi)

@pytest.mark.parametrize('piece',[11,12])
@pytest.mark.parametrize('motion',['normal_tilt','view_x','screen_x'])
def test_inside_actual_angular_cover_has_complete_measure_and_matched_motion(piece,motion):
    from tools.opaque_inside_angle import inside_material_cover
    from tools.opaque_original_motion_reference import mapped_inputs
    from tests.packaging.test_opaque_motion_tangent import frozen
    from tools.opaque_arb_jets import backend
    with backend(profile='ball',fast_conversion=True):
        p,curve,cover,charts=inside_material_cover(frozen(),piece=piece,motion=motion,anchor=0)
        assert p['patch'].inside and p['inside_angle'] and p['motion']==motion and p['piece']==piece
        assert cover.radial.lo>0 and cover.edge_slope.zero and not cover.events
        assert sum(Q(1,2**r['depth']) for r in charts)==1 and len(charts)==len(cover.charts)
        for index,c in enumerate(cover.charts):
            row=cover.map_exit_angle(index,Jet(Q(1,2)),Jet(Q(1,2)),p['prepared_angle'].radial) if c['high'] is None else cover.map_angle_cover(index,Jet(Q(1,2)),Jet(Q(1,2)),p['prepared_angle'].radial)
            native=mapped_inputs(p,1-row['u'],row['s'],flatten=False)
            if row.get('coordinate_kind')=='angle':
                for k in ((0,0),(0,1)):assert max(native[0][k].lo,row['angle'][k].lo)<=min(native[0][k].hi,row['angle'][k].hi)
            assert (native[3]*row['jacobian'])[0,0].lo>0

@pytest.mark.parametrize('piece',[0,8,9])
def test_inside_profile_refuses_noninside_geometry(piece):
    from tools.opaque_inside_angle import inside_material_cover
    from tests.packaging.test_opaque_motion_tangent import frozen
    from tools.opaque_arb_jets import backend
    with backend(profile='ball',fast_conversion=True):
        with pytest.raises(ValueError,match='inside curved'):inside_material_cover(frozen(),piece=piece,motion='normal_tilt',anchor=0)

@pytest.mark.parametrize('piece',[11,12])
@pytest.mark.parametrize('motion',['normal_tilt','view_x','screen_x'])
def test_inside_shared_shading_callbacks_match_independent_vector_radiance(piece,motion):
    from tools.opaque_inside_angle import inside_material_cover
    from tests.packaging.test_opaque_motion_tangent import frozen
    from tools.opaque_arb_jets import backend
    with backend(profile='ball',fast_conversion=True):
        data=frozen();p,curve,cover,charts=inside_material_cover(data,piece=piece,motion=motion,anchor=0)
        for index in (0,len(charts)//2,len(charts)-1):
            t,y=Jet(Q(1,2)),Jet(Q(1,3));c=cover.charts[index]
            row=cover.map_exit_angle(index,t,y,p['prepared_angle'].radial) if c['high'] is None else cover.map_angle_cover(index,t,y,p['prepared_angle'].radial)
            radius,jac=p['patch'].radial_map(1-row['u'],row['s']);f=p['fields'](radius,row['s'])
            expected=curve.radiance(f['n'],f['v'],[t.lift(k) for k in (1,1,2)],[t.lift(a) for a in data['material']['albedo']],[t.lift(.04)]*3)
            charts[index]['prepare'](Q(1,2),Q(1,3),(I(Q(1,2)),I(Q(1,3))))
            actual=charts[index]['field'](t,y)
            for a,v in zip(data['material']['albedo'],expected):
                got=actual[0]+actual[1]*I(a);want=v*jac*row['jacobian']
                for k in ((0,0),(0,1)):assert max(got[k].lo,want[k].lo)<=min(got[k].hi,want[k].hi)

@pytest.mark.parametrize('fail',[False,True])
def test_inside_complete_reference_keeps_identity_and_terminal_failure(monkeypatch,fail):
    import tools.opaque_inside_angle as m
    from types import SimpleNamespace
    from tools.opaque_shading_intervals import Unresolved
    seen=[]
    def cover(data,**kw):seen.append(kw);return dict(face=5743,horizon_cache=SimpleNamespace(report=lambda:{}),prepared_angle=SimpleNamespace(report=lambda:{})),None,SimpleNamespace(cells=3,depth=1,evaluations=2),[]
    monkeypatch.setattr(m,'inside_material_cover',cover)
    def integrate(*a,**kw):
        if fail:raise Unresolved('retained test failure')
        return dict(value=[I(0),I(0)],derivative=[I(0),I(0)],cells=3,depth=1)
    monkeypatch.setattr(m,'integrate_square',integrate)
    if fail:
        with pytest.raises(Unresolved) as e:m.inside_piece_reference({'material':{'albedo':[.2,.4,.6]}},piece=11,motion='view_x',anchor=0)
        assert e.value.details['piece']==11 and e.value.details['motion']=='view_x' and e.value.details['anchor']==0
    else:
        r=m.inside_piece_reference({'material':{'albedo':[.2,.4,.6]}},piece=11,motion='view_x',anchor=0)
        assert (r['piece'],r['face'],r['motion'],r['anchor'])==(11,5743,'view_x',0)
        assert r['inside_angle'] and r['status']=='complete_original_piece_integrated'
    assert seen[0]['piece']==11 and seen[0]['motion']=='view_x' and seen[0]['anchor']==0

@pytest.mark.parametrize('piece',[11,12])
@pytest.mark.parametrize('motion',['normal_tilt','view_x','screen_x'])
def test_inside_unsquared_residual_retains_independent_mixed_field(piece,motion):
    from tools.opaque_inside_angle import InsideAngle
    from tools.opaque_algebraic_angle_inverse import UnsquaredAngleResidual
    from tools.opaque_original_motion_reference import prepare
    from tests.packaging.test_opaque_motion_tangent import frozen
    from tools.opaque_arb_jets import backend
    with backend(profile='ball',fast_conversion=True):
        p=prepare(frozen(),piece=piece,motion=motion,anchor=0,tight_discriminant=True);angle=InsideAngle(p);residual=UnsquaredAngleResidual(angle)
        for axis in (0,1):
            u=Jet.variable(Q(1,2),order=4) if axis==0 else Jet(Q(1,2),order=4)
            s=Jet.variable(Q(1,3),order=4) if axis==1 else Jet(Q(1,3),order=4)
            target=u.lift(Q(1,256),tangent=-Q(1,2**60));got=residual(u,s,target)
            radius,_=p['patch'].radial_map(1-u,s);fields=p['fields'](radius,s);n,v=fields['n'],fields['v'];dot=lambda a,b:sum((x*y for x,y in zip(a,b)),u.lift(0))
            want=dot(n,v)-target*(dot(n,n)*dot(v,v)).sqrt()
            for k in set(got.c)|set(want.c):assert max(got[k].lo,want[k].lo)<=min(got[k].hi,want[k].hi)

@pytest.mark.parametrize('piece',[11,12])
@pytest.mark.parametrize('motion',['normal_tilt','view_x','screen_x'])
def test_inside_physical_inverse_matches_independent_field_and_known_motion(piece,motion):
    from tools.opaque_inside_angle import InsidePhysicalAngleResidual
    from tools.opaque_original_motion_reference import prepare
    from tests.packaging.test_opaque_motion_tangent import frozen
    from tools.opaque_arb_jets import backend
    with backend(profile='ball',fast_conversion=True):
        p=prepare(frozen(),piece=piece,motion=motion,anchor=0,tight_discriminant=True);field=InsidePhysicalAngleResidual(p['fields'].coefficients,compiled_series=True,contraction_stop=True)
        rho=Jet.variable(Q(999,1000),order=2)+Jet(Q(0),order=2).lift(0,tangent=-Q(1,2**60));s=Jet(Q(1,3),order=2)
        fields=p['fields'](1-rho,s);n,v=fields['n'],fields['v'];dot=lambda a,b:sum((x*y for x,y in zip(a,b)),rho.lift(0));B=dot(n,n)*dot(v,v);target=dot(n,v)/B.sqrt()
        residual=field(rho,s,target)
        for k in set(residual.c):assert residual[k].lo<=0<=residual[k].hi
        lower=(1-p['patch'].root(Jet(s[0,0])))[0,0];got=field.inverse(target,s,lower=lower)
        for i in range(3):
            for j in (0,1):assert got[i,j].lo<=rho[i,j].lo<=rho[i,j].hi<=got[i,j].hi
        assert field._slope(got[0,0],s[0,0]).lo>0 and field.report()['coordinate']=='reversed inside physical radius'
