"""Physical-root, measure, motion and profile obligations for polynomial light."""
from fractions import Fraction as Q
from types import SimpleNamespace
import pytest
from tools.opaque_shading_intervals import I,Jet,Unresolved
from tools.opaque_arb_jets import backend
from tests.packaging.test_opaque_motion_tangent import frozen
from tests.packaging.test_opaque_motion_extension import PROFILE

@pytest.mark.parametrize('bad',[dict(polynomial_light=1),dict(polynomial_light=True),dict(polynomial_light=True,physical_angle=True)])
def test_polynomial_light_refuses_unqualified_outside_before_geometry(monkeypatch,bad):
    import tools.opaque_original_motion_reference as m
    monkeypatch.setattr(m,'prepare',lambda *a,**kw:pytest.fail('invalid profile reached geometry'))
    for method in (m.angular_material_cover,m.angular_piece_reference):
        with pytest.raises(ValueError,match='polynomial light'):method({},**bad)

@pytest.mark.parametrize('family',['outside','inside'])
@pytest.mark.parametrize('failure',[False,True])
def test_polynomial_light_complete_and_failed_identity(monkeypatch,family,failure):
    import tools.opaque_original_motion_reference as outside
    import tools.opaque_inside_angle as inside
    m=outside if family=='outside' else inside;name='angular_material_cover' if family=='outside' else 'inside_material_cover';seen=[]
    def cover(data,**kw):
        seen.append(kw);return dict(face=5742,prepared_angle=SimpleNamespace(report=lambda:{}),horizon_cache=SimpleNamespace(report=lambda:{})),None,SimpleNamespace(cells=1,depth=0,evaluations=0),[]
    monkeypatch.setattr(m,name,cover)
    def integrate(*a,**kw):
        assert kw['max_cells']==256 and kw['max_depth']==10 and kw['tolerance']=='3e-13';kw['checkpoint']()
        if failure:raise Unresolved('retained binding failure')
        return dict(value=[I(0)]*2,derivative=[I(0)]*2)
    monkeypatch.setattr(m,'integrate_square',integrate)
    method=m.angular_piece_reference if family=='outside' else m.inside_piece_reference
    kw=dict(piece=9 if family=='outside' else 11,motion='normal_tilt',anchor=0,tolerance='3e-13',polynomial_light=True)
    if family=='outside':kw.update(PROFILE,curved_transverse=True)
    if failure:
        with pytest.raises(Unresolved) as e:method({'material':{'albedo':[.2,.4,.6]}},**kw)
        assert e.value.details['polynomial_light'] is True
    else:assert method({'material':{'albedo':[.2,.4,.6]}},**kw)['polynomial_light'] is True
    assert seen[0]['polynomial_light'] is True and seen[0]['motion']=='normal_tilt'

@pytest.mark.parametrize('piece',[8,11])
@pytest.mark.parametrize('motion',['normal_tilt','view_x','screen_x'])
@pytest.mark.parametrize('order',[1,4])
def test_polynomial_light_physical_callbacks_keep_angle_measure_and_mixed_motion(monkeypatch,piece,motion,order):
    import tools.opaque_original_motion_reference as m
    from tools.opaque_inside_angle import inside_material_cover
    from tools.opaque_physical_angle import physical_angle_chart
    from tools.opaque_polynomial_inputs import PolynomialCosines
    from tools.opaque_loss_extension import bounded_cosines
    captures=[];old=m.TangentCorrection
    def correction(e,inputs,**kw):captures.append(inputs);return old(e,inputs,**kw)
    monkeypatch.setattr(m,'TangentCorrection',correction)
    with backend(profile='ball',fast_conversion=True):
        if piece==8:p,curve,cover,charts=m.angular_material_cover(frozen(),piece=piece,motion=motion,anchor=0,**PROFILE,curved_transverse=True,polynomial_light=True)
        else:p,curve,cover,charts=inside_material_cover(frozen(),piece=piece,motion=motion,anchor=0,polynomial_light=True)
        assert p['polynomial_light'] is True and isinstance(p['polynomial_light_model'],PolynomialCosines)
        index=next(i for i,c in enumerate(cover.charts) if c['low'] is not None and c['high'] is not None)
        selected=captures[index];baseline=dict(p);baseline.pop('polynomial_light_model');bounds=bounded_cosines(lambda u,s:list(m.mapped_inputs(p,1-u if piece==11 else u,s,flatten=False)[:2]),I(0,1),I(0,1),monotone=True,partial=True)
        base=m.angular_material_chart(baseline,curve,cover,index,bounds);ordinary=captures[-1]
        t=Jet.variable(I(Q(1,2)),order=order);y=Jet(I(Q(1,3)),order=order)
        row=physical_angle_chart(p['physical_angle'],cover,index,t,y,lower=lambda s:(1-p['patch'].root(Jet(s[0,0])) if piece==11 else p['patch'].root(Jet(s[0,0])))[0,0]);assert row is not None
        radius=1-row['radius'] if piece==11 else row['radius'];expected_jac=radius*radius.lift(p['patch'].det[0],tangent=p['patch'].det[1])*row['jacobian']
        seen=[];method=p['polynomial_light_model'].light_highlight
        def light(*a,**kw):seen.append(kw);return method(*a,**kw)
        monkeypatch.setattr(p['polynomial_light_model'],'light_highlight',light)
        new=selected(t,y);previous=ordinary(t,y)
        assert seen==[dict(material_only=True)] and not new[2].c
        for i in range(order+1):
            for j in (0,1):
                k=(i,j)
                assert (new[0][k].lo,new[0][k].hi)==(row['angle'][k].lo,row['angle'][k].hi)
                assert (new[3][k].lo,new[3][k].hi)==(expected_jac[k].lo,expected_jac[k].hi)
                assert max(new[1][k].lo,previous[1][k].lo)<=min(new[1][k].hi,previous[1][k].hi)
        box=(I(Q(1,2)),I(Q(1,3)));charts[index]['prepare'](Q(1,2),Q(1,3),box);base['prepare'](Q(1,2),Q(1,3),box)
        new=charts[index]['field'](t,y);previous=base['field'](t,y)
        assert any(kw==dict(material_only=False) for kw in seen)
        for a,b in zip(new,previous):
            for i in range(order+1):
                for j in (0,1):assert max(a[i,j].lo,b[i,j].lo)<=min(a[i,j].hi,b[i,j].hi)



def test_saved_curved_polynomial_light_cost_and_complete_bound_components(record_property):
    import os,json,hashlib,time
    from pathlib import Path
    import tools.opaque_original_motion_reference as m
    from tools.opaque_tensor_smooth import rule,quarter_factors
    pin=Path(os.environ['GFFX_CURVED_DIAGNOSTIC']);assert hashlib.sha256(pin.read_bytes()).hexdigest()==os.environ['GFFX_CURVED_DIAGNOSTIC_SHA']
    spec=json.loads(pin.read_bytes());src=Path(spec['path']);assert hashlib.sha256(src.read_bytes()).hexdigest()==spec['sha256'];saved=json.loads(src.read_bytes())
    assert saved['status']=='qualified' and saved['case']==dict(piece=8,motion='normal_tilt',anchor=0) and len(spec['regions'])==3
    start=time.monotonic();rows=[];construction=[]
    def check():
        if time.monotonic()-start>=90:raise Unresolved('saved curved polynomial light deadline')
    with backend(profile='ball',fast_conversion=True):
        families={}
        for selected in (False,True):
            before=time.monotonic();p,curve,cover,charts=m.angular_material_cover(frozen(),piece=8,motion='normal_tilt',anchor=0,checkpoint=check,**PROFILE,curved_transverse=True,polynomial_light=selected)
            families[selected]=charts;construction.append(dict(polynomial_light=selected,seconds=time.monotonic()-before,charts=len(charts),cells=cover.cells));check()
        _,_,(order_s,order_t),(factor_s,factor_t)=rule(checkpoint=check);quarters=quarter_factors(checkpoint=check)
        for index,region in enumerate(spec['regions']):
            chart=region['chart'];a,b,c,d=map(Q,region['box']);box=(I(I(a).lo,I(b).hi),I(I(c).lo,I(d).hi));cases=[]
            for selected in ([False,True] if index%2==0 else [True,False]):
                before=time.monotonic();row=families[selected][chart];selected_rule=row.get('tensor_rule');assert selected_rule in (None,'gauss3x8-local4');row['prepare']((a+b)/2,(c+d)/2,box)
                os_,ot_=(order_s,order_t) if selected_rule else (10,10)
                from math import factorial
                fs=factor_s if selected_rule else I(Q(factorial(5)**4,11*factorial(10)**2))
                ordinary=row['correction_integrator'](box);tangent=row['tangent_correction_integrator'](box)
                ds=row['field'](Jet.variable(box[0],order=os_),Jet(box[1],order=os_));dts=[]
                parts=4 if selected_rule else 1
                for part in range(parts):
                    lo,hi=c+(d-c)*Q(part,parts),c+(d-c)*Q(part+1,parts);dts.append(row['field'](Jet(box[0],order=ot_),Jet.variable(I(I(lo).lo,I(hi).hi),order=ot_)));check()
                hs,ht=I(b-a),I(d-c);radii=[]
                for k in range(2):
                    channel=[]
                    for motion in (0,1):
                        es=ht*hs**(os_+1)*ds[k][os_,motion].bound*fs
                        et=hs*ht**(ot_+1)*(sum((I(v[k][ot_,motion].bound)*mass for v,mass in zip(dts,quarters)),I(0)) if selected_rule else I(dts[0][k][ot_,motion].bound)*fs);channel.append(str((es+et).hi))
                    radii.append(channel)
                cases.append(dict(tensor_rule=selected_rule,polynomial_light=selected,seconds=time.monotonic()-before,ordinary_widths=[str(v.width) for v in ordinary],combined_tangent_widths=[str(v.width) for v in tangent],smooth_remainder_radii=radii));check()
            rows.append(dict(chart=chart,box=region['box'],cases=cases))
    p=Path(os.environ['GFFX_FULL_REFERENCE']);assert not p.exists();p.write_text(json.dumps(dict(status='diagnostic_complete',construction=construction,rows=rows,seconds=time.monotonic()-start,selection_sha256=os.environ['GFFX_CURVED_DIAGNOSTIC_SHA'],scope='Fresh old/new construction and alternating local comparison of ordinary residual, combined tangent and both smooth remainder orders on three saved regions; no complete-cover or whole-piece/runtime acceptance.'),indent=2),encoding='utf8');record_property('curved_polynomial_light_diagnostic',str(p))


@pytest.mark.parametrize('piece',[8,11])
@pytest.mark.parametrize('motion',['normal_tilt','view_x','screen_x'])
@pytest.mark.parametrize('order',[1,4])
def test_horizon_material_query_forwards_only_consumed_fields(monkeypatch,piece,motion,order):
    import tools.opaque_original_motion_reference as m
    from tools.opaque_inside_angle import inside_material_cover
    captured=[];ctor=m.TangentCorrection
    def correction(e,inputs,**kw):captured.append(inputs);return ctor(e,inputs,**kw)
    monkeypatch.setattr(m,'TangentCorrection',correction)
    with backend(profile='ball',fast_conversion=True):
        if piece==8:p,curve,cover,charts=m.angular_material_cover(frozen(),piece=piece,motion=motion,anchor=0,**PROFILE,curved_transverse=True,polynomial_light=True)
        else:p,curve,cover,charts=inside_material_cover(frozen(),piece=piece,motion=motion,anchor=0,polynomial_light=True)
        assert cover.charts[0]['low'] is None
        original=m.mapped_inputs;seen=[]
        def inputs(*a,**kw):
            assert type(kw.get('material_only'))==bool,'horizon material query dropped its explicit consumption flag'
            selected=kw['material_only'];seen.append(selected);got=original(*a,**kw);full=original(*a,**dict(kw,material_only=False))
            for index in (0,1,3):
                for i in range(a[1].order+1):
                    for j in (0,1):assert (got[index][i,j].lo,got[index][i,j].hi)==(full[index][i,j].lo,full[index][i,j].hi)
            if selected:assert not got[2].c
            else:assert got[2].c
            return got
        monkeypatch.setattr(m,'mapped_inputs',inputs)
        t=Jet.variable(I(Q(1,2)),order=order);y=Jet(I(Q(1,3)),order=order)
        captured[0](t,y);assert seen==[True]
        charts[0]['prepare'](Q(1,2),Q(1,3),(I(Q(1,2)),I(Q(1,3))));charts[0]['field'](t,y)
        assert False in seen and True in seen
