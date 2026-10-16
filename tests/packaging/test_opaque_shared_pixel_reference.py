"""One conserved complete-pixel reference pool across independent visible pieces."""
import json,copy
from pathlib import Path
from fractions import Fraction as Q
import pytest
from tools.opaque_full_material_reference import full_pixel
from tools.opaque_principled_reference import PreparedPrincipled
from tools.opaque_shading_intervals import I,Jet

def fixture():
    from tools.corpus_registry import load_entry
    # The installed frozen loss resource is read as data, never evaluated by native code.
    from gffx.torch.material import PrincipledBudget
    import torch
    budget=PrincipledBudget.bundled('ggx-smith-loss-cpu-f64-v1',device='cpu',dtype=torch.float64)
    triangles=[[[0,0],[1,0],[0,1]],[[1,0],[1,1],[0,1]]]
    faces=[dict(face=i,corners=tri,clip_w=[1]*3,normal=[[0,0,1]]*3,view=[[.1+.2*i,.2,1]]*3) for i,tri in enumerate(triangles)]
    return dict(faces=faces,selected=dict(origin=[0,0],visible_parts=[dict(face=i,triangle=t) for i,t in enumerate(triangles)]),budget={k:v.tolist() for k,v in budget._data.items()},material=dict(roughness=1.,f0=[.04]*3,albedo=[.2,.4,.6],direction=[1.,1.,2.]))

def test_shared_pixel_pool_matches_closed_piecewise_constant_integral():
    data=fixture();result=full_pixel(data,shared_piece_budget=True,material_only_inputs=True);curve=PreparedPrincipled(data['budget'],1.)
    expected=[I(0)]*3
    for f in data['faces']:
        rgb=curve.radiance([Jet(v) for v in f['normal'][0]],[Jet(v) for v in f['view'][0]],[Jet(v) for v in data['material']['direction']],[Jet(v) for v in data['material']['albedo']],[Jet(v) for v in data['material']['f0']])
        expected=[a+b[0,0]/2 for a,b in zip(expected,rgb)]
    for actual,closed in zip(result['value'],expected):
        assert max(actual.lo,closed.lo)<=min(actual.hi,closed.hi)
        assert (actual-closed).bound<I('1e-50').lo
        assert actual.width<=I('2e-12').lo
    assert result['cells']<=256 and result['shared_piece_budget']

def test_shared_pool_closures_keep_distinct_piece_fields_and_charge_one_pool(monkeypatch):
    import tools.opaque_full_material_reference as m
    data=fixture();calls=[]
    original=m.integrate_square
    def audited(field,**kw):
        calls.append(kw);assert field is None
        charts=kw['chart_cover'];assert len(charts)==2 and [c['depth'] for c in charts]==[0,0] and kw['disjoint_roots'] is True
        values=[]
        for c in charts:
            c['prepare'](0,0,(I('.5'),I('.5')))
            values.append([str(q[0,0]) for q in c['field'](Jet('.5'),Jet('.5'))])
        assert values[0]!=values[1]
        assert kw['tolerance']=='5e-13' and kw['max_cells']==256 and kw['max_depth']==10
        return original(field,**kw)
    monkeypatch.setattr(m,'integrate_square',audited)
    result=m.full_pixel(data,shared_piece_budget=True,material_only_inputs=True)
    assert len(calls)==1 and result['cells']>=2

def test_shared_reference_preserves_legacy_default_and_refuses_campaign_mixing():
    data=fixture();a=full_pixel(data,material_only_inputs=True);b=full_pixel(data,material_only_inputs=True,shared_piece_budget=True)
    for x,y in zip(a['value'],b['value']):assert max(x.lo,y.lo)<=min(x.hi,y.hi)
    with pytest.raises(ValueError,match='shared'):full_pixel(data,shared_piece_budget=True,selected_piece=0)
    with pytest.raises(ValueError,match='shared'):full_pixel(data,shared_piece_budget=1)


def test_shared_curved_reference_tight_root_preserves_field_and_first_jet(monkeypatch):
    import tools.opaque_full_material_reference as m
    from tools.opaque_quadratic_sqrt import QuadraticSqrt
    data=fixture();data['faces']=data['faces'][:1];data['selected']['visible_parts']=data['selected']['visible_parts'][:1]
    data['faces'][0]['normal']=[[1,0,-.25],[1,0,.75],[1,0,.75]]
    data['faces'][0]['view']=[[0,0,1]]*3
    class Captured(Exception):pass
    charts=[]
    def capture(field,**kw):charts.append(kw['chart_cover'][0]);raise Captured()
    monkeypatch.setattr(m,'integrate_square',capture)
    for tight in (False,True):
        with pytest.raises(Captured):m.full_pixel(data,shared_piece_budget=True,tight_discriminant=tight,material_only_inputs=True)
    results=[]
    for c in charts:
        c['prepare'](0,0,(I('.5'),I('.5')))
        results.append(c['field'](Jet.variable('.5',order=10),Jet('.5',order=10)))
    for a,b in zip(*results):
        for i in range(11):
            assert max(a[i,0].lo,b[i,0].lo)<=min(a[i,0].hi,b[i,0].hi)
    with pytest.raises(ValueError,match='discriminant'):m.full_pixel(data,tight_discriminant=1)


def test_shared_aggregate_residual_keeps_distinct_fields_and_complete_bounds(monkeypatch):
    import tools.opaque_full_material_reference as m
    from tools.opaque_residual_range import ResidualRangeIndex
    charts=[]
    class Captured(Exception):pass
    def capture(field,**kw):charts.extend(kw['chart_cover']);raise Captured()
    monkeypatch.setattr(m,'integrate_square',capture)
    with pytest.raises(Captured):m.full_pixel(fixture(),shared_piece_budget=True,aggregate_residual=True,material_only_inputs=True)
    indexes=[];values=[]
    for chart in charts:
        chart['prepare'](0,0,(I('.2','.4'),I('.3','.5')))
        correction=chart['correction_integrator'];indexes.append(correction.range_index)
        pooled=correction.envelope_integral((I('.2','.4'),I('.3','.5')))
        correction.range_index=ResidualRangeIndex(correction.extension)
        separate=correction.envelope_integral((I('.2','.4'),I('.3','.5')))
        for a,b in zip(pooled,separate):assert (a.lo,a.hi)==(b.lo,b.hi)
        values.append([str(v[0,0]) for v in chart['field'](Jet('.3'),Jet('.4'))])
    assert indexes[0] is indexes[1] and indexes[0] is not None
    assert values[0]!=values[1]
    for kwargs in ({'aggregate_residual':True},{'aggregate_residual':1,'shared_piece_budget':True}):
        with pytest.raises(ValueError,match='aggregate'):m.full_pixel(fixture(),**kwargs)

    from tools.opaque_peano_material import MaterialCorrection
    for flags in (dict(curved=False,envelope_fallback=True),dict(curved=False,planar_envelope=1),dict(curved=True,planar_envelope=True,envelope_fallback=True),dict(curved=False,planar_envelope=True)):
        with pytest.raises(ValueError):MaterialCorrection(None,None,**flags)


def test_shared_localized_smooth_remainder_keeps_one_budget_and_closed_answer(monkeypatch):
    import tools.opaque_full_material_reference as m
    original=m.integrate_square;calls=[]
    def audit(field,**kw):
        calls.append(kw)
        assert kw['max_cells']==256 and kw['max_depth']==10 and kw['tolerance']=='5e-13'
        assert all(c['remainder_partition']==(4,4) and c['refinement_field'] is None for c in kw['chart_cover'])
        return original(field,**kw)
    data=fixture();baseline=m.full_pixel(data,shared_piece_budget=True,material_only_inputs=True)
    monkeypatch.setattr(m,'integrate_square',audit)
    localized=m.full_pixel(data,shared_piece_budget=True,material_only_inputs=True,smooth_partition=(4,4))
    assert len(calls)==1
    for a,b in zip(baseline['value'],localized['value']):
        assert max(a.lo,b.lo)<=min(a.hi,b.hi)
        assert (a-b).bound<I('1e-48').lo
    assert localized['smooth_partition']==(4,4)
    for kw in ({'smooth_partition':(4,4)}, {'shared_piece_budget':True,'smooth_partition':(3,4)}, {'shared_piece_budget':True,'smooth_partition':[4,4]}):
        with pytest.raises(ValueError):m.full_pixel(data,**kw)


def test_shared_stable_view_base_preserves_full_value_and_spatial_jets(monkeypatch):
    import tools.opaque_full_material_reference as m
    data=fixture();data['faces']=data['faces'][:1];data['selected']['visible_parts']=data['selected']['visible_parts'][:1]
    data['faces'][0]['normal']=[[1,0,-.25],[1,0,.75],[1,0,.75]]
    data['faces'][0]['view']=[[0,0,1]]*3
    charts=[]
    class Captured(Exception):pass
    def capture(field,**kw):charts.append(kw['chart_cover'][0]);raise Captured()
    monkeypatch.setattr(m,'integrate_square',capture)
    for stable in (False,True):
        with pytest.raises(Captured):m.full_pixel(data,shared_piece_budget=True,aggregate_residual=True,material_only_inputs=True,tight_discriminant=True,stable_view_base=stable)
    box=(I('.15','.30'),I('.3','.6'));indices=[];values=[]
    for c in charts:
        c['prepare'](0,0,box);e=c['correction_integrator'].extension
        assert e.bounds[0].lo>0 and e.bounds[0].hi>=4*e.bounds[0].lo
        indices.append(e.indices[0]);rows=[]
        for x,y in ((Q(1,5),Q(2,5)),(Q(1,4),Q(1,2))):
            for axis in (0,1):
                args=[Jet(x,order=3),Jet(y,order=3)];args[axis]=Jet.variable(args[axis][0,0],order=3)
                rows.append([a+b for a,b in zip(c['field'](*args),c['correction_integrator'].ordinary(*args))])
        values.append(rows)
    assert indices[0]>=0 and indices[1]==-1
    for aa,bb in zip(*values):
        for a,b in zip(aa,bb):
            for i in range(4):assert max(a[i,0].lo,b[i,0].lo)<=min(a[i,0].hi,b[i,0].hi)
    for kw in ({'stable_view_base':True},{'stable_view_base':1,'shared_piece_budget':True,'aggregate_residual':True},{'stable_view_base':True,'shared_piece_budget':True}):
        with pytest.raises(ValueError):m.full_pixel(data,**kw)


@pytest.mark.parametrize('option',['tight_light_slope','tight_cosine_slopes','slope_partition'])
def test_shared_tight_light_slope_retains_complete_spatial_coefficients(monkeypatch,option):
    import tools.opaque_full_material_reference as m
    data=fixture();data['faces']=data['faces'][:1];data['selected']['visible_parts']=data['selected']['visible_parts'][:1]
    data['faces'][0]['normal']=[[1,0,-.25],[1,0,.75],[1,0,.75]];data['faces'][0]['view']=[[0,0,1]]*3
    charts=[]
    class Captured(Exception):pass
    def capture(field,**kw):charts.append(kw['chart_cover'][0]);raise Captured()
    monkeypatch.setattr(m,'integrate_square',capture)
    for tight in (False,True):
        selected={option:tight} if option!='slope_partition' else dict(slope_partition=(2,4) if tight else None,tight_cosine_slopes=True)
        with pytest.raises(Captured):m.full_pixel(data,shared_piece_budget=True,tight_discriminant=True,material_only_inputs=True,**selected)
    for axis in (0,1):
        args=[Jet(I('.2','.4'),order=3),Jet(I('.3','.6'),order=3)];args[axis]=Jet.variable(args[axis][0,0],order=3)
        for role_index in (0,1):
            a,b=[c['correction_integrator'].inputs(*args)[role_index] for c in charts]
            assert a[1,0].lo<=b[1,0].lo<=b[1,0].hi<=a[1,0].hi
            for k in (0,2,3):assert (a[k,0].lo,a[k,0].hi)==(b[k,0].lo,b[k,0].hi)
            role=charts[1]['correction_integrator'].role_inputs(role_index,*args)
            assert (role[1,0].lo,role[1,0].hi)==(b[1,0].lo,b[1,0].hi)
    for kw in ({option:True},{option:1,'shared_piece_budget':True}):
        with pytest.raises(ValueError):m.full_pixel(data,**kw)
