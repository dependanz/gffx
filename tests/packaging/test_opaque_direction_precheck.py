"""A work-selection precheck cannot certify or discard an integration bound."""
from types import SimpleNamespace
import pytest
from tools.opaque_shading_intervals import I,Jet,Unresolved

@pytest.mark.parametrize('kind,expected',[('positive',(1,0)),('negative',(1,0)),('turn',(0,)),('constant',(1,0)),('no_knot',(1,0)),('second_role_turn',(0,))])
def test_direction_precheck_uses_both_roles_and_only_relevant_knots(kind,expected):
    from tools.opaque_direction_precheck import DirectionBoundary
    def inputs(x,y):
        q=y*y+y.lift('.5') if kind in ('turn','second_role_turn') else y.lift('.5') if kind=='constant' else y+y.lift('.5') if kind=='positive' else -y+y.lift('.5') if kind=='negative' else y*y+y.lift(2)
        return [y.lift('.75'),q] if kind=='second_role_turn' else [q,y.lift('.75')]
    c=SimpleNamespace(inputs=inputs,checkpoint=lambda:None);p=object.__new__(DirectionBoundary);p.c=c;p.proof=SimpleNamespace(index=SimpleNamespace(upper=[I('.5').hi,I(1).hi],lower=[I('.5').lo,I(1).lo]));p.prechecks=[]
    assert p.candidate_directions((I(0),I('-.1','.1')))==expected

@pytest.mark.parametrize('kind',['cancel','unknown'])
def test_direction_precheck_propagates_cancel_and_unknown_geometry(kind):
    from tools.opaque_direction_precheck import DirectionBoundary
    def inputs(*a):raise ValueError('unknown geometry')
    def check():
        if kind=='cancel':raise Unresolved('cancel')
    p=object.__new__(DirectionBoundary);p.c=SimpleNamespace(inputs=inputs,checkpoint=check);p.prechecks=[]
    with pytest.raises(Unresolved if kind=='cancel' else ValueError):p.candidate_directions((I(0),I(0,1)))

@pytest.mark.parametrize('permit',[False,True])
@pytest.mark.parametrize('outcome',['pass','refusal','unknown','cancel','disjoint'])
def test_direction_selection_keeps_complete_intersections(monkeypatch,permit,outcome):
    from tools.opaque_direction_precheck import DirectionBoundary
    from tools import opaque_value_tail as v
    c=SimpleNamespace(checkpoint=lambda:None,preferred_split_axis=0,envelope_integral=lambda b:[I(0),I(-1,1)]);calls=[]
    class Proof:
        def __init__(self,*a,**k):pass
        def __call__(self,b,*,integrated_axis):
            calls.append(integrated_axis)
            if outcome=='cancel':raise Unresolved('stop')
            if outcome=='unknown':raise ValueError('unknown proof')
            if outcome=='disjoint':return [I(0),I(2,3)]
            if outcome=='refusal' and integrated_axis==1:raise ValueError('aggregate angular crossings need certified monotonicity')
            return [I(0),I('-.1','.1')]
    monkeypatch.setattr(v,'ValueCenteredBoundary',Proof);monkeypatch.setattr(DirectionBoundary,'candidate_directions',lambda self,b:(1,0) if permit else (0,));p=DirectionBoundary(c,tolerance='1e-14',piece=11)
    if outcome in ('cancel','unknown','disjoint'):
        with pytest.raises(Unresolved if outcome=='cancel' else ValueError):p((I('.5','.51'),I('.5','.6')))
    else:
        q=p((I('.5','.51'),I('.5','.6')));assert q[1].width==I('.2').lo
        assert calls==([1,0] if permit and outcome=='refusal' else [1] if permit else [0])

@pytest.mark.parametrize('sx,sy',[(1,1),(-1,1),(1,-1),(-1,-1)])
def test_prechecked_boundary_keeps_independent_closed_integral(monkeypatch,sx,sy):
    from tools.opaque_direction_precheck import DirectionBoundary
    from tools import opaque_boundary_material as boundary
    from tests.packaging import test_opaque_boundary_material as f
    from tools.opaque_decimal_convolution import decimal_convolution
    monkeypatch.setattr(DirectionBoundary,'should_refine',lambda self,b,old:True)
    class Adapter(DirectionBoundary):
        def __init__(self,c,**kw):super().__init__(c,tolerance='1e-14',piece=11)
    monkeypatch.setattr(boundary,'BoundaryMaterial',Adapter)
    with decimal_convolution():f.test_transport_keeps_independent_log_squared_rectangle(0,sx,sy,2)
@pytest.mark.parametrize('name',['standard','original13-offline-extended-v1'])
def test_centered_tail_requires_explicit_profile(name):
    from tools.opaque_reference_profile import worker_profile,EXTENDED
    if name=='standard':
        with pytest.raises(ValueError):worker_profile('offline-direction-tail',name)
    else:assert worker_profile('offline-direction-tail',name)==EXTENDED

@pytest.mark.parametrize('index',[11])
def test_centered_tail_preflight_routes_both_pieces(monkeypatch,tmp_path,index):
    import os,json,hashlib
    from pathlib import Path
    from tools import opaque_full_material_reference as m
    from tools.opaque_reference_profile import EXTENDED
    from tests.packaging import test_opaque_direction_preflight as worker
    from tools.opaque_direction_precheck import DirectionBoundary
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    f=tmp_path/'manifest.json';f.write_text(json.dumps(dict(source=[dict(path='tools/opaque_direction_precheck.py',sha256=sha(Path('tools/opaque_direction_precheck.py')))],reference_profile=EXTENDED.record(),piece=index)))
    dest=tmp_path/'mock.json';monkeypatch.setenv('GFFX_EXPERIMENT_MANIFEST',str(f));monkeypatch.setenv('GFFX_EXPERIMENT_MANIFEST_SHA',sha(f));monkeypatch.setenv('GFFX_OFFLINE_PIECE',str(index));monkeypatch.setenv('GFFX_FULL_REFERENCE',str(dest));seen={}
    def square(field,**kw):seen['square']=kw
    def full(data,**kw):
        seen['full']=kw;m.integrate_square(None,tolerance='1e-14',correction_integrator=SimpleNamespace(curved=True,envelope_fallback=True,aggregate_fallback=True));return {'mock_routing_only':True}
    monkeypatch.setattr(m,'integrate_square',square);monkeypatch.setattr(m,'full_pixel',full)
    worker.test_original_direction_tail_preflight(monkeypatch,lambda *a:None)
    assert seen['full']==dict(selected_piece=index,material_only_inputs=True,reference_profile=EXTENDED.name)
    assert seen['square']['seed_shape']==(64,8) and isinstance(seen['square']['correction_integrator'],DirectionBoundary)
    assert seen['square']['correction_integrator'].directions==(0,)
    assert json.loads(dest.read_bytes())['seed_shape']==[64,8]
