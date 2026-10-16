"""Bounded tail scheduling preserves complete enclosures and fail-closed errors."""
from types import SimpleNamespace
import pytest
from tools.opaque_shading_intervals import I,Unresolved

@pytest.mark.parametrize('piece,expected',[(11,[1,0]),(12,[0])])
@pytest.mark.parametrize('kind',['tight','broad','pass','fallback','all_refuse','unknown','cancel','disjoint','channels'])
def test_tail_direction_selection_and_complete_evidence(monkeypatch,piece,expected,kind):
    from tools import opaque_tail_boundary as m
    calls=[];old=[I(0),I(-1,1)];c=SimpleNamespace(checkpoint=lambda:None,preferred_split_axis=0,envelope_integral=lambda b:old)
    box=(I('.3','.3125'),I('.2','.25'))
    if kind=='tight':old[1]=I(0)
    if kind=='broad':box=(I(0,1),I(0,1))
    class Proof:
        def __init__(self,*a,**k):pass
        def __call__(self,b,*,integrated_axis):
            calls.append(integrated_axis);c.preferred_split_axis=1
            if kind=='cancel':raise Unresolved('cancel')
            if kind=='unknown':raise ValueError('unknown fault')
            if kind=='all_refuse' or kind=='fallback' and len(calls)==1:raise ValueError('aggregate angular crossings need certified monotonicity')
            if kind=='disjoint':return [I(0),I(2,3)]
            if kind=='channels':return [I(0)]
            return [I(0),I('-.25','.5')]
    monkeypatch.setattr(m,'CenteredBoundaryMaterial',Proof);p=m.TailBoundary(c,tolerance='1e-14',piece=piece)
    if kind in ('unknown','cancel','disjoint','channels'):
        with pytest.raises(Unresolved if kind=='cancel' else ValueError):p(box)
        assert calls==expected[:1];return
    got=p(box)
    if kind in ('tight','broad'):assert not calls and got==old
    elif kind=='all_refuse' or kind=='fallback' and piece==12:
        assert calls==expected and got==old and c.preferred_split_axis==0
    else:
        assert calls==(expected if kind=='fallback' else expected[:1]);assert got[1].lo==I('-.25').lo and got[1].hi==I('.5').hi

@pytest.mark.parametrize('piece',[0,10,13,'11',True])
def test_tail_refuses_unqualified_piece(piece):
    from tools.opaque_tail_boundary import TailBoundary
    with pytest.raises(ValueError):TailBoundary(None,tolerance='1e-14',piece=piece)

@pytest.mark.parametrize('piece',[11,12])
@pytest.mark.parametrize('sx,sy',[(1,1),(-1,1),(1,-1),(-1,-1)])
def test_tail_keeps_independent_closed_integral(monkeypatch,piece,sx,sy):
    from tools.opaque_tail_boundary import TailBoundary
    from tools import opaque_boundary_material as boundary
    from tests.packaging import test_opaque_boundary_material as f
    from tools.opaque_decimal_convolution import decimal_convolution
    monkeypatch.setattr(TailBoundary,'should_refine',lambda self,b,old:True)
    class Adapter(TailBoundary):
        def __init__(self,c,**kw):super().__init__(c,tolerance='1e-14',piece=piece)
    monkeypatch.setattr(boundary,'BoundaryMaterial',Adapter)
    with decimal_convolution():f.test_transport_keeps_independent_log_squared_rectangle(1 if piece==11 else 0,sx,sy,2)

@pytest.mark.parametrize('name',['standard','original13-offline-extended-v1'])
def test_centered_tail_requires_explicit_profile(name):
    from tools.opaque_reference_profile import worker_profile,EXTENDED
    if name=='standard':
        with pytest.raises(ValueError):worker_profile('offline-centered-tail',name)
    else:assert worker_profile('offline-centered-tail',name)==EXTENDED

@pytest.mark.parametrize('index',[11,12])
def test_centered_tail_preflight_routes_both_pieces(monkeypatch,tmp_path,index):
    import os,json,hashlib
    from pathlib import Path
    from tools import opaque_full_material_reference as m
    from tools.opaque_reference_profile import EXTENDED
    from tests.packaging import test_opaque_centered_preflight as worker
    from tools.opaque_tail_boundary import TailBoundary
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    f=tmp_path/'manifest.json';f.write_text(json.dumps(dict(source=[dict(path='tools/opaque_tail_boundary.py',sha256=sha(Path('tools/opaque_tail_boundary.py')))],reference_profile=EXTENDED.record(),piece=index)))
    dest=tmp_path/'mock.json';monkeypatch.setenv('GFFX_EXPERIMENT_MANIFEST',str(f));monkeypatch.setenv('GFFX_EXPERIMENT_MANIFEST_SHA',sha(f));monkeypatch.setenv('GFFX_OFFLINE_PIECE',str(index));monkeypatch.setenv('GFFX_FULL_REFERENCE',str(dest));seen={}
    def square(field,**kw):seen['square']=kw
    def full(data,**kw):
        seen['full']=kw;m.integrate_square(None,tolerance='1e-14',correction_integrator=SimpleNamespace(curved=True,envelope_fallback=True,aggregate_fallback=True));return {'mock_routing_only':True}
    monkeypatch.setattr(m,'integrate_square',square);monkeypatch.setattr(m,'full_pixel',full)
    worker.test_original_centered_tail_preflight(monkeypatch,lambda *a:None)
    assert seen['full']==dict(selected_piece=index,material_only_inputs=True,reference_profile=EXTENDED.name)
    assert seen['square']['seed_shape']==(64,8) and isinstance(seen['square']['correction_integrator'],TailBoundary)
    assert seen['square']['correction_integrator'].directions==((1,0) if index==11 else (0,))
    assert json.loads(dest.read_bytes())['seed_shape']==[64,8]
