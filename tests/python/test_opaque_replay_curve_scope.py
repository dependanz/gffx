"""Prepared material curves belong only to one explicit guarded replay call."""
import os,torch,pytest
from test_opaque_full_original import original
from gffx.torch import render
from gffx.torch import _principled_groups as groups


def inputs():
    s,c,_=original(os.environ.get('GFFX_KNOT_DEVICE','cpu'));a=s.prims[1].parameters['albedo']
    return s,c,a.new_ones((1,1,4,3)),a.new_zeros((1,1,4))


def test_replay_prepares_one_curve_per_call_for_forward_and_backward(monkeypatch):
    s,c,g,gm=inputs();calls=[];old=groups._prepare_material_curve
    def traced(*args,**kw):calls.append(1);return old(*args,**kw)
    monkeypatch.setattr(groups,'_prepare_material_curve',traced)
    for _ in range(2):render.opaque_albedo_replay(s,c,(64,62,1,4),g,gm)
    assert len(calls)==2


def test_replay_curve_scope_is_cleared_after_exception(monkeypatch):
    from gffx.torch import _opaque_stream as stream
    s,c,g,gm=inputs();original_evaluate=stream._evaluate
    def fail(*args,**kw):raise RuntimeError('injected replay error')
    monkeypatch.setattr(stream,'_evaluate',fail)
    with pytest.raises(RuntimeError,match='injected replay error'):render.opaque_albedo_replay(s,c,(64,62,1,4),g,gm)
    assert groups._active_albedo_curve.get() is None
    monkeypatch.setattr(stream,'_evaluate',original_evaluate)
    out=render.opaque_albedo_replay(s,c,(64,62,1,4),g,gm)
    assert torch.isfinite(out['albedo_gradient']).all() and groups._active_albedo_curve.get() is None


def test_replay_curve_reuse_refuses_roughness_mutation(monkeypatch):
    s,c,g,gm=inputs();old=groups._prepare_material_curve;calls=[]
    def mutate(*args,**kw):
        result=old(*args,**kw)
        if not calls:
            with torch.no_grad():s.prims[1].parameters['roughness'].sub_(.0001)
        calls.append(1);return result
    monkeypatch.setattr(groups,'_prepare_material_curve',mutate)
    with pytest.raises(RuntimeError,match='modified'):render.opaque_albedo_replay(s,c,(64,62,1,4),g,gm)
    assert groups._active_albedo_curve.get() is None


def test_curve_scope_refuses_different_storage_and_general_gradients():
    s,c,g,gm=inputs();p=s.prims[1].parameters;a=p['albedo'];r=p['roughness'];b=p['budget']
    keys=('albedo','f0','roughness');n=a.new_tensor([[0.,0.,1.]]);values=(n,n,n[:,None],a,p['f0'],r)
    with groups._fixed_albedo_curve(a,r,b):
        assert groups._call_curve_state(keys,{'budget':b},values) is not None
        detached=tuple(x.detach() for x in values)
        assert groups._call_curve_state(keys,{'budget':b},detached) is not None
        other=(*values[:3],a.clone(),*values[4:])
        assert groups._call_curve_state(keys,{'budget':b},other) is None
        changing=(n.clone().requires_grad_(),*values[1:])
        assert groups._call_curve_state(keys,{'budget':b},changing) is None
    assert groups._active_albedo_curve.get() is None


@pytest.mark.parametrize('enabled',[False,True])
def test_replay_restores_callers_autograd_thread_setting(enabled,monkeypatch):
    from gffx.torch import _opaque_stream as stream
    import threading
    s,c,g,gm=inputs();caller=threading.get_ident();threads=[];old=groups._prepare_material_curve
    def traced(*args,**kw):threads.append(threading.get_ident());return old(*args,**kw)
    monkeypatch.setattr(groups,'_prepare_material_curve',traced)
    with torch.autograd.set_multithreading_enabled(enabled):
        render.opaque_albedo_replay(s,c,(64,62,1,4),g,gm)
        assert torch.autograd.is_multithreading_enabled() is enabled
        assert threads==[caller]
        def fail(*args,**kw):raise RuntimeError('injected thread scope failure')
        monkeypatch.setattr(stream,'_evaluate',fail)
        with pytest.raises(RuntimeError,match='thread scope failure'):render.opaque_albedo_replay(s,c,(64,62,1,4),g,gm)
        assert torch.autograd.is_multithreading_enabled() is enabled
    assert groups._active_albedo_curve.get() is None


def test_replay_curve_tensors_are_released_at_public_return(monkeypatch):
    import weakref,gc
    s,c,g,gm=inputs();refs=[];old=groups._prepare_material_curve
    def traced(*args,**kw):
        result=old(*args,**kw);refs.extend(weakref.ref(t) for t in result[1:]);return result
    monkeypatch.setattr(groups,'_prepare_material_curve',traced)
    out=render.opaque_albedo_replay(s,c,(64,62,1,4),g,gm);gc.collect()
    assert refs and all(r() is None for r in refs)
    assert all(not value.requires_grad for value in out.values())
