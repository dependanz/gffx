"""One public forward window may reuse static projection, never its backward graph."""
import os,json,hashlib,weakref,gc,torch,pytest
from pathlib import Path
from test_opaque_full_original import original


def test_window_prepares_static_geometry_once_and_releases_it(monkeypatch):
    from gffx.torch import render,_opaque_scene as scene
    s,c,_=original(os.environ.get('GFFX_KNOT_DEVICE','cpu'));s.prims[1].parameters['albedo'].requires_grad_()
    calls=[];refs=[];base=scene._project
    def counted(*args,**kw):
        result=base(*args,**kw);calls.append(True);refs.append(weakref.ref(result[2]));return result
    monkeypatch.setattr(scene,'_project',counted)
    out=render.opaque_window(s,c,(10,48,1,16),cull_mode=2)
    assert len(calls)==1,len(calls)
    gc.collect();assert all(ref() is None for ref in refs),'forward geometry retained on autograd context'
    B=Path('D:/devspace/gffx/qualification-20260913/m1-continuation-20260913/opaque177-20261002')
    saved=json.loads((B/('original-'+c.fx.device.type+'1.json')).read_bytes())['cases'][0]
    assert {k:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for k,v in out.items()}==saved['hashes']


def test_window_reuse_refuses_mutation_before_next_tile(monkeypatch):
    from gffx.torch import render,_opaque_stream as stream
    s,c,_=original(os.environ.get('GFFX_KNOT_DEVICE','cpu'));a=s.prims[1].parameters['albedo'];base=stream._evaluate;calls=[]
    def changed(*args,**kw):
        result=base(*args,**kw)
        if not calls:
            with torch.no_grad():a.add_(.001)
        calls.append(True);return result
    monkeypatch.setattr(stream,'_evaluate',changed)
    with pytest.raises(RuntimeError,match='modified during forward window'):
        render.opaque_window(s,c,(10,48,1,16),cull_mode=2,tile_pixels=1)
    assert len(calls)==1
