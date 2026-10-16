"""Forward-only prepared geometry within one bounded overflowing pixel."""
import os,json,hashlib,torch,pytest
from pathlib import Path
from test_opaque_full_original import original


def test_pixel_prepares_geometry_once_without_retaining_it_for_backward(monkeypatch):
    from gffx.torch import render,_opaque_pixel as pixel,_opaque_scene as scene
    dev=os.environ.get('GFFX_KNOT_DEVICE','cpu');s,c,_=original(dev);a=s.prims[1].parameters['albedo'];a.requires_grad_()
    calls=[];count=[0];base_project=scene._project;base_evaluate=pixel._evaluate
    def project(*args,**kw):count[0]+=1;return base_project(*args,**kw)
    def evaluate(*args,**kw):
        before=count[0];result=base_evaluate(*args,**kw);calls.append((torch.is_grad_enabled(),count[0]-before));return result
    monkeypatch.setattr(scene,'_project',project);monkeypatch.setattr(pixel,'_evaluate',evaluate)
    out=render.opaque_window(s,c,(15,48,1,1),cull_mode=2)
    forward=[n for grad,n in calls if not grad];assert len(forward)>1 and forward[0] in (0,1) and forward[1:]==[0]*(len(forward)-1),forward
    g,=torch.autograd.grad(out['colour'].sum(),a)
    # The outer replay performs one new forward; every differentiated batch still prepares its own geometry.
    differentiated=[n for grad,n in calls if grad];assert differentiated and all(n==1 for n in differentiated)
    B=Path('D:/devspace/gffx/qualification-20260913/m1-continuation-20260913/opaque178-20261002')
    old=json.loads((B/('pixels-'+dev+'1.json')).read_bytes())['cases'][1]
    bits={k:v.detach().cpu().contiguous().numpy().tobytes().hex() for k,v in {**out,'albedo_sum_gradient':g}.items()}
    assert bits==old['bytes']


def test_forward_reuse_refuses_input_mutation_between_batches(monkeypatch):
    from test_opaque_pixel_batches import crowded,call
    from gffx.torch import _opaque_pixel as pixel
    s,c,a=crowded(os.environ.get('GFFX_KNOT_DEVICE','cpu'));original_evaluate=pixel._evaluate;calls=[]
    def changed(*args,**kw):
        result=original_evaluate(*args,**kw)
        if not calls:
            with torch.no_grad():a.add_(.001)
        calls.append(True);return result
    monkeypatch.setattr(pixel,'_evaluate',changed)
    with pytest.raises(RuntimeError,match='modified during forward'):call(s,c)
    assert len(calls)==1
