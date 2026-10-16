"""Conservative closed-window face selection preserves all candidate contributions."""
import os
import pytest
import torch
from test_opaque_full_original import original


def test_forward_native_quadrature_gets_only_window_faces_and_dense_backward(monkeypatch):
    from gffx.torch import render
    dev=os.environ.get('GFFX_KNOT_DEVICE','cpu');s,c,_=original(dev);a=s.prims[1].parameters['albedo'];a.requires_grad_();calls=[]
    for name in ['opaque_material_quadrature','opaque_material_quadrature_batch']:
        base=getattr(render,name)
        def observed(v,f,*args,_base=base,**kw):
            calls.append((torch.is_grad_enabled(),len(f)));return _base(v,f,*args,**kw)
        monkeypatch.setattr(render,name,observed)
    out=render.opaque_window(s,c,(15,48,1,1),cull_mode=2)
    assert calls and all(n<=32 for grad,n in calls if not grad),calls
    torch.autograd.grad(out['colour'].sum(),a)
    assert any(grad and n>32 for grad,n in calls),'dense gradient preparation changed'


def test_unselected_material_cut_nonfinite_still_refuses(monkeypatch):
    from gffx.torch import render,_opaque_scene as scene
    s,c,_=original(os.environ.get('GFFX_KNOT_DEVICE','cpu'));packed,_,pixels,keep,_=scene._project(s,c,2);q=pixels[packed.faces.long(),:2]
    outside=keep & ((q[...,0].amax(-1)<48)|(q[...,0].amin(-1)>49)|(q[...,1].amax(-1)<15)|(q[...,1].amin(-1)>16))
    index=int(torch.nonzero(outside)[0]);old=scene._cuts
    def corrupt(packed,clip):
        cuts=old(packed,clip);extra=clip.new_zeros((len(packed.faces),3,1));extra[index]=float('nan');return torch.cat((cuts,extra),-1)
    monkeypatch.setattr(scene,'_cuts',corrupt)
    with pytest.raises(ValueError,match='finite'):render.opaque_window(s,c,(15,48,1,1),cull_mode=2)


def test_unreferenced_out_of_range_projected_vertex_still_refuses(monkeypatch):
    from gffx.torch import render,_opaque_scene as scene
    s,c,_=original(os.environ.get('GFFX_KNOT_DEVICE','cpu'));old=scene._project
    def corrupt(*args,**kw):
        packed,camera,pixels,keep,clip=old(*args,**kw);pixels=pixels.clone();pixels[0,0]=1048577.;return packed,camera,pixels,keep,clip
    monkeypatch.setattr(scene,'_project',corrupt)
    with pytest.raises(ValueError,match='bounded|range'):render.opaque_window(s,c,(15,48,1,1),cull_mode=2)
