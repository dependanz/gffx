"""Connect the qualified native sample VJP only to explicit fixed-albedo replay."""
import os,torch,pytest
from test_opaque_full_original import original
from gffx.torch import render


def inputs():
 s,c,_=original(os.environ.get('GFFX_KNOT_DEVICE','cpu'));a=s.prims[1].parameters['albedo']
 return s,c,a.new_ones((1,1,4,3)),a.new_zeros((1,1,4))


def test_replay_uses_native_sample_vjp_for_ordinary_pixels(monkeypatch):
 from gffx.torch import _opaque_scene as scene
 s,c,g,gm=inputs();calls=[];old=torch.ops.gffx.shading_opaque_albedo
 def observed(*a):calls.append(a[0].shape[0]);return old(*a)
 def forbidden(*a,**kw):raise AssertionError('eager sample composition still used')
 monkeypatch.setattr(scene,'_sample_integral_eager',forbidden);monkeypatch.setattr(torch.ops.gffx,'shading_opaque_albedo',observed)
 out=render.opaque_albedo_replay(s,c,(64,62,1,4),g,gm)
 assert calls==[4] and torch.isfinite(out['albedo_gradient']).all()


def test_native_replay_failure_is_not_hidden(monkeypatch):
 s,c,g,gm=inputs()
 def failure(*args):raise RuntimeError('injected sample VJP failure')
 monkeypatch.setattr(torch.ops.gffx,'shading_opaque_albedo',failure)
 with pytest.raises(RuntimeError,match='injected'):
  render.opaque_albedo_replay(s,c,(64,62,1,4),g,gm)


def test_general_window_differentiation_does_not_select_explicit_native_vjp(monkeypatch):
 s,c,g,gm=inputs();a=s.prims[1].parameters['albedo'];a.requires_grad_()
 def forbidden(*args):raise AssertionError('explicit native VJP in general autograd')
 monkeypatch.setattr(torch.ops.gffx,'shading_opaque_albedo',forbidden)
 out=render.opaque_window(s,c,(64,62,1,1));grad=torch.autograd.grad(out['colour'].sum(),a)[0]
 assert torch.isfinite(grad).all()
