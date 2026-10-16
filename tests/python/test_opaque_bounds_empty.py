"""Forward-only face bounds and bitwise-zero integration preserve the old result."""
import os,torch,pytest
from types import SimpleNamespace
from test_opaque_full_original import original


def test_forward_window_face_bounds_once(monkeypatch):
    from gffx.torch import render,_opaque_scene as scene
    s,c,_=original(os.environ.get('GFFX_KNOT_DEVICE','cpu'))
    calls=[];base=scene._face_bounds
    def counted(*a):calls.append(1);return base(*a)
    monkeypatch.setattr(scene,'_face_bounds',counted)
    result=render.opaque_window(s,c,(0,0,1,32),cull_mode=2,tile_pixels=1)
    assert len(calls)==1
    for v in result.values():assert v.detach().cpu().numpy().tobytes()==bytes(v.numel()*8)


@pytest.mark.parametrize('kind',['positive_zero','negative_zero','tiny','nan'])
def test_empty_sample_reduction_guard(monkeypatch,kind):
    from gffx.torch import _opaque_scene as scene,_curve_math as math
    device=os.environ.get('GFFX_KNOT_DEVICE','cpu')
    q=torch.zeros((1,1,800,4),dtype=torch.float64,device=device)
    if kind=='negative_zero':q[...,3]=-0.
    if kind=='tiny':q[0,0,0,3]=float.fromhex('0x0.0000000000001p-1022')
    if kind=='nan':q[0,0,0,3]=float('nan')
    owners=torch.full((1,1,800),-1,dtype=torch.int32,device=device)
    packed=SimpleNamespace(vertices=q.new_empty((0,3)),faces=owners.new_empty((0,3)))
    camera=SimpleNamespace(batch_size=1);calls=[];base=math.sum_last
    expected_alpha=base(q[...,3]);expected_rgb=base((q.new_zeros((1,1,800,3))*q[...,3,None]).transpose(-1,-2))
    def counted(v):calls.append(1);return base(v)
    monkeypatch.setattr(math,'sum_last',counted)
    with torch.no_grad():rgb,alpha=scene._sample_integral(packed,camera,None,None,owners,q,{})
    assert len(calls)==(0 if kind=='positive_zero' else 2)
    for actual,expected in ((rgb,expected_rgb),(alpha,expected_alpha)):
        assert actual.cpu().numpy().tobytes()==expected.cpu().numpy().tobytes()
