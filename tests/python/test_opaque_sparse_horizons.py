"""Exact forward horizon coefficients on selected faces; dense gradient path retained."""
import os,torch,pytest
from test_opaque_full_original import original


@pytest.mark.parametrize('batch',[1,2])
def test_forward_horizons_visit_only_selected_faces_and_match_dense(monkeypatch,batch):
    from gffx.torch import _opaque_scene as scene,_reference_channels as channels
    s,c,_=original(os.environ.get('GFFX_KNOT_DEVICE','cpu'))
    if batch==2:
        from dataclasses import replace
        from gffx.torch.scene import Scene
        m=s.prims[0];s=Scene([replace(m,vertices=torch.stack((m.vertices,m.vertices+m.vertices.new_tensor([.001,0,0])))),*s.prims[1:]])
    packed,c,pixels,keep,clip=scene._project(s,c,2);normal=scene._normals(packed);corners=pixels[packed.faces.long(),:2]
    selected=keep&(corners[...,0].amax(-1)>=48)&(corners[...,0].amin(-1)<=49)&(corners[...,1].amax(-1)>=15)&(corners[...,1].amin(-1)<=16)
    count=int(selected.sum());assert 0<count<32*batch
    with torch.enable_grad():expected=scene._horizon_cuts(packed,c,clip,normal,selected)
    sizes=[];base=channels.sum3
    def observed(x):sizes.append(x.shape[0]);return base(x)
    monkeypatch.setattr(channels,'sum3',observed)
    with torch.no_grad():actual=scene._horizon_cuts(packed,c,clip,normal,selected)
    assert sizes and max(sizes)<=count,(count,max(sizes))
    for a,b in zip(actual,expected):
        if b is None:assert a is None
        else:assert a.cpu().numpy().tobytes()==b.detach().cpu().numpy().tobytes()


def test_extreme_finite_inputs_retain_dense_horizon_validation(monkeypatch):
    from gffx.torch import _opaque_scene as scene,_reference_channels as channels
    s,c,_=original(os.environ.get('GFFX_KNOT_DEVICE','cpu'));packed,c,_,keep,clip=scene._project(s,c,2);normal=scene._normals(packed)*1e51
    selected=keep.clone();selected[1:]=False;selected[0]=True
    sizes=[];base=channels.sum3
    def observed(x):sizes.append(x.shape[0]);return base(x)
    monkeypatch.setattr(channels,'sum3',observed)
    with torch.no_grad():scene._horizon_cuts(packed,c,clip,normal,selected)
    assert max(sizes)==len(packed.faces)
