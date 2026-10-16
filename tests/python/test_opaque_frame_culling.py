"""Frame visibility identity: installed pixels and gradients, not scale admission."""
import copy,importlib.util,os
from pathlib import Path
import pytest,torch


def fixture():
    p=Path(__file__).with_name('test_opaque_window.py')
    spec=importlib.util.spec_from_file_location('frame_cull_fixture',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m.fixture(1,os.environ.get('GFFX_KNOT_DEVICE','cpu'))


@pytest.mark.parametrize('mode',[1,2,3])
def test_explicit_frame_culling_matches_installed_pixels_and_vjp(tmp_path,monkeypatch,mode):
    from tools.opaque_frame_execution import execute,replay
    from gffx.torch import render
    s,c,v=fixture();seen=[];original=render.opaque_window
    def traced(*args,**kwargs):
        seen.append(kwargs['cull_mode']);return original(*args,**kwargs)
    monkeypatch.setattr(render,'opaque_window',traced)
    receipt=execute(s,c,tmp_path/'frame',identity={'case':'culling'},cull_mode=mode)
    assert receipt['identity']['cull_mode']==mode
    gradients=replay(s,c,receipt)
    assert seen==[mode,mode]
    expected=original(s,c,(0,0,1,1),cull_mode=mode)
    for key in ('colour','mask'):
        assert (tmp_path/'frame'/(key+'.f64')).read_bytes()==expected[key].detach().cpu().numpy().tobytes()
    loss=(expected['colour']*v.new_tensor([.2,-.3,.7])).sum()+expected['mask'].sum()*.1
    g=torch.autograd.grad(loss,v,allow_unused=True)[0] if loss.requires_grad else None
    if g is None:g=torch.zeros_like(v)
    assert gradients[0].cpu().numpy().tobytes()==g.cpu().numpy().tobytes()
    altered=copy.deepcopy(receipt);altered['identity']['cull_mode']=mode%3+1
    with pytest.raises(ValueError,match='identity'):replay(s,c,altered)
    assert seen==[mode,mode]


def test_frame_culling_invalid_modes_refused_before_storage(tmp_path):
    from tools.opaque_frame_execution import execute
    s,c,_=fixture()
    for mode in (True,0,4,2.0,'back'):
        with pytest.raises(ValueError,match='cull'):execute(s,c,tmp_path/'invalid',identity={},cull_mode=mode)
        assert not (tmp_path/'invalid').exists()


def test_default_frame_culling_preserves_legacy_receipt(tmp_path):
    from tools.opaque_frame_execution import execute,replay
    s,c,_=fixture();receipt=execute(s,c,tmp_path/'legacy',identity={})
    assert receipt['identity']['profile']=='public-window-pixel-replay-v1'
    assert 'cull_mode' not in receipt['identity'];replay(s,c,receipt)
