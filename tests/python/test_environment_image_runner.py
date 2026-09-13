"""Image loss assembly must use the global denominator, not equal tile weights."""
import importlib
import pytest
import torch


def module(): return importlib.import_module('tools.environment_image_runner')


def test_image_adjoint_assembly_weights_unequal_tiles_by_pixel_count():
    one=torch.ones((32,32,3),dtype=torch.float64)
    parts=[dict(start=0,end=2,values=[[1.,2.,3.]]*2,gradient=one.tolist(),loss=2.),
           dict(start=2,end=3,values=[[4.,5.,6.]],gradient=(4*one).tolist(),loss=5.)]
    out=module().combine(parts,3)
    assert out['loss']==pytest.approx(3.)
    torch.testing.assert_close(out['gradient'],2*one)
    assert out['values'].tolist()==[[1.,2.,3.]]*2+[[4.,5.,6.]]


@pytest.mark.parametrize('bad',['gap','overlap','missing','nonfinite'])
def test_image_adjoint_assembly_rejects_partial_or_invalid_evidence(bad):
    part=dict(start=0,end=2,values=[[1.,2.,3.]]*2,gradient=torch.zeros((32,32,3)).tolist(),loss=1.)
    if bad=='gap': part['start']=1
    elif bad=='overlap': part['end']=3
    elif bad=='nonfinite': part['loss']=float('nan')
    parts=[] if bad=='missing' else [part]
    with pytest.raises(ValueError): module().combine(parts,2)


def test_image_adjoint_worker_rejects_wrong_schedule_hash_before_loading(tmp_path):
    path=tmp_path/'schedule.json';path.write_text('not json')
    with pytest.raises(ValueError,match='hash'):module().read_schedule(path,'0'*64)


def test_image_adjoint_next_worker_reserves_watchdog_time():
    module().launch_guard(0.,6000.)
    with pytest.raises(TimeoutError): module().launch_guard(5821.,6000.)
