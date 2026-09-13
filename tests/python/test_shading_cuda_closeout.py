"""Coordinator controls separate from actual CUDA numerical qualification."""
import pytest


def test_cuda_closeout_stops_after_first_failed_stage(monkeypatch,tmp_path):
    from tools import shading_cuda_closeout as driver
    calls=[]
    def failure(*args,**kwargs):
        calls.append(args)
        return 1
    monkeypatch.setattr(driver,'stage',failure)
    with pytest.raises(RuntimeError,match='no retry'):
        driver.run(tmp_path,tmp_path/'new',tmp_path/'wheel.whl','0'*64)
    assert len(calls)==1


def test_cuda_closeout_rejects_changed_wheel_before_device_execution(tmp_path):
    from tools import shading_cuda_closeout as driver
    path=tmp_path/'bad.whl';path.write_bytes(b'not a wheel')
    with pytest.raises(ValueError,match='identity'):
        driver.identity(path,'0'*64)


def test_cuda_closeout_restores_cpu_memory_policy_after_failure(monkeypatch,tmp_path):
    from tools import shading_cuda_closeout as driver
    original=driver.cost.PROFILE
    seen=[]
    def fail(*args,**kwargs):
        seen.append(driver.cost.PROFILE['sampled_working_set_limit_bytes'])
        raise RuntimeError('worker launch failed')
    monkeypatch.setattr(driver.cost,'supervise',fail)
    with pytest.raises(RuntimeError):
        driver.stage(tmp_path,tmp_path/'new.json',tmp_path/'wheel.whl','0'*64,'chains')
    assert seen==[2*1024**3] and driver.cost.PROFILE is original
