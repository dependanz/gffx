"""Named admission must wait for every gate and keep the CUDA exception scoped."""
import importlib
import json
import pytest


def module():return importlib.import_module('tools.cuda_bundle_probe')


def test_cuda_bundle_probe_restores_cpu_limits_after_interruption(monkeypatch,tmp_path):
    m=module();original=m.old.cost.PROFILE;seen=[]
    def stop(*args,**kwargs):
        seen.append(m.old.cost.PROFILE.copy());raise RuntimeError('interrupted')
    monkeypatch.setattr(m.old.cost,'supervise',stop)
    with pytest.raises(RuntimeError):m.stage(tmp_path,tmp_path/'new.json',tmp_path/'x.whl','a'*64)
    assert m.old.cost.PROFILE is original
    assert seen==[dict(original,sampled_working_set_limit_bytes=2*1024**3)]


def test_cuda_bundle_probe_does_not_admit_after_parent_failure(monkeypatch,tmp_path):
    m=module();calls=[]
    def failed(*args,**kwargs):raise RuntimeError('parent rejected')
    monkeypatch.setattr(m.old,'run',failed)
    monkeypatch.setattr(m,'stage',lambda *args: calls.append(args))
    with pytest.raises(RuntimeError):m.run(tmp_path,tmp_path/'new',tmp_path/'x.whl','a'*64)
    assert not calls
    assert json.loads((tmp_path/'new'/'admission.json').read_text())['status']=='stopped'
