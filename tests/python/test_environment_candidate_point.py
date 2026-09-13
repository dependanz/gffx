"""Actual public material sampling, not a substituted analytic candidate."""
import importlib
import math
import time
import pytest
import torch


def tool():return importlib.import_module('tools.environment_candidate_point')


def test_public_environment_sampling_calls_full_material_and_integrator(monkeypatch):
    from gffx.torch import material,light
    calls=[];integrations=[];original=material.classic;integrate=light.integrate
    def observed(n,v,l,**kwargs):
        calls.append(len(l));assert not torch.is_grad_enabled()
        return original(n,v,l,**kwargs)
    def observed_integrator(*args):
        integrations.append(True);return integrate(*args)
    monkeypatch.setattr(material,'classic',observed)
    monkeypatch.setattr(light,'integrate',observed_integrator)
    result=tool().sample(1.,model='classic',order=8,deadline=time.perf_counter()+30)
    expected=[a+.1*25/24 for a in (.2,.4,.6)]
    for estimate in result['estimates']:assert estimate==pytest.approx(expected,abs=1e-12)
    assert len(calls)==len(integrations)==26
    assert max(calls)==10 and min(calls)==8 and sum(calls)==256


def test_public_environment_sampling_stops_before_expired_work():
    with pytest.raises(TimeoutError):tool().sample(.4,model='classic',order=8,deadline=0)


@pytest.mark.parametrize('order',[0,7,256])
def test_public_environment_sampling_rejects_unqualified_orders(order):
    with pytest.raises(ValueError):tool().sample(.4,model='classic',order=order,deadline=math.inf)
