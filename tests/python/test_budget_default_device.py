"""An explicit resource device must override ambient tensor defaults."""
import pytest
import torch


@pytest.mark.parametrize('route',['bundle','envelope','transfer'])
def test_budget_explicit_cpu_overrides_ambient_meta_device(route):
    from gffx.torch.material import PrincipledBudget
    resource='ggx-smith-loss-cpu-f64-v1'
    baseline=PrincipledBudget.bundled(resource,device='cpu',dtype=torch.float64)
    with torch.device('meta'):
        if route=='bundle':
            actual=PrincipledBudget.bundled(resource,device='cpu',dtype=torch.float64)
        elif route=='envelope':
            actual=PrincipledBudget.from_envelope(baseline.to_envelope(),device='cpu',dtype=torch.float64)
        else:
            actual=baseline.to(device='cpu',dtype=torch.float64)
    assert actual.metadata==baseline.metadata
    assert set(actual._data)==set(baseline._data)
    for key,tensor in actual._data.items():
        assert tensor.device==torch.device('cpu')
        torch.testing.assert_close(tensor,baseline._data[key],rtol=0,atol=0)
