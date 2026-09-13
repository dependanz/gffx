"""Unsupported graph modes must not be mistaken for validated shading exports."""
import pytest
import torch


@pytest.fixture(autouse=True)
def serial():
    previous = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def module_for(model):
    from gffx.torch import material
    class Shader(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.budget = material.PrincipledBudget.bundled('ggx-smith-loss-cpu-f64-v1',device='cpu',dtype=torch.float64) if model=='principled' else None
        def forward(self, albedo):
            n = albedo.new_tensor([[0.,0.,1.]])
            if model=='classic':
                return material.classic(n,n,n,albedo=albedo,specular=albedo.new_tensor(.1),exponent=albedo.new_tensor(2.))
            return material.principled(n,n,n,albedo=albedo,roughness=albedo.new_tensor(.63),f0=albedo.new_tensor(.04),budget=self.budget)
    return Shader()


@pytest.mark.parametrize('model',['classic','principled'])
def test_shading_fullgraph_rejection_never_calls_a_fallback_backend(model):
    shader = module_for(model)
    albedo = torch.tensor([.2,.3,.4],dtype=torch.float64)
    expected = shader(albedo)
    calls = []
    def backend(graph, inputs):
        calls.append(graph)
        return graph.forward
    # fullgraph=True deliberately forbids graph breaks and eager fallback.
    # An Unsupported compiler result is a negative capability, not a passing export.
    from torch._dynamo.exc import Unsupported
    with pytest.raises(Unsupported,match='(?i)(bool|data.dependent|tensor)'):
        torch.compile(shader,backend=backend,fullgraph=True)(albedo)
    assert calls == []
    torch.testing.assert_close(shader(albedo),expected,atol=0,rtol=0)
    with pytest.raises(ValueError): shader(albedo.new_tensor([-.1,.3,.4]))


@pytest.mark.parametrize('model',['classic','principled'])
@pytest.mark.parametrize('strict',[True,False])
def test_shading_export_rejection_preserves_eager_domain_checks(model,strict):
    shader = module_for(model)
    albedo = torch.tensor([.2,.3,.4],dtype=torch.float64)
    from torch._dynamo.exc import Unsupported
    from torch.fx.experimental.symbolic_shapes import GuardOnDataDependentSymNode
    with pytest.raises((Unsupported,GuardOnDataDependentSymNode),match='(?i)(bool|data.dependent|tensor)'):
        torch.export.export(shader,(albedo,),strict=strict)
    assert bool(torch.isfinite(shader(albedo)).all())
    with pytest.raises(ValueError): shader(albedo.new_tensor([float('nan'),.3,.4]))
