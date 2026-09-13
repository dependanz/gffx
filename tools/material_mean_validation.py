"""Test oracle: the frozen mean quadrature's value and first partial derivatives.

This is not a differentiable public query or an alternative physical model.
Complete each weighted view's backward before evaluating the next view; return
detached measurements only AFTER all required parameter derivatives are taken.
"""
import torch
from tools import material_loss_reference as reference
from tools.material_budget_reference import _validate, _resolution
from tools.shading_energy_probe import _gauss_unit


def average_value_and_gradients(roughness, f0, *, resolution=128,
                                view_resolution=64, method='sinh'):
    _validate(roughness=roughness, f0=f0)
    _resolution(resolution)
    _resolution(view_resolution, maximum=64)
    if 2*resolution**2*view_resolution > 2**21:
        raise ValueError('requested quadrature work exceeds 2097152 light samples')
    if method not in ('sinh', 'log'):
        raise ValueError('unknown loss coordinate method')
    nodes, weights = _gauss_unit(view_resolution)
    values, gradients = [], []
    with torch.enable_grad():
        for index, weight in enumerate(weights):
            # Fresh independent leaves give partials at the supplied parameter
            # values, without consuming or accumulating into the caller's graph.
            r = roughness.detach().clone().requires_grad_()
            f = f0.detach().clone().requires_grad_()
            a = r.square()
            bound = torch.asinh(1/a)
            # Preserve the frozen vector arithmetic/node values exactly. Only
            # these tiny coordinate vectors are repeated, not light graphs.
            u = nodes*bound
            cosine = (a*torch.sinh(u))[index]
            jacobian = (a*torch.cosh(u)*bound)[index]
            loss = reference.directional(cosine, r, f, resolution=resolution, method=method)
            term = loss*cosine*jacobian*weight
            partials = torch.autograd.grad(term, (r, f))
            values.append(term.detach())
            gradients.append(torch.stack(partials).detach())
            del term, loss, partials, cosine, jacobian, u, bound, a, r, f
    return 2*torch.stack(values).sum(), tuple(2*torch.stack(gradients).sum(dim=0))
