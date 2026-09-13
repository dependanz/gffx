"""Experimental immutable full-angle loss interpolant with an exact mean.

This is a CPU float64 research resource, not an admitted material/runtime API.
Queries use no quadrature. The independent numerical integrator is test-only.
Piecewise slopes are right-sided at interior knots and left-sided at the end.
"""
from dataclasses import dataclass
import math

import torch

from tools.material_budget_reference import _resolution, _validate
from tools.material_loss_endpoints import endpoints
from tools.shading_energy_probe import _gauss_unit


def _tensor(value):
    return torch.tensor(value, dtype=torch.float64)


def _freeze(value):
    return tuple(_freeze(item) if isinstance(item, list) else item for item in value)


def _log_mean(logx, logy):
    """Logarithmic mean, including x=y and its correct derivatives."""
    z = (logy-logx)/2
    near = z.abs() < 1e-3
    safe = torch.where(near, torch.ones_like(z), z)
    ratio = torch.where(near, 1+z.square()/6+z.pow(4)/120+z.pow(6)/5040,
                        torch.sinh(safe)/safe)
    return torch.exp((logx+logy)/2)*ratio


@dataclass(frozen=True, init=False)
class LossResource:
    roughness_nodes: tuple
    cosine_nodes: tuple
    losses: tuple

    def __init__(self, roughness_nodes, cosine_nodes, losses):
        for value in (roughness_nodes, cosine_nodes, losses):
            if (not isinstance(value, torch.Tensor) or value.dtype != torch.float64
                    or value.device.type != 'cpu' or not bool(torch.isfinite(value).all())):
                raise ValueError('resource data requires finite CPU float64 tensors')
        r, c, v = roughness_nodes, cosine_nodes, losses
        if (r.ndim != 1 or c.ndim != 1 or not 2 <= r.numel() <= 129
                or not 2 <= c.numel() <= 513 or v.shape != (r.numel(), c.numel(), 2)):
            raise ValueError('invalid resource shape or node limit')
        if not bool(((r > 0)&(r <= 1)).all() and (r[1:] > r[:-1]).all()
                    and c[0] == 0 and c[-1] == 1 and (c[1:] > c[:-1]).all()):
            raise ValueError('invalid resource axes')
        if not bool((v <= 1).all() and (v[:, 1:] > 0).all()
                    and (v[:, 0, 0] > 0).all() and (v[:, 0, 1] == 0).all()
                    and (v[..., 0] >= v[..., 1]).all()):
            raise ValueError('invalid positive endpoint loss data')
        object.__setattr__(self, 'roughness_nodes', _freeze(r.tolist()))
        object.__setattr__(self, 'cosine_nodes', _freeze(c.tolist()))
        object.__setattr__(self, 'losses', _freeze(v.tolist()))

    def _at_roughness(self, roughness):
        _validate(roughness=roughness)
        rn, cn, values = map(_tensor, (self.roughness_nodes, self.cosine_nodes, self.losses))
        if not bool((roughness >= rn[0])&(roughness <= rn[-1])):
            raise ValueError('roughness outside this resource coverage')
        # Selection is discrete; interpolation retains the continuous parameter.
        i = min(int(torch.searchsorted(rn, roughness, right=True))-1, len(rn)-2)
        u = (roughness.log()-rn[i].log())/(rn[i+1].log()-rn[i].log())
        logs = (1-u)*values[i, 1:].log()+u*values[i+1, 1:].log()
        horizon_zero = torch.exp((1-u)*values[i, 0, 0].log()+u*values[i+1, 0, 0].log())
        horizon = torch.stack((horizon_zero, horizon_zero*0))
        return cn, logs, horizon

    @staticmethod
    def _direction(cosine, cn, logs, horizon):
        if bool(cosine < cn[1]):
            u = cosine/cn[1]
            return (1-u)*horizon+u*logs[0].exp()
        i = min(int(torch.searchsorted(cn, cosine, right=True))-1, len(cn)-2)
        u = (cosine.log()-cn[i].log())/(cn[i+1].log()-cn[i].log())
        return torch.exp((1-u)*logs[i-1]+u*logs[i])

    @staticmethod
    def _mean(cn, logs, horizon):
        first = cn[1].square()*(horizon+2*logs[0].exp())/3
        left, right = cn[1:-1], cn[2:]
        logx = 2*left.log()[:, None]+logs[:-1]
        logy = 2*right.log()[:, None]+logs[1:]
        return first+(2*(right/left).log()[:, None]*_log_mean(logx, logy)).sum(dim=0)

    def evaluate(self, cosine, roughness, f0):
        """Return separate loss/energy and their exactly consistent angular mean."""
        _validate(cosine=cosine, f0=f0)
        cn, logs, horizon = self._at_roughness(roughness)
        pair = self._direction(cosine, cn, logs, horizon)
        mean_pair = self._mean(cn, logs, horizon)
        loss, mean = (1-f0)*pair[0]+f0*pair[1], (1-f0)*mean_pair[0]+f0*mean_pair[1]
        if not bool(torch.isfinite(loss)&torch.isfinite(mean)&(mean > 0)):
            raise ValueError('resource arithmetic is not representable')
        return dict(loss=loss, energy=1-loss, mean_loss=mean, mean_energy=1-mean)

    def integrate_for_test(self, roughness, f0, *, order=32):
        """Independent per-segment quadrature, never called by evaluate()."""
        _validate(f0=f0)
        _resolution(order, maximum=64)
        cn, logs, horizon = self._at_roughness(roughness)
        nodes, weights = _gauss_unit(order)
        curve = (1-nodes[:, None])*horizon+nodes[:, None]*logs[0].exp()
        first = (2*cn[1].square()*nodes[:, None]*curve*weights[:, None]).sum(dim=0)
        a, b = cn[1:-1], cn[2:]
        width = (b/a).log()
        c = a[:, None]*torch.exp(width[:, None]*nodes)
        curve = torch.exp((1-nodes[None, :, None])*logs[:-1, None, :]
                          +nodes[None, :, None]*logs[1:, None, :])
        rest = (2*c[..., None].square()*width[:, None, None]*curve*weights[None, :, None]).sum(dim=(0, 1))
        pair = first+rest
        return (1-f0)*pair[0]+f0*pair[1]


def generation_orders(cosine, roughness, radial, azimuth):
    """Same-work allocation; an integration rule, never a material domain cutoff."""
    _resolution(radial)
    _resolution(azimuth)
    if radial < 32 or radial % 2 or azimuth > 256:
        raise ValueError('grazing allocation requires even radial >=32 and azimuth <=256')
    return (radial//2, azimuth*2) if cosine < roughness*roughness/16 else (radial, azimuth)


def generation_pair(cosine, roughness, *, resolution=280, azimuth_resolution=32):
    """Paired endpoint integral with the declared conditional generation allocation."""
    radial, azimuth = generation_orders(cosine, roughness, resolution, azimuth_resolution)
    return endpoints(cosine, roughness, resolution=radial, azimuth_resolution=azimuth)


def generate(*, roughness_nodes=65, cosine_nodes=257, resolution=128, roughness_min=.001,
             azimuth_resolution=None, grazing_azimuth=False):
    """Bounded shared-geometry generation; caller controls serial CPU execution."""
    _resolution(resolution)
    azimuth_resolution = resolution if azimuth_resolution is None else azimuth_resolution
    _resolution(azimuth_resolution)
    if type(grazing_azimuth) is not bool:
        raise ValueError('grazing_azimuth must be bool')
    if grazing_azimuth:
        generation_orders(0., roughness_min, resolution, azimuth_resolution)
    if type(roughness_nodes) is not int or not 2 <= roughness_nodes <= 129:
        raise ValueError('roughness node limit is 129')
    if type(cosine_nodes) is not int or not 17 <= cosine_nodes <= 513 or (cosine_nodes-1) % 8:
        raise ValueError('cosine nodes must be 8k+1 in [17,513]')
    if not math.isfinite(roughness_min) or not .001 <= roughness_min < 1:
        raise ValueError('experimental roughness minimum is .001, not a public floor')
    if roughness_nodes*cosine_nodes*2*resolution*azimuth_resolution > 1_200_000_000:
        raise ValueError('generation exceeds 1200000000 shared geometry samples')
    rn = torch.exp(torch.linspace(math.log(roughness_min), 0, roughness_nodes, dtype=torch.float64))
    rn[0], rn[-1] = roughness_min, 1.
    a = roughness_min**2
    lower = (cosine_nodes-1)//8
    low = a*torch.logspace(-4, 0, lower+1, dtype=torch.float64)[:-1]
    high = a*torch.sinh(torch.linspace(math.asinh(1), math.asinh(1/a), cosine_nodes-1-lower, dtype=torch.float64))
    cn = torch.cat((_tensor([0]), low, high))
    cn[-1] = 1.
    paired = generation_pair if grazing_azimuth else endpoints
    with torch.no_grad():
        values = torch.stack([torch.stack([paired(c, r, resolution=resolution,
                  azimuth_resolution=azimuth_resolution) for c in cn]) for r in rn])
    return LossResource(rn, cn, values)
