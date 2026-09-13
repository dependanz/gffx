"""Bounded residual diagnostics, not a verified replacement for average()."""
import argparse
import torch

from tools import material_budget_sharp as sharp
from tools import material_loss_reference as loss
from tools.material_budget_reference import _validate, _resolution
from tools.shading_energy_probe import _gauss_unit, directional_albedo


def residual(roughness, f0, *, resolution=128, view_resolution=64, method="transformed"):
    """Explicit experimental methods under the same 2097152 light-sample cap.

    'legacy' subtracts the regular mean from one. 'direct' integrates 1-E on
    regular view nodes. 'transformed' moves view nodes to c=alpha*sinh(u).
    'independent' uses the VNDF oracle at those moved nodes (detached diagnostic).
    None is admitted as an accurate tiny-budget constructor.
    """
    _validate(roughness=roughness, f0=f0)
    _resolution(resolution)
    _resolution(view_resolution, maximum=64)
    if 2*resolution**2*view_resolution > 2**21:
        raise ValueError("requested quadrature work exceeds 2097152 light samples")
    if method not in ("legacy", "direct", "transformed", "independent"):
        raise ValueError("unknown residual diagnostic method")
    if method == "legacy":
        return 1-sharp.average(roughness, f0, resolution=resolution,
                               view_resolution=view_resolution)
    nodes, weights = _gauss_unit(view_resolution)
    if method == "direct":
        cosines, jacobian = nodes, torch.ones_like(nodes)
    else:
        alpha = roughness.square()
        bound = torch.asinh(1/alpha)
        u = nodes*bound
        cosines = alpha*torch.sinh(u)
        jacobian = alpha*torch.cosh(u)*bound
    if method == "independent":
        if roughness.requires_grad or f0.requires_grad:
            raise ValueError("independent oracle is detached, not a gradient reference")
        values = torch.tensor([directional_albedo(c.item(), roughness.item(),
            f0=f0.item(), resolution=resolution, radial_rule="tail-dense")
            for c in cosines], dtype=torch.float64)
    else:
        values = torch.stack([sharp.directional(c, roughness, f0,
            resolution=resolution) for c in cosines])
    result = 2*((1-values)*cosines*jacobian*weights).sum()
    if not bool(torch.isfinite(result)):
        raise ValueError("residual is not representable")
    return result


def study():
    """Six fixed pairs, serial; returns evidence, not pass/fail decisions."""
    rows = []
    with torch.no_grad():
        for r in (.001, .003, .01):
            for f in (1-1e-8, 1.):
                args = tuple(torch.tensor(x, dtype=torch.float64) for x in (r, f))
                methods = {}
                for method in ("legacy", "direct", "transformed", "independent"):
                    a, b = [residual(*args, resolution=n, view_resolution=v,
                        method=method).item() for n, v in ((64,32),(128,64))]
                    methods[method] = dict(coarse=a, fine=b,
                        relative_change=abs(a-b)/abs(b) if b else None)
                ref = methods["transformed"]["fine"]
                independent = methods["independent"]["fine"]
                for method in ('sinh','log'):
                    a,b=[loss.average(*args,resolution=n,view_resolution=v,method=method).item()
                         for n,v in ((64,32),(128,64))]
                    methods['loss-'+method]=dict(coarse=a,fine=b,relative_change=abs(a-b)/abs(b))
                repaired=methods['loss-sinh']['fine']
                rows.append(dict(roughness=r, f0=f, methods=methods,
                    independent_disagreement=abs(ref-independent)/abs(ref) if ref else None,
                    loss_coordinate_disagreement=abs(repaired-methods['loss-log']['fine'])/abs(repaired)))
    return dict(rows=rows, scope="Direct-loss repair passes the declared finite residual gates. Old unresolved estimates are preserved. Coordinate agreement is not an independent physical model or full-domain proof; no fast-table extension or public material.")


if __name__ == "__main__":
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    torch.set_num_threads(1)
    print(json.dumps(study(), indent=2))
