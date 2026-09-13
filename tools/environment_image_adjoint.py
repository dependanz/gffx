"""Bounded full-colour Jacobians followed by one native texture VJP per point.

Qualification tooling only: all lighting still runs through the public material
and integrator. Never form losses on partial angular sums.
"""
import math

import torch

from tools import environment_loss as old


def sum_jacobians(colour, chunks, *, deadline):
    old.check_deadline(deadline)
    if (colour.shape != (1, 3) or colour.dtype != torch.float64
            or colour.device.type != 'cpu' or not bool(torch.isfinite(colour).all())
            or not bool(((colour >= 0) & (colour <= 1)).all())):
        raise ValueError('one finite CPU float64 RGB colour in [0,1] required')
    if not isinstance(chunks, (list, tuple)) or not 1 <= len(chunks) <= 64 or not all(callable(x) for x in chunks):
        raise ValueError('one to 64 bounded angular callbacks required')
    total = torch.zeros_like(colour)
    jacobian = colour.new_zeros((3, 3))
    for callback in chunks:
        old.check_deadline(deadline)
        leaf = colour.detach().requires_grad_(True)
        rgb = callback(leaf)
        if rgb.shape != (1, 3) or rgb.dtype != colour.dtype or rgb.device != colour.device or not bool(torch.isfinite(rgb).all()):
            raise ValueError('finite RGB callback result required')
        total += rgb.detach()
        for channel in range(3):
            old.check_deadline(deadline)
            derivative, = torch.autograd.grad(rgb[0, channel], leaf, retain_graph=channel < 2)
            if not bool(torch.isfinite(derivative).all()):
                raise ValueError('finite colour Jacobian required')
            jacobian[channel] += derivative.detach()[0]
        del rgb, leaf, derivative
    old.check_deadline(deadline)
    return total, jacobian


def contribution(colour, n, v, d, w, model, environment, budget):
    from gffx.torch import material, light
    if model == 'classic':
        brdf = material.classic(n, v, d, albedo=colour,
            specular=colour.new_tensor(.1), exponent=colour.new_tensor(2.))
    else:
        brdf = material.principled(n, v, d, albedo=colour,
            roughness=colour.new_tensor(1.), f0=colour.new_tensor(.04), budget=budget)
    packet = dict(measure='solid_angle', direction=d[None],
        radiance=old.base.old.field(d, old.base.bands(environment))[None],
        valid=torch.ones((1, len(d)), dtype=torch.bool), solid_angle=w[None])
    return light.integrate(n, brdf, packet)


def angular_pass(tex, n, v, uv, target, *, model, environment, budget,
                 orders, split, deadline, chunk_size=1024):
    old.direct.validate(n, v, uv, model, deadline)
    old.base.bands(environment)
    if len(n) > 4 or len(orders) != len(n) or any(type(k) is not int or k not in (8, 16, 32, 64, 128) for k in orders):
        raise ValueError('bounded angular orders/pixels required')
    if split not in (.5, .25) or type(chunk_size) is not int or not 1 <= chunk_size <= 1024:
        raise ValueError('angular work limit')
    if any(math.ceil(4*k*k/chunk_size) > 64 for k in orders):
        raise ValueError('at most 64 angular chunks per point')
    if (tex.shape != (32, 32, 3) or tex.dtype != torch.float64 or tex.device.type != 'cpu'
            or not bool(torch.isfinite(tex).all()) or not bool(((tex >= 0) & (tex <= 1)).all())
            or target.shape != n.shape or target.dtype != torch.float64 or target.device.type != 'cpu'
            or target.requires_grad or not bool(torch.isfinite(target).all())):
        raise ValueError('texture/target domain')
    from gffx.torch import render
    values = []; gradient = torch.zeros_like(tex); calls = 0; directions = 0
    for i, order in enumerate(orders):
        old.check_deadline(deadline)
        leaf = tex.detach().requires_grad_(True)
        pyramid, offsets = render.texture_pyramid(leaf, levels=1)
        colour = render.texture(pyramid, offsets, 32, 32, uv[i:i+1].contiguous(), channel_count=3,
            filter=render.FILTER_BILINEAR, mip_filter=render.MIP_NEAREST,
            wrap_u=render.WRAP_REPEAT, wrap_v=render.WRAP_REPEAT)
        q, w = old.base.old.quadrature(order, split)
        d = q @ old.base.old.frame(n[i].tolist(), v[i].tolist())[1]
        chunks = [lambda a, start=start, end=min(start+chunk_size, len(q)):
            contribution(a, n[i:i+1], v[i:i+1], d[start:end], w[start:end], model, environment, budget)
            for start in range(0, len(q), chunk_size)]
        total, jacobian = sum_jacobians(colour.detach(), chunks, deadline=deadline)
        cotangent = 2*(total-target[i:i+1])/(3*len(n))
        derivative, = torch.autograd.grad(colour, leaf, grad_outputs=cotangent @ jacobian)
        gradient += derivative.detach()
        values.append(total[0]); calls += len(chunks); directions += len(q)
        del derivative, leaf, pyramid, colour, chunks
    values = torch.stack(values)
    old.check_deadline(deadline)
    return dict(values=values, gradient=gradient,
        loss=float((values-target).square().sum()/(3*len(n))), calls=calls,
        directions=directions, colour_backwards=3*calls, texture_backwards=len(n))
