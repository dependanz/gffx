"""Tensor-first material compositions. Classic is empirical, not energy-conserving.

Principled is experimental CPU float64 until a physical budget is admitted.
Callbacks are trusted user code, not automatically energy-conserving materials.
"""
import math
from collections.abc import Mapping
import torch

from . import shade
from ._reference_channels import sum3, expand
from .light import _vectors, _shared, _unit, _finite
from ._principled_budget import PrincipledBudget, WORKSPACE_BYTES
from ._registry import Registry, parameters as _parameters

__all__ = ['classic', 'principled', 'PrincipledBudget', 'custom', 'Registry']


def _dot(a, b):
    value = sum3(a*b)
    if bool((value.abs() > 1 + 8*torch.finfo(value.dtype).eps).any()):
        raise ValueError('normalized dot product is not representable')
    # Correct endpoint roundoff without changing the uncorrected derivative.
    return value + (value.clamp(-1, 1)-value).detach()


def _rgb(value, batch, points, name, *, grey=False):
    if value.shape == (3,) or (grey and value.ndim == 0):
        return expand(value,(batch, points, 3))
    if value.ndim == 2:
        value = value[None]
    if (value.ndim != 3 or value.shape[0] not in (1, batch) or
            value.shape[1] != points or value.shape[2] not in ((1, 3) if grey else (3,))):
        raise ValueError(f'{name} shape/batch/channels do not match material points')
    return expand(value,(batch, points, 3))


def _geometry(normals, view, directions, two_sided, *, _layout_only=False):
    if type(two_sided) is not bool:
        raise TypeError('two_sided must be a bool')
    n = _vectors(normals, 'normals')
    b, p = n.shape[:2]
    v = _shared(view, b, (p, 3), 'view')
    if directions.ndim == 2 and directions.shape[-1] == 3:
        l = directions[None, None].expand(b, p, -1, 3)
    else:
        l = directions[None] if directions.ndim == 3 else directions
        if (l.ndim != 4 or l.shape[0] not in (1, b) or l.shape[1] != p or l.shape[-1] != 3):
            raise ValueError('directions shape/batch must be [Q,3], [P,Q,3] or [B,P,Q,3]')
        l = l.expand(b, p, -1, 3)
    if _layout_only:
        return n, v, l
    # Validate even shared vectors whose expansion will be empty. Masks and
    # orientation are discrete; inactive normalization must not enter backward.
    with torch.no_grad():
        _unit(normals, 'normals'); _unit(view, 'view'); _unit(directions, 'directions')
        nn, vv, ll = _unit(n, 'normals'), _unit(v, 'view'), _unit(l, 'directions')
        nv = _dot(nn, vv)
        sign = torch.where((nv < 0) & two_sided, -torch.ones_like(nv), torch.ones_like(nv))
        active = ((nv*sign) > 0)[..., None] & (_dot(nn[:, :, None]*sign[..., None, None], ll) > 0)
    return n, v, l, sign, active


def _zero_connection(*values):
    # An empty reduction retains a zero gradient without overflowing a sum of
    # large finite values or copying an entire noncontiguous input to flatten it.
    return sum(value.narrow(0, 0, 0).sum() if value.ndim else value*0 for value in values)


def classic(normals, view, directions, *, albedo, specular, exponent, two_sided=False):
    """Return albedo/pi + RGB specular*Blinn(N.H, exponent) on active pairs.

    Normals own [P,3]/[B,P,3]; output is [P,Q,3]/[B,P,Q,3]. View and light
    vectors are normalized. Back/horizon pairs have zero values and gradients.
    two_sided explicitly flips normals facing away from the viewer. This returns
    a BRDF, without lighting, visibility, textures, colour conversion or clipping.
    """
    shade._inputs(normals=normals, view=view, directions=directions,
                  albedo=albedo, specular=specular, exponent=exponent)
    n, v, l, sign, active = _geometry(normals, view, directions, two_sided)
    b, p, q = active.shape
    shade._domain(albedo, 'albedo'); shade._domain(specular, 'specular')
    shade._domain(exponent, 'exponent', upper=None)
    a = _rgb(albedo, b, p, 'albedo')
    s = _rgb(specular, b, p, 'specular', grey=True)
    e = shade._scalar(exponent, n[..., 0], 'exponent')
    result = _zero_connection(normals, view, directions, albedo, specular, exponent).expand(b, p, q, 3).clone()
    if bool(active.any()):
        nn = _unit(n[:, :, None].expand_as(l)[active], 'normals') * sign[..., None].expand_as(active)[active, None]
        vv = _unit(v[:, :, None].expand_as(l)[active], 'view')
        ll = _unit(l[active], 'directions')
        h = _unit(vv+ll, 'half-vector')
        cosine = _dot(nn, h)
        aa = a[:, :, None].expand_as(l)[active]
        ss = s[:, :, None].expand_as(l)[active]
        ee = e[..., None].expand_as(active)[active]
        result[active] = _finite(shade.lambert(aa) + ss*shade.blinn_phong(cosine, ee)[:, None])
    return result[0] if normals.ndim == 2 else result


def _principled_reference(normals, view, directions, *, albedo, roughness, f0, budget,
               two_sided=False, workspace_limit_bytes=WORKSPACE_BYTES,
               allow_experimental_budget=False):
    """GGX/Smith/Schlick plus reciprocal matte, without lighting or clipping.

    CPU float64 material calls accept at most 1024 point/light pairs and
    8 forward points or 4 gradient-bearing points, subject to the 64 MiB
    workspace ceiling. Standalone budget queries still have 32-entry limits.
    A synthetic budget requires a separate material opt-in.
    All per-call losses keep the current graph; nothing is cached across calls.
    """
    if not isinstance(budget, PrincipledBudget):
        raise TypeError('budget must be a PrincipledBudget')
    if type(allow_experimental_budget) is not bool:
        raise TypeError('allow_experimental_budget must be a bool')
    if not allow_experimental_budget and budget._metadata['admission'] != 'accepted':
        raise ValueError('experimental budget requires explicit material opt-in')
    if type(workspace_limit_bytes) is not int:
        raise TypeError('workspace_limit_bytes must be integer bytes')
    if not 0 <= workspace_limit_bytes <= WORKSPACE_BYTES:
        raise ValueError('workspace limit must lie within 64 MiB')
    shade._inputs(normals=normals, view=view, directions=directions,
                  albedo=albedo, roughness=roughness, f0=f0)
    if normals.dtype != torch.float64:
        raise TypeError('budget requires float64 material tensors')
    if normals.device != budget._data['roughness_nodes'].device:
        raise ValueError('budget requires material tensors on its resource device')
    lo, hi = budget._metadata['domain']['roughness']
    if normals.device.type=='cuda':
        flags=torch.stack((((albedo<0)|(albedo>1)).any(),
                           ((f0<0)|(f0>1)).any(),
                           ((roughness<lo)|(roughness>hi)).any())).tolist()
        for bad,message in zip(flags,('albedo outside its allowed domain',
                'f0 outside its allowed domain','roughness outside resource domain')):
            if bad:raise ValueError(message)
    else:
        shade._domain(albedo, 'albedo'); shade._domain(f0, 'f0')
        if bool(((roughness < lo) | (roughness > hi)).any()):
            raise ValueError('roughness outside resource domain')
    n, v, l = _geometry(normals, view, directions, two_sided, _layout_only=True)
    b, p, q = l.shape[:3]
    a = _rgb(albedo, b, p, 'albedo')
    f = _rgb(f0, b, p, 'f0', grey=True)
    r = shade._scalar(roughness, n[..., 0], 'roughness')
    backward = torch.is_grad_enabled() and any(x.requires_grad for x in
                   (normals, view, directions, albedo, roughness, f0))
    # Point curves are shared within this call; light pairs gather only their
    # required coefficients. Count the full layout before any active mask.
    if b*p*q and (b*p > (4 if backward else 8) or b*p*q > 1024):
        raise ValueError('material workspace budget exceeded: batch limit before expanded allocation')
    count = b*p*3 if q else 0
    needed = 3*budget.workspace(count, backward=backward)
    needed += 8*b*p*q*(192 if backward else 64)
    if needed > workspace_limit_bytes:
        raise ValueError('material workspace budget exceeded before expanded allocation')
    n, v, l, sign, active = _geometry(normals, view, directions, two_sided)
    result = _zero_connection(normals, view, directions, albedo, roughness, f0).expand(b, p, q, 3).clone()
    if bool(active.any()):
        points = active.any(-1)
        nn = _unit(n[points], 'normals')*sign[points, None]
        vv = _unit(v[points], 'view')
        rp = r[points, None].expand(-1, 3)
        fp = f[points]
        nv = _dot(nn, vv)
        # Mapping active pairs to active points reuses view loss and mean over Q.
        index = (points.reshape(-1).long().cumsum(0)-1).reshape(b, p)
        indices = index[..., None].expand_as(active)[active]
        nn, vv = nn[indices], vv[indices]
        ll = _unit(l[active], 'directions')
        h = _unit(vv+ll, 'half-vector')
        rr, ff = rp[indices], fp[indices]
        nl, nh, vh = _dot(nn, ll), _dot(nn, h), _dot(vv, h)
        view_loss, mean, loss = budget._material_losses(nv, nl, r[points], fp, indices)
        distribution = shade.ggx_distribution(nh, rr[:, 0])
        visibility = shade.smith_visibility(nv[indices], nl, rr[:, 0])
        specular = (distribution*visibility)[:, None]*shade.fresnel_schlick(vh, ff)
        matte = a[:, :, None].expand_as(l)[active]*view_loss[indices]*loss/(math.pi*mean[indices])
        result[active] = _finite(specular+matte)
    return result[0] if normals.ndim == 2 else result

def _principled_values(normals,view,directions,albedo,roughness,f0,budget):
    """Check original values once; no shape expansion or persistent validation state."""
    if not isinstance(budget,PrincipledBudget):raise TypeError("budget must be a PrincipledBudget")
    shade._inputs(normals=normals, view=view, directions=directions,
                  albedo=albedo, roughness=roughness, f0=f0)
    if normals.dtype != torch.float64:
        raise TypeError('budget requires float64 material tensors')
    if normals.device != budget._data['roughness_nodes'].device:
        raise ValueError('budget requires material tensors on its resource device')
    lo, hi = budget._metadata['domain']['roughness']
    if normals.device.type=='cuda':
        flags=torch.stack((((albedo<0)|(albedo>1)).any(),
                           ((f0<0)|(f0>1)).any(),
                           ((roughness<lo)|(roughness>hi)).any())).tolist()
        for bad,message in zip(flags,('albedo outside its allowed domain',
                'f0 outside its allowed domain','roughness outside resource domain')):
            if bad:raise ValueError(message)
    else:
        shade._domain(albedo, 'albedo'); shade._domain(f0, 'f0')
        if bool(((roughness < lo) | (roughness > hi)).any()):
            raise ValueError('roughness outside resource domain')


def _principled_layout(normals, view, directions, *, albedo, roughness, f0, budget,
                       two_sided=False, workspace_limit_bytes=WORKSPACE_BYTES,
                       allow_experimental_budget=False, _values_checked=False):
    """Validate one original admitted group; also used by renderer recomputation."""
    if not isinstance(budget, PrincipledBudget):
        raise TypeError('budget must be a PrincipledBudget')
    if type(allow_experimental_budget) is not bool:
        raise TypeError('allow_experimental_budget must be a bool')
    if not allow_experimental_budget and budget._metadata['admission'] != 'accepted':
        raise ValueError('experimental budget requires explicit material opt-in')
    if type(workspace_limit_bytes) is not int:
        raise TypeError('workspace_limit_bytes must be integer bytes')
    if not 0 <= workspace_limit_bytes <= WORKSPACE_BYTES:
        raise ValueError('workspace limit must lie within 64 MiB')
    if not _values_checked:
        _principled_values(normals,view,directions,albedo,roughness,f0,budget)
    n, v, l = _geometry(normals, view, directions, two_sided, _layout_only=True)
    b, p, q = l.shape[:3]
    a = _rgb(albedo, b, p, 'albedo')
    f = _rgb(f0, b, p, 'f0', grey=True)
    r = shade._scalar(roughness, n[..., 0], 'roughness')
    # Preserve shape validation, then use the qualified adjacent-pair reduction
    # for shared roughness VJPs on both devices. Forward values are unchanged.
    r = expand(roughness, r.shape)
    backward = torch.is_grad_enabled() and any(x.requires_grad for x in
                   (normals, view, directions, albedo, roughness, f0))
    # Point curves are shared within this call; light pairs gather only their
    # required coefficients. Count the full layout before any active mask.
    if b*p*q and (b*p > (4 if backward else 8) or b*p*q > 1024):
        raise ValueError('material workspace budget exceeded: batch limit before expanded allocation')
    count = b*p*3 if q else 0
    needed = 3*budget.workspace(count, backward=backward)
    needed += 8*b*p*q*(192 if backward else 64)
    if needed > workspace_limit_bytes:
        raise ValueError('material workspace budget exceeded before expanded allocation')
    return (n.reshape(b*p,3),v.reshape(b*p,3),l.reshape(b*p,q,3),
            a.reshape(b*p,3),r.reshape(b*p),f.reshape(b*p,3)),(b,p,q),backward


def principled(normals, view, directions, *, albedo, roughness, f0, budget,
               two_sided=False, workspace_limit_bytes=WORKSPACE_BYTES,
               allow_experimental_budget=False):
    """GGX/Smith/Schlick plus reciprocal matte, without lighting or clipping.

    CPU float64 material calls accept at most 1024 point/light pairs and
    8 forward points or 4 gradient-bearing points, subject to the 64 MiB
    workspace ceiling. Standalone budget queries still have 32-entry limits.
    A synthetic budget requires a separate material opt-in.
    All per-call losses keep the current graph; nothing is cached across calls.
    """
    values,(b,p,q),backward=_principled_layout(normals,view,directions,albedo=albedo,
        roughness=roughness,f0=f0,budget=budget,two_sided=two_sided,
        workspace_limit_bytes=workspace_limit_bytes,allow_experimental_budget=allow_experimental_budget)
    if not b*p*q:
        # Preserve original empty-layout validation and zero connections, including
        # shared vectors whose expansion is empty. No material work is dispatched.
        return _principled_reference(normals,view,directions,albedo=albedo,roughness=roughness,
            f0=f0,budget=budget,two_sided=two_sided,workspace_limit_bytes=workspace_limit_bytes,
            allow_experimental_budget=allow_experimental_budget)
    from ._principled_native import evaluate
    value=evaluate(budget,*values,two_sided,backward)
    result=value.reshape(b,p,q,3)
    return result[0] if normals.ndim==2 else result



def custom(function, normals, view, directions, *, parameters, two_sided=False):
    """Evaluate a trusted BRDF callback on normalized active pairs only."""
    if not callable(function):
        raise TypeError('function must be callable')
    _parameters(parameters, normals)
    shade._inputs(normals=normals, view=view, directions=directions)
    n, v, l, sign, active = _geometry(normals, view, directions, two_sided)
    b, p, q = active.shape
    fields = {}
    for key, value in parameters.items():
        if value.ndim == 0:
            value = value.reshape(1)
        if value.ndim == 1 and value.shape[0] > 0:
            value = value.expand(b, p, -1)
        else:
            if value.ndim == 2:
                value = value[None]
            if (value.ndim != 3 or value.shape[0] not in (1, b) or
                    value.shape[1] != p or value.shape[2] < 1):
                raise ValueError('parameter '+key+' shape must be scalar/channels or point channels')
            value = value.expand(b, p, -1)
        fields[key] = value
    result = _zero_connection(normals, view, directions, *parameters.values()).expand(b, p, q, 3).clone()
    if bool(active.any()):
        nn = _unit(n[:, :, None].expand_as(l)[active], 'normals')*sign[..., None].expand_as(active)[active, None]
        vv = _unit(v[:, :, None].expand_as(l)[active], 'view')
        ll = _unit(l[active], 'directions')
        packed = {k: value[:, :, None].expand(b, p, q, value.shape[-1])[active] for k, value in fields.items()}
        value = function(nn, vv, ll, parameters=packed)
        shade._inputs(normals=normals, output=value)
        if value.shape != nn.shape or bool((value < 0).any()):
            raise ValueError('custom BRDF output must be nonnegative [active_pairs,3]')
        result[active] = value
    return result[0] if normals.ndim == 2 else result
