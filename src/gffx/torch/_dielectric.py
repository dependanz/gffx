"""Finite geometric and optical operations for the first solid-glass path.

Only PyTorch and the Python standard library are used.  ``TriangleQuery`` is a
two-sided, chunked triangle query, not a screen-space depth lookup.  Its initial
cost is O(number of rays * number of faces); pairwise temporary storage is
bounded by ``ray_chunk * face_chunk``.  It is intended for the small Studio D
controls and previews, not a real-time acceleration claim.

Distances are world-space lengths because ray directions must be unit vectors.
The renderer owns medium transitions and bounded path transport.  This module
does not infer a glass thickness from a single front/back depth pair.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import NamedTuple

import torch


class InterfaceResult(NamedTuple):
    reflected: torch.Tensor
    transmitted: torch.Tensor
    fresnel: torch.Tensor
    tir: torch.Tensor
    cos_i: torch.Tensor
    cos_t: torch.Tensor


class Hit(NamedTuple):
    face_index: torch.Tensor
    distance: torch.Tensor
    barycentric: torch.Tensor
    normal: torch.Tensor
    hit: torch.Tensor


def dielectric(*args, **kwargs):
    """Material identity marker; a delta interface is not an ordinary BRDF."""
    raise NotImplementedError(
        "solid dielectric material requires the dielectric transmission renderer"
    )


def _float_tensor(value, name, *, like=None):
    if not isinstance(value, torch.Tensor) or value.layout != torch.strided:
        raise TypeError(name + " must be a strided tensor")
    if value.dtype not in (torch.float32, torch.float64):
        raise TypeError(name + " must use float32 or float64")
    if value.device.type not in ("cpu", "cuda"):
        raise ValueError(name + " must use CPU or CUDA")
    if like is not None:
        if value.dtype != like.dtype:
            raise TypeError(name + " precision must match the other input")
        if value.device != like.device:
            raise ValueError(name + " device must match the other input")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(name + " must be finite")
    return value


def validate_parameters(parameters):
    """Validate the complete, constant optical parameter record.

    ``absorption`` is an RGB Beer-Lambert coefficient per world-space unit.
    It is not a surface tint or an alpha value.  The first implementation uses
    smooth interfaces, so roughness/textures/opacity are not silently accepted.
    """
    if not isinstance(parameters, Mapping):
        raise TypeError("dielectric parameters must be a mapping")
    if set(parameters) != {"ior", "absorption"}:
        raise ValueError("dielectric parameters must be exactly ior and absorption")
    ior = _float_tensor(parameters["ior"], "dielectric ior")
    absorption = _float_tensor(parameters["absorption"], "dielectric absorption", like=ior)
    if ior.ndim != 0 or not bool((ior >= 1) & (ior <= 4)):
        raise ValueError("dielectric ior must be a scalar in [1,4]")
    if absorption.shape != (3,) or not bool((absorption >= 0).all()):
        raise ValueError("dielectric absorption must be nonnegative RGB")


def make_material(cls, name, *, ior=1.5, absorption=(0., 0., 0.), dtype=None, device=None):
    """Shared authoring constructor used by ``Material.dielectric``."""
    from ._scene_builders import tensors

    ior, absorption = tensors(ior, absorption, dtype=dtype, device=device)
    parameters = {"ior": ior, "absorption": absorption}
    validate_parameters(parameters)
    return cls(name, dielectric, parameters)


def _unit(value, name):
    # Rescaling first avoids an unnecessary overflow/underflow while computing
    # a norm.  Finite input vectors can nevertheless have a squared norm that
    # is not representable at their original scale.
    scale = value.abs().amax(dim=-1, keepdim=True)
    if not bool((scale > 0).all()):
        raise ValueError(name + " must be nonzero")
    scaled = value / scale
    return scaled / torch.linalg.vector_norm(scaled, dim=-1, keepdim=True)


def _length(value):
    """Vector length without squaring the vector's original magnitude."""
    scale = value.abs().amax(dim=-1, keepdim=True)
    scaled = value / torch.where(scale > 0, scale, 1)
    return (scale * torch.linalg.vector_norm(scaled, dim=-1, keepdim=True))[..., 0]


def _field(value, reference, shape, name, *, allow_infinite=False):
    if isinstance(value, torch.Tensor):
        if value.layout != torch.strided or value.dtype != reference.dtype:
            raise TypeError(name + " must match the ray floating precision")
        if value.device != reference.device:
            raise ValueError(name + " must match the ray device")
    else:
        value = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    try:
        value = torch.broadcast_to(value, shape)
    except RuntimeError as error:
        raise ValueError(name + " is not broadcastable to the ray shape") from error
    valid = ~torch.isnan(value) if allow_infinite else torch.isfinite(value)
    if not bool(valid.all()):
        raise ValueError(name + (" must not be NaN" if allow_infinite else " must be finite"))
    return value


def interface(incident, normal, eta_i, eta_t):
    """Reflect/refract at one oriented, smooth dielectric interface.

    ``incident`` points along light/ray travel toward the surface. ``normal``
    points into the incident medium, opposing that direction. Both inputs are
    normalized here and may broadcast over their leading dimensions. IORs are
    positive scalar or per-ray values. The returned Fresnel value is the exact
    unpolarized dielectric *power* reflectance, not a Schlick fit.

    Total internal reflection produces unit reflection, Fresnel=1, and a zero
    transmitted vector. The caller must not launch the zero vector as a ray.
    Equal IORs produce exactly zero reflection weight and unchanged direction,
    including at grazing incidence.
    """
    _float_tensor(incident, "incident direction")
    _float_tensor(normal, "interface normal", like=incident)
    if incident.ndim < 1 or normal.ndim < 1 or incident.shape[-1] != 3 or normal.shape[-1] != 3:
        raise ValueError("incident direction and normal must have a final xyz dimension")
    try:
        incident, normal = torch.broadcast_tensors(incident, normal)
    except RuntimeError as error:
        raise ValueError("incident direction and normal shapes do not broadcast") from error
    incident = _unit(incident, "incident direction")
    normal = _unit(normal, "interface normal")
    cosine = -(incident * normal).sum(dim=-1)
    tolerance = 16 * torch.finfo(incident.dtype).eps
    if not bool((cosine >= -tolerance).all()):
        raise ValueError("interface normal must oppose the incident direction")
    cosine = cosine.clamp(0, 1)
    shape = incident.shape[:-1]
    eta_i = _field(eta_i, incident, shape, "incident ior")
    eta_t = _field(eta_t, incident, shape, "transmitted ior")
    if not bool(((eta_i > 0) & (eta_t > 0)).all()):
        raise ValueError("interface IORs must be positive")

    ratio = eta_i / eta_t
    if not bool(torch.isfinite(ratio).all()):
        raise ValueError("interface IOR ratio exceeds the working precision")
    tangent = incident + cosine[..., None] * normal
    sin_t = ratio * _length(tangent).clamp_max(1)
    same = eta_i == eta_t
    tir = (sin_t > 1) & ~same
    cos_t = (1 - sin_t.clamp_max(1).square()).clamp_min(0).sqrt()
    cos_t = torch.where(same, cosine, cos_t)
    reflected = _unit(incident + 2 * cosine[..., None] * normal, "reflected direction")
    safe_ratio = torch.where(tir | same, torch.ones_like(ratio), ratio)
    transmitted = safe_ratio[..., None] * tangent - cos_t[..., None] * normal
    # A TIR branch has no real transmitted direction. Replace it before unit
    # normalization, then expose an explicit finite zero in the returned record.
    transmitted = torch.where((tir | same)[..., None], incident, transmitted)
    transmitted = _unit(transmitted, "transmitted direction")
    transmitted = torch.where(tir[..., None], torch.zeros_like(transmitted), transmitted)

    tiny = torch.finfo(incident.dtype).tiny
    eta_scale = torch.maximum(eta_i, eta_t)
    ni, nt = eta_i / eta_scale, eta_t / eta_scale
    rs = (ni * cosine - nt * cos_t) / (ni * cosine + nt * cos_t).clamp_min(tiny)
    rp = (nt * cosine - ni * cos_t) / (nt * cosine + ni * cos_t).clamp_min(tiny)
    fresnel = ((rs.square() + rp.square()) * .5).clamp(0, 1)
    fresnel = torch.where(cosine == 0, torch.ones_like(fresnel), fresnel)
    fresnel = torch.where(tir, torch.ones_like(fresnel), fresnel)
    fresnel = torch.where(same, torch.zeros_like(fresnel), fresnel)
    return InterfaceResult(reflected, transmitted, fresnel, tir, cosine, cos_t)


def beer(absorption, distance):
    """RGB Beer-Lambert attenuation over a finite world-space path length."""
    _float_tensor(absorption, "absorption")
    if absorption.ndim < 1 or absorption.shape[-1] != 3 or not bool((absorption >= 0).all()):
        raise ValueError("absorption must be nonnegative RGB")
    if isinstance(distance, torch.Tensor):
        _float_tensor(distance, "distance", like=absorption)
    else:
        distance = torch.as_tensor(distance, dtype=absorption.dtype, device=absorption.device)
        if not bool(torch.isfinite(distance).all()):
            raise ValueError("distance must be finite")
    if not bool((distance >= 0).all()):
        raise ValueError("distance must be nonnegative")
    try:
        exponent = absorption * distance[..., None]
    except RuntimeError as error:
        raise ValueError("absorption and distance shapes do not broadcast") from error
    return torch.exp(-exponent)


class TriangleQuery:
    """Two-sided nearest triangle intersections, with bounded pairwise storage.

    Unit ray directions are required. A hit must satisfy ``t_min < t <= t_max``.
    Bounds can be scalar or per-ray tensors; ``t_max=inf`` is allowed. Exact
    distance ties select the smallest input face index, independent of chunks.
    ``ignore_faces`` optionally excludes one input face per ray (-1 means none).

    Degenerate/near-parallel intersections are rejected using a dtype-scaled
    determinant tolerance. A small barycentric roundoff tolerance closes shared
    edges; admitted barycentrics are clamped and renormalized. No screen-space
    data, native extension, or installed geometry package is used.
    """
    def __init__(self, vertices, faces, *, ray_chunk=256, face_chunk=256):
        _float_tensor(vertices, "vertices")
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("vertices must have shape [N,3]")
        if not isinstance(faces, torch.Tensor) or faces.layout != torch.strided:
            raise TypeError("faces must be a strided integer tensor")
        if faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype not in (torch.int32, torch.int64):
            raise TypeError("faces must be int32/int64 with shape [F,3]")
        if faces.device != vertices.device:
            raise ValueError("face and vertex devices must match")
        if faces.numel() and not bool(((faces >= 0) & (faces < len(vertices))).all()):
            raise ValueError("face vertex indices are out of bounds")
        if type(ray_chunk) is not int or type(face_chunk) is not int or min(ray_chunk, face_chunk) < 1:
            raise ValueError("ray_chunk and face_chunk must be positive integers")
        if ray_chunk * face_chunk > 1_048_576:
            raise ValueError("ray_chunk * face_chunk must not exceed 1048576 pairs")
        self.vertices = vertices
        self.faces = faces
        self.ray_chunk = ray_chunk
        self.face_chunk = face_chunk
        triangles = vertices[faces.long()]
        self.a = triangles[:, 0]
        self.e1 = triangles[:, 1] - self.a
        self.e2 = triangles[:, 2] - self.a
        cross = torch.linalg.cross(self.e1, self.e2)
        if not bool(torch.isfinite(cross).all()):
            raise ValueError("triangle geometry exceeds the working precision")
        self.area2 = _length(cross)
        if not bool(torch.isfinite(self.area2).all()):
            raise ValueError("triangle area exceeds the working precision")
        self.normal = cross / torch.where(self.area2 > 0, self.area2, 1)[:, None]
        self._vertex_version = vertices._version
        self._face_version = faces._version

    def intersect(self, origins, directions, *, t_min=0., t_max=float("inf"), ignore_faces=None):
        _float_tensor(origins, "ray origins", like=self.vertices)
        _float_tensor(directions, "ray directions", like=self.vertices)
        if origins.ndim != 2 or origins.shape[1] != 3 or directions.shape != origins.shape:
            raise ValueError("ray origins/directions must have the same shape [R,3]")
        if self.vertices._version != self._vertex_version or self.faces._version != self._face_version:
            raise ValueError("triangle inputs changed; construct a fresh TriangleQuery")
        norm2 = directions.square().sum(dim=-1)
        if not bool((torch.abs(norm2 - 1) <= 64 * torch.finfo(origins.dtype).eps).all()):
            raise ValueError("ray directions must be unit length so distances are world-space")
        count = len(origins)
        minimum = _field(t_min, origins, (count,), "t_min")
        maximum = _field(t_max, origins, (count,), "t_max", allow_infinite=True)
        if not bool(((minimum >= 0) & (maximum >= minimum)).all()):
            raise ValueError("ray bounds require 0 <= t_min <= t_max")
        if ignore_faces is not None:
            if (not isinstance(ignore_faces, torch.Tensor) or ignore_faces.layout != torch.strided
                    or ignore_faces.dtype not in (torch.int32, torch.int64)):
                raise TypeError("ignore_faces must be an int32/int64 tensor")
            if ignore_faces.device != origins.device or ignore_faces.shape != (count,):
                raise ValueError("ignore_faces must match the ray device and have shape [R]")
            if not bool(((ignore_faces >= -1) & (ignore_faces < len(self.faces))).all()):
                raise ValueError("ignore_faces contains an out-of-range face")
        index = torch.full((count,), -1, dtype=torch.int64, device=origins.device)
        distance = origins.new_full((count,), float("inf"))
        barycentric = origins.new_zeros((count, 3))
        normals = origins.new_zeros((count, 3))
        if not len(self.faces) or not count:
            return Hit(index, distance, barycentric, normals, index >= 0)
        epsilon = torch.finfo(origins.dtype).eps
        # Accumulate query results functionally, preserving autograd graphs.
        index_parts, distance_parts, barycentric_parts, normal_parts = [], [], [], []
        for first_ray in range(0, count, self.ray_chunk):
            last_ray = min(first_ray + self.ray_chunk, count)
            ray_origins = origins[first_ray:last_ray]
            ray_directions = directions[first_ray:last_ray]
            ray_count = last_ray - first_ray
            closest = origins.new_full((ray_count,), float("inf"))
            chosen = torch.full((ray_count,), -1, dtype=torch.int64, device=origins.device)
            weights = origins.new_zeros((ray_count, 3))
            rows = torch.arange(last_ray - first_ray, device=origins.device)
            for first_face in range(0, len(self.faces), self.face_chunk):
                last_face = min(first_face + self.face_chunk, len(self.faces))
                a = self.a[first_face:last_face]
                e1 = self.e1[first_face:last_face]
                e2 = self.e2[first_face:last_face]
                p = torch.linalg.cross(ray_directions[:, None, :], e2[None, :, :])
                determinant = (p * e1[None, :, :]).sum(dim=-1)
                eligible = determinant.abs() > (8 * epsilon * self.area2[first_face:last_face])[None, :]
                eligible = eligible & (self.area2[first_face:last_face][None, :] > 0)
                reciprocal = 1 / torch.where(eligible, determinant, 1)
                delta = ray_origins[:, None, :] - a[None, :, :]
                u = (delta * p).sum(dim=-1) * reciprocal
                q = torch.linalg.cross(delta, e1[None, :, :])
                v = (ray_directions[:, None, :] * q).sum(dim=-1) * reciprocal
                candidate = (e2[None, :, :] * q).sum(dim=-1) * reciprocal
                eligible = eligible & (u >= -8 * epsilon) & (v >= -8 * epsilon) & (u + v <= 1 + 8 * epsilon)
                # D02 isolated precision candidate: resolve only float32
                # triangle-edge ambiguity using the exact stored float32
                # geometry/ray values promoted to float64 for evaluation.
                # No change to the original float32 distance/weights.
                if ray_origins.dtype == torch.float32:
                    edge_band = 8 * epsilon
                    near_edge = eligible & (
                        (u.abs() <= edge_band)
                        | (v.abs() <= edge_band)
                        | ((u + v) >= (1 - edge_band))
                    )
                    if bool(near_edge.any()):
                        rr, ff = near_edge.nonzero(as_tuple=True)
                        o64 = ray_origins[rr].to(torch.float64)
                        d64 = ray_directions[rr].to(torch.float64)
                        # Reconstruct the edges in f64 from the *stored*
                        # float32 vertices, not from f32-subtracted edges.
                        # This distinguishes coordinates quantized before
                        # subtraction from coordinates quantized afterward.
                        vertices64 = self.vertices[self.faces[first_face + ff].long()].to(torch.float64)
                        a64 = vertices64[:, 0]
                        e164 = vertices64[:, 1] - a64
                        e264 = vertices64[:, 2] - a64
                        p64 = torch.linalg.cross(d64, e264)
                        det64 = (e164 * p64).sum(dim=-1)
                        q64 = torch.linalg.cross(o64 - a64, e164)
                        u64 = ((o64 - a64) * p64).sum(dim=-1) / det64
                        v64 = (d64 * q64).sum(dim=-1) / det64
                        double_tol = 16 * torch.finfo(torch.float64).eps
                        high_precision_inside = (
                            torch.isfinite(u64) & torch.isfinite(v64)
                            & (det64 != 0)
                            & (u64 >= -double_tol)
                            & (v64 >= -double_tol)
                            & (u64 + v64 <= 1 + double_tol)
                        )
                        edge_keep = torch.ones_like(eligible)
                        edge_keep[rr, ff] = high_precision_inside
                        eligible = eligible & edge_keep
                eligible = eligible & (candidate > minimum[first_ray:last_ray, None])
                eligible = eligible & (candidate <= maximum[first_ray:last_ray, None])
                eligible = eligible & torch.isfinite(candidate)
                if ignore_faces is not None:
                    face_ids = torch.arange(first_face, last_face, device=origins.device)
                    eligible = eligible & (face_ids[None, :] != ignore_faces[first_ray:last_ray, None])
                candidate = torch.where(eligible, candidate, float("inf"))
                best_distance, local_index = candidate.min(dim=1)
                # Strict '<' retains the earlier (smaller) global face on ties.
                replace = best_distance < closest
                selected_u = u[rows, local_index]
                selected_v = v[rows, local_index]
                selected_weights = torch.stack((1 - selected_u - selected_v, selected_u, selected_v), dim=-1)
                selected_weights = selected_weights.clamp(0, 1)
                selected_weights = selected_weights / selected_weights.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(origins.dtype).tiny)
                weights = torch.where(replace[:, None], selected_weights, weights)
                chosen = torch.where(replace, local_index + first_face, chosen)
                closest = torch.minimum(best_distance, closest)
            normal_parts.append(torch.where(
                (chosen >= 0)[:, None], self.normal[chosen.clamp_min(0)], 0
            ))
            index_parts.append(chosen)
            distance_parts.append(closest)
            barycentric_parts.append(weights)
        index = torch.cat(index_parts)
        distance = torch.cat(distance_parts)
        barycentric = torch.cat(barycentric_parts)
        normals = torch.cat(normal_parts)
        return Hit(index, distance, barycentric, normals, index >= 0)
