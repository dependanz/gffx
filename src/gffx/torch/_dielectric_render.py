"""Bounded geometric dielectric transport for the first Studio D controls.

Opaque-only scenes keep the established raster path. Scenes using a dielectric
surface use exact triangle queries here, including queries after rays refract.
This is a small forward reference path. Optional plane caustics add bounded
photon transport; neither route claims real-time performance.

The admitted optical domain is closed, outward-oriented, non-overlapping solid
dielectrics, with the camera and opaque/emissive endpoints in air. A hollow
mesh may have any number of separate wall crossings: its cavity remains air.
One current object medium is tracked; nested/intersecting solids are rejected
when their unsupported transitions are encountered. Geometric face normals
define the optical interfaces. Camera transport branches at exact dielectric
Fresnel reflection/refraction, bounded by depth and retained path weight.

Radiance eta-squared factors telescope to one on the admitted air-to-air
paths. The stored weights are Fresnel power weights with Beer absorption;
they must not be extended to endpoints inside media without adding the
appropriate radiance transport factors. Discarded weights are diagnostics,
not universal image-error bounds. Direct lighting uses an explicitly separate
straight-ray Fresnel/Beer shadow approximation by default. Explicit plane
caustics replace that approximation on selected receiver meshes only.
"""
from __future__ import annotations

import math
import time

import torch

from . import material
from ._dielectric import TriangleQuery, beer, dielectric, interface, validate_parameters
from ._raster_material import pbr, unit, validate_material


MAX_INTERFACE_EVENTS = 1024
MIN_PATH_WEIGHT = 1e-4
MAX_SHADOW_INTERFACES = 32
ACTIVE_PATHS_PER_PRIMARY_BUDGET = 32


def _notify(callback, **event):
    if callback is not None:
        callback(event)


def _epsilon(points):
    """A positive query distance, without moving the absorption origin."""
    return 16 * torch.finfo(points.dtype).eps * points.abs().amax(-1).clamp_min(1.)


def _constant(value, name, like, shapes):
    if not isinstance(value, torch.Tensor) or tuple(value.shape) not in shapes:
        raise TypeError(name + " must be a constant authored tensor")
    if value.dtype != like.dtype or value.device != like.device:
        raise ValueError(name + " must match scene precision and device")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(name + " must be finite")


def _validate_materials(packed):
    """Initial glass scenes deliberately admit constant opaque fields only."""
    from ._deferred import TextureSource

    for mat in packed.materials:
        if any(isinstance(v, TextureSource) for v in mat.parameters.values()):
            raise NotImplementedError("dielectric transport currently requires constant material fields; textures are not admitted")
        if mat.evaluator is dielectric:
            validate_parameters(mat.parameters)
            _constant(mat.parameters['ior'], 'dielectric IOR', packed.vertices, {()})
            _constant(mat.parameters['absorption'], 'dielectric absorption', packed.vertices, {(3,)})
            if mat.emission is not None:
                raise NotImplementedError("emissive dielectric volumes are not admitted")
        elif mat.evaluator is pbr:
            validate_material(mat.parameters)
            allowed = {'base_color', 'roughness', 'metallic', 'ior', 'coat', 'coat_roughness'}
            if set(mat.parameters) - allowed:
                raise NotImplementedError("dielectric scenes currently admit constant PBR without subsurface or normal maps")
            for key, value in mat.parameters.items():
                _constant(value, 'PBR ' + key, packed.vertices, {(3,)} if key == 'base_color' else {()})
        elif mat.evaluator is material.classic:
            if set(mat.parameters) - {'albedo', 'specular', 'exponent'}:
                raise NotImplementedError("dielectric scenes admit the diffuse/glossy classic fields only")
            for key, value in mat.parameters.items():
                shapes = {(3,)} if key == 'albedo' else ({(), (3,)} if key == 'specular' else {()})
                _constant(value, 'classic ' + key, packed.vertices, shapes)
                if bool((value < 0).any()) or (key != 'exponent' and bool((value > 1).any())):
                    raise ValueError('classic ' + key + ' is outside its material domain')
        else:
            raise NotImplementedError("dielectric scenes require Material.dielectric, pbr, diffuse, glossy or emissive")
        if mat.emission is not None:
            _constant(mat.emission, 'emission', packed.vertices, {(3,)})
            if bool((mat.emission < 0).any()):
                raise ValueError('emission must be nonnegative')


def _validate_solids(packed, is_glass):
    """Check edge closure/orientation; this is not a self-intersection proof."""
    offsets = packed.object_face_offsets[0].detach().cpu().tolist()
    for owner, name in enumerate(packed.mesh_names):
        first, last = offsets[owner:owner + 2]
        if first == last or not bool(is_glass[packed.face_material[first].clamp_min(0)]):
            continue
        if int(packed.face_material[first]) < 0:
            continue
        faces = packed.faces[first:last]
        corners = packed.vertices[faces.long()]
        cross = torch.linalg.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        if bool((torch.linalg.vector_norm(cross, dim=-1) == 0).any()):
            raise ValueError('dielectric mesh has a degenerate face: ' + name)
        edges = {}
        for a, b, c in faces.detach().cpu().tolist():
            for u, v in ((a, b), (b, c), (c, a)):
                key = (min(u, v), max(u, v))
                count, orientation = edges.get(key, (0, 0))
                edges[key] = count + 1, orientation + (1 if u < v else -1)
        if any(count != 2 or orientation != 0 for count, orientation in edges.values()):
            raise ValueError('dielectric mesh must be closed with consistently oriented shared edges: ' + name)
        # Shift before the volume sum to avoid cancellation under translation.
        centre = corners.reshape(-1, 3).mean(0)
        local = corners - centre
        volume6 = (local[:, 0] * torch.linalg.cross(local[:, 1], local[:, 2])).sum()
        if not bool(torch.isfinite(volume6)) or not bool(volume6 > 0):
            raise ValueError('dielectric mesh must have outward winding and positive enclosed volume: ' + name)


class _Transport:
    def __init__(self, packed, camera, *, sources, background, environment, shadows,
                 query_chunk, diagnostics):
        self.packed, self.camera = packed, camera
        self.sources, self.background, self.environment = sources, background, environment
        self.shadows, self.stats = shadows, diagnostics
        self.caustic_plane = None
        self.query = TriangleQuery(packed.vertices, packed.faces,
                                   ray_chunk=query_chunk, face_chunk=256)
        self.is_glass = torch.tensor([m.evaluator is dielectric for m in packed.materials],
                                    dtype=torch.bool, device=packed.vertices.device)
        _validate_solids(packed, self.is_glass)
        face_number = torch.arange(len(packed.faces), device=packed.vertices.device)
        self.face_owner = torch.bucketize(face_number, packed.object_face_offsets[0, 1:].long(), right=True)
        count = len(packed.mesh_names)
        self.ior = packed.vertices.new_ones(count)
        self.sigma = packed.vertices.new_zeros((count, 3))
        offsets = packed.object_face_offsets[0].detach().cpu().tolist()
        for owner, (first, last) in enumerate(zip(offsets[:-1], offsets[1:])):
            if first == last:
                continue
            m = int(packed.face_material[first])
            if m >= 0 and bool(self.is_glass[m]):
                self.ior[owner] = packed.materials[m].parameters['ior']
                self.sigma[owner] = packed.materials[m].parameters['absorption']

    def miss(self, directions):
        if self.background is not None:
            return self.background.expand(len(directions), 3)
        if self.environment is not None:
            value = self.environment.sample(directions)
            if value.shape != directions.shape or not bool(torch.isfinite(value).all()):
                raise ValueError('environment.sample must return finite RGB for each direction')
            return value
        return torch.zeros_like(directions)

    def glass_mask(self, face):
        if not len(self.is_glass):
            return torch.zeros_like(face, dtype=torch.bool)
        m = self.packed.face_material[face]
        return (m >= 0) & self.is_glass[m.clamp_min(0).long()]

    def crossing(self, face, directions, geometric, medium):
        owner = self.face_owner[face]
        entering = (directions * geometric).sum(-1) < 0
        bad_entry = entering & (medium >= 0)
        bad_exit = ~entering & (medium != owner)
        if bool(bad_entry.any()):
            raise NotImplementedError('dielectric transport encountered nested/intersecting glass or inconsistent winding; only one occupied solid medium is admitted')
        if bool(bad_exit.any()):
            raise NotImplementedError('dielectric transport encountered an unmatched exit; the camera/near plane must be outside closed, outward-oriented glass')
        eta = self.ior[owner]
        eta_i = torch.where(entering, torch.ones_like(eta), eta)
        eta_t = torch.where(entering, eta, torch.ones_like(eta))
        normal = torch.where(entering[:, None], geometric, -geometric)
        optical = interface(directions, normal, eta_i, eta_t)
        next_medium = torch.where(entering, owner, -torch.ones_like(owner))
        return optical, next_medium

    def attenuate(self, weight, medium, distance):
        inside = medium >= 0
        if bool(inside.any()):
            weight = weight.clone()
            weight[inside] *= beer(self.sigma[medium[inside]], distance[inside])
        return weight

    def shadow(self, points, directions, distance, receiver_face):
        """Default straight transmission, or opaque visibility on caustic receivers."""
        if self.caustic_plane is not None:
            selected = self.caustic_plane.receiver_face_mask[receiver_face]
            if bool(selected.any()):
                value = torch.ones_like(points)
                value[selected] = self.caustic_plane.opaque_visibility(
                    points[selected], directions[selected], distance[selected], receiver_face[selected])
                if bool((~selected).any()):
                    value[~selected] = self.shadow(points[~selected], directions[~selected],
                                                   distance[~selected], receiver_face[~selected])
                return value
        n = len(points)
        result = torch.ones_like(points)
        if not n:
            return result
        active = torch.arange(n, device=points.device)
        origin, direction = points, directions
        remaining = distance - _epsilon(points) * 2
        medium = torch.full((n,), -1, dtype=torch.long, device=points.device)
        previous = receiver_face
        weight = torch.ones_like(points)
        # A near-coincident source has no positive segment to test.
        keep = remaining > _epsilon(origin)
        active, origin, direction = active[keep], origin[keep], direction[keep]
        remaining, medium, previous, weight = remaining[keep], medium[keep], previous[keep], weight[keep]
        for _ in range(MAX_SHADOW_INTERFACES):
            if not len(active):
                break
            hit = self.query.intersect(origin, direction, t_min=_epsilon(origin),
                                       t_max=remaining, ignore_faces=previous)
            self.stats['shadow_query_rays'] += len(active)
            missed = ~hit.hit
            if bool((missed & (medium >= 0)).any()):
                raise NotImplementedError('a direct-light segment ended inside glass; lights and opaque endpoints must be in air')
            if bool(missed.any()):
                result[active[missed]] = weight[missed]
            live = hit.hit
            if not bool(live.any()):
                active = active[:0]
                break
            active, origin, direction = active[live], origin[live], direction[live]
            remaining, medium, weight = remaining[live], medium[live], weight[live]
            face, step = hit.face_index[live], hit.distance[live]
            geometric = hit.normal[live]
            weight = self.attenuate(weight, medium, step)
            glass = self.glass_mask(face)
            if bool((~glass & (medium >= 0)).any()):
                raise NotImplementedError('opaque geometry embedded inside a dielectric volume is not admitted')
            if bool((~glass).any()):
                result[active[~glass]] = 0
            if not bool(glass.any()):
                active = active[:0]
                break
            active, origin, direction = active[glass], origin[glass], direction[glass]
            remaining, medium, weight = remaining[glass], medium[glass], weight[glass]
            face, step, geometric = face[glass], step[glass], geometric[glass]
            optical, next_medium = self.crossing(face, direction, geometric, medium)
            weight *= (1 - optical.fresnel)[:, None]
            self.stats['shadow_glass_interfaces'] += len(active)
            origin = origin + step[:, None] * direction
            remaining = remaining - step
            medium, previous = next_medium, face
            ended_in_glass = ((remaining <= _epsilon(origin)) & (medium >= 0)
                              & (weight.amax(-1) > 0))
            if bool(ended_in_glass.any()):
                raise NotImplementedError('a direct-light endpoint lies inside glass; emitters must be in air')
            live = (weight.amax(-1) > 0) & (remaining > _epsilon(origin))
            if bool((~live).any()):
                result[active[~live]] = weight[~live]
            active, origin, direction = active[live], origin[live], direction[live]
            remaining, medium, previous, weight = remaining[live], medium[live], previous[live], weight[live]
        if len(active):
            self.stats['shadow_paths_truncated'] += len(active)
            result[active] = 0
        return result

    def shade(self, face, barycentric, points, direction, geometric):
        """Constant opaque BRDFs, sampled direct sources and optional image IBL."""
        value = torch.zeros_like(points)
        material_ids = self.packed.face_material[face]
        normal = geometric
        attribute = self.packed.attributes.get('normals')
        if attribute is not None:
            smooth = (attribute.values[face] * barycentric[..., None]).sum(1)
            usable = attribute.present[face] & (torch.linalg.vector_norm(smooth, dim=-1) > 0)
            smooth = unit(smooth)
            usable &= (smooth * geometric).sum(-1) > 0
            normal = torch.where(usable[:, None], smooth, geometric)
        view = -direction
        # Prevent interpolated normals from lighting a geometrically back face.
        front = (geometric * view).sum(-1) > 0
        for number, mat in enumerate(self.packed.materials):
            if mat.evaluator is dielectric:
                continue
            selected = (material_ids == number) & front
            if not bool(selected.any()):
                continue
            p, n, g, v = points[selected], normal[selected], geometric[selected], view[selected]
            local_face = face[selected]
            n = torch.where(((n * v).sum(-1) > 0)[:, None], n, g)
            colour = torch.zeros_like(p)
            params = dict(mat.parameters)
            for source in self.sources:
                if source['kind'] == 'sun':
                    incoming_direction = source['direction'].expand(len(p), 3)
                    incoming = source['irradiance'].expand(len(p), 3)
                    distance = p.new_full((len(p),), float('inf'))
                else:
                    delta = source['position'] - p
                    distance = torch.linalg.vector_norm(delta, dim=-1)
                    incoming_direction = unit(delta)
                    incoming = source['intensity'] / distance.square().clamp_min(1e-12)[:, None]
                    if 'normal' in source:
                        incoming = incoming * (-(incoming_direction * source['normal']).sum(-1)).clamp_min(0)[:, None]
                    if source['kind'] == 'spot':
                        cosine = -(incoming_direction * source['direction']).sum(-1)
                        outer = math.cos(source['angle'] / 2)
                        inner = outer + (1 - outer) * source['blend']
                        t = (((cosine - outer) / max(inner - outer, 1e-12)).clamp(0, 1)
                             if source['blend'] else (cosine >= outer).to(p.dtype))
                        incoming = incoming * (t * t * (3 - 2 * t))[:, None]
                positive = ((n * incoming_direction).sum(-1) > 0) & (incoming.amax(-1) > 0)
                if not bool(positive.any()):
                    continue
                visibility = torch.ones_like(p)
                if self.shadows:
                    visibility[positive] = self.shadow(p[positive], incoming_direction[positive],
                                                       distance[positive], local_face[positive])
                evaluator = pbr if mat.evaluator is pbr else material.classic
                brdf = evaluator(n, v, incoming_direction[:, None], _geometric_normals=g, **params)[:, 0]
                colour += brdf * incoming * visibility * (n * incoming_direction).sum(-1).clamp_min(0)[:, None]
            if self.caustic_plane is not None:
                receiving = self.caustic_plane.receiver_face_mask[local_face]
                if bool(receiving.any()):
                    colour[receiving] += self.caustic_plane.gather(
                        p[receiving], n[receiving], g[receiving], v[receiving], mat)
            if self.environment is not None:
                from ._raster_lighting import surface
                colour += surface(p, n, v, params, environment=self.environment,
                                  probes=(), irradiance=None,
                                  classic=mat.evaluator is material.classic)
            if mat.emission is not None:
                colour += mat.emission
            value[selected] = colour
        return value

    def trace(self, origins, directions, near, far):
        """Trace one primary block; all scratch scales with this bounded block."""
        count = len(origins)
        colour = torch.zeros_like(origins)
        discarded = torch.zeros_like(origins)
        active = torch.arange(count, device=origins.device)
        origin, direction = origins, directions
        weight = torch.ones_like(origins)
        medium = torch.full((count,), -1, dtype=torch.long, device=origins.device)
        previous = torch.full_like(medium, -1)
        lo, hi = near, far
        primary_face = torch.full_like(medium, -1)
        primary_distance = origins.new_full((count,), float('inf'))
        max_active = max(256, count * ACTIVE_PATHS_PER_PRIMARY_BUDGET)
        for depth in range(MAX_INTERFACE_EVENTS + 1):
            if not len(active):
                break
            self.stats['maximum_active_paths'] = max(self.stats['maximum_active_paths'], len(active))
            self.stats['camera_query_rays'] += len(active)
            hit = self.query.intersect(origin, direction, t_min=lo, t_max=hi, ignore_faces=previous)
            if depth == 0:
                primary_face = hit.face_index
                primary_distance = hit.distance
            missed = ~hit.hit
            if bool((missed & (medium >= 0)).any()):
                raise ValueError('a ray escaped while still inside glass; check closure, winding and geometric scale')
            if bool(missed.any()):
                colour.index_add_(0, active[missed], weight[missed] * self.miss(direction[missed]))
                self.stats['camera_misses_in_air'] += int(missed.sum())
            live = hit.hit
            if not bool(live.any()):
                active = active[:0]
                break
            active, origin, direction = active[live], origin[live], direction[live]
            weight, medium = weight[live], medium[live]
            face, distance = hit.face_index[live], hit.distance[live]
            geometric, barycentric = hit.normal[live], hit.barycentric[live]
            weight = self.attenuate(weight, medium, distance)
            point = origin + distance[:, None] * direction
            glass = self.glass_mask(face)
            if bool((~glass & (medium >= 0)).any()):
                raise NotImplementedError('opaque/emissive endpoints inside glass require radiance medium transport and are not admitted')
            opaque = ~glass
            if bool(opaque.any()):
                radiance = self.shade(face[opaque], barycentric[opaque], point[opaque],
                                      direction[opaque], geometric[opaque])
                colour.index_add_(0, active[opaque], weight[opaque] * radiance)
                self.stats['opaque_endpoint_paths'] += int(opaque.sum())
            if not bool(glass.any()):
                active = active[:0]
                break
            active, point, direction = active[glass], point[glass], direction[glass]
            weight, medium = weight[glass], medium[glass]
            face, geometric = face[glass], geometric[glass]
            if depth == MAX_INTERFACE_EVENTS:
                discarded.index_add_(0, active, weight)
                self.stats['depth_limited_paths'] += len(active)
                active = active[:0]
                break
            optical, next_medium = self.crossing(face, direction, geometric, medium)
            self.stats['camera_glass_interfaces'] += len(active)
            self.stats['total_internal_reflection_events'] += int(optical.tir.sum())
            reflected_weight = weight * optical.fresnel[:, None]
            transmitted_weight = weight * (1 - optical.fresnel)[:, None]
            child_active = torch.cat((active, active))
            child_origin = torch.cat((point, point))
            child_direction = torch.cat((optical.reflected, optical.transmitted))
            child_weight = torch.cat((reflected_weight, transmitted_weight))
            child_medium = torch.cat((medium, next_medium))
            child_previous = torch.cat((face, face))
            positive = child_weight.amax(-1) > 0
            retained = child_weight.amax(-1) >= MIN_PATH_WEIGHT
            tiny = positive & ~retained
            if bool(tiny.any()):
                discarded.index_add_(0, child_active[tiny], child_weight[tiny])
                self.stats['weight_limited_paths'] += int(tiny.sum())
            active, origin = child_active[retained], child_origin[retained]
            direction, weight = child_direction[retained], child_weight[retained]
            medium, previous = child_medium[retained], child_previous[retained]
            if len(active) > max_active:
                # Stable ordering makes equal-weight selection deterministic.
                order = torch.argsort(weight.amax(-1), descending=True, stable=True)
                drop, keep = order[max_active:], order[:max_active]
                discarded.index_add_(0, active[drop], weight[drop])
                self.stats['capacity_limited_paths'] += len(drop)
                active, origin, direction = active[keep], origin[keep], direction[keep]
                weight, medium, previous = weight[keep], medium[keep], previous[keep]
            lo, hi = _epsilon(origin), float('inf')
        return colour, primary_face, primary_distance, discarded


def _primary_rays(camera, pixels, dx, dy):
    width = camera.image_width
    # Integer coordinates plus a Python float otherwise choose the default
    # float dtype before a zero-dimensional float64 intrinsic can promote it.
    x = ((pixels % width).to(camera.fx.dtype) + .5 - dx - camera.cx[0]) / camera.fx[0]
    y = -((pixels // width).to(camera.fy.dtype) + .5 - dy - camera.cy[0]) / camera.fy[0]
    linear = camera.pose[0, :3, :3]
    position = camera.pose[0, :3, 3]
    if camera.projection_kind == 'perspective':
        local = torch.stack((x, y, -torch.ones_like(x)), -1)
        world = local @ linear.T
        scale = torch.linalg.vector_norm(world, dim=-1)
        origin = position.expand(len(pixels), 3)
    elif camera.projection_kind == 'orthographic':
        local = torch.stack((x, y, torch.zeros_like(x)), -1)
        origin = local @ linear.T + position
        world = (-linear[:, 2]).expand(len(pixels), 3)
        scale = torch.linalg.vector_norm(world, dim=-1)
    else:
        raise NotImplementedError('dielectric camera projection must be perspective or orthographic')
    direction = world / scale[:, None]
    return origin, direction, scale * camera.near, scale * camera.far


def _primary_ray_batch(camera, start, stop, dx, dy, generation_chunk):
    """Combine original camera-ray groups without changing their arithmetic.

    Matrix multiplication may round differently for a different row count.
    Transport blocks begin on generation-chunk boundaries, so each call below
    retains the exact pixel group used before transport batching was enlarged.
    """
    parts = [
        _primary_rays(camera, torch.arange(first, min(stop, first + generation_chunk),
                                          device=camera.pose.device), dx, dy)
        for first in range(start, stop, generation_chunk)
    ]
    if len(parts) == 1:
        return parts[0]
    return tuple(torch.cat(values) for values in zip(*parts))


def render(scene, camera, *, supersample=1, shadows=False, shadow_map_size=512,
           background=None, outputs=('colour', 'mask'), fragment_block=262144,
           progress=None, environment=None, probes=(), irradiance=None, mirrors=(),
           caustics=None, _context=None, _clip_plane=None):
    """Forward dielectric candidate through the ordinary raster scene API.

    ``outputs=('colour','mask','transmission')`` exposes transport diagnostics.
    Primary mask is geometric coverage (glass counts as covered). Optional
    depth/face_index describe the centre primary ray, not a refracted endpoint.
    No-glass callers are routed to the original raster implementation.
    """
    if type(supersample) is not int or not 1 <= supersample <= 16:
        raise ValueError('supersample must be an integer in [1,16]')
    if type(fragment_block) is not int or fragment_block < 1:
        raise ValueError('fragment_block must be positive')
    if type(shadows) is not bool:
        raise TypeError('shadows must be boolean')
    if type(shadow_map_size) is not int or not 16 <= shadow_map_size <= 4096:
        raise ValueError('shadow_map_size must be in [16,4096]')
    if set(outputs) - {'colour', 'mask', 'depth', 'face_index', 'timings', 'transmission', 'caustics'}:
        raise ValueError('unsupported dielectric raster output')
    if caustics is None and 'caustics' in outputs:
        raise ValueError('caustics evidence output requires an enabled caustics configuration')
    if probes or irradiance is not None or mirrors or _clip_plane is not None:
        raise NotImplementedError('initial dielectric transport does not admit probes, irradiance volumes, planar mirrors or clip planes')
    from ._area import expand_emitters
    from ._deferred import _prepare
    from ._raster import _light_sources

    started = time.perf_counter()
    packed, camera = _prepare(expand_emitters(scene), camera)
    if camera.batch_size != 1:
        raise ValueError('dielectric raster accepts one frame at a time')
    if camera.fx.dtype != packed.vertices.dtype:
        raise ValueError('dielectric camera precision must match scene precision')
    if not bool((camera.pose[0, 3] == camera.pose.new_tensor([0., 0., 0., 1.])).all()):
        raise ValueError('dielectric camera pose must be affine')
    _validate_materials(packed)
    tracked = [packed.vertices, camera.pose, camera.fx, camera.fy, camera.cx, camera.cy]
    tracked.extend(a.values for a in packed.attributes.values())
    tracked.extend(packed.world_matrices.values())
    for prim in (*packed.materials, *packed.lights):
        tracked.extend(v for v in prim.parameters.values() if isinstance(v, torch.Tensor))
        if getattr(prim, 'emission', None) is not None:
            tracked.append(prim.emission)
    if environment is not None:
        for value in (*environment.levels, environment.irradiance):
            if value.dtype != packed.vertices.dtype or value.device != packed.vertices.device:
                raise ValueError('environment precision/device must match scene')
            tracked.append(value)
    if isinstance(background, torch.Tensor):
        tracked.append(background)
    if torch.is_grad_enabled() and any(t.requires_grad for t in tracked):
        raise ValueError('dielectric raster is forward-only; use torch.no_grad() explicitly')
    if background is not None:
        background = torch.as_tensor(background, dtype=packed.vertices.dtype, device=packed.vertices.device)
        if background.shape != (3,) or not bool(torch.isfinite(background).all()) or bool((background < 0).any()):
            raise ValueError('background must be finite nonnegative RGB')
    stats = dict(
        algorithm='bounded exact triangle dielectric reference',
        optical_normals='geometric triangle normals',
        admitted_media='one closed non-overlapping solid at a time; camera and opaque endpoints in air',
        radiance_eta_factors='telescope to one on completed admitted air-to-air paths',
        direct_shadow_model='straight-ray Fresnel and Beer; no refracted caustics' if shadows else 'disabled',
        mask_semantics='geometric primary coverage including glass',
        query_complexity='bounded-memory O(rays * triangles), no acceleration hierarchy',
        maximum_interface_events=MAX_INTERFACE_EVENTS,
        minimum_path_weight=MIN_PATH_WEIGHT,
        active_paths_per_primary_budget=ACTIVE_PATHS_PER_PRIMARY_BUDGET,
        active_path_capacity_formula='max(256, block_primary_count * 32); shared by the primary block',
        maximum_shadow_interfaces=MAX_SHADOW_INTERFACES,
        camera_query_rays=0, camera_glass_interfaces=0, camera_misses_in_air=0,
        opaque_endpoint_paths=0, total_internal_reflection_events=0,
        depth_limited_paths=0, weight_limited_paths=0, capacity_limited_paths=0,
        maximum_active_paths=0, shadow_query_rays=0, shadow_glass_interfaces=0,
        shadow_paths_truncated=0,
        discarded_weight_interpretation='sum of omitted RGB path weights per primary sample; not a universal radiance-error bound',
    )
    with torch.no_grad():
        generation_chunk = min(fragment_block, 256 if packed.vertices.is_cuda else 64)
        transport_limit = min(fragment_block, 2048 if packed.vertices.is_cuda else 64)
        # Whole original ray groups preserve their camera matrix arithmetic.
        # fragment_block remains an upper bound, including nonmultiples of 256.
        query_chunk = (transport_limit // generation_chunk) * generation_chunk
        stats.update(primary_ray_generation_chunk=generation_chunk,
                     transport_ray_chunk=query_chunk, query_face_chunk=256)
        sources = _light_sources(packed, packed.vertices, packed.faces, shadow_map_size, False, None)
        transport = _Transport(packed, camera, sources=sources, background=background,
                               environment=environment, shadows=shadows,
                               query_chunk=query_chunk, diagnostics=stats)
        if caustics is not None:
            from ._caustic_plane import build as build_caustics
            transport.caustic_plane = build_caustics(transport, caustics, progress=progress)
            stats['caustics'] = transport.caustic_plane.stats
            stats['direct_shadow_model'] = 'Selected plane: opaque direct plus directional photon caustics; other surfaces: straight-ray Fresnel and Beer'
        if packed.vertices.is_cuda:
            torch.cuda.synchronize(packed.vertices.device)
        preparation_end = time.perf_counter()
        h, w = camera.image_height, camera.image_width
        count = h * w
        colour = packed.vertices.new_zeros((count, 3))
        mask = packed.vertices.new_zeros(count)
        discarded = torch.zeros_like(colour)
        centre_faces = torch.full((count,), -1, dtype=torch.long, device=packed.vertices.device)
        centre_depth = packed.vertices.new_full((count,), float('inf'))
        sample_count = supersample ** 2
        total_blocks = sample_count * ((count + query_chunk - 1) // query_chunk)
        completed = 0
        _notify(progress, stage='dielectric transport', completed=0, total=total_blocks)
        for sample in range(sample_count):
            row, column = divmod(sample, supersample)
            dx = .5 - (column + .5) / supersample
            dy = .5 - (row + .5) / supersample
            for start in range(0, count, query_chunk):
                stop = min(count, start + query_chunk)
                origins, directions, near, far = _primary_ray_batch(
                    camera, start, stop, dx, dy, generation_chunk)
                result, face, distance, lost = transport.trace(origins, directions, near, far)
                colour[start:stop] += result / sample_count
                mask[start:stop] += (face >= 0).to(mask.dtype) / sample_count
                discarded[start:stop] += lost / sample_count
                if supersample == 1 and set(outputs) & {'depth', 'face_index'}:
                    centre_faces[start:stop] = face
                    for first in range(0, stop - start, generation_chunk):
                        block = slice(first, first + generation_chunk)
                        step = distance[block]
                        points = origins[block] + torch.where(torch.isfinite(step), step, 0.)[:, None] * directions[block]
                        view = camera.world_to_view()[0]
                        view_depth = -(points @ view[:3, :3].T + view[:3, 3])[:, 2]
                        centre_depth[start + first:min(stop, start + first + generation_chunk)] = torch.where(
                            face[block] >= 0, camera.view_depth_to_ndc(view_depth), float('inf'))
                completed += 1
                if completed == total_blocks or completed % 8 == 0:
                    _notify(progress, stage='dielectric transport', completed=completed,
                            total=total_blocks, elapsed_seconds=time.perf_counter() - started)
        if supersample != 1 and set(outputs) & {'depth', 'face_index'}:
            view = camera.world_to_view()[0]
            for start in range(0, count, generation_chunk):
                stop = min(count, start + generation_chunk)
                pixels = torch.arange(start, stop, device=packed.vertices.device)
                origins, directions, near, far = _primary_rays(camera, pixels, 0., 0.)
                hit = transport.query.intersect(origins, directions, t_min=near, t_max=far)
                centre_faces[start:stop] = hit.face_index
                distance = torch.where(hit.hit, hit.distance, 0.)
                points = origins + distance[:, None] * directions
                view_depth = -(points @ view[:3, :3].T + view[:3, 3])[:, 2]
                centre_depth[start:stop] = torch.where(hit.hit, camera.view_depth_to_ndc(view_depth), float('inf'))
        if not bool(torch.isfinite(colour).all()) or bool((colour < -1e-7).any()):
            raise ValueError('dielectric colour is not finite nonnegative radiance')
        stats['discarded_path_weight_max_rgb'] = discarded.amax(0).detach().cpu().tolist()
        stats['discarded_path_weight_mean_rgb'] = discarded.mean(0).detach().cpu().tolist()
        stats['pixels_with_discarded_weight'] = int((discarded.amax(-1) > 0).sum())
        stats['triangle_count'] = len(packed.faces)
        stats['primary_samples'] = count * sample_count
        if packed.vertices.is_cuda:
            torch.cuda.synchronize(packed.vertices.device)
        ended = time.perf_counter()
        result = dict(colour=colour.reshape(1, h, w, 3), mask=mask.reshape(1, h, w),
                      depth=centre_depth.reshape(1, h, w), face_index=centre_faces.reshape(1, h, w),
                      transmission=stats,
                      timings=dict(preparation_seconds=preparation_end - started,
                                   primary_render_seconds=ended - preparation_end,
                                   total_seconds=ended - started))
        if 'caustics' in outputs:
            result['caustics'] = transport.caustic_plane.evidence()
        _notify(progress, stage='complete', elapsed_seconds=ended - started)
        return {key: result[key] for key in outputs}
