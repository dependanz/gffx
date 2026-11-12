"""Bounded, opt-in glass-to-plane photon transport for the raster glass path.

This is an experimental finite-sample density estimate, not a replacement for
the normal renderer. A selected horizontal receiver uses opaque direct-light
visibility plus light paths whose first hit is glass. Other surfaces keep the
existing straight-segment shadow approximation. The two terms are disjoint.

The full selected rectangle is discretized in world coordinates. Photon power
and incoming directions are retained separately from receiver materials. At
every camera endpoint, including endpoints reached through glass, the existing
BRDF is evaluated for each nearby photon and the current view direction:

    sum_j f_r(x, wi_j, wo) * Phi_j / (cell_area * kernel_size**2).

Phi already measures power crossing the surface. There is no second receiver
cosine. Filtering does not mix albedos, normalize a crop, or renormalize a
truncated boundary kernel. Finite cells and a fixed filter introduce spatial
bias. The ordinary camera path still has its separate depth/weight limits.
"""
from __future__ import annotations

from collections.abc import Mapping
import copy
import hashlib
import math
import time

import torch

from . import light, material
from ._dielectric import TriangleQuery
from ._raster_material import pbr


REVISION = '2-photon-float64-directional-plane-caustics'
TERMINAL_CODES = {
    'initial_miss': 0,
    'receiver': 1,
    'escape': 2,
    'other_surface': 3,
    'depth_limit': 4,
}
_PAIR_BLOCK = 65536


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _seed(seed, purpose, index):
    value = ('gffx-plane-photons-v1:%s:%s:%s' % (seed, purpose, index)).encode('ascii')
    return int.from_bytes(hashlib.sha256(value).digest()[:8], 'little') & ((1 << 63) - 1)


def _sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def _configuration(value):
    if not isinstance(value, Mapping):
        raise TypeError('caustics must be None or a configuration mapping')
    allowed = {'receivers', 'photons', 'seed', 'cell_size', 'kernel_size', 'max_interfaces'}
    _require(not set(value) - allowed, 'unknown caustics configuration field')
    receivers = value.get('receivers')
    _require(isinstance(receivers, (list, tuple)) and len(receivers) > 0
             and all(type(name) is str and name for name in receivers)
             and len(set(receivers)) == len(receivers),
             'caustics.receivers must be a nonempty sequence of distinct mesh names')
    options = dict(receivers=tuple(receivers), photons=value.get('photons', 65536),
                   seed=value.get('seed', 1701), cell_size=value.get('cell_size', .02),
                   kernel_size=value.get('kernel_size', 3),
                   max_interfaces=value.get('max_interfaces', 256))
    _require(type(options['photons']) is int and 16 <= options['photons'] <= 262144,
             'caustics.photons must be an integer in [16,262144]')
    _require(type(options['seed']) is int and 0 <= options['seed'] < 2 ** 31,
             'caustics.seed must be an integer in [0,2^31)')
    size = options['cell_size']
    _require(type(size) in (int, float) and math.isfinite(size) and size > 0,
             'caustics.cell_size must be finite and positive')
    options['cell_size'] = float(size)
    _require(type(options['kernel_size']) is int and options['kernel_size'] in (1, 3),
             'caustics.kernel_size must be 1 or 3')
    _require(type(options['max_interfaces']) is int and 1 <= options['max_interfaces'] <= 256,
             'caustics.max_interfaces must be an integer in [1,256]')
    return options


class PlaneCaustics:
    """One render's immutable photon map plus small mutable gather counters."""

    def __init__(self, transport, options):
        self.transport = transport
        self.options = _configuration(options)
        packed = transport.packed
        self.device, self.dtype = packed.vertices.device, packed.vertices.dtype
        _require(self.device.type in ('cpu', 'cuda') and self.dtype in (torch.float32, torch.float64),
                 'plane caustics require CPU/CUDA float32 or float64 geometry')
        _require(transport.shadows, 'plane caustics require shadows=True')
        _require(transport.environment is None, 'plane caustics do not yet admit environment lighting')
        _require(bool(packed.lights) and all(lamp.evaluator is light.rectangle for lamp in packed.lights),
                 'plane caustics require explicit rectangular lights only')
        _require(len(transport.sources) > 0
                 and all(source.get('kind') == 'point' and 'normal' in source and 'area' in source
                         for source in transport.sources),
                 'plane caustics require the accepted rectangle-light quadrature')
        self._prepare_receiver()
        self._prepare_sources()
        # A valid grazing flight through the authored tumbler rim can be less
        # than the float32 self-hit threshold. Photon positions, intersections
        # and optical directions therefore use float64 from launch onward.
        # Keep the accepted camera state and optical methods unchanged. This
        # shallow transport view only supplies widened IOR/absorption arrays
        # to the same strict owner-transition operation; no shared input is
        # mutated. This is a reference-precision choice, not a speed claim.
        self.photon_dtype = torch.float64
        self.photon_transport = copy.copy(transport)
        self.photon_transport.ior = transport.ior.to(self.photon_dtype)
        self.photon_transport.sigma = transport.sigma.to(self.photon_dtype)
        self.query = TriangleQuery(packed.vertices.to(self.photon_dtype), packed.faces,
                                   ray_chunk=2048, face_chunk=256)
        self.photon_transport.query = self.query
        self.tensors = {}
        self.stats = {
            'schema': 'gffx-plane-caustics-1', 'revision': REVISION,
            'configuration': dict(self.options, receivers=list(self.options['receivers'])),
            'receiver': self.receiver_metadata, 'sources': self.source_metadata,
            'scope': 'Selected horizontal rectangular receiver only; other surfaces retain straight-segment shadows.',
            'partition': 'Opaque direct visibility plus only paths whose first interaction is glass.',
            'receiver_estimator': 'Per-photon existing BRDF at camera endpoint view, times power/(physical cell area * kernel area in cells); no extra receiver cosine.',
            'medium_model': 'One occupied glass owner at a time; disjoint solids and hollow closed boundaries are supported, nested/intersecting solids are rejected when encountered.',
            'spatial_bias': 'Finite world-space cells and fixed box filter; no image/ROI fit or boundary renormalization.',
            'camera_dependence': 'Map stores incoming directions; all opaque camera endpoints use their own view and material, including endpoints seen through glass.',
            'light_transport_scope': 'Direct rectangular emitters followed by zero or more glass reflections/transmissions and one receiver endpoint; no diffuse continuation.',
            'native_acceleration': False,
            'photon_geometry_dtype': 'float64',
            'camera_geometry_dtype': str(self.dtype),
            'precision_policy': 'Photon-only float64 geometry, launch/transport directions and interface state preserve sub-float32-threshold rim flights; accepted camera/core and strict medium validation are unchanged.',
            'gather_calls': 0, 'gather_receiver_endpoints': 0,
            'gather_brdf_pairs': 0, 'maximum_gather_pairs_per_block': 0,
        }

    def _prepare_receiver(self):
        packed = self.transport.packed
        names = list(packed.mesh_names)
        _require(all(name in names for name in self.options['receivers']),
                 'a caustic receiver mesh does not exist')
        _require(len(set(names)) == len(names), 'caustic receiver lookup requires unique packed mesh names')
        offsets = packed.object_face_offsets[0].detach().cpu().tolist()
        vertices = packed.vertices.detach().cpu().double()
        faces = packed.faces.detach().cpu().long()
        face_material = packed.face_material.detach().cpu().long()
        mask = torch.zeros(len(faces), dtype=torch.bool)
        bounds, levels, areas = [], [], []
        scale = max(1., float(vertices.abs().max())) if len(vertices) else 1.
        tolerance = max(2e-6, 32 * torch.finfo(self.dtype).eps * scale)
        self.geometry_tolerance = tolerance
        normal_attribute = packed.attributes.get('normals')
        for name in self.options['receivers']:
            owner = names.index(name)
            first, last = offsets[owner:owner + 2]
            local_faces = faces[first:last]
            _require(len(local_faces) == 2, 'each caustic receiver mesh must be one two-triangle rectangle: ' + name)
            unique = torch.unique(local_faces.reshape(-1))
            _require(len(unique) == 4, 'each caustic receiver rectangle must have four shared corners: ' + name)
            points = vertices[unique]
            lower, upper = points.amin(0), points.amax(0)
            _require(float(upper[1] - lower[1]) <= tolerance,
                     'caustic receiver rectangles must be horizontal: ' + name)
            _require(float(upper[0] - lower[0]) > tolerance
                     and float(upper[2] - lower[2]) > tolerance,
                     'caustic receiver rectangle is degenerate: ' + name)
            # Each coordinate belongs to its low or high side, and every corner
            # appears once. This rejects sheared quads without relying on area.
            bits = []
            for point in points:
                corner = []
                for axis in (0, 2):
                    dl, dh = abs(float(point[axis] - lower[axis])), abs(float(point[axis] - upper[axis]))
                    _require(min(dl, dh) <= tolerance, 'caustic receiver is not axis aligned: ' + name)
                    corner.append(0 if dl < dh else 1)
                bits.append(tuple(corner))
            _require(set(bits) == {(0, 0), (0, 1), (1, 0), (1, 1)},
                     'caustic receiver has repeated rectangle corners: ' + name)
            shared = set(local_faces[0].tolist()) & set(local_faces[1].tolist())
            _require(len(shared) == 2, 'caustic receiver triangles must share a diagonal: ' + name)
            diagonal = vertices[torch.tensor(sorted(shared), dtype=torch.long)]
            _require(abs(float(diagonal[1, 0] - diagonal[0, 0])) > tolerance
                     and abs(float(diagonal[1, 2] - diagonal[0, 2])) > tolerance,
                     'caustic receiver triangles must share the rectangle diagonal: ' + name)
            triangles = vertices[local_faces]
            cross = torch.linalg.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
            norm = torch.linalg.vector_norm(cross, dim=-1)
            _require(bool((norm > 0).all()) and bool((cross[:, 1] > 0).all())
                     and float((cross[:, [0, 2]] / norm[:, None]).abs().max()) <= 1e-5,
                     'caustic receiver triangles must face +Y: ' + name)
            rectangle_area = float((upper[0] - lower[0]) * (upper[2] - lower[2]))
            _require(abs(float(norm.sum()) * .5 - rectangle_area) <= tolerance * max(1., rectangle_area),
                     'caustic receiver triangles do not cover their rectangle: ' + name)
            mids = face_material[first:last]
            _require(bool((mids >= 0).all()) and not bool(self.transport.is_glass[mids.to(self.device)].any()),
                     'caustic receivers must have opaque materials: ' + name)
            _require(all(packed.materials[int(mid)].evaluator in (pbr, material.classic) for mid in mids),
                     'caustic receivers require the existing PBR or classic BRDF')
            if normal_attribute is not None:
                present = normal_attribute.present[first:last].detach().cpu()
                values = normal_attribute.values[first:last].detach().cpu().double()[present]
                if values.numel():
                    lengths = torch.linalg.vector_norm(values, dim=-1)
                    _require(bool((lengths > 0).all()), 'caustic receiver normals must be nonzero')
                    normals = values / lengths[..., None]
                    _require(float((normals - torch.tensor([0., 1., 0.])).abs().max()) <= 1e-5,
                             'caustic receiver shading normals must match the plane')
            mask[first:last] = True
            levels.append(float(points[:, 1].mean()))
            bounds.append((float(lower[0]), float(upper[0]), float(lower[2]), float(upper[2])))
            areas.append(rectangle_area)
        _require(max(levels) - min(levels) <= tolerance, 'caustic receivers must be coplanar')
        lower_x, upper_x = min(b[0] for b in bounds), max(b[1] for b in bounds)
        lower_z, upper_z = min(b[2] for b in bounds), max(b[3] for b in bounds)
        for i, a in enumerate(bounds):
            for b in bounds[i + 1:]:
                overlap_x = min(a[1], b[1]) - max(a[0], b[0])
                overlap_z = min(a[3], b[3]) - max(a[2], b[2])
                _require(not (overlap_x > tolerance and overlap_z > tolerance),
                         'caustic receiver rectangles overlap')
        area = (upper_x - lower_x) * (upper_z - lower_z)
        _require(abs(sum(areas) - area) <= tolerance * max(1., area),
                 'caustic receivers must cover one full rectangle without holes')
        target = self.options['cell_size']
        nx, nz = round((upper_x - lower_x) / target), round((upper_z - lower_z) / target)
        _require(nx > 0 and nz > 0 and nx * nz <= 1_048_576,
                 'caustic receiver grid must contain between 1 and 1048576 cells')
        _require(abs((upper_x - lower_x) - nx * target) <= tolerance
                 and abs((upper_z - lower_z) - nz * target) <= tolerance,
                 'caustic cell_size must divide the full receiver extents')
        self.receiver_face_mask = mask.to(self.device)
        self.floor_y = sum(levels) / len(levels)
        self.floor_bounds = (lower_x, upper_x, lower_z, upper_z)
        self.nx, self.nz = nx, nz
        self.dx, self.dz = (upper_x - lower_x) / nx, (upper_z - lower_z) / nz
        self.cell_area = self.dx * self.dz
        self.receiver_metadata = {
            'meshes': list(self.options['receivers']), 'faces': int(mask.sum()),
            'plane_y': self.floor_y, 'normal': [0., 1., 0.],
            'bounds_xz': list(self.floor_bounds), 'grid_shape': [nz, nx],
            'cell_width_x': self.dx, 'cell_width_z': self.dz, 'cell_area': self.cell_area,
            'requested_cell_size': target, 'physical_rectangle_area': area,
            'sum_mesh_rectangle_areas': sum(areas), 'geometry_tolerance': tolerance,
            'grid_material_policy': 'Store incident photon power/direction, not shaded colour; evaluate the receiving endpoint material after gathering.',
        }

    def _prepare_sources(self):
        packed = self.transport.packed
        glass_faces = self.transport.glass_mask(torch.arange(len(packed.faces), device=self.device))
        _require(bool(glass_faces.any()), 'plane caustics require at least one active dielectric mesh')
        vertex_ids = torch.unique(packed.faces[glass_faces].long().reshape(-1))
        glass_vertices = packed.vertices[vertex_ids].detach().cpu().double()
        centre = (glass_vertices.amin(0) + glass_vertices.amax(0)) * .5
        exact_radius = torch.linalg.vector_norm(glass_vertices - centre, dim=-1).max()
        radius = float(exact_radius) * (1 + 64 * torch.finfo(self.dtype).eps) + 2e-6
        sources = self.transport.sources
        count = len(sources)
        _require(self.options['photons'] % count == 0,
                 'caustics.photons must be divisible by the number of accepted emitter quadrature samples')
        position = torch.stack([s['position'].detach().cpu() for s in sources])
        normal = torch.stack([s['normal'].detach().cpu().double() for s in sources])
        intensity = torch.stack([s['intensity'].detach().cpu().double() for s in sources])
        _require(bool(torch.isfinite(position).all()) and bool(torch.isfinite(normal).all())
                 and bool(torch.isfinite(intensity).all()) and bool((intensity >= 0).all()),
                 'caustic emitter quadrature must be finite with nonnegative intensity')
        delta = centre - position.double()
        distance = torch.linalg.vector_norm(delta, dim=-1)
        axis = normal.clone()
        cosine = torch.zeros(count, dtype=torch.float64)
        omega = torch.full((count,), 2 * math.pi, dtype=torch.float64)
        integral = torch.full((count,), math.pi, dtype=torch.float64)
        modes, minimum_cosines = [], []
        for i in range(count):
            mode = 'full_emitting_hemisphere'
            minimum = 0.
            if float(distance[i]) > radius:
                cap_axis = delta[i] / distance[i]
                sine = radius / float(distance[i])
                cap_cosine = math.sqrt(max(0., 1 - sine * sine))
                ndot = float((normal[i] * cap_axis).sum())
                minimum = ndot * cap_cosine - math.sqrt(max(0., 1 - ndot * ndot)) * sine
                if minimum > 0:
                    axis[i] = cap_axis
                    cosine[i] = cap_cosine
                    omega[i] = 2 * math.pi * (1 - cap_cosine)
                    integral[i] = math.pi * sine * sine * ndot
                    mode = 'union_glass_bound_cap'
            modes.append(mode)
            minimum_cosines.append(minimum if mode == 'union_glass_bound_cap' else 0.)
        self.source_count = count
        self.source_position_cpu, self.source_normal_cpu = position, normal
        self.source_intensity_cpu = intensity
        self.axis_cpu, self.cosine_cpu, self.omega_cpu = axis, cosine, omega
        self.analytic_launch_flux = (intensity * integral[:, None]).sum(0)
        self.source_metadata = {
            'sample_count': count, 'quadrature': 'Unmodified accepted _light_sources output; equal packet counts per source sample.',
            'positions': position.tolist(), 'normals': normal.tolist(),
            'sample_intensity_rgb': intensity.tolist(),
            'union_glass_bound_center': centre.tolist(), 'union_glass_bound_radius': radius,
            'union_glass_vertex_radius_before_padding': float(exact_radius),
            'proposal_modes': modes, 'proposal_axis': axis.tolist(),
            'proposal_cos_theta_max': cosine.tolist(), 'proposal_solid_angle': omega.tolist(),
            'minimum_emission_cosine': minimum_cosines,
            'analytic_launched_flux_rgb': self.analytic_launch_flux.tolist(),
            'normalization': 'intensity_s * source_count * solid_angle_s * emitter_cosine / all_launches; misses remain in the denominator.',
            'first_hit_policy': 'Accept any glass first hit in the single union proposal; no per-object duplicate proposals.',
        }

    def _launch(self):
        count, seed, sources = self.options['photons'], self.options['seed'], self.source_count
        samples = torch.empty((count, 2), dtype=torch.float64)
        for i in range(sources):
            engine = torch.quasirandom.SobolEngine(2, scramble=True,
                                                   seed=_seed(seed, 'direction', i) % (2 ** 31))
            samples[i::sources] = engine.draw(count // sources, dtype=torch.float64)
        source_ids = torch.arange(count) % sources
        axes = self.axis_cpu[source_ids]
        helper = torch.zeros_like(axes)
        helper[:, 1] = 1.
        helper[axes[:, 1].abs() > .9] = torch.tensor([1., 0., 0.], dtype=torch.float64)
        tangent = torch.linalg.cross(helper, axes)
        tangent /= torch.linalg.vector_norm(tangent, dim=-1)[:, None]
        bitangent = torch.linalg.cross(axes, tangent)
        cosine = 1 - samples[:, 0] * (1 - self.cosine_cpu[source_ids])
        sine = (1 - cosine.square()).clamp_min(0).sqrt()
        azimuth = 2 * math.pi * samples[:, 1]
        direction = (axes * cosine[:, None] + tangent * (sine * azimuth.cos())[:, None]
                     + bitangent * (sine * azimuth.sin())[:, None]).to(self.photon_dtype)
        direction /= torch.linalg.vector_norm(direction, dim=-1)[:, None]
        position = self.source_position_cpu[source_ids].to(self.photon_dtype).clone()
        emission_cosine = (direction.double() * self.source_normal_cpu[source_ids]).sum(-1).clamp_min(0)
        flux = (self.source_intensity_cpu[source_ids] * sources
                * self.omega_cpu[source_ids, None] * emission_cosine[:, None] / count)
        _require(bool(torch.isfinite(flux).all()) and bool((flux >= 0).all()), 'invalid photon launch power')
        return source_ids, position, direction, flux

    @torch.no_grad()
    def trace(self, *, progress=None, launch_chunk=None):
        """Trace one roulette packet per launch with owner-specific Beer optical depth.

        ``launch_chunk`` is private validation control. Random streams use the
        original photon ID and interface number, independently of compaction.
        """
        from ._dielectric_render import _epsilon
        started = time.perf_counter()
        count, seed = self.options['photons'], self.options['seed']
        maximum = self.options['max_interfaces']
        chunk = 2048 if launch_chunk is None else launch_chunk
        _require(type(chunk) is int and 1 <= chunk <= 4096, 'photon launch chunk must be in [1,4096]')
        source_ids, positions, directions, launch_flux = self._launch()
        terminal = torch.full((count,), -1, dtype=torch.long)
        terminal_medium = torch.full((count,), -1, dtype=torch.long)
        terminal_optical = torch.zeros((count, 3), dtype=torch.float64)
        terminal_distance = torch.zeros(count, dtype=torch.float64)
        first_face = torch.full((count,), -1, dtype=torch.long)
        first_owner = torch.full((count,), -1, dtype=torch.long)
        events_all = torch.zeros(count, dtype=torch.long)
        receiver_position = torch.zeros((count, 3), dtype=self.photon_dtype)
        receiver_incoming = torch.zeros((count, 3), dtype=self.photon_dtype)
        receiver_face = torch.full((count,), -1, dtype=torch.long)
        counters = dict(query_rays=0, glass_interfaces=0, reflection_events=0,
                        transmission_events=0, total_internal_reflection_events=0,
                        maximum_active_packets=0)
        random_streams = {}
        sigma = self.transport.sigma.double()
        def finish(ids, code, medium, optical, distance, events, point=None, incoming=None, face=None):
            cpu_ids = ids.cpu()
            _require(bool((terminal[cpu_ids] == -1).all()), 'a photon was terminated twice')
            terminal[cpu_ids] = code
            terminal_medium[cpu_ids] = medium.cpu()
            terminal_optical[cpu_ids] = optical.cpu()
            terminal_distance[cpu_ids] = distance.cpu()
            events_all[cpu_ids] = events.cpu()
            if point is not None:
                receiver_position[cpu_ids] = point.cpu()
                receiver_incoming[cpu_ids] = incoming.cpu()
                receiver_face[cpu_ids] = face.cpu()
        def draws(event, ids):
            if event not in random_streams:
                generator = torch.Generator(device='cpu').manual_seed(_seed(seed, 'fresnel', event))
                random_streams[event] = torch.rand(count, generator=generator, dtype=torch.float32).to(self.device)
            return random_streams[event][ids]
        _sync(self.device)
        for begin in range(0, count, chunk):
            end = min(begin + chunk, count)
            ids = torch.arange(begin, end, device=self.device)
            origin = positions[begin:end].to(self.device)
            direction = directions[begin:end].to(self.device)
            hit = self.query.intersect(origin, direction, t_min=_epsilon(origin))
            counters['query_rays'] += len(ids)
            first_face[begin:end] = hit.face_index.cpu()
            glass = hit.hit & self.transport.glass_mask(hit.face_index.clamp_min(0))
            terminal[ids[~glass].cpu()] = TERMINAL_CODES['initial_miss']
            first_owner[ids[glass].cpu()] = self.transport.face_owner[hit.face_index[glass]].cpu()
            ids, origin, direction = ids[glass], origin[glass], direction[glass]
            origin = origin + hit.distance[glass, None] * direction
            face, geometric = hit.face_index[glass], hit.normal[glass]
            medium = torch.full((len(ids),), -1, dtype=torch.long, device=self.device)
            optical = torch.zeros((len(ids), 3), dtype=torch.float64, device=self.device)
            glass_distance = torch.zeros(len(ids), dtype=torch.float64, device=self.device)
            events = torch.zeros(len(ids), dtype=torch.long, device=self.device)
            for event in range(maximum):
                if not len(ids):
                    break
                counters['maximum_active_packets'] = max(counters['maximum_active_packets'], len(ids))
                interface, transmitted_medium = self.photon_transport.crossing(face, direction, geometric, medium)
                reflect = draws(event, ids) < interface.fresnel
                _require(not bool((interface.tir & ~reflect).any()), 'a TIR packet was transmitted')
                direction = torch.where(reflect[:, None], interface.reflected, interface.transmitted)
                medium = torch.where(reflect, medium, transmitted_medium)
                events += 1
                counters['glass_interfaces'] += len(ids)
                counters['reflection_events'] += int(reflect.sum())
                counters['transmission_events'] += int((~reflect).sum())
                counters['total_internal_reflection_events'] += int(interface.tir.sum())
                hit = self.query.intersect(origin, direction, t_min=_epsilon(origin), ignore_faces=face)
                counters['query_rays'] += len(ids)
                missed = ~hit.hit
                _require(not bool((missed & (medium >= 0)).any()), 'a caustic photon escaped while inside glass')
                if bool(missed.any()):
                    finish(ids[missed], TERMINAL_CODES['escape'], medium[missed], optical[missed],
                           glass_distance[missed], events[missed])
                live = hit.hit
                if not bool(live.any()):
                    ids = ids[:0]
                    break
                ids, origin, direction = ids[live], origin[live], direction[live]
                medium, optical, glass_distance, events = medium[live], optical[live], glass_distance[live], events[live]
                step = hit.distance[live]
                inside = medium >= 0
                if bool(inside.any()):
                    optical[inside] += sigma[medium[inside]] * step[inside, None].double()
                    glass_distance[inside] += step[inside].double()
                origin = origin + step[:, None] * direction
                face, geometric = hit.face_index[live], hit.normal[live]
                glass = self.transport.glass_mask(face)
                _require(not bool((~glass & (medium >= 0)).any()), 'a caustic photon ended on opaque geometry inside glass')
                receiver = self.receiver_face_mask[face]
                if bool(receiver.any()):
                    _require(bool(((direction[receiver] * geometric[receiver]).sum(-1) < 0).all()),
                             'a caustic photon reached the back of the receiver')
                    finish(ids[receiver], TERMINAL_CODES['receiver'], medium[receiver], optical[receiver],
                           glass_distance[receiver], events[receiver], origin[receiver], -direction[receiver], face[receiver])
                other = ~glass & ~receiver
                if bool(other.any()):
                    finish(ids[other], TERMINAL_CODES['other_surface'], medium[other], optical[other],
                           glass_distance[other], events[other])
                ids, origin, direction = ids[glass], origin[glass], direction[glass]
                medium, optical, glass_distance, events = medium[glass], optical[glass], glass_distance[glass], events[glass]
                face, geometric = face[glass], geometric[glass]
            if len(ids):
                finish(ids, TERMINAL_CODES['depth_limit'], medium, optical, glass_distance, events)
            if progress is not None:
                progress(dict(stage='caustic photon transport', completed=end, total=count,
                              elapsed_seconds=time.perf_counter() - started))
        _sync(self.device)
        _require(bool((terminal >= 0).all()), 'an emitted caustic packet has no terminal account')
        _require(bool((terminal_medium[terminal != TERMINAL_CODES['depth_limit']] == -1).all()),
                 'a completed caustic path remains in glass')
        _require(bool(torch.isfinite(terminal_optical).all()) and bool((terminal_optical >= 0).all()),
                 'invalid caustic optical depth')
        terminal_flux = launch_flux * torch.exp(-terminal_optical)
        receiver_ids = torch.nonzero(terminal == TERMINAL_CODES['receiver'], as_tuple=False).flatten()
        self.tensors = dict(
            launch_source_id=source_ids, launch_position=positions, launch_direction=directions,
            launch_flux=launch_flux, first_face_index=first_face, first_glass_owner=first_owner,
            terminal_code=terminal, terminal_medium=terminal_medium,
            terminal_optical_depth=terminal_optical, terminal_glass_distance=terminal_distance,
            terminal_flux=terminal_flux, interface_count=events_all,
            receiver_photon_id=receiver_ids, receiver_position=receiver_position[receiver_ids],
            receiver_incoming=receiver_incoming[receiver_ids], receiver_face_index=receiver_face[receiver_ids],
            receiver_flux=terminal_flux[receiver_ids],
        )
        self.stats.update(photons=count, seed=seed, launch_chunk=chunk,
                          ray_query_chunk=self.query.ray_chunk,
                          query_face_chunk=self.query.face_chunk,
                          terminal_codes=dict(TERMINAL_CODES),
                          terminal_counts={k: int((terminal == code).sum()) for k, code in TERMINAL_CODES.items()},
                          first_glass_packets=int((first_owner >= 0).sum()),
                          first_glass_owner_counts={self.transport.packed.mesh_names[int(owner)]: int((first_owner == owner).sum())
                                                    for owner in torch.unique(first_owner[first_owner >= 0]).tolist()},
                          counters=counters,
                          trace_seconds=time.perf_counter() - started,
                          random_stream='CPU scrambled Sobol per source; CPU seeded Fresnel stream indexed by original photon ID and interface number, independent of launch chunks.')
        self._prepare_gather()
        return self

    def _indices(self, points):
        if not len(points):
            return torch.empty(0, dtype=torch.long, device=points.device)
        lower_x, upper_x, lower_z, upper_z = self.floor_bounds
        tolerance = self.geometry_tolerance * 2
        _require(bool(torch.isfinite(points).all())
                 and float((points[:, 1] - self.floor_y).abs().max()) <= tolerance
                 and bool(((points[:, 0] >= lower_x - tolerance) & (points[:, 0] <= upper_x + tolerance)
                           & (points[:, 2] >= lower_z - tolerance) & (points[:, 2] <= upper_z + tolerance)).all()),
                 'a caustic receiver point is outside its declared physical plane')
        x = torch.floor((points[:, 0].double() - lower_x) / self.dx).long().clamp(0, self.nx - 1)
        z = torch.floor((points[:, 2].double() - lower_z) / self.dz).long().clamp(0, self.nz - 1)
        return z * self.nx + x

    def _prepare_gather(self):
        # Accumulate on CPU in float64 for an inspectable deterministic power
        # ledger. The camera gather uses the original directional packets.
        data = self.tensors
        bins = self._indices(data['receiver_position'])
        raw = torch.zeros((self.nz * self.nx, 3), dtype=torch.float64)
        raw.index_add_(0, bins, data['receiver_flux'])
        raw = raw.reshape(self.nz, self.nx, 3)
        kernel = self.options['kernel_size']
        filtered = torch.nn.functional.avg_pool2d(raw.permute(2, 0, 1)[None], kernel,
                                                  stride=1, padding=kernel // 2,
                                                  count_include_pad=True)[0].permute(1, 2, 0).contiguous()
        data.update(receiver_bin_index=bins, raw_flux_grid=raw, filtered_flux_grid=filtered)
        order = torch.argsort(bins, stable=True)
        counts = torch.bincount(bins, minlength=self.nz * self.nx)
        self.bin_counts = counts.to(self.device)
        self.bin_starts = (counts.cumsum(0) - counts).to(self.device)
        # The incident directions remain float64 in the evidence. BRDF calls
        # use the receiver material's authored dtype, as all ordinary shading
        # does; only the geometric photon transport requires wider precision.
        self.sorted_incoming = data['receiver_incoming'][order].to(device=self.device, dtype=self.dtype)
        self.sorted_flux = data['receiver_flux'][order].to(self.device)
        terminal, launch, final = data['terminal_code'], data['launch_flux'], data['terminal_flux']
        buckets = {name: final[terminal == code].sum(0) for name, code in TERMINAL_CODES.items()}
        absorbed = (launch - final).sum(0)
        launched = launch.sum(0)
        first_glass = launch[data['first_glass_owner'] >= 0].sum(0)
        raw_sum, filtered_sum = raw.sum((0, 1)), filtered.sum((0, 1))
        self.stats['ledger'] = {
            'units': 'Power at the authored radiant scale; all packets use the same all-launch denominator.',
            'launched_flux_rgb': launched.tolist(),
            'analytic_launched_flux_rgb': self.analytic_launch_flux.tolist(),
            'launch_relative_error_rgb': ((launched - self.analytic_launch_flux).abs()
                                          / self.analytic_launch_flux.clamp_min(1e-30)).tolist(),
            'first_glass_launch_flux_rgb': first_glass.tolist(),
            'terminal_flux_rgb': {name: value.tolist() for name, value in buckets.items()},
            'absorbed_flux_rgb': absorbed.tolist(),
            'accounted_flux_rgb': (sum(buckets.values()) + absorbed).tolist(),
            'closure_error_rgb': (sum(buckets.values()) + absorbed - launched).tolist(),
            'depth_residual_fraction_of_first_glass_flux_rgb':
                (buckets['depth_limit'] / first_glass.clamp_min(1e-30)).tolist(),
            'raw_grid_flux_rgb': raw_sum.tolist(),
            'raw_grid_vs_receiver_flux_error_rgb': (raw_sum - buckets['receiver']).tolist(),
            'filtered_grid_flux_rgb': filtered_sum.tolist(),
            'filter_boundary_spill_flux_rgb': (raw_sum - filtered_sum).tolist(),
            'interpretation': 'A packet power account, not a universal radiance-error bound or convergence proof.',
        }

    def opaque_visibility(self, points, directions, distance, receiver_faces):
        from ._dielectric_render import _epsilon
        visibility = torch.ones_like(points)
        remaining = distance - 2 * _epsilon(points)
        active = remaining > _epsilon(points)
        if bool(active.any()):
            hit = self.transport.query.intersect(points[active], directions[active],
                                                 t_min=_epsilon(points[active]),
                                                 t_max=remaining[active], ignore_faces=receiver_faces[active])
            visibility[active] = (~hit.hit).to(points.dtype)[:, None]
            self.transport.stats['shadow_query_rays'] += int(active.sum())
        return visibility

    def gather(self, points, normals, geometric, view, mat):
        """Evaluate the existing BRDF for directional photons in each footprint."""
        count = len(points)
        result = torch.zeros((count, 3), dtype=torch.float64, device=self.device)
        self.stats['gather_calls'] += 1
        self.stats['gather_receiver_endpoints'] += count
        if not count or not len(self.sorted_flux):
            return result.to(points.dtype)
        bins = self._indices(points)
        x, z = bins % self.nx, bins // self.nx
        radius = self.options['kernel_size'] // 2
        offsets = [(dz, dx) for dz in range(-radius, radius + 1) for dx in range(-radius, radius + 1)]
        neighbours, valid = [], []
        for dz, dx in offsets:
            nx, nz = x + dx, z + dz
            valid.append((nx >= 0) & (nx < self.nx) & (nz >= 0) & (nz < self.nz))
            neighbours.append((nz.clamp(0, self.nz - 1) * self.nx + nx.clamp(0, self.nx - 1)))
        neighbours = torch.stack(neighbours, 1).reshape(-1)
        valid = torch.stack(valid, 1).reshape(-1)
        counts = torch.where(valid, self.bin_counts[neighbours], 0)
        cumulative = counts.cumsum(0)
        total = int(cumulative[-1])
        if not total:
            return result.to(points.dtype)
        starts = cumulative - counts
        evaluator = pbr if mat.evaluator is pbr else material.classic
        kernel_cells = self.options['kernel_size'] ** 2
        scale = 1. / (self.cell_area * kernel_cells)
        for begin in range(0, total, _PAIR_BLOCK):
            end = min(begin + _PAIR_BLOCK, total)
            pairs = torch.arange(begin, end, device=self.device)
            entry = torch.searchsorted(cumulative, pairs, right=True)
            target = entry // kernel_cells
            photon = self.bin_starts[neighbours[entry]] + pairs - starts[entry]
            incoming = self.sorted_incoming[photon]
            brdf = evaluator(normals[target], view[target], incoming[:, None],
                             _geometric_normals=geometric[target], **dict(mat.parameters))[:, 0]
            result.index_add_(0, target, brdf.double() * self.sorted_flux[photon] * scale)
            self.stats['maximum_gather_pairs_per_block'] = max(self.stats['maximum_gather_pairs_per_block'], end - begin)
        self.stats['gather_brdf_pairs'] += total
        _require(bool(torch.isfinite(result).all()) and bool((result >= 0).all()),
                 'caustic BRDF gather produced invalid radiance')
        return result.to(points.dtype)

    def evidence(self):
        """Explicit requested output; tensors are CPU arrays with original dtypes."""
        return {'metadata': self.stats, 'tensors': self.tensors}


@torch.no_grad()
def build(transport, options, *, progress=None):
    return PlaneCaustics(transport, options).trace(progress=progress)
