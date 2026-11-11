"""Small known-result controls for opt-in directional plane caustics.

These unittest fixtures need only the existing GFFX runtime and PyTorch. The
host can run the same class on CUDA by setting GFFX_CAUSTIC_TEST_DEVICE=cuda.
Full authored-scene visual comparison is a separate experiment.
"""
from __future__ import annotations

import math
import os
import unittest
from unittest import mock

import torch
import gffx as g

from gffx.torch import _caustic_plane as module
from gffx.torch._area import expand_emitters
from gffx.torch._deferred import _prepare
from gffx.torch._dielectric_render import _Transport, _validate_materials
from gffx.torch._raster import _light_sources


DEVICE = os.environ.get('GFFX_CAUSTIC_TEST_DEVICE', 'cpu')


def box(name, lower, upper, material, device=DEVICE):
    vertices = torch.tensor([[x, y, z] for x in (lower[0], upper[0])
                             for y in (lower[1], upper[1])
                             for z in (lower[2], upper[2])], dtype=torch.float32, device=device)
    quads = [(0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4),
             (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5)]
    centre = vertices.mean(0)
    faces = []
    for a, b, c, d in quads:
        normal = torch.linalg.cross(vertices[b] - vertices[a], vertices[c] - vertices[a])
        if float((normal * (vertices[a] - centre)).sum()) < 0:
            b, d = d, b
        faces.extend(((a, b, c), (a, c, d)))
    return g.Mesh(name, vertices, torch.tensor(faces, dtype=torch.int32, device=device), material=material)


def fixture(*, ior=1., absorbing=False, two=True, receiver='diffuse', size=6, device=DEVICE):
    floor = (g.Material.pbr('floor_mat', (.3, .5, .7), roughness=.65, device=device)
             if receiver == 'pbr' else g.Material.diffuse('floor_mat', (.3, .5, .7), device=device))
    boxes = [(-.4, .34, -.4, .4, .46, .4)]
    if two:
        boxes.append((-.4, .66, -.4, .4, .74, .4))
    sigmas = [(1., 0., 0.), (0., 2., 3.)] if absorbing else [(0., 0., 0.), (0., 0., 0.)]
    prims = [floor, g.Mesh.quad('floor', center=(0., 0., 0.), u=(2., 0., 0.), v=(0., 0., -2.),
                               material='floor_mat', device=device)]
    for i, b in enumerate(boxes):
        prims.extend([g.Material.dielectric('glass_mat_%d' % i, ior=ior, absorption=sigmas[i], device=device),
                      box('glass_%d' % i, b[:3], b[3:], 'glass_mat_%d' % i, device)])
    prims.append(g.Light.rectangle('lamp', center=(0., 1.3, 0.), u=(.2, 0., 0.), v=(0., 0., .2),
                                  radiance=(2., 3., 4.), samples=1, visible=False, device=device))
    camera = g.Camera.look_at((.75, 1.1, 1.3), (0., .25, 0.), size=(size, size),
                              fov=50., near=.01, far=5., device=device)
    return g.Scene(prims), camera, boxes, sigmas


def transport(scene, camera):
    packed, resolved = _prepare(expand_emitters(scene), camera)
    _validate_materials(packed)
    sources = _light_sources(packed, packed.vertices, packed.faces, 32, False, None)
    return _Transport(packed, resolved, sources=sources, background=None, environment=None,
                      shadows=True, query_chunk=128, diagnostics={'shadow_query_rays': 0})


def plane(scene=None, camera=None, **options):
    if scene is None:
        scene, camera, _, _ = fixture()
    config = dict(receivers=['floor'], photons=256, seed=1701, cell_size=.1, kernel_size=3, max_interfaces=64)
    config.update(options)
    return module.PlaneCaustics(transport(scene, camera), config)


def synthetic_map(value, *, kernel_size=1, corner=False, incoming=(.8, .6, 0.), flux=(.03, .04, .05)):
    value.options['kernel_size'] = kernel_size
    x, z = (0, 0) if corner else (value.nx // 2, value.nz // 2)
    point = torch.tensor([[value.floor_bounds[0] + (x + .5) * value.dx, value.floor_y,
                           value.floor_bounds[2] + (z + .5) * value.dz]], dtype=torch.float32)
    power = torch.tensor([flux], dtype=torch.float64)
    value.tensors = dict(receiver_position=point, receiver_incoming=torch.tensor([incoming], dtype=torch.float32),
                         receiver_flux=power, terminal_code=torch.tensor([module.TERMINAL_CODES['receiver']]),
                         launch_flux=power, terminal_flux=power, first_glass_owner=torch.tensor([1]))
    value._prepare_gather()
    return point.to(DEVICE), power.to(DEVICE)


class TestPlaneCaustics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if DEVICE == 'cuda' and not torch.cuda.is_available():
            raise unittest.SkipTest('CUDA unavailable')
        cls.scene, cls.camera, cls.boxes, cls.sigmas = fixture(absorbing=True)
        cls.base = plane(cls.scene, cls.camera).trace(launch_chunk=64)

    def test_01_independent_two_owner_beer_lengths_and_null_endpoints(self):
        data = self.base.tensors
        origins, directions = data['launch_position'].double(), data['launch_direction'].double()
        optical = torch.zeros_like(origins)
        distance = torch.zeros(len(origins), dtype=torch.float64)
        for bounds, sigma in zip(self.boxes, self.sigmas):
            lo, hi = origins.new_tensor(bounds[:3]), origins.new_tensor(bounds[3:])
            a, b = (lo - origins) / directions, (hi - origins) / directions
            enter, leave = torch.minimum(a, b).amax(-1), torch.maximum(a, b).amin(-1)
            length = (leave - enter.clamp_min(0)).clamp_min(0)
            optical += length[:, None] * origins.new_tensor(sigma)
            distance += length
        torch.testing.assert_close(data['terminal_optical_depth'], optical, rtol=2e-5, atol=8e-6)
        torch.testing.assert_close(data['terminal_glass_distance'], distance, rtol=2e-5, atol=4e-6)
        expected_power = data['launch_flux'] * torch.exp(-optical)
        torch.testing.assert_close(data['terminal_flux'], expected_power, rtol=2e-5, atol=1e-9)
        ids = data['receiver_photon_id']
        time_to_plane = -origins[ids, 1] / directions[ids, 1]
        endpoints = origins[ids] + time_to_plane[:, None] * directions[ids]
        torch.testing.assert_close(data['receiver_position'].double(), endpoints, rtol=2e-5, atol=4e-6)
        # Some rays pass through both separate owners, proving cavity air is
        # not assigned either solid's absorption coefficient.
        self.assertGreater(int((data['interface_count'] >= 4).sum()), 0)
        self.assertEqual(self.base.stats['counters']['reflection_events'], 0)
        self.assertEqual(self.base.stats['terminal_counts']['depth_limit'], 0)
        ledger = self.base.stats['ledger']
        self.assertLess(max(abs(x) for x in ledger['closure_error_rgb']), 1e-12)
        self.assertTrue(all(x > 0 for x in ledger['absorbed_flux_rgb']))

    def test_02_packet_identity_is_independent_of_launch_chunk(self):
        alternate = plane(self.scene, self.camera).trace(launch_chunk=37)
        for key in self.base.tensors:
            torch.testing.assert_close(alternate.tensors[key], self.base.tensors[key], atol=0., rtol=0., msg=key)

    def test_03_union_bound_samples_each_source_once_and_keeps_misses(self):
        base = self.base
        data = base.tensors
        self.assertEqual(base.source_count, 1)
        self.assertEqual(base.source_metadata['proposal_modes'], ['union_glass_bound_cap'])
        self.assertEqual(len(data['launch_position']), base.options['photons'])
        self.assertGreater(base.stats['terminal_counts']['initial_miss'], 0)
        normal = torch.tensor(base.source_metadata['normals'], dtype=torch.float64)
        intensity = torch.tensor(base.source_metadata['sample_intensity_rgb'], dtype=torch.float64)
        omega = torch.tensor(base.source_metadata['proposal_solid_angle'], dtype=torch.float64)
        s = data['launch_source_id']
        cosine = (data['launch_direction'].double() * normal[s]).sum(-1).clamp_min(0)
        expected = intensity[s] * base.source_count * omega[s, None] * cosine[:, None] / base.options['photons']
        torch.testing.assert_close(data['launch_flux'], expected, rtol=0., atol=0.)
        self.assertLess(max(base.stats['ledger']['launch_relative_error_rgb']), .01)

    def test_04_oblique_lambertian_gather_has_no_second_cosine(self):
        value = plane(kernel_size=1)
        points, power = synthetic_map(value, incoming=(.8, .6, 0.))
        normal = points.new_tensor([[0., 1., 0.]])
        mat = next(m for m in value.transport.packed.materials if m.name == 'floor_mat')
        observed = value.gather(points, normal, normal, normal, mat)
        expected = power * points.new_tensor([.3, .5, .7]).double() / (math.pi * value.cell_area)
        torch.testing.assert_close(observed.double(), expected, rtol=2e-6, atol=2e-6)

    def test_05_pbr_closed_form_direction_view_and_material_response(self):
        scene, camera, _, _ = fixture(receiver='pbr')
        value = plane(scene, camera, kernel_size=1)
        points, power = synthetic_map(value, incoming=(0., 1., 0.))
        n = points.new_tensor([[0., 1., 0.]])
        mat = next(m for m in value.transport.packed.materials if m.name == 'floor_mat')
        actual = value.gather(points, n, n, n, mat).double()
        base = points.new_tensor([.3, .5, .7]).double()
        roughness, f0 = float(mat.parameters['roughness']), .04
        expected = power / value.cell_area * ((1 - f0) * base / math.pi + f0 / (4 * math.pi * roughness ** 4))
        torch.testing.assert_close(actual, expected, rtol=3e-6, atol=3e-6)
        # A different receiving material in the same cell must reuse the same
        # incident photons, not their original endpoint's pre-shaded colour.
        other = g.Material.pbr('other', (.8, .2, .1), roughness=.65, device=DEVICE)
        expected_other = power / value.cell_area * ((1 - f0) * points.new_tensor([.8, .2, .1]).double() / math.pi
                                                   + f0 / (4 * math.pi * roughness ** 4))
        torch.testing.assert_close(value.gather(points, n, n, n, other).double(), expected_other, rtol=3e-6, atol=3e-6)
        # Known oblique GGX values, evaluated here with scalar mathematics.
        points, power = synthetic_map(value, incoming=(.6, .8, 0.))
        v = points.new_tensor([[.6, .8, 0.]])
        a2, nl, nv, nh = roughness ** 4, .8, .8, .8
        distribution = a2 / (math.pi * (nh * nh * (a2 - 1) + 1) ** 2)
        visibility = .5 / (nl * math.sqrt(nv * nv * (1 - a2) + a2)
                             + nv * math.sqrt(nl * nl * (1 - a2) + a2))
        expected_oblique = power / value.cell_area * ((1 - f0) * base / math.pi + distribution * visibility * f0)
        observed_oblique = value.gather(points, n, n, v, mat).double()
        torch.testing.assert_close(observed_oblique, expected_oblique, rtol=3e-6, atol=3e-6)
        self.assertGreater(float((observed_oblique - actual).abs().max()), .01)

    def test_06_boundary_filter_loses_mass_without_renormalization(self):
        value = plane()
        points, power = synthetic_map(value, corner=True, kernel_size=3)
        total = value.tensors['raw_flux_grid'].sum((0, 1))
        torch.testing.assert_close(value.tensors['filtered_flux_grid'].sum((0, 1)), total * 4 / 9,
                                   rtol=1e-12, atol=1e-12)
        spill = torch.tensor(value.stats['ledger']['filter_boundary_spill_flux_rgb'])
        torch.testing.assert_close(spill.double(), total * 5 / 9, rtol=1e-7, atol=1e-9)
        n = points.new_tensor([[0., 1., 0.]])
        mat = next(m for m in value.transport.packed.materials if m.name == 'floor_mat')
        expected = power * points.new_tensor([.3, .5, .7]).double() / (math.pi * value.cell_area * 9)
        torch.testing.assert_close(value.gather(points, n, n, n, mat).double(), expected, rtol=2e-6, atol=1e-6)

    def test_07_rejects_overlapping_receiver_triangles_and_nonplane(self):
        scene, camera, _, _ = fixture()
        prims = list(scene.prims)
        i = next(i for i, p in enumerate(prims) if isinstance(p, g.Mesh) and p.name == 'floor')
        original = prims[i]
        vertices = original.vertices
        # Find two triangles with positive Y normal and a shared boundary
        # edge: they overlap while their summed area equals the rectangle.
        triangles = []
        for triple in ((0, 1, 2), (0, 1, 3)):
            a, b, c = triple
            if float(torch.linalg.cross(vertices[b] - vertices[a], vertices[c] - vertices[a])[1]) < 0:
                b, c = c, b
            triangles.append((a, b, c))
        prims[i] = g.Mesh('floor', vertices, torch.tensor(triangles, dtype=torch.int32, device=DEVICE), material='floor_mat')
        with self.assertRaisesRegex(ValueError, 'diagonal'):
            plane(g.Scene(prims), camera)
        bent = original.vertices.clone()
        bent[0, 1] += .1
        prims[i] = g.Mesh('floor', bent, original.faces, material='floor_mat')
        with self.assertRaisesRegex(ValueError, 'horizontal'):
            plane(g.Scene(prims), camera)

    def test_08_public_forwarding_and_default_off(self):
        scene, camera, _, _ = fixture(two=False, size=3)
        kwargs = dict(integrator='raster', shadows=True, background=(0., 0., 0.),
                      outputs=('colour', 'mask', 'transmission'))
        default = g.render(scene, camera, **kwargs)
        disabled = g.render(scene, camera, caustics=None, **kwargs)
        torch.testing.assert_close(default['colour'], disabled['colour'], rtol=0., atol=0.)
        torch.testing.assert_close(default['mask'], disabled['mask'], rtol=0., atol=0.)
        self.assertNotIn('caustics', disabled['transmission'])
        config = dict(receivers=['floor'], photons=64, cell_size=.1, seed=1701, kernel_size=3, max_interfaces=64)
        enabled = g.render(scene, camera, caustics=config, **dict(kwargs, outputs=('colour', 'mask', 'face_index', 'transmission', 'caustics')))
        self.assertEqual(enabled['caustics']['metadata']['revision'], module.REVISION)
        self.assertEqual(len(enabled['caustics']['tensors']['launch_flux']), 64)
        self.assertGreater(enabled['transmission']['caustics']['gather_receiver_endpoints'], 0)
        primary_floor = int(((enabled['face_index'] >= 0) & (enabled['face_index'] < 2)).sum())
        self.assertGreater(enabled['transmission']['caustics']['gather_receiver_endpoints'], primary_floor,
                           'caustic shading must include floor endpoints reached through glass')
        self.assertTrue(bool(torch.isfinite(enabled['colour']).all()))
        with self.assertRaisesRegex(ValueError, 'shadows=True'):
            g.render(scene, camera, integrator='raster', shadows=False, caustics=config)
        with self.assertRaisesRegex(ValueError, 'integrator=raster'):
            g.render(scene, camera, integrator='deferred', caustics=config)
        opaque = g.Scene([p for p in scene.prims if not (isinstance(p, g.Mesh) and p.name.startswith('glass_'))])
        with self.assertRaisesRegex(ValueError, 'active dielectric'):
            g.render(opaque, camera, integrator='raster', caustics=config)

    def test_09_emitting_hemisphere_fallback_normalizes_cosine_power(self):
        scene, camera, _, _ = fixture()
        prims = list(scene.prims)
        i = next(i for i, p in enumerate(prims) if isinstance(p, g.Light))
        # In the union sphere's empty side region but outside both glass slabs.
        prims[i] = g.Light.rectangle('lamp', center=(.5, .54, 0.), u=(0., .1, 0.), v=(0., 0., .1),
                                     radiance=(2., 3., 4.), samples=1, visible=False, device=DEVICE)
        value = plane(g.Scene(prims), camera, photons=1024)
        self.assertEqual(value.source_metadata['proposal_modes'], ['full_emitting_hemisphere'])
        _, _, directions, flux = value._launch()
        normal = value.source_normal_cpu[0]
        self.assertTrue(bool(((directions.double() * normal).sum(-1) >= 0).all()))
        expected = value.source_intensity_cpu[0] * math.pi
        torch.testing.assert_close(flux.sum(0), expected, rtol=.002, atol=1e-9)

    def test_10_multiple_lights_use_source_probability_and_quadrature_weights(self):
        scene, camera, _, _ = fixture()
        extra = g.Light.rectangle('other_lamp', center=(.3, 1.4, .1), u=(.15, 0., 0.), v=(0., 0., .15),
                                   radiance=(.5, 1., .25), samples=2, visible=False, device=DEVICE)
        value = plane(g.Scene([*scene.prims, extra]), camera, photons=1280)
        source_ids, _, directions, flux = value._launch()
        self.assertEqual(value.source_count, 5)  # one original + four area samples
        self.assertEqual(torch.bincount(source_ids).tolist(), [256] * 5)
        cosine = (directions.double() * value.source_normal_cpu[source_ids]).sum(-1).clamp_min(0)
        expected = (value.source_intensity_cpu[source_ids] * 5
                    * value.omega_cpu[source_ids, None] * cosine[:, None] / 1280)
        torch.testing.assert_close(flux, expected, rtol=0., atol=0.)
        # The analytic cap integrals have no photon count, hit count, or
        # per-object denominator; compare each RGB power component.
        torch.testing.assert_close(flux.sum(0), value.analytic_launch_flux, rtol=.01, atol=1e-8)

    def test_11_gather_crosses_pair_blocks_without_dropping_photons(self):
        value = plane(kernel_size=3)
        center, _ = synthetic_map(value, incoming=(0., 1., 0.), kernel_size=3)
        corner = center.detach().cpu().clone()
        corner[0, 0] = value.floor_bounds[0] + .5 * value.dx
        corner[0, 2] = value.floor_bounds[2] + .5 * value.dz
        n_center, n_corner = 40000, 5000
        total = n_center + n_corner
        value.tensors['receiver_position'] = torch.cat((center.cpu().repeat(n_center, 1), corner.repeat(n_corner, 1)))
        value.tensors['receiver_incoming'] = torch.tensor([[0., 1., 0.]]).repeat(total, 1)
        flux = torch.tensor([[1e-7, 2e-7, 3e-7]], dtype=torch.float64).repeat(total, 1)
        value.tensors.update(receiver_flux=flux, terminal_flux=flux, launch_flux=flux,
                             terminal_code=torch.full((total,), module.TERMINAL_CODES['receiver']),
                             first_glass_owner=torch.ones(total, dtype=torch.long))
        value._prepare_gather()
        neighbour = center.clone()
        neighbour[:, 0] += value.dx
        points = torch.cat((center, neighbour, corner.to(DEVICE)))
        normal = points.new_tensor([[0., 1., 0.]]).expand(3, 3)
        mat = next(m for m in value.transport.packed.materials if m.name == 'floor_mat')
        actual = value.gather(points, normal, normal, normal, mat).double()
        counts = points.new_tensor([n_center, n_center, n_corner]).double()
        expected = (counts[:, None] * flux[:1].to(DEVICE)
                    * points.new_tensor([.3, .5, .7]).double() / (math.pi * value.cell_area * 9))
        torch.testing.assert_close(actual, expected, rtol=3e-6, atol=1e-8)
        self.assertEqual(value.stats['gather_brdf_pairs'], n_center * 2 + n_corner)
        self.assertEqual(value.stats['maximum_gather_pairs_per_block'], 65536)

    def test_12_authored_tumbler_rim_keeps_sub_float32_threshold_tir_flight(self):
        # Captured from authored D01 photon57806, seed1701, N65536. It enters
        # the top rim only0.135micrometres from an inner-wall TIR event. A
        # float32 self-hit threshold skipped that real wall and corrupted the
        # medium state. No threshold/owner guard is relaxed in this regression.
        from gffx.examples.studio_d import tumbler
        from gffx.torch._dielectric_render import _epsilon
        scene, camera, _, _ = fixture(two=False)
        prims = [p for p in scene.prims if not (isinstance(p, g.Mesh) and p.name.startswith('glass_'))
                 and not (isinstance(p, g.Material) and p.name.startswith('glass_mat_'))]
        prims.extend([g.Material.dielectric('cup_glass', ior=1.5, absorption=(.12, .025, .008), device=DEVICE),
                      tumbler('cup', device=DEVICE)])
        value = plane(g.Scene(prims), camera, photons=16, max_interfaces=64)
        origin = torch.tensor([[-.6000000238418579, 2.299999952316284, .6499999761581421]],
                              dtype=torch.float64, device=DEVICE)
        direction = torch.tensor([[.5365786552429199, -.7996941804885864, -.2693931758403778]],
                                 dtype=torch.float64, device=DEVICE)
        direction /= torch.linalg.vector_norm(direction, dim=-1, keepdim=True)
        first = value.query.intersect(origin, direction, t_min=_epsilon(origin))
        self.assertEqual(int(first.face_index[0]), 90)  # two floor faces + local rim88
        point = origin + first.distance[:, None] * direction
        optical, medium = value.photon_transport.crossing(first.face_index, direction, first.normal,
                                                        torch.tensor([-1], device=DEVICE))
        second = value.query.intersect(point, optical.transmitted, t_min=_epsilon(point),
                                       ignore_faces=first.face_index)
        self.assertEqual(int(second.face_index[0]), 139)  # local inner wall137
        self.assertAlmostEqual(float(second.distance[0]), 1.3526871238339822e-7, delta=5e-13)
        self.assertLess(float(second.distance[0]), float(_epsilon(point.float())[0]))
        inner, _ = value.photon_transport.crossing(second.face_index, optical.transmitted, second.normal, medium)
        self.assertTrue(bool(inner.tir[0]))
        self.assertEqual(value.query.vertices.dtype, torch.float64)
        self.assertEqual(value.transport.query.vertices.dtype, torch.float32)
        # Replay the observed launch and the original photon ID's random
        # choices, while exercising the production trace and strict media.
        launch_power = torch.tensor([[.00017065726272685565] * 3], dtype=torch.float64)
        value._launch = lambda: (torch.zeros(16, dtype=torch.long), origin.cpu().repeat(16, 1),
                                 direction.cpu().repeat(16, 1), launch_power.repeat(16, 1))
        random = torch.rand
        def witness_draws(count, *, generator, dtype):
            self.assertEqual(count, 16)
            return random(65536, generator=generator, dtype=dtype)[57806].repeat(16)
        with mock.patch.object(module.torch, 'rand', side_effect=witness_draws):
            value.trace()
        self.assertTrue(bool((value.tensors['terminal_code'] == module.TERMINAL_CODES['receiver']).all()))
        self.assertTrue(bool((value.tensors['terminal_medium'] == -1).all()))
        self.assertTrue(bool((value.tensors['interface_count'] == 9).all()))
        expected = torch.tensor([[.39333448464220083, 0., .29678693851766436]], dtype=torch.float64).repeat(16, 1)
        torch.testing.assert_close(value.tensors['receiver_position'], expected, atol=2e-10, rtol=2e-10)
        self.assertEqual(value.stats['counters']['total_internal_reflection_events'], 16 * 7)


if __name__ == '__main__':
    unittest.main(verbosity=2)
