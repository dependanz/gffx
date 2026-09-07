"""Finite light-equation illustrations, not a public gffx shading implementation."""
import argparse
import math
import subprocess

import torch
from gffx.dashboard import ArrayView, connect
from tools import light_contract_reference as reference
from .reporter import Reporter


def examples():
    distances = torch.arange(1, 17, dtype=torch.float64)/2
    points = torch.zeros(16, 3, dtype=torch.float64)
    points[:, 2] = -distances
    source = torch.zeros(1, 3, dtype=torch.float64)
    intensity = torch.full((1, 3), 4., dtype=torch.float64)
    weights = reference.point(points, source, intensity)['irradiance'][:, 0, 0]
    angles = torch.arange(181, dtype=torch.float64)
    radians = angles*math.pi/180
    normals = torch.stack((radians.sin(), torch.zeros_like(radians), radians.cos()), -1)
    direction = torch.tensor([0., 0., 1.], dtype=torch.float64)
    coeff = reference.basis(direction)[:, None].expand(9, 3)
    sh = reference.spherical_harmonics(normals, coeff)[:, 0]
    cosine = normals[:, 2].clamp_min(0)

    axis = (torch.arange(128, dtype=torch.float64)+.5)/64-1
    y, x = torch.meshgrid(axis, axis, indexing='ij')
    mask = x*x+y*y <= 1
    z = (1-x*x-y*y).clamp_min(0).sqrt()
    surface = torch.stack((x, -y, z), -1)
    surface[~mask] = direction
    light = torch.tensor([.8, .2, .3], dtype=torch.float64)
    light = light/light.norm()
    sphere_sh = reference.spherical_harmonics(surface.reshape(-1, 3), reference.basis(light)[:, None].expand(9, 3)).reshape(128, 128, 3)[..., 0]
    exact = (surface*light).sum(-1).clamp_min(0)
    images = []
    for values in (exact, sphere_sh):
        # Shared display scale, not tone mapping in the math. Negative ringing is magenta.
        image = (values.clamp(0, 17/16)/(17/16)*255).round().byte()[..., None].expand(128, 128, 3).clone()
        image[values < 0] = torch.tensor([210, 45, 170], dtype=torch.uint8)
        image[~mask] = torch.tensor([32, 38, 48], dtype=torch.uint8)
        images.append(image)
    return dict(distance=list(zip(distances.tolist(), weights.tolist(), [4.]*16)),
                angle=list(zip(angles.tolist(), cosine.tolist(), sh.tolist())),
                image=torch.cat(images, 1),
                coincident_valid=bool(reference.point(source, source, intensity)['valid'][0, 0]))


def publish(address, run, *, studio=False, device='cpu'):
    data = examples()
    reporter = Reporter(address, run)
    scope = 'independent light contract recipes; not public gffx shading'
    provenance = {'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  'dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True))}
    if not reporter.send('start', kind='demo', provenance=provenance):
        raise RuntimeError(reporter.error)
    artifacts = []
    with connect(address, run=run, name='finite light contract examples', wait_acks=True) as dash:
        if studio:
            from .showcase import publish_clips
            torch.set_num_threads(2)
            artifacts.extend(publish_clips(dash,modes=('lights','studio','approach'),device=device))
        def add(kind, name, value):
            path = 'lighting/'+name
            getattr(dash, kind)(path, value, step=0)
            artifacts.append(dict(kind=kind, path=path, label=name))
        add('text', 'how-to-read', 'Independent equation illustrations, NOT public gffx light APIs. Distance plot: x is a sample index; distance=(index+1)/2 world units. A point source of intensity 4 gives 4/r^2, while a distant light stays at 4. Angle plot: x is degrees away from the light; exact cosine goes dark at 90 degrees, nine-coefficient SH smooths the boundary and can ring negative. No shadow or visibility test is performed. Sphere image: exact directional response LEFT, band-2 SH RIGHT, same 0 to 17/16 linear grayscale display scale. Magenta explicitly marks negative SH values, not a material color. These are analytic sphere normals, not a mesh rasterizer or a rendered gffit scene.')
        add('image', 'exact-left-sh-right', ArrayView('uint8', tuple(data['image'].shape), bytes(data['image'].reshape(-1).tolist())))
        for index, (distance, point, distant) in enumerate(data['distance']):
            dash.scalar('lighting/distance/point', point, step=index)
            dash.scalar('lighting/distance/distant', distant, step=index)
        artifacts.append(dict(kind='plot', path='lighting/distance', label='Point falloff vs constant distant light'))
        for degree, exact, sh in data['angle']:
            dash.scalar('lighting/angle/exact', exact, step=int(degree))
            dash.scalar('lighting/angle/sh', sh, step=int(degree))
        artifacts.append(dict(kind='plot', path='lighting/angle', label='Exact cosine vs signed SH approximation'))
        add('record', 'checked-values', dict(point_at_half_unit=16., point_at_two_units=1.,
              sh_aligned=17/16, sh_opposed=1/16, minimum_sampled_sh=min(row[2] for row in data['angle']),
              coincident_valid=data['coincident_valid'], scope=scope))
    if not reporter.send('demo', page='shading/lights-demo', title='Finite lighting equations', artifacts=artifacts):
        raise RuntimeError(reporter.error)
    if not reporter.send('finish', exit_code=0):
        raise RuntimeError(reporter.error)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--address', required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--studio',action='store_true')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args = parser.parse_args()
    publish(args.address, args.run,studio=args.studio,device=args.device)
