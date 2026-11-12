"""D01 authoring: a solid glass sphere and a closed, hollow glass tumbler.

Geometry and lighting live here; optical transport lives in gffx.torch.
Run the small examples/studio_d.py entry point for a first still.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import struct
import zipfile

import torch
import gffx as g


def tumbler(name, *, center=(.5, .02, 0.), radius=.31, height=.90,
            wall=.055, bottom=.09, segments=24, material='cup_glass',
            dtype=torch.float32, device='cpu'):
    """Revolve a closed glass cross-section, including the bottom and top rim.

    The cavity is open to air at the top. Inner and outer walls belong to one
    closed solid boundary and have opposite outward orientations.
    """
    if not 0 < wall < radius or not 0 < bottom < height:
        raise ValueError('tumbler wall and bottom must fit inside its dimensions')
    if type(segments) is not int or segments < 6:
        raise ValueError('tumbler segments must be an integer >=6')
    profile = [(0., 0.), (radius, 0.), (radius, height),
               (radius-wall, height), (radius-wall, bottom), (0., bottom)]
    vertices, rings = [], []
    for r, y in profile:
        if r == 0:
            rings.append([len(vertices)] * segments)
            vertices.append([center[0], center[1]+y, center[2]])
        else:
            indices = []
            for j in range(segments):
                angle = math.tau*j/segments
                indices.append(len(vertices))
                vertices.append([center[0]+r*math.cos(angle), center[1]+y,
                                 center[2]+r*math.sin(angle)])
            rings.append(indices)
    faces = []
    for inner, outer in zip(rings[:-1], rings[1:]):
        for j in range(segments):
            k = (j+1) % segments
            for triangle in ((inner[j], outer[j], outer[k]),
                             (inner[j], outer[k], inner[k])):
                if len(set(triangle)) == 3:
                    faces.append(triangle)
    return g.Mesh(name, torch.tensor(vertices, dtype=dtype, device=device),
                  torch.tensor(faces, dtype=torch.int32, device=device),
                  material=material)


def build_scene(*, device='cpu', dtype=torch.float32, size=(192, 192),
                azimuth_degrees=0., sphere_segments=24, sphere_rings=12,
                cup_segments=24):
    """Return (scene, camera) in world units interpreted as metres for D01."""
    colours = {'tile_light': (.72, .72, .68), 'tile_dark': (.055, .075, .09),
               'panel_red': (.65, .035, .02), 'panel_blue': (.025, .13, .60),
               'panel_white': (.78, .78, .78)}
    prims = [g.Material.pbr(name, colour, roughness=.65, dtype=dtype, device=device)
             for name, colour in colours.items()]
    prims.extend([
        g.Material.dielectric('sphere_glass', ior=1.5, absorption=(.015, .035, .08),
                              dtype=dtype, device=device),
        g.Material.dielectric('cup_glass', ior=1.5, absorption=(.12, .025, .008),
                              dtype=dtype, device=device),
        g.Mesh.sphere('glass_sphere', center=(-.50, .40, .03), radius=.38,
                      segments=sphere_segments, rings=sphere_rings,
                      material='sphere_glass', dtype=dtype, device=device),
        tumbler('glass_tumbler', segments=cup_segments, dtype=dtype, device=device),
    ])
    step = .30
    for iz in range(8):
        for ix in range(10):
            prims.append(g.Mesh.quad('tile_%02d_%02d' % (ix, iz),
                center=((ix-4.5)*step, 0., (iz-3.5)*step), u=(step, 0., 0.),
                v=(0., 0., -step), material='tile_light' if (ix+iz)%2 else 'tile_dark',
                dtype=dtype, device=device))
    for index, colour in enumerate(['panel_red', 'panel_white', 'panel_blue',
                                     'panel_white', 'panel_red', 'panel_white',
                                     'panel_blue', 'panel_white']):
        prims.append(g.Mesh.quad('backdrop_%02d' % index,
            center=((index-3.5)*.29, .68, -.86), u=(.29, 0., 0.), v=(0., 1.34, 0.),
            material=colour, dtype=dtype, device=device))
    prims.extend([
        g.Light.rectangle('key', center=(-.6, 2.3, .65), u=(1.65, 0., 0.),
            v=(0., 0., 1.4), radiance=(4., 4., 4.), samples=1, visible=True,
            dtype=dtype, device=device),
        g.Light.rectangle('fill', center=(-1.7, 1.05, .25), u=(0., 1.5, 0.),
            v=(0., 0., 1.2), radiance=(1.3, 1.5, 1.9), samples=1, visible=True,
            dtype=dtype, device=device),
    ])
    angle = math.radians(azimuth_degrees)
    x, z = 1.65, 3.15
    eye = (x*math.cos(angle)+z*math.sin(angle), 1.65,
           z*math.cos(angle)-x*math.sin(angle))
    camera = g.Camera.look_at(eye, (0., .48, 0.), size=size, fov=39.,
                              near=.025, far=12., dtype=dtype, device=device)
    return g.Scene(prims), camera


def write_bundle(path, scene, camera):
    """Write this constant-material D01 fixture as the existing gffx-scene-1 format."""
    from gffx.torch.scene import Material, Mesh, Light
    from gffx.torch._dielectric import dielectric
    from gffx.torch._raster_material import pbr
    from gffx.torch import light
    payloads = {}
    def array(name, value, integer=False):
        value = value.detach().to(device='cpu', dtype=torch.int32 if integer else torch.float32).contiguous()
        code = 'i' if integer else 'f'
        filename = 'arrays/'+name+'.bin'
        payloads[filename] = struct.pack('<'+code*value.numel(), *value.reshape(-1).tolist())
        return {'file': filename, 'dtype': 'int32' if integer else 'float32', 'shape': list(value.shape)}
    def plain(value):
        return value.detach().cpu().tolist() if isinstance(value, torch.Tensor) else value
    spec = {'format': 'gffx-scene-1', 'materials': [], 'meshes': [], 'lights': [], 'textures': {}}
    for prim in scene.prims:
        if prim.transform is not None or prim.parent is not None:
            raise ValueError('D01 fixture writer expects already authored world-space prims')
        if isinstance(prim, Light):
            if prim.evaluator is not light.rectangle:
                raise ValueError('D01 fixture writer expects rectangle lights')
            parameters = {k: plain(v) for k, v in prim.parameters.items()}
            parameters['visible'] = prim.visible
            spec['lights'].append({'name': prim.name, 'kind': 'rectangle', 'parameters': parameters})
        elif isinstance(prim, Material):
            if prim.evaluator not in (dielectric, pbr):
                raise ValueError('D01 fixture writer expects PBR or dielectric materials')
            spec['materials'].append({'name': prim.name,
                'kind': 'dielectric' if prim.evaluator is dielectric else 'pbr',
                'parameters': {k: plain(v) for k, v in prim.parameters.items()}})
        elif isinstance(prim, Mesh):
            attributes = {}
            for key, attribute in prim.attributes.items():
                values = attribute.values
                if attribute.domain == 'vertex': values = values[prim.faces.long()]
                elif attribute.domain != 'face_varying':
                    raise ValueError('D01 fixture writer expects vertex or face-varying attributes')
                attributes[key] = array(prim.name+'_'+key, values)
            spec['meshes'].append({'name': prim.name, 'material': prim.material,
                'vertices': array(prim.name+'_vertices', prim.vertices),
                'faces': array(prim.name+'_faces', prim.faces, True), 'attributes': attributes})
        else:
            raise ValueError('Unsupported D01 fixture prim')
    spec['camera'] = {key: float(getattr(camera, key)[0]) for key in ('fx','fy','cx','cy')}
    spec['camera'].update(height=camera.image_height, width=camera.image_width,
                          near=camera.near, far=camera.far,
                          pose=array('camera_pose', camera.pose[0]))
    path = Path(path)
    with zipfile.ZipFile(path, 'x', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.writestr('scene.json', json.dumps(spec, indent=2, allow_nan=False)+'\n')
        for name, data in payloads.items(): archive.writestr(name, data)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu','cuda'], default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--size', type=int, default=192)
    parser.add_argument('--samples', type=int, default=1, help='Grid dimension; total samples is its square')
    parser.add_argument('--output', type=Path, default=Path('studio_d_d01.png'))
    options = parser.parse_args()
    scene, camera = build_scene(device=options.device, size=(options.size, options.size))
    result = g.render(scene, camera, integrator='raster', supersample=options.samples,
                      shadows=True, background=(.035, .045, .065),
                      outputs=('colour','mask','timings','transmission'))
    g.io.write_png(str(options.output), result['colour'][0])
    print(json.dumps(result['transmission'], indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
