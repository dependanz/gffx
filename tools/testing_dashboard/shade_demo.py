"""Finite elementary shading illustrations; no public shading or scene claim."""
import argparse
import ctypes
import hashlib
import math
from pathlib import Path
import subprocess

import torch
from gffx.dashboard import ArrayView, connect
from tools import shade_contract_reference as shade
from .reporter import Reporter


@torch.no_grad()
def examples(*, public=False):
    if public:
        from gffx.torch import shade as shader
    else:
        shader = shade
    # Analytic sphere normals; there is no mesh, geometry inference or hidden renderer.
    y, x = torch.meshgrid((torch.arange(360, dtype=torch.float64)+.5-180)/142,
                          (torch.arange(320, dtype=torch.float64)+.5-160)/142, indexing='ij')
    visible = x.square()+y.square() < 1
    z = (1-x.square()-y.square()).clamp_min(0).sqrt()
    normals = torch.stack((x, -y, z), -1).reshape(-1, 3)
    normals[~visible.flatten()] = normals.new_tensor([0., 0., 1.])
    light = normals.new_tensor([-.45, .55, 1.]); light /= light.norm()
    view = normals.new_tensor([0., 0., 1.]); half = light+view; half /= half.norm()
    nv = normals[:, 2]; nl = (normals@light).clamp(0, 1); nh = (normals@half).clamp(0, 1)
    vh = torch.full_like(nv, float(view@half))
    color = normals.new_tensor([.12, .62, .48]).expand(len(nv), 3)
    matte = shader.lambert(color)
    classic = .8*matte+.2*shader.blinn_phong(nh, nv.new_tensor(64.))[:, None]
    values = [matte, classic]
    for roughness in (.25, .65):
        r = nv.new_tensor(roughness)
        values.append((shader.ggx_distribution(nh, r)*shader.smith_visibility(nv, nl, r))[:, None]*
                      shader.fresnel_schlick(vh, nv.new_tensor([.8, .43, .14])))
    images = []
    for value in values:
        linear = 4*value*nl[:, None]+.025*color
        display = (linear/(1+linear)).clamp(0, 1).pow(1/2.2)
        display[~visible.flatten()] = display.new_tensor([.035, .05, .075])
        images.append((display.reshape(360, 320, 3)*255).round().byte())
    degrees = torch.arange(181, dtype=torch.float64)
    cosine = (degrees*math.pi/180).cos().clamp_min(0)
    half_cosine = ((1+(degrees*math.pi/180).cos())/2).clamp_min(0).sqrt()
    curves = {'matte': cosine/math.pi,
              'blinn_16': shader.blinn_phong(half_cosine, nv.new_tensor(16.))*cosine,
              'blinn_64': shader.blinn_phong(half_cosine, nv.new_tensor(64.))*cosine}
    for roughness in (.25, .65):
        r = nv.new_tensor(roughness)
        curves['ggx_'+str(roughness)] = (shader.ggx_distribution(half_cosine, r)*
            shader.smith_visibility(torch.ones_like(cosine), cosine, r)*
            shader.fresnel_schlick(half_cosine, nv.new_tensor([.04]*3))[:, 0]*cosine)
    return {'image': torch.cat(images, 1), 'angles': degrees.tolist(), 'curves': curves,
            'scope': ('Public elementary gffx.torch.shade functions' if public else 'Phase-0 elementary recipes') +
                     '; analytic spheres, not complete public gffx materials or a rasterized scene'}


def publish(address, run, *, public=False):
    torch.set_num_threads(2)
    data = examples(public=public); reporter = Reporter(address, run)
    if public:
        from gffx.torch import shade as shader
    else:
        shader = shade
    source = b''.join(Path(p).read_bytes() for p in (__file__, shader.__file__))
    provenance = {'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  'dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True)),
                  'artifact_sha256': hashlib.sha256(source).hexdigest()}
    def send(event, **kwargs):
        if not reporter.send(event, **kwargs): raise RuntimeError(reporter.error)
    send('start', kind='demo', provenance=provenance)
    with connect(address, run=run, wait_acks=True) as dash:
        image = data['image'].contiguous()
        dash.image('lobes/spheres', ArrayView('uint8', tuple(image.shape),
            ctypes.string_at(image.data_ptr(), image.numel())), step=0)
        dash.text('lobes/read-me', 'Left to right: teal Lambert; teal plus classic Blinn highlight (exponent 64); '
                  'gold-colored single-scattering GGX at roughness 0.25; the same at 0.65. '
                  'All use one light, the same analytic sphere and the same display transform. '
                  'The small fixed fill and x/(1+x), gamma 1/2.2 display conversion are illustrative. '
                  'This is NOT the revised principled material, a metal workflow, mesh rendering, shadows or energy acceptance. '
                  'Plot x is angle in degrees from the normal-facing camera to the light; y is BRDF times cosine '
                  'for unit incident weight. The two Blinn curves use unit highlight weight; GGX uses F0=0.04. '
                  'Their heights are not brightness-normalized. The numeric unit-environment check below is separate.', step=0)
        for name, values in data['curves'].items():
            for degree, value in zip(data['angles'], values.tolist()):
                dash.scalar('lobes/angular-response/'+name, value, step=int(degree))
        dash.record('lobes/analytic-checks', {'lambert_unit_albedo_brdf': 1/math.pi,
            'ggx_r1_unit_fresnel_reflectance': 1-math.log(2), 'blinn_n2_normal_view_reflectance': 25/24,
            'scope': data['scope'], 'note': 'closed-form examples, not a substitute for independent test results'}, step=0)
    send('demo', page='shading/public-lobes-demo' if public else 'shading/lobes-demo', title='How a surface catches light', artifacts=[
        {'kind': kind, 'path': 'lobes/'+path, 'label': label} for kind, path, label in [
            ('image', 'spheres', 'Matte, classic highlight, narrow GGX, broad GGX'),
            ('text', 'read-me', 'How to read this comparison'),
            ('plot', 'angular-response', 'Response as the light moves: degrees'),
            ('record', 'analytic-checks', 'Three independently checkable values')]])
    send('finish', exit_code=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--address', required=True); parser.add_argument('--run', required=True)
    parser.add_argument('--public', action='store_true', help='Use the installed public shading adapter')
    args = parser.parse_args(); publish(args.address, args.run, public=args.public)
