"""Finite public-material comparison using an explicitly synthetic loss table."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import zipfile
import torch
from gffx.torch import material, light
from .image_reference_demo import display_rgb


def synthetic_envelope():
    payload = dict(roughness_nodes=[.001, 1.], cosine_nodes=[0., 1.],
                   losses=[[[.8, 0.], [.6, .3]], [[.9, 0.], [.7, .4]]])
    blob = json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return dict(format_version=1, model='ggx-correlated-smith-schlick-reciprocal-matte-v1',
        representation='bounded-cubic-loss-v1', dtype='float64',
        domain=dict(roughness=[.001, 1.], cosine=[0, 1], f0=[0, 1]),
        generator=dict(source_sha256=None, policy='synthetic demonstration only'),
        evidence_id='synthetic-not-physical-acceptance',
        payload_sha256=hashlib.sha256(blob).hexdigest(), payload=payload)


@torch.no_grad()
def evaluate(*, size=128):
    if type(size) is not int or not 8 <= size <= 128:
        raise ValueError('demo size must lie within 8..128')
    axis = torch.linspace(-1, 1, size, dtype=torch.float64)
    y, x = torch.meshgrid(axis, axis, indexing='ij')
    mask = x.square()+y.square() < .98**2
    normals = torch.stack((x[mask], -y[mask], (1-x[mask].square()-y[mask].square()).sqrt()), -1)
    view = torch.tensor([0., 0., 1.], dtype=torch.float64).expand_as(normals)
    colour = torch.tensor([.18, .42, .65], dtype=torch.float64).expand_as(normals).clone()
    stripe = ((x[mask]*8).floor().long()+ (y[mask]*8).floor().long()) % 2 == 0
    colour[stripe] *= .55
    directions = torch.tensor([[.7, .5, 1.4]], dtype=torch.float64)
    resource = material.PrincipledBudget.from_envelope(synthetic_envelope(),
                   device='cpu', dtype=torch.float64, allow_unverified=True)
    def lambert(n, v, l, *, parameters):
        return parameters['colour']/math.pi
    images = {}
    # Explicit bounded forward-only display tiles, not a hidden renderer graph
    # chunker. Every call has at most 8*3=24 budget entries; no retained backward.
    for name in ('classic', 'principled-r025', 'principled-r070', 'builtin-lambert', 'custom-lambert'):
        pixels = torch.zeros_like(normals)
        for start in range(0, len(normals), 8):
            end = start+8
            n, v, a = normals[start:end], view[start:end], colour[start:end]
            if name.startswith('principled'):
                roughness = .25 if name.endswith('025') else .7
                brdf = material.principled(n, v, directions, albedo=a,
                    roughness=torch.tensor(roughness, dtype=torch.float64),
                    f0=torch.tensor(.04, dtype=torch.float64), budget=resource,
                    allow_experimental_budget=True)
            elif name == 'custom-lambert':
                brdf = material.custom(lambert, n, v, directions, parameters={'colour': a})
            else:
                brdf = material.classic(n, v, directions, albedo=a,
                    specular=torch.tensor(.12 if name == 'classic' else 0., dtype=torch.float64),
                    exponent=torch.tensor(24., dtype=torch.float64))
            packet = dict(measure='delta', direction=directions.expand(len(n), 1, 3),
                irradiance=torch.tensor([2., 1.8, 1.6], dtype=torch.float64).expand(len(n), 1, 3),
                valid=torch.ones((len(n), 1), dtype=torch.bool))
            pixels[start:end] = light.integrate(n, brdf, packet)
        image = torch.full((size, size, 3), .018, dtype=torch.float64)
        image[mask] = pixels
        images[name] = image
    return dict(images=images, covered_pixels=int(mask.sum()), tile_points=8,
        custom_max_abs_error=float((images['builtin-lambert']-images['custom-lambert']).abs().max()),
        scope='Experimental synthetic principled budget: interface demonstration, NOT physical accuracy or energy acceptance. '
              'Analytic sphere normals with direct lighting only; no rasterizer, shadows, environment transport, CUDA or M1 claim. '
              'Classic is empirical. Built-in and custom Lambert should match. Display uses sRGB and clipping; metrics are linear.')


def publish(address, run, result, record):
    from gffx.dashboard import ArrayView, connect
    from .reporter import Reporter
    reporter = Reporter(address, run)
    def send(event, **values):
        if not reporter.send(event, **values):
            raise RuntimeError(reporter.error)
    send('start', kind='demo', provenance=dict(artifact_sha256=record['candidate_wheel_sha256']))
    artifacts = []
    with connect(address, run=run, wait_acks=True) as dash:
        dash.text('materials/read-me', result['scope'], step=0)
        artifacts.append(dict(kind='text', path='materials/read-me', label='Read first: experimental resource'))
        for name, linear in result['images'].items():
            pixels = display_rgb(linear)
            path = 'materials/'+name
            dash.image(path, ArrayView('uint8', tuple(pixels.shape), bytes(pixels.flatten().tolist())), step=0)
            artifacts.append(dict(kind='image', path=path, label=name))
        dash.record('materials/evidence', record, step=0)
        artifacts.append(dict(kind='record', path='materials/evidence', label='Scope, equality and wheel provenance'))
    send('demo', page='shading/complete-material-demo', title='Complete materials and custom callbacks — experimental', artifacts=artifacts)
    send('finish', exit_code=0)


def main():
    import gffx
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheel', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--address'); parser.add_argument('--run')
    args = parser.parse_args()
    if bool(args.address) != bool(args.run):
        parser.error('address and run required together')
    if args.report.exists():
        parser.error('report already exists')
    torch.set_num_threads(1)
    root = Path(gffx.__file__).resolve().parent.parent
    with zipfile.ZipFile(args.wheel) as archive:
        for name in archive.namelist():
            if name.startswith('gffx/') and not name.endswith('/') and archive.read(name) != (root/name).read_bytes():
                raise ValueError('installed wheel mismatch: '+name)
    result = evaluate()
    record = {key: value for key, value in result.items() if key != 'images'}
    record.update(candidate_wheel_sha256=hashlib.sha256(args.wheel.read_bytes()).hexdigest(),
                  producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  synthetic_payload_sha256=synthetic_envelope()['payload_sha256'])
    with args.report.open('x', encoding='utf-8') as handle:
        json.dump(record, handle, indent=2, allow_nan=False)
    print(json.dumps(record))
    if result['custom_max_abs_error'] > 1e-12:
        raise ValueError('custom Lambert equivalence failed')
    if args.address:
        publish(args.address, args.run, result, record)


if __name__ == '__main__':
    main()
