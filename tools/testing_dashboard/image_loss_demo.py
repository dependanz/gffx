"""Finite direct-image fitting snapshots, published only after strict replay."""
import argparse
import json
from pathlib import Path
import torch
from tools import image_loss_gradients as fixture
from .image_reference_demo import display_rgb


def panels(report):
    ids = report['indices']; result = report['result']; images = []
    for values in (result['target'], result['passes'][0]['values'], result['passes'][1]['values']):
        image = torch.zeros((128, 128, 3), dtype=torch.float64)
        image.reshape(-1, 3)[ids] = torch.tensor(values, dtype=torch.float64)
        images.append(display_rgb(image))
    gradient = torch.tensor(result['passes'][0]['gradient'], dtype=torch.float64)
    scale = float(gradient.abs().max())
    if not torch.isfinite(gradient).all() or scale <= 0: raise ValueError('finite nonzero display gradient required')
    return dict(images=torch.cat(images, dim=1), gradient=(gradient.abs()/scale*255).round().to(torch.uint8), gradient_scale=scale)


def publish(reports, sha256, address, run):
    from gffx.dashboard import ArrayView, connect
    from .reporter import Reporter
    reporter = Reporter(address, run)
    def send(event, **values):
        if not reporter.send(event, **values): raise RuntimeError(reporter.error)
    send('start', kind='demo', provenance=dict(artifact_sha256=fixture.WHEEL_HASH))
    artifacts = []
    with connect(address, run=run, wait_acks=True) as dash:
        dash.text('image-loss/scope', 'What this proves: image error reaches the original colour texture through gffx shading and native bilinear sampling. '
                  'Both complete 128x128 images include 6,536 covered surface pixels; black is background and is excluded from the loss. '
                  'LEFT: independent target. MIDDLE: initial public output. RIGHT: public output after one gradient step. '
                  'The two fitting passes use freshly built four-pixel graphs; no full image graph is retained. '
                  'This is deliberately a small correctness step, not a converged fit: improvements can be hard to see. '
                  'sRGB conversion is display only. Lighting, geometry, UVs, roughness and visibility stay fixed; this is not environment fitting or M1 completion.', step=0)
        artifacts.append(dict(kind='text', path='image-loss/scope', label='How to read this fitting check'))
        for report in reports:
            if report['pixels'] != 6536: continue
            model = report['model']; base = 'image-loss/'+model; p = panels(report)
            for name, label in (('images', 'Target / before / after one step'), ('gradient', 'Source texture: absolute RGB derivatives, normalized for display')):
                image = p[name]
                dash.image(base+'/'+name, ArrayView('uint8', tuple(image.shape), bytes(image.flatten().tolist())), step=0)
                artifacts.append(dict(kind='image', path=base+'/'+name, label=model+': '+label))
            for index, row in enumerate(report['result']['passes']):
                dash.scalar(base+'/loss/mean-squared-RGB', row['loss'], step=index)
            artifacts.append(dict(kind='plot', path=base+'/loss', label=model+': mean squared linear RGB error (two measurements, not a training history)'))
            dash.record(base+'/measurements', dict(gradient_display_scale=p['gradient_scale'],
                        passes=[{key: value for key, value in row.items() if key not in ('texture', 'values', 'gradient')} for row in report['result']['passes']],
                        supervisor=report['supervisor']), step=0)
            artifacts.append(dict(kind='record', path=base+'/measurements', label=model+': exact losses, errors and sampled worker limits'))
        dash.record('image-loss/evidence', dict(cohort_sha256=sha256, wheel_sha256=fixture.WHEEL_HASH,
                    workers=[{key: r[key] for key in ('model', 'pixels', 'supervisor')} for r in reports],
                    scope='Fixed direct light; texture-colour gradients only; no automatic replay or simulation'), step=0)
        artifacts.append(dict(kind='record', path='image-loss/evidence', label='Exact evidence identity and qualification boundary'))
    send('demo', page='shading/image-loss-gradients-demo', title='Image error to source texture', artifacts=artifacts)
    send('finish', exit_code=0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True); p.add_argument('--sha256', required=True)
    p.add_argument('--address', required=True); p.add_argument('--run', required=True)
    args = p.parse_args(); torch.set_num_threads(1)
    cohort = fixture.replay(args.root, args.sha256)
    reports = [json.loads((args.root/e['path']).read_text(encoding='utf-8')) for e in cohort['workers']]
    publish(reports, args.sha256, args.address, args.run)


if __name__ == '__main__': main()
