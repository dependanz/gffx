"""Publish four saved, hash-checked comparisons without repeating shading work."""
import argparse
from pathlib import Path

from tools.principled_direct_images import MANIFEST, MANIFEST_HASH, WHEEL_HASH, load_bundle, read_measured
from .image_reference_demo import display_rgb


def publish(address, run, results):
    from gffx.dashboard import ArrayView, connect
    from .reporter import Reporter
    reporter = Reporter(address, run)
    def send(event, **values):
        if not reporter.send(event, **values): raise RuntimeError(reporter.error)
    send('start', kind='demo', provenance=dict(artifact_sha256=WHEEL_HASH))
    artifacts = []
    with connect(address, run=run, wait_acks=True) as dash:
        dash.text('principled/read-me',
            'Complete principled direct lighting using the accepted table, compared with the independent '
            'scalar evaluator. Four saved albedo fields, roughness 1, F0 .04, the same white directional '
            'light and unchanged geometry. Each group is reference, public result, then error divided by '
            '1e-10 + 1e-8*abs(reference): black is zero error; white reaches the tolerance. '
            'Display-only sRGB/clipping does not affect the stored linear arrays. The head is a numerical '
            'fixture, not an artistic showcase. No independent raster, complete environment, GPU, '
            'training-memory, Cycles or real-model M1 claim. Each image was a separate capped CPU worker; '
            'this page replays saved data and starts no renderer.', step=0)
        artifacts.append(dict(kind='text', path='principled/read-me', label='What these images prove'))
        for colour, item in results.items():
            error = (item['actual']-item['reference']).abs()/(1e-10+1e-8*item['reference'].abs())
            for name, pixels in (('reference', display_rgb(item['reference'])),
                                 ('public', display_rgb(item['actual'])),
                                 ('error-over-tolerance', (error.clamp(0, 1)*255).round().byte())):
                path = 'principled/'+colour+'/'+name
                dash.image(path, ArrayView('uint8', (128, 128, 3), bytes(pixels.flatten().tolist())), step=0)
                artifacts.append(dict(kind='image', path=path, label=colour+' — '+name))
        records = {colour: dict(report_sha256=item['report_sha256'], metrics=item['metrics'],
                    supervisor=item['record']['supervisor'], arrays=item['record']['arrays'],
                    source_hashes=item['record']['sources']) for colour, item in results.items()}
        dash.record('principled/evidence', dict(wheel_sha256=WHEEL_HASH,
                    manifest_sha256=MANIFEST_HASH, images=records), step=0)
        artifacts.append(dict(kind='record', path='principled/evidence', label='Linear errors, limits and identities'))
    send('demo', page='shading/principled-direct-demo', title='Principled direct images: independent comparison', artifacts=artifacts)
    send('finish', exit_code=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--input', nargs=2, action='append', required=True, metavar=('REPORT', 'SHA256'))
    parser.add_argument('--address', required=True); parser.add_argument('--run', required=True)
    args = parser.parse_args()
    if len(args.input) != 4: parser.error('exactly four named reports required')
    bundle = load_bundle(args.root, path=MANIFEST, sha256=MANIFEST_HASH)
    results = {}
    for path, digest in args.input:
        item = read_measured(Path(path), digest, bundle['buffers']['covered'])
        colour = item['record']['albedo']
        if colour in results: raise ValueError('duplicate albedo report')
        results[colour] = item
    if set(results) != {'constant', 'vertex', 'gradient', 'checker'}: raise ValueError('missing image pass')
    publish(args.address, args.run, results)
    print('Published four measured images / twelve comparison panels; no shading repeated.')


if __name__ == '__main__': main()
