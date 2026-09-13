"""Finite reviewed animation transport; loading/publishing imports no renderer.

The render probe writes raw cache arrays once. Publication verifies the immutable
report, wheel, sources and all arrays before any network message is sent.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

CAP = 16 * 1024**2


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            value.update(block)
    return value.hexdigest()


def publication_provenance(report_sha256):
    from gffx.dashboard._server import check_provenance
    return check_provenance({'artifact_sha256': report_sha256})


def cache_payload(data, directory):
    """Render-worker-only writer, never called by the cached publisher."""
    from tools.testing_dashboard.scene_showcase import view
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    total = 0
    def save(tensor):
        nonlocal total
        array = view(tensor)
        total += len(array.data)
        if total > CAP:
            raise MemoryError('finite animation cache exceeds16MiB')
        sha = hashlib.sha256(array.data).hexdigest()
        path = directory / (sha + '.bin')
        if path.exists():
            if path.is_symlink() or digest(path) != sha:
                raise ValueError('existing cache identity differs')
        else:
            with path.open('xb') as stream:
                stream.write(array.data)
        return dict(dtype=array.dtype, shape=list(array.shape), sha256=sha, bytes=len(array.data))
    images = [save(value) for value in data['images']]
    clip = []
    for frame in data['clip']:
        row = {key: save(frame[key]) for key in ('vertices', 'faces', 'colors', 'normals')}
        row['markers'] = frame['markers'].tolist()
        row['max_native_reference_error'] = frame['max_native_reference_error']
        clip.append(row)
    return dict(images=images, clip=clip, maximum_error=data['maximum_error'], logical_bytes=total)


def load_cached(report, wheel, expected, source_root, report_sha256):
    from gffx.dashboard import ArrayView
    report, root = Path(report).resolve(), Path(source_root).resolve()
    if report.stat().st_size > 1024**2 or digest(report) != report_sha256:
        raise ValueError('report identity differs')
    row = json.loads(report.read_text(encoding='utf-8'))
    if row.get('status') != 'passed' or row.get('mode') != 'render':
        raise ValueError('report is not a passed render')
    supervision = row.get('supervisor', {})
    if supervision.get('exit_code') != 0 or supervision.get('failure'):
        raise ValueError('render supervision failed')
    if row.get('wheel', {}).get('sha256') != expected or digest(wheel) != expected:
        raise ValueError('wheel identity differs')
    if not row.get('sources'):
        raise ValueError('source identities missing')
    for name, source_hash in row['sources'].items():
        source = (root / name).resolve()
        if not source.is_relative_to(root) or digest(source) != source_hash:
            raise ValueError('source identity differs')
    cache = row.get('cache', {})
    if len(cache.get('images', [])) != 3 or len(cache.get('clip', [])) != 24:
        raise ValueError('expected three images and24frames')
    total = 0
    array_root = report.parent / 'arrays'
    if array_root.is_symlink() or array_root.resolve().parent != report.parent:
        raise ValueError('cache root escapes report')
    def read(entry, dtype, rank):
        nonlocal total
        shape, sha = entry.get('shape'), entry.get('sha256', '')
        if entry.get('dtype') != dtype or not isinstance(shape, list) or len(shape) != rank:
            raise ValueError('cache array kind differs')
        if any(type(value) is not int or not 0 < value <= 10000 for value in shape):
            raise ValueError('cache shape outside bounds')
        if shape[-1] != 3 or len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
            raise ValueError('cache identity malformed')
        size = math.prod(shape) * (1 if dtype == 'uint8' else 4)
        total += size
        if size != entry.get('bytes') or total > CAP:
            raise ValueError('cache size outside bounds')
        path = array_root / (sha + '.bin')
        if path.is_symlink() or path.resolve().parent != array_root.resolve():
            raise ValueError('cache path identity differs')
        if path.stat().st_size != size or digest(path) != sha:
            raise ValueError('cache array identity differs')
        return ArrayView(dtype, tuple(shape), path.read_bytes())
    images = [read(entry, 'uint8', 3) for entry in cache['images']]
    if any(image.shape not in ((180, 320, 3), (720, 1280, 3)) for image in images):
        raise ValueError('unreviewed image dimensions')
    clip = []
    for source in cache['clip']:
        frame = {key: read(source[key], 'int32' if key == 'faces' else 'float32', 2)
                 for key in ('vertices', 'faces', 'colors', 'normals')}
        if not (frame['vertices'].shape == frame['colors'].shape == frame['normals'].shape):
            raise ValueError('mesh attribute shapes differ')
        markers = source.get('markers')
        if not isinstance(markers, list) or len(markers) != 3 or any(
                not isinstance(point, list) or len(point) != 3 or any(
                    type(x) not in (int, float) or not math.isfinite(x) for x in point) for point in markers):
            raise ValueError('invalid marker coordinates')
        error = source.get('max_native_reference_error')
        if type(error) not in (int, float) or not math.isfinite(error) or not 0 <= error <= 2e-6:
            raise ValueError('unqualified displayed comparison')
        frame.update(markers=markers, max_native_reference_error=error)
        clip.append(frame)
    if total != cache.get('logical_bytes') or cache.get('maximum_error') != max(frame['max_native_reference_error'] for frame in clip):
        raise ValueError('cache summary differs')
    return row, dict(images=images, clip=clip, maximum_error=cache['maximum_error'])


def send_payload(dash, data, evidence):
    artifacts = []
    def add(kind, path, label, *values, **kwargs):
        getattr(dash, kind)(path, *values, step=0, **kwargs)
        artifacts.append(dict(kind=kind, path=path, label=label))
    add('image', 'bloom/hero', 'KINETIC BLOOM · a five-joint deformable sculpture', data['images'][0])
    add('text', 'bloom/guide', 'What is actually moving?',
        'The turquoise and coral sails share a five-joint skeleton. Two corrective shapes breathe the surface; '
        'Rodrigues rotations, forward kinematics, nonidentity inverse bind and LBS move every vertex. '
        'Play the finite24-frame loop and inspect the cached studio stills. Embedded markers are barycentric. '
        'This is authored motion, not physics or an animation scheduler. Playback FPS is not measured solver throughput. '
        'Native rasterization with inherited display lighting makes the stills, not Principled/Cycles acceptance. '
        'The separate independent-target recovery test, not this authored clip, is numerical fitting evidence. '
        'Cached evidence: ' + json.dumps(evidence, sort_keys=True))
    queue = dash.queue('bloom/motion', 'mesh', recent_window=24, snapshot_every=1)
    for index, frame in enumerate(data['clip']):
        queue.push(frame['vertices'], frame['faces'], colors=frame['colors'], normals=frame['normals'], step=index)
        dash.scalar('bloom/error/maximum-coordinate', frame['max_native_reference_error'], step=index)
        for axis, name in enumerate(('x', 'y', 'z')):
            dash.scalar('bloom/landmark/' + name, frame['markers'][1][axis], step=index)
    artifacts.extend((dict(kind='mesh', path='bloom/motion', label='24 native-rig poses · stationary plinth'),
        dict(kind='plot', path='bloom/landmark', label='Embedded sail marker · x/y/z position'),
        dict(kind='plot', path='bloom/error', label='Displayed vertex comparison · maximum coordinate error')))
    for index in (1, 2):
        add('image', f'bloom/still-{index}', f'Pose study {index + 1}', data['images'][index])
    add('text', 'bloom/comparison', 'What the numerical comparison covers',
        f"Maximum absolute displayed-coordinate difference: {data['maximum_error']:.9g}. "
        'Independent corrective addition and skinning use the same joint matrices, so this does not independently '
        'validate FK/Rodrigues; separate hierarchy, matrix-exponential and derivative tests do. Synthetic geometry only.')
    return artifacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('report', 'wheel', 'source-root', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('sha256', 'report-sha256', 'address', 'run'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('preserve previous publication outcome')
    row, data = load_cached(args.report, args.wheel, args.sha256, args.source_root, args.report_sha256)
    from gffx.dashboard import connect
    from tools.testing_dashboard.reporter import Reporter
    provenance = publication_provenance(args.report_sha256)
    reporter = Reporter(args.address, args.run)
    outcome = dict(status='started', run=args.run, report_sha256=args.report_sha256, wheel_sha256=args.sha256)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(outcome, stream, indent=2)
    def send(event, **kwargs):
        if not reporter.send(event, **kwargs):
            raise RuntimeError(reporter.error)
    try:
        send('start', kind='demo', provenance=provenance)
        with connect(args.address, run=args.run, wait_acks=True, provenance=provenance) as dash:
            artifacts = send_payload(dash, data, {'render_report_sha256': args.report_sha256,
                'wheel_sha256': args.sha256, 'sources': row['sources'], 'cached_arrays': True})
        send('demo', page='animation/deformation-demo', title='Kinetic Bloom · a differentiable articulated sculpture', artifacts=artifacts)
        send('finish', exit_code=0)
        outcome['status'] = 'passed'
    except Exception as error:
        outcome.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        args.output.write_text(json.dumps(outcome, indent=2), encoding='utf-8')
    print(json.dumps(outcome))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
