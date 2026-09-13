"""Cached animation evidence must be validated before any dashboard message."""
import hashlib
import json

import pytest


def fixture(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'demo.py').write_text('source', encoding='utf-8')
    wheel = tmp_path / 'wheel.whl'
    wheel.write_bytes(b'wheel')
    arrays = tmp_path / 'arrays'
    arrays.mkdir()
    def save(dtype, shape, blob):
        sha = hashlib.sha256(blob).hexdigest()
        (arrays / (sha + '.bin')).write_bytes(blob)
        return dict(dtype=dtype, shape=shape, sha256=sha, bytes=len(blob))
    image = save('uint8', [180, 320, 3], b'\x42' * (180 * 320 * 3))
    value = save('float32', [1, 3], bytes(12))
    topology = save('int32', [1, 3], bytes(12))
    frame = dict(vertices=value, normals=value, colors=value, faces=topology,
                 markers=[[0., 0., 0.]] * 3, max_native_reference_error=0.)
    row = dict(status='passed', mode='render', supervisor=dict(exit_code=0, failure=None),
               wheel=dict(sha256=hashlib.sha256(b'wheel').hexdigest()),
               sources={'demo.py': hashlib.sha256(b'source').hexdigest()},
               cache=dict(images=[image] * 3, clip=[frame] * 24, maximum_error=0.,
                          logical_bytes=3 * 180 * 320 * 3 + 24 * 4 * 12))
    report = tmp_path / 'render.json'
    report.write_text(json.dumps(row), encoding='utf-8')
    return report, wheel, row['wheel']['sha256'], source


def test_animation_cached_payload_loads_exact_finite_frames(tmp_path):
    from tools.animation_showcase_cached import digest, load_cached
    report, wheel, expected, root = fixture(tmp_path)
    row, payload = load_cached(report, wheel, expected, root, digest(report))
    assert len(payload['images']) == 3 and len(payload['clip']) == 24
    assert payload['clip'][0]['vertices'].shape == (1, 3)
    assert row['cache']['logical_bytes'] < 16 * 1024**2


@pytest.mark.parametrize('damage', ['report', 'wheel', 'source', 'array'])
def test_animation_cached_payload_rejects_tampered_identity(tmp_path, damage):
    from tools.animation_showcase_cached import digest, load_cached
    report, wheel, expected, root = fixture(tmp_path)
    sha = digest(report)
    raw = next((tmp_path / 'arrays').glob('*.bin'))
    {'report': report, 'wheel': wheel, 'source': root / 'demo.py', 'array': raw}[damage].write_bytes(b'changed')
    with pytest.raises(ValueError, match='identity'):
        load_cached(report, wheel, expected, root, sha)


@pytest.mark.parametrize('damage', ['budget', 'marker', 'frames'])
def test_animation_cached_payload_rejects_invalid_bounds(tmp_path, damage):
    from tools.animation_showcase_cached import digest, load_cached
    report, wheel, expected, root = fixture(tmp_path)
    row = json.loads(report.read_text())
    if damage == 'budget':
        row['cache']['images'][0].update(shape=[10000, 10000, 3], bytes=300000000)
    elif damage == 'marker':
        row['cache']['clip'][0]['markers'][0][0] = float('nan')
    else:
        row['cache']['clip'].pop()
    report.write_text(json.dumps(row))
    with pytest.raises(ValueError):
        load_cached(report, wheel, expected, root, digest(report))


def test_animation_publication_uses_real_provenance_contract(tmp_path, monkeypatch):
    from tools import animation_showcase_cached as cached
    from tools.testing_dashboard import reporter
    from gffx.dashboard._server import check_provenance
    import gffx.dashboard
    import sys
    report, wheel, expected, root = fixture(tmp_path)
    report_hash = cached.digest(report)
    events, images, frames = [], [], []
    class Reporter:
        def __init__(self, *args):
            pass
        def send(self, event, **values):
            if event == 'start':
                check_provenance(values['provenance'])
            events.append((event, values))
            return True
    class Dash:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def image(self, *args, **kwargs):
            images.append(args)
        def text(self, *args, **kwargs):
            pass
        def scalar(self, *args, **kwargs):
            pass
        def queue(self, *args, **kwargs):
            assert kwargs == dict(recent_window=24, snapshot_every=1)
            return self
        def push(self, *args, **kwargs):
            frames.append(args)
    def connect(*args, **kwargs):
        assert check_provenance(kwargs['provenance']) == {'artifact_sha256': report_hash}
        return Dash()
    monkeypatch.setattr(reporter, 'Reporter', Reporter)
    monkeypatch.setattr(gffx.dashboard, 'connect', connect)
    outcome = tmp_path / 'publish.json'
    monkeypatch.setattr(sys, 'argv', ['cached', '--report', str(report), '--wheel', str(wheel),
        '--sha256', expected, '--report-sha256', report_hash, '--source-root', str(root),
        '--output', str(outcome), '--address', 'http://unused', '--run', 'fixture'])
    assert cached.main() == 0
    assert events[0][1]['provenance'] == {'artifact_sha256': report_hash}
    assert events[-1] == ('finish', {'exit_code': 0})
    assert len(images) == 3 and len(frames) == 24
    assert json.loads(outcome.read_text())['status'] == 'passed'
    with pytest.raises(ValueError, match='previous'):
        cached.main()
