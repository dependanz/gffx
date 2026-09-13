"""Finite image harness contracts, separate from physical resource holdouts."""
import importlib
import hashlib
import json
import math
import os
from pathlib import Path
import time

import pytest
import torch


def tool():
    return importlib.import_module('tools.principled_direct_images')


def test_image_reference_source_pin_matches_saved_corpus_revision():
    from tools import shading_independent_evaluator as scalar
    # Independently read from immutable classic-images-v1 manifest sources.
    expected = '8e1dd0a473034264e79617930b433e1d0ef33aeab98da32154215271034d641c'
    assert hashlib.sha256(Path(scalar.__file__).read_bytes()).hexdigest() == expected
    assert tool().SCALAR_HASH == expected


@pytest.fixture(scope='module')
def budget():
    from gffx.torch.material import PrincipledBudget
    return PrincipledBudget.bundled('ggx-smith-loss-cpu-f64-v1', device='cpu', dtype=torch.float64)


def buffers():
    mask = torch.zeros((128, 128), dtype=torch.bool)
    mask[60, 60:69] = True
    n = torch.zeros((128, 128, 3), dtype=torch.float64)
    n[mask] = torch.tensor([0., 0., 1.], dtype=torch.float64)
    a = torch.zeros_like(n); a[mask] = torch.tensor([.2, .4, .6], dtype=torch.float64)
    return dict(covered=mask, normal=n, view=n.clone(), **{'albedo/constant': a})


@pytest.mark.parametrize('cosine', [0., .00001, .019, .463, 1.])
def test_prepared_scalar_curve_matches_original_without_reconstruction(cosine, monkeypatch):
    from tools.shading_independent_evaluator import LossCurve
    from tools.testing_dashboard.complete_material_demo import synthetic_envelope
    curve = LossCurve(synthetic_envelope()['payload'])
    expected = (curve.loss(cosine, 1., .04), curve.mean(1., .04))
    prepared = tool().PreparedCurve(curve, roughness=1.)
    monkeypatch.setattr(curve, '_angular', lambda *a: pytest.fail('reconstructed per pixel'))
    assert prepared.loss(cosine, 1., .04) == pytest.approx(expected[0], abs=1e-13)
    assert prepared.mean(1., .04) == pytest.approx(expected[1], abs=1e-13)
    with pytest.raises(ValueError, match='roughness'):
        prepared.loss(cosine, .7, .04)


def test_direct_tiles_compare_independent_pixels_and_keep_background(budget, monkeypatch):
    from gffx.torch import material
    calls = []
    original = material.principled
    def observed(n, *args, **kwargs):
        calls.append((len(n), torch.is_grad_enabled()))
        return original(n, *args, **kwargs)
    monkeypatch.setattr(material, 'principled', observed)
    result = tool().evaluate(buffers(), 'constant', budget, deadline=time.perf_counter()+30)
    assert calls == [(8, False), (1, False)]
    assert result['metrics']['passed']
    assert result['metrics']['covered_pixels'] == 9
    assert torch.count_nonzero(result['actual'][~buffers()['covered']]) == 0
    assert not result['actual'].requires_grad
    assert result['actual'][60, 60, 2] > result['actual'][60, 60, 0] > 0


def test_prepared_real_curve_retains_log_segments_and_independent_means(budget):
    from tools.shading_independent_evaluator import LossCurve
    original = LossCurve(budget.to_envelope()['payload'])
    prepared = tool().PreparedCurve(original, roughness=1.)
    for cosine in (.000001, .019, .463, 1.):
        assert prepared.loss(cosine, 1., .04) == pytest.approx(original.loss(cosine, 1., .04), abs=1e-13)
    assert prepared.mean(1., .04) == pytest.approx(original.mean(1., .04), abs=1e-13)


@pytest.mark.parametrize('field', ['normal', 'view', 'albedo/constant'])
def test_image_tool_rejects_training_graphs_before_public_calls(field, budget, monkeypatch):
    from gffx.torch import material
    data = buffers(); data[field].requires_grad_()
    monkeypatch.setattr(material, 'principled', lambda *a, **k: pytest.fail('called before preflight'))
    with pytest.raises(ValueError, match='forward-only'):
        tool().evaluate(data, 'constant', budget, deadline=time.perf_counter()+30)


@pytest.mark.parametrize('change', ['empty', 'shape', 'dtype', 'nonfinite', 'zero-normal', 'colour'])
def test_image_tool_rejects_invalid_buffers_before_reference_work(change, budget):
    data = buffers()
    if change == 'empty': data['covered'].zero_()
    if change == 'shape': data['normal'] = data['normal'][:2]
    if change == 'dtype': data['normal'] = data['normal'].float()
    if change == 'nonfinite': data['view'][60, 60, 0] = math.nan
    if change == 'zero-normal': data['normal'][60, 60] = 0
    if change == 'colour': data['albedo/constant'][60, 60, 0] = 2
    with pytest.raises(ValueError):
        tool().evaluate(data, 'constant', budget, deadline=time.perf_counter()+30)


def test_image_tool_honours_expired_deadline(budget):
    with pytest.raises(TimeoutError):
        tool().evaluate(buffers(), 'constant', budget, deadline=0)


def test_image_comparator_rejects_wrong_channels_light_and_background(budget):
    from tools.shading_image_reference import compare_images
    result = tool().evaluate(buffers(), 'constant', budget, deadline=time.perf_counter()+30)
    for wrong in (result['actual'].flip(-1), result['actual']*2, result['actual']+.01):
        assert not compare_images(wrong, result['reference'], buffers()['covered'])['passed']


def test_explicit_public_tiles_preserve_shared_parameter_gradients(budget):
    from gffx.torch import material
    n = torch.tensor([[.1, .2, 1.]]*8, dtype=torch.float64, requires_grad=True)
    v = torch.tensor([[.2, .1, 1.]]*8, dtype=torch.float64, requires_grad=True)
    l = torch.tensor([[1., 1., 2.]], dtype=torch.float64, requires_grad=True)
    a = torch.tensor([.2, .4, .6], dtype=torch.float64, requires_grad=True)
    r = torch.tensor(.73, dtype=torch.float64, requires_grad=True)
    f = torch.tensor(.04, dtype=torch.float64, requires_grad=True)
    def shade(start, end):
        return material.principled(n[start:end], v[start:end], l, albedo=a, roughness=r, f0=f, budget=budget)
    # Eight points fit forward, but not this complete-material backward budget.
    with pytest.raises(ValueError, match='workspace budget exceeded'):
        shade(0, 8)
    whole = shade(0, 4)
    tiled = torch.cat([shade(0, 2), shade(2, 4)])
    assert torch.allclose(whole, tiled, atol=1e-12, rtol=1e-12)
    inputs = (n, v, l, a, r, f)
    for x, y in zip(torch.autograd.grad(whole.sum(), inputs), torch.autograd.grad(tiled.sum(), inputs)):
        assert torch.isfinite(x).all()
        assert torch.allclose(x, y, atol=1e-11, rtol=1e-10)


def test_image_worker_refuses_existing_payload_before_launch(tmp_path, monkeypatch):
    module = tool()
    output = tmp_path/'result.json'
    output.with_suffix('.actual.bin').write_bytes(b'preserve')
    monkeypatch.setattr(module, 'supervise', lambda *a, **k: pytest.fail('launched overwrite'))
    with pytest.raises(FileExistsError):
        module.launch(tmp_path, output, tmp_path/'wheel.whl', 'constant')
    assert output.with_suffix('.actual.bin').read_bytes() == b'preserve'


def test_saved_image_reader_rejects_changed_report_before_loading_arrays(tmp_path):
    path = tmp_path/'report.json'; path.write_bytes(b'{}')
    with pytest.raises(ValueError, match='report identity'):
        tool().read_measured(path, '0'*64, buffers()['covered'])


@pytest.mark.parametrize('colour', ['constant', 'vertex', 'gradient', 'checker'])
def test_saved_full_head_image_rechecks_hashes_and_linear_errors(colour, request):
    root = os.environ.get('GFFX_DIRECT_IMAGE_REPORTS')
    corpus = os.environ.get('GFFX_DIRECT_IMAGE_CORPUS')
    if not root or not corpus:
        pytest.skip('explicit measured image reports and immutable corpus required')
    from tools.shading_image_bundle import load_bundle
    from tools.testing_dashboard.pytest_plugin import Evidence
    module = tool()
    mask = load_bundle(Path(corpus), path=module.MANIFEST, sha256=module.MANIFEST_HASH)['buffers']['covered']
    path = Path(root)/('constant-verified.json' if colour == 'constant' else colour+'.json')
    result = module.read_measured(path, hashlib.sha256(path.read_bytes()).hexdigest(), mask)
    evidence = Evidence(request.node.nodeid)
    with evidence.check('Saved image integrity and linear agreement'):
        assert result['metrics']['passed']
        assert result['metrics']['covered_pixels'] == 6536
        assert result['metrics']['background_equal']
    evidence.record('Measured direct-image errors', result['metrics'])


@pytest.mark.parametrize('change', ['payload', 'path', 'worker', 'mask'])
def test_saved_image_rejects_tampering_and_failed_supervision(change, tmp_path):
    root = os.environ.get('GFFX_DIRECT_IMAGE_REPORTS')
    corpus = os.environ.get('GFFX_DIRECT_IMAGE_CORPUS')
    if not root or not corpus:
        pytest.skip('explicit measured image reports and immutable corpus required')
    module = tool(); source = Path(root)/'constant-verified.json'
    record = json.loads(source.read_bytes())
    path = tmp_path/source.name
    for name in ('actual', 'reference'):
        filename = record['arrays'][name]['path']
        (tmp_path/filename).write_bytes((Path(root)/filename).read_bytes())
    mask = module.load_bundle(Path(corpus), path=module.MANIFEST, sha256=module.MANIFEST_HASH)['buffers']['covered'].clone()
    if change == 'payload':
        target = tmp_path/record['arrays']['actual']['path']
        blob = bytearray(target.read_bytes()); blob[0] ^= 1; target.write_bytes(blob)
    if change == 'path': record['arrays']['actual']['path'] = '../unexpected.bin'
    if change == 'worker': record['supervisor']['exit_code'] = 1
    if change == 'mask': mask[0, 0] = True
    raw = json.dumps(record, allow_nan=False).encode(); path.write_bytes(raw)
    with pytest.raises(ValueError):
        module.read_measured(path, hashlib.sha256(raw).hexdigest(), mask)
