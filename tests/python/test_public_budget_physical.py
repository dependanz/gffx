"""Fixed fresh public-query cohort; opt-in saved fixtures, never grid generation."""
import importlib.util
import json
import os
from pathlib import Path
import time
import pytest
from tools import public_budget_probe as probe
from tools import material_loss_cost as cost


@pytest.fixture
def measurement(request, monkeypatch, tmp_path):
    location = os.environ.get('GFFX_PUBLIC_BUDGET_FIXTURES')
    if not location:
        pytest.skip('requires explicit supervised public-budget worker')
    import torch
    from gffx.torch.material import PrincipledBudget
    deadline = float(os.environ['GFFX_PUBLIC_BUDGET_DEADLINE'])
    check = lambda: cost.check_deadline(deadline)
    check()
    spec = importlib.util.spec_from_file_location('public_budget_controls', Path(__file__).with_name('test_material_loss_resource.py'))
    controls = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(controls)
    controls.SCREENS['public_v1'] = (probe.ROUGHNESSES, probe.VIEWS, probe.FRESNELS, .0027)
    original_save = controls.save_rows
    def save_rows(request, root, name, rows):
        Path(os.environ['GFFX_PUBLIC_BUDGET_ROWS']).write_text(json.dumps(rows, indent=2, allow_nan=False), encoding='utf-8')
        original_save(request, root, name, rows)
    monkeypatch.setattr(controls, 'save_rows', save_rows)
    resources = []
    for name, digest in cost.HASHES.items():
        check()
        data = cost.load_json(Path(location)/(name+'.json'), digest)
        public = PrincipledBudget.from_envelope(probe.envelope(data, name), device='cpu', dtype=torch.float64, allow_unverified=True)
        assert public.metadata['admission'] == 'experimental'
        resources.append(probe.PublicComparison(public, cost.construct(data), check))
    yield controls, (*resources, tmp_path)
    check()


def test_public_directional_values_and_gradients(measurement, request):
    controls, resources = measurement
    controls.test_full_resource_held_out_directional_accuracy_and_gradients(resources, request, 'public_v1')


def test_public_mean_values_and_gradients(measurement, request):
    controls, resources = measurement
    controls.test_full_resource_mean_matches_reference_and_its_gradients(resources, request, 'public_v1')


def test_public_furnace_energy_balance(measurement, request):
    controls, resources = measurement
    controls.test_full_resource_independently_integrated_furnace(resources, request, 'public_v1')


@pytest.mark.parametrize('stage', ['guard', 'directional', 'mean', 'furnace'])
def test_saved_supervised_stage_evidence(stage, request):
    """Publish already measured rows without repeating heavy computation."""
    from tools.testing_dashboard.pytest_plugin import Evidence
    location = os.environ.get('GFFX_PUBLIC_BUDGET_REPORTS')
    if not location:
        pytest.skip('explicit saved measurement reports required')
    report = json.loads((Path(location)/(stage+'.json')).read_text(encoding='utf-8'))
    evidence = Evidence(request.node.nodeid)
    evidence.record('supervised-physical-measurements', report)
    rows, supervisor = report['rows'], report['supervisor']
    with evidence.check('Complete supervised provenance'):
        assert report['mode'] == stage
        assert report['status'] == stage+' passed; not public admission'
        assert report['passed'] is True
        assert len(rows) == dict(guard=80, directional=80, mean=16, furnace=192)[stage]
        assert report['references'] == probe.FROZEN and report['fixture_hashes'] == cost.HASHES
        assert report['wheel']['sha256'] == probe.WHEEL_SHA
        assert report['worker_pid'] == supervisor['monitored_pid']
        assert supervisor['exit_code'] == 0 and supervisor['failure'] is None
        assert 0 < supervisor['elapsed_seconds'] <= 180
        assert 0 < supervisor['sampled_peak_working_set_bytes'] <= 1024**3
    with evidence.check('Unchanged numerical acceptance thresholds'):
        if stage == 'guard':
            assert probe.guard_passed(rows)
        elif stage == 'directional':
            assert all(r['absolute_error'] <= 5e-4 and r['relative_error'] <= .001 and
                       r['refinement'] <= .001 and max(r['gradient_ratios']) <= 1 for r in rows)
        elif stage == 'mean':
            assert all(r['relative_error'] <= .001 and max(r['gradient_ratios']) <= 1 and
                       r['numerical_error'] <= 1e-11 and max(r['numerical_gradient_errors']) <= 1e-11 for r in rows)
        else:
            assert all(0 <= r['energy'] <= 1.001 and r['white_error'] < .001 and r['refinement'] < .001 for r in rows)
