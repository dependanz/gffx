"""Measurement plumbing, not new material-accuracy or performance acceptance.

Each test names a safeguard: trusted fixture loading, honest storage accounting,
real parameter gradients, and bounded execution. Synthetic fields keep this
suite inexpensive; the separate CLI measures the frozen physical fixtures.
"""
import hashlib
import json
import pytest

torch = pytest.importorskip('torch')


def fixture_data():
    return dict(roughness_nodes=[.001, 1.], cosine_nodes=[0., .1, 1.],
                losses=[[[.8, 0.], [.7, .2], [.6, .3]],
                        [[.9, 0.], [.8, .3], [.7, .4]]])


def save(tmp_path, value):
    path = tmp_path / 'resource.json'
    blob = json.dumps(value).encode()
    path.write_bytes(blob)
    return path, hashlib.sha256(blob).hexdigest()


def test_fixture_hash_checked_before_parsing(tmp_path):
    """A changed saved grid must fail before any JSON decoding or tensor work."""
    from tools.material_loss_cost import load_json
    path = tmp_path / 'resource.json'
    path.write_bytes(b'not json')
    with pytest.raises(ValueError, match='SHA-256'):
        load_json(path, '0'*64)


@pytest.mark.parametrize('kind', ['bytes', 'nodes', 'shape', 'nonfinite', 'keys'])
def test_loader_rejects_unbounded_or_malformed_inputs(tmp_path, kind):
    """Enforce byte, shape and numeric boundaries before allocating tensors."""
    from tools.material_loss_cost import load_json, MAX_FILE_BYTES
    data = fixture_data()
    if kind == 'nodes':
        data['roughness_nodes'] = [.01]*130
    elif kind == 'shape':
        data['losses'][0][1] = [.5]
    elif kind == 'nonfinite':
        data['losses'][0][1][0] = float('nan')
    elif kind == 'keys':
        data['surprise'] = 1
    path, digest = save(tmp_path, data)
    if kind == 'bytes':
        path.write_bytes(b' '*(MAX_FILE_BYTES+1))
    with pytest.raises(ValueError):
        load_json(path, digest)


def test_storage_counts_constants_without_calling_them_resident_tensors(tmp_path):
    """File size, float64-equivalent constants and Python storage are distinct."""
    from tools.material_loss_cost import load_json, construct, storage
    path, digest = save(tmp_path, fixture_data())
    resource = construct(load_json(path, digest))
    report = storage(resource)
    assert report['raw_grid_float64_bytes'] == (2+3+2*3*2)*8
    assert report['all_constants_float64_equivalent_bytes'] > report['raw_grid_float64_bytes']
    assert report['python_object_bytes'] > report['raw_grid_float64_bytes']
    assert report['persistent_tensor_bytes'] == 0
    assert 'not RSS' in report['qualification']


class AnalyticResource:
    def evaluate(self, c, r, f):
        return dict(loss=c.square()+r*f, mean_loss=r.square()*f)


def test_loop_batch_retains_all_three_parameter_gradients():
    """Backpropagate a complete batch, not disconnected scalar placeholders."""
    from tools.material_loss_cost import inputs, run_batch
    c, r, f = inputs(8, gradients=True)
    result = run_batch(AnalyticResource(), (c, r, f), backward=True)
    expected = (c.square()+r*f+r.square()*f).sum()
    assert result['objective'] == pytest.approx(expected.item())
    assert result['gradient_sums'] == pytest.approx([
        (2*c).sum().item(), (f+2*r*f).sum().item(), (r+r.square()).sum().item()])
    assert c.grad is None and r.grad is None and f.grad is None


def test_no_grad_forward_matches_backward_forward_value():
    """Timing modes must evaluate the same workload, with graph cost explicit."""
    from tools.material_loss_cost import inputs, run_batch
    a = run_batch(AnalyticResource(), inputs(1, gradients=False), backward=False)
    b = run_batch(AnalyticResource(), inputs(1, gradients=True), backward=True)
    assert a['objective'] == b['objective']
    assert a['gradient_sums'] is None


@pytest.mark.parametrize('size', [0, True, 33, 1000000])
def test_batch_limit_rejected_before_input_allocation(size):
    """The measurement tool cannot silently scale this laptop workload up."""
    from tools.material_loss_cost import inputs
    with pytest.raises(ValueError, match='batch'):
        inputs(size, gradients=True)


def test_nonfinite_gradient_is_a_failed_measurement():
    """A finite forward result does not excuse an invalid derivative."""
    from tools.material_loss_cost import inputs, run_batch
    class BadResource:
        def evaluate(self, c, r, f):
            return dict(loss=(c-c).sqrt()+r*f, mean_loss=r*f)
    with pytest.raises(ValueError, match='gradient'):
        run_batch(BadResource(), inputs(1, gradients=True), backward=True)


def test_deadline_stops_before_next_case():
    """Expired work budgets raise rather than quietly extending the experiment."""
    from tools.material_loss_cost import check_deadline
    with pytest.raises(TimeoutError):
        check_deadline(0)


def test_profile_is_fixed_and_does_not_claim_vectorized_support():
    """The saved report must say precisely what was and was not measured."""
    from tools.material_loss_cost import PROFILE
    assert PROFILE['batch_sizes'] == [1, 8, 32]
    assert PROFILE['warmups'] == 1 and PROFILE['repetitions'] == 3
    assert PROFILE['batch_implementation'] == 'scalar loop with retained batch graph'
    assert PROFILE['generation'] == 'not measured; hash-verified saved fixtures'
    assert PROFILE['watchdog_seconds'] == 180


def test_timing_excludes_warmup_and_records_three_raw_samples(monkeypatch):
    """Keep reproducible raw timings, rather than choosing the fastest sample."""
    import tools.material_loss_cost as cost
    calls = []
    ticks = iter(range(16))
    monkeypatch.setattr(cost.time, 'perf_counter', lambda: next(ticks))
    result, report = cost.samples(lambda: calls.append(1) or len(calls), 100)
    assert result == 4 and len(calls) == 4
    assert report == dict(seconds=[1, 1, 1], median_seconds=1)


def test_constructor_rejects_invalid_physical_knots(tmp_path):
    """Valid JSON does not bypass the frozen resource's physical validation."""
    from tools.material_loss_cost import load_json, construct
    data = fixture_data()
    data['losses'][0][1][0] = 1.1
    path, digest = save(tmp_path, data)
    with pytest.raises(ValueError, match='endpoint loss'):
        construct(load_json(path, digest))


def test_working_set_monitor_reads_this_process():
    """Exercise the real OS memory query, without allocating a stress workload."""
    import os
    import sys
    from tools.material_loss_cost import working_set_reader
    if os.name != 'nt' and not sys.platform.startswith('linux'):
        pytest.skip('measurement supervisor targets Windows/Linux')
    read, close = working_set_reader(os.getpid())
    try:
        assert read() > 0
    finally:
        close()


def test_supervisor_stops_over_memory_budget_and_preserves_partial_report(tmp_path, monkeypatch):
    """A simulated limit violation stops only the owned worker, retaining evidence."""
    import tools.material_loss_cost as cost
    class Child:
        pid = 123
        returncode = None
        killed = False
        def poll(self): return self.returncode
        def kill(self): self.killed = True; self.returncode = -9
        def wait(self): return self.returncode
    child = Child()
    closed = []
    monkeypatch.setattr(cost.subprocess, 'Popen', lambda _: child)
    monkeypatch.setattr(cost, 'working_set_reader', lambda _: (
        lambda: cost.PROFILE['sampled_working_set_limit_bytes']+1,
        lambda: closed.append(True)))
    output = tmp_path/'report.json'
    assert cost.supervise(tmp_path, output) == 1
    report = json.loads(output.read_text())
    assert child.killed and closed == [True]
    assert report['status'] == 'stopped'
    assert 'MemoryError' in report['supervisor']['failure']


def test_supervisor_refuses_to_overwrite_prior_evidence(tmp_path, monkeypatch):
    """Repeating an experiment needs a new result path, not a replaced failure."""
    import tools.material_loss_cost as cost
    output = tmp_path/'report.json'
    output.write_text('saved failure')
    monkeypatch.setattr(cost.subprocess, 'Popen', lambda _: pytest.fail('must not launch'))
    with pytest.raises(ValueError, match='previous experiment'):
        cost.supervise(tmp_path, output)
    assert output.read_text() == 'saved failure'


def test_worker_launch_bypasses_venv_redirector_but_keeps_import_paths(tmp_path):
    """Windows venv launchers must not hide the real PyTorch worker from its monitor."""
    import sys
    from tools.material_loss_cost import worker_command
    command = worker_command(tmp_path, tmp_path/'result.json')
    assert command[0] == getattr(sys, '_base_executable', sys.executable)
    assert command[1] == '-c'
    assert repr(sys.path) in command[2]
    assert '--worker' in command
