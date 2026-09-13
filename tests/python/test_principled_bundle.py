"""Installed-package identity and loading tests, not new physical holdouts."""
import copy
import hashlib
import importlib
import json
import os
import tarfile
from pathlib import Path
import pytest
import torch
import zipfile
from gffx.torch.material import PrincipledBudget, principled

RESOURCE_ID = 'ggx-smith-loss-cpu-f64-v1'


@pytest.fixture(autouse=True)
def serial():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.fixture(scope='module')
def accepted():
    return PrincipledBudget.bundled(RESOURCE_ID, device='cpu', dtype=torch.float64)


def test_bundle_has_exact_profile_and_no_experimental_opt_in(accepted):
    metadata = accepted.metadata
    assert metadata['admission'] == 'accepted'
    assert metadata['resource_id'] == RESOURCE_ID
    assert metadata['acceptance_profile_id'] == 'public-budget-v0.1'
    assert metadata['acceptance_record_sha256'] == 'df424fcb92630ccd2bc70f8cf26997bcc3b0929ad1d56d3ca22e6ad1f4b98131'
    assert metadata['payload_sha256'] == 'eb9393c7afd6e1a4964019643ee827dff72696896bd05786e8a2ec908ca538e5'
    assert metadata['domain'] == dict(roughness=[.001, 1.], cosine=[0, 1], f0=[0, 1])
    assert metadata['supported_query_entries'] == 32
    n = torch.tensor([[0., 0., 1.]], dtype=torch.float64)
    roughness = torch.tensor(.213, dtype=torch.float64, requires_grad=True)
    value = principled(n, n, n, albedo=torch.tensor([.4, .4, .4], dtype=torch.float64),
                       roughness=roughness, f0=torch.tensor(.07, dtype=torch.float64), budget=accepted)
    assert value.shape == (1, 1, 3) and bool(torch.isfinite(value).all())
    assert bool(torch.isfinite(torch.autograd.grad(value.sum(), roughness)[0]))


def test_accepted_envelope_roundtrip_and_transfer_copy_storage(accepted, tmp_path):
    envelope = accepted.to_envelope()
    path = tmp_path/'budget.json'
    path.write_text(json.dumps(envelope), encoding='utf-8')
    candidates = [PrincipledBudget.from_envelope(envelope, device='cpu', dtype=torch.float64),
                  PrincipledBudget.load(path, device='cpu', dtype=torch.float64),
                  accepted.to(device='cpu', dtype=torch.float64)]
    c, r, f = [torch.tensor(x, dtype=torch.float64) for x in (.463, .213, .47)]
    for item in candidates:
        assert item is not accepted and item.metadata['admission'] == 'accepted'
        assert torch.equal(item.directional_loss(c, r, f), accepted.directional_loss(c, r, f))
        assert item._data['losses'].data_ptr() != accepted._data['losses'].data_ptr()
    envelope['payload']['losses'][0][0][0] = .1
    assert accepted.to_envelope()['payload'] != envelope['payload']


def test_evidence_label_cannot_grant_or_remove_exact_data_admission(accepted):
    envelope = accepted.to_envelope()
    envelope['evidence_id'] = 'untrusted user label, not authority'
    assert PrincipledBudget.from_envelope(envelope, device='cpu', dtype=torch.float64).metadata['admission'] == 'accepted'
    envelope['generator']['policy'] = 'different generator with copied evidence'
    with pytest.raises(ValueError, match='accepted catalog'):
        PrincipledBudget.from_envelope(envelope, device='cpu', dtype=torch.float64)
    experimental = PrincipledBudget.from_envelope(envelope, device='cpu', dtype=torch.float64, allow_unverified=True)
    assert experimental.metadata['admission'] == 'experimental'
    assert experimental.metadata['acceptance_profile_id'] is None
    assert experimental.to(device='cpu', dtype=torch.float64).metadata['admission'] == 'experimental'


@pytest.mark.parametrize('field', ['format_version','model','representation','dtype','domain','generator_source','generator_policy','payload_sha256','payload'])
def test_each_identity_mutation_rejected_before_tensor_allocation(accepted, monkeypatch, field):
    envelope = accepted.to_envelope()
    if field == 'domain': envelope[field]['roughness'][0] = .002
    elif field == 'generator_source': envelope['generator']['source_sha256'] = '0'*64
    elif field == 'generator_policy': envelope['generator']['policy'] = 'other'
    elif field == 'payload':
        envelope['payload']['losses'][20][20][0] *= .999
        envelope['payload_sha256'] = hashlib.sha256(json.dumps(envelope['payload'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    elif field == 'format_version': envelope[field] = 2
    elif field == 'payload_sha256': envelope[field] = '0'*64
    else: envelope[field] = 'unknown'
    with monkeypatch.context() as m:
        m.setattr(torch, 'tensor', lambda *a, **k: pytest.fail('tensor allocation before identity rejection'))
        with pytest.raises(ValueError):
            PrincipledBudget.from_envelope(envelope, device='cpu', dtype=torch.float64)


@pytest.mark.parametrize('resource_id', ['latest', '../ggx-smith-loss-cpu-f64-v1', 'unknown', 'GGX-SMITH-LOSS-CPU-F64-V1'])
def test_unknown_bundle_never_reads_or_substitutes(resource_id, monkeypatch):
    catalog = importlib.import_module('gffx.torch._budget_catalog')
    with monkeypatch.context() as m:
        m.setattr(catalog, 'files', lambda *a: pytest.fail('unknown resource accessed package data'))
        with pytest.raises(ValueError):
            PrincipledBudget.bundled(resource_id, device='cpu', dtype=torch.float64)


@pytest.mark.parametrize('device,dtype', [('cpu',torch.float32), ('cuda',torch.float64), ('mps',torch.float64)])
def test_bundle_unsupported_lane_rejected_before_file_access(monkeypatch, device, dtype):
    catalog = importlib.import_module('gffx.torch._budget_catalog')
    with monkeypatch.context() as m:
        m.setattr(catalog, 'files', lambda *a: pytest.fail('unsupported lane accessed package data'))
        with pytest.raises(ValueError):
            PrincipledBudget.bundled(RESOURCE_ID, device=device, dtype=dtype)


@pytest.mark.parametrize('kind', ['missing','corrupt','oversized'])
def test_bundled_file_failure_is_closed_without_fallback(tmp_path, monkeypatch, kind):
    catalog = importlib.import_module('gffx.torch._budget_catalog')
    path = tmp_path/'data'/'principled'/(RESOURCE_ID+'.json')
    path.parent.mkdir(parents=True)
    if kind != 'missing': path.write_bytes(b'x'*(8*1024**2+1 if kind == 'oversized' else 20))
    with monkeypatch.context() as m:
        m.setattr(catalog, 'files', lambda *a: tmp_path)
        m.setattr(torch, 'tensor', lambda *a, **k: pytest.fail('allocation after corrupt bundle'))
        with pytest.raises(OSError if kind == 'missing' else ValueError):
            PrincipledBudget.bundled(RESOURCE_ID, device='cpu', dtype=torch.float64)


def test_accepted_queries_need_no_further_resource_io(accepted, monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(Path, 'open', lambda *a, **k: pytest.fail('query file read'))
        c, r, f = [torch.tensor(x, dtype=torch.float64, requires_grad=True) for x in (.463, .213, .47)]
        loss = accepted.directional_loss(c, r, f)+accepted.mean_loss(r, f)
        assert all(bool(torch.isfinite(g)) for g in torch.autograd.grad(loss, (c,r,f)))


def test_installed_bundle_bytes_match_wheel_and_have_no_dependencies():
    import gffx
    location = os.environ.get('GFFX_BUNDLE_WHEEL')
    if not location:
        pytest.skip('explicit built wheel required for artifact identity check')
    name = 'gffx/data/principled/'+RESOURCE_ID+'.json'
    with zipfile.ZipFile(location) as archive:
        data = archive.read(name)
        assert hashlib.sha256(data).hexdigest() == '5e095d1aa417dcba460f7cb2a5f72b1cf8c857d793cba1da1b6e77b9fc769cb7'
        record = archive.read('gffx/data/principled/public-budget-v0.1.json')
        assert hashlib.sha256(record).hexdigest() == 'df424fcb92630ccd2bc70f8cf26997bcc3b0929ad1d56d3ca22e6ad1f4b98131'
        assert json.loads(record)['payload_sha256'] == json.loads(data)['payload_sha256']
        installed = Path(gffx.__file__).resolve().parent.parent
        members = [p for p in archive.namelist() if p.startswith('gffx/') and not p.endswith('/')]
        assert all((installed/p).read_bytes() == archive.read(p) for p in members)
        metadata = next(p for p in archive.namelist() if p.endswith('.dist-info/METADATA'))
        assert 'Requires-Dist:' not in archive.read(metadata).decode('utf-8')
        assert not any('/tools/' in p or '/build/' in p for p in archive.namelist())


def test_source_archive_preserves_exact_bundle_and_catalog():
    location = os.environ.get('GFFX_BUNDLE_SDIST')
    if not location:
        pytest.skip('explicit built source archive required')
    with tarfile.open(location) as archive:
        for suffix, digest in (
            ('src/gffx/data/principled/'+RESOURCE_ID+'.json', '5e095d1aa417dcba460f7cb2a5f72b1cf8c857d793cba1da1b6e77b9fc769cb7'),
            ('src/gffx/data/principled/public-budget-v0.1.json', 'df424fcb92630ccd2bc70f8cf26997bcc3b0929ad1d56d3ca22e6ad1f4b98131'),
        ):
            matches = [m for m in archive.getmembers() if m.name.endswith('/'+suffix)]
            assert len(matches) == 1 and matches[0].isfile()
            assert hashlib.sha256(archive.extractfile(matches[0]).read()).hexdigest() == digest
        assert any(m.name.endswith('/src/gffx/torch/_budget_catalog.py') for m in archive.getmembers())
    attributes = Path(__file__).resolve().parents[2]/'.gitattributes'
    assert 'src/gffx/data/principled/*.json -text' in attributes.read_text(encoding='utf-8')


@pytest.mark.parametrize('stage', ['directional', 'mean'])
def test_bundled_queries_reproduce_validated_public_values(accepted, stage):
    location = os.environ.get('GFFX_BUNDLE_REFERENCE_REPORTS')
    if not location:
        pytest.skip('explicit frozen physical reports required for numerical replay')
    from tools.package_principled_budget import REPORTS
    blob = (Path(location)/(stage+'.json')).read_bytes()
    assert hashlib.sha256(blob).hexdigest() == REPORTS[stage]
    rows = json.loads(blob)['rows']
    with torch.no_grad():
        for row in rows:
            r = torch.tensor(row['roughness'], dtype=torch.float64)
            f = torch.tensor(row['f0'], dtype=torch.float64)
            if stage == 'directional':
                c = torch.tensor(row['cosine'], dtype=torch.float64)
                actual, expected = accepted.directional_loss(c, r, f).item(), row['loss']
            else:
                actual, expected = accepted.mean_loss(r, f).item(), row['mean']
            assert actual == expected, (stage, row)
