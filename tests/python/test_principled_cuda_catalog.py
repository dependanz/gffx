"""Named CUDA admission is distinct from a CPU ID or an experimental cast."""
import copy
import importlib.util
import json
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[2]
CUDA_ID='ggx-smith-loss-cuda-f64-v1'


@pytest.fixture
def catalog():
    spec=importlib.util.spec_from_file_location('candidate_budget_catalog',ROOT/'src/gffx/torch/_budget_catalog.py')
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value


@pytest.fixture
def envelope():
    return json.loads((ROOT/'src/gffx/data/principled/ggx-smith-loss-cpu-f64-v1.json').read_text())


def test_cuda_catalog_uses_a_distinct_named_profile(catalog,envelope):
    cpu=catalog.acceptance(envelope,device='cpu',dtype='float64')
    assert catalog.cuda_acceptance(envelope,device='cuda',dtype='float64')[:2]==(CUDA_ID,'public-budget-cuda-v0.1')
    assert cpu[0]=='ggx-smith-loss-cpu-f64-v1'
    assert catalog.acceptance(envelope,device='cuda',dtype='float64') is None
    assert catalog.cuda_acceptance(envelope,device='cpu',dtype='float64') is None
    assert catalog.cuda_acceptance(envelope,device='cuda',dtype='float32') is None


@pytest.mark.parametrize('field',['payload_sha256','model','generator'])
def test_cuda_catalog_rejects_changed_physical_identity(catalog,envelope,field):
    changed=copy.deepcopy(envelope);changed[field]='changed'
    assert catalog.cuda_acceptance(changed,device='cuda',dtype='float64') is None


def test_cuda_catalog_reuses_exact_payload_without_duplicating_data(catalog):
    cpu=catalog.read_bundle('ggx-smith-loss-cpu-f64-v1',byte_limit=4*1024**2)
    assert catalog.read_bundle(CUDA_ID,byte_limit=4*1024**2)==cpu
    assert not (ROOT/'src/gffx/data/principled'/f'{CUDA_ID}.json').exists()


def test_cuda_catalog_ships_an_exact_finite_acceptance_record(catalog):
    import hashlib
    raw=(ROOT/'src/gffx/data/principled/public-budget-cuda-v0.1.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==catalog.CUDA_RECORD_SHA256
    row=json.loads(raw)
    assert row['resource_id']==CUDA_ID and row['payload_sha256']==catalog.PAYLOAD_SHA256
    assert row['counts']==dict(chain=15,directional=80,mean=16,furnace=192,repeat=20)
    assert row['qualification_wheel_sha256']=='c1702a82604698fd791fee9e49f5ab6f3c038546091b9a6b4c543215b03d5590'
    assert set(row['reports_sha256'])=={'chain','directional','mean','furnace','repeat'}
    assert 'float32' in row['not_qualified'] and 'Cycles fidelity' in row['not_qualified']
