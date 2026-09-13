"""Saved numerical prefixes must not masquerade as complete image evidence."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import pytest


def module():return importlib.import_module('tools.environment_image_evidence')


def test_environment_image_evidence_rejects_unknown_profile(tmp_path):
    with pytest.raises(ValueError,match='profile'):
        module().snapshot(tmp_path,'invented','uniform','0'*64)


def test_environment_image_evidence_rejects_hash_before_parsing(tmp_path):
    with pytest.raises(ValueError,match='hash'):
        module().snapshot(tmp_path,'classic','uniform','0'*64,blob=b'not json')


@pytest.fixture(scope='module')
def saved():
    value=os.environ.get('GFFX_ENV_IMAGE_EVIDENCE_ROOT')
    if not value:pytest.skip('requires explicit saved whole-image prefix binding')
    root=Path(value)
    raw=(root/'whole-classic-uniform'/'cohort.json').read_bytes()
    return root,raw,hashlib.sha256(raw).hexdigest()


def test_environment_image_saved_prefix_is_truthfully_partial_or_complete(saved):
    root,raw,digest=saved
    result=module().snapshot(root,'classic','uniform',digest,blob=raw)
    assert result['workers']>0 and result['pixels_per_image']==6536
    assert result['complete']==(json.loads(raw)['status']==module().runner.PASSED)
    assert result['validated_pixels'][0]<=6536


@pytest.mark.parametrize('mutation',['source','schedule','counts','allowance','ownership','complete'])
def test_environment_image_evidence_rejects_rehashed_false_claims(saved,mutation):
    root,raw,_=saved;row=json.loads(raw)
    if mutation=='source':row['sources']={}
    elif mutation=='schedule':row['schedule_sha256']='0'*64
    elif mutation=='counts':row['declared_counts']['calls']+=1
    elif mutation=='allowance':row['aggregate_seconds']=999999
    elif mutation=='ownership':row['workers'][0]['index']=1
    else:
        row['status']=module().runner.PASSED;row['workers']=row['workers'][:1]
    blob=json.dumps(row).encode()
    with pytest.raises(ValueError):
        module().snapshot(root,'classic','uniform',hashlib.sha256(blob).hexdigest(),blob=blob)


def test_environment_image_complete_request_rejects_a_valid_prefix(saved):
    root,raw,_=saved;row=json.loads(raw)
    row.update(status='working',workers=row['workers'][:1],passes=[])
    blob=json.dumps(row).encode()
    with pytest.raises(ValueError,match='incomplete image'):
        module().snapshot(root,'classic','uniform',hashlib.sha256(blob).hexdigest(),
                          blob=blob,require_complete=True)


@pytest.mark.parametrize('model',['classic','principled'])
@pytest.mark.parametrize('environment',['uniform','affine'])
def test_environment_image_complete_profile_replays_all_pixels_and_fresh_descent(saved,model,environment):
    root,_,_=saved
    raw=(root/('whole-'+model+'-'+environment)/'cohort.json').read_bytes()
    result=module().snapshot(root,model,environment,hashlib.sha256(raw).hexdigest(),
                             blob=raw,require_complete=True)
    assert result['complete'] and result['workers']==result['total_workers']==104
    assert result['validated_pixels']==[6536,6536] and len(result['gradients'])==2
    assert all(after<before for before,after in zip(*result['losses']))
    assert 0<result['peak_host_bytes']<=1024**3
