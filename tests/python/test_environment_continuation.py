"""Approved continuation must reuse immutable evidence, not restart its budget."""
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import time

import pytest


def tool(): return importlib.import_module('tools.environment_continuation')


def test_continuation_schedule_contains_only_unfinished_tiles():
    assert tool().remaining()==list(range(32,52))
    assert tool().PROFILE['additional_seconds']==900
    assert tool().PROFILE['maximum_additional_evaluations']==2440*(87296+65536)


def test_continuation_reserves_watchdog_before_launch():
    with pytest.raises(TimeoutError):tool().launch_guard(time.perf_counter()-721)
    tool().launch_guard(time.perf_counter())


def test_parent_digest_rejected_before_child_reads(tmp_path):
    (tmp_path/'cohort.json').write_text('{}',encoding='utf-8')
    with pytest.raises(ValueError,match='parent identity'):
        tool().parent_manifest(tmp_path)


@pytest.mark.parametrize('mutation',['status','prefix','duplicate','path','classic','reason','duration'])
def test_parent_layout_rejects_unqualified_prefix(mutation):
    from tools.environment_full_images import PROFILE
    row=dict(status='stopped',profile=PROFILE,error='TimeoutError: cooperative measurement deadline reached',
             elapsed_seconds=1634.1123321999912,images={'classic':{}},
             tiles={m:[dict(path=f'{m}-{i:02d}.json',sha256='a'*64) for i in range(n)]
                    for m,n in [('classic',52),('principled',32)]})
    tool().validate_parent_layout(row)
    if mutation=='status':row['status']='working'
    elif mutation=='prefix':row['tiles']['principled'].pop()
    elif mutation=='duplicate':row['tiles']['principled'][-1]=row['tiles']['principled'][0]
    elif mutation=='path':row['tiles']['classic'][0]['path']='../classic-00.json'
    elif mutation=='classic':row['images']={}
    elif mutation=='reason':row['error']='numerical tolerance failed'
    else:row['elapsed_seconds']=float('nan')
    with pytest.raises(ValueError):tool().validate_parent_layout(row)


def test_continuation_refuses_existing_output_before_launch(tmp_path):
    with pytest.raises(FileExistsError):tool().run(tmp_path,tmp_path,tmp_path,tmp_path/'wheel.whl')


def test_generator_source_identity_is_frozen(monkeypatch):
    from tools import environment_full_images as original
    assert original.source_identities()==tool().FROZEN_SOURCES
    monkeypatch.setattr(original,'source_identities',lambda:{})
    with pytest.raises(ValueError,match='source'):tool().check_sources()


def test_saved_parent_prefix_validates_without_shading(monkeypatch):
    root=os.environ.get('GFFX_ENV_PARENT');corpus=os.environ.get('GFFX_FULL_ENV_CORPUS')
    if not root or not corpus:pytest.skip('explicit saved parent and corpus required')
    from tools import environment_full_images as original
    monkeypatch.setattr(original,'candidate',lambda *a,**k:pytest.fail('replay must not shade'))
    bundle=original.load_bundle(Path(corpus),path=original.MANIFEST,sha256=original.MANIFEST_HASH)
    result=tool().validate_parent(Path(root),bundle['buffers'])
    assert len(result['prefix'])==32
    assert result['classic']['metrics']['passed']


def test_continuation_replay_refuses_changed_summary(tmp_path):
    (tmp_path/'continuation.json').write_text('{}',encoding='utf-8')
    with pytest.raises(ValueError,match='continuation identity'):
        tool().replay(tmp_path,'0'*64,tmp_path,{})


def test_saved_continuation_replays_both_complete_images():
    root=os.environ.get('GFFX_ENV_CONTINUATION');digest=os.environ.get('GFFX_ENV_CONTINUATION_SHA256')
    parent=os.environ.get('GFFX_ENV_PARENT');corpus=os.environ.get('GFFX_FULL_ENV_CORPUS')
    if not all((root,digest,parent,corpus)):pytest.skip('explicit completed continuation required')
    from tools import environment_full_images as original
    bundle=original.load_bundle(Path(corpus),path=original.MANIFEST,sha256=original.MANIFEST_HASH)
    result=tool().replay(Path(root),digest,Path(parent),bundle['buffers'])
    assert set(result['images'])=={'classic','principled'}
    assert all(x['metrics']['covered_pixels']==6536 and x['metrics']['passed'] for x in result['images'].values())
    assert result['record']['additional_elapsed_seconds']<=900
    assert len(result['record']['tiles'])==20


def test_dashboard_projection_preserves_execution_segments():
    from tools.testing_dashboard.environment_continuation_demo import projection
    parent=dict(status='stopped',images={'classic':{'metrics':{}}},
                tiles={'classic':list(range(52)),'principled':list(range(32))})
    record=dict(status=tool().COMPLETE,images={'principled':{'metrics':{}}},
                tiles=list(range(32,52)),original_elapsed_seconds=1634.0,
                additional_elapsed_seconds=600.0,cumulative_elapsed_seconds=2234.0,
                reused_tiles={'classic':52,'principled':32})
    result=dict(parent=parent,record=record,images={'classic':{},'principled':{}},sha256='a'*64)
    before=copy.deepcopy(result)
    value=projection(result);cohort=value['cohort'];segments=cohort['segments']
    assert result==before
    assert segments['original_status']=='stopped'
    assert segments['original_sha256']==tool().PARENT_HASH
    assert segments['new_tiles']==20 and segments['additional_budget_seconds']==900
    assert segments['original_elapsed_seconds']+segments['additional_elapsed_seconds']==cohort['elapsed_seconds']
    assert cohort['tiles']['principled']==list(range(52))
    assert set(cohort['images'])=={'classic','principled'}
