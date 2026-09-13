"""Saved full-image results must retain numerical and ownership evidence."""
import copy
import importlib
import json
import math

import pytest
import torch


def tool(): return importlib.import_module('tools.testing_dashboard.environment_full_demo')


def test_replay_rejects_changed_cohort_before_reading_tiles(tmp_path):
    (tmp_path/'cohort.json').write_text('{}',encoding='utf-8')
    with pytest.raises(ValueError,match='identity'):
        tool().load(tmp_path,'0'*64,torch.zeros((128,128),dtype=torch.bool))


@pytest.mark.parametrize('change', ['nan','threshold','streak','count','estimate','short'])
def test_replay_recomputes_convergence_instead_of_trusting_flags(change):
    estimates=[[.3,.5,.7],[.3,.5,.7]]
    history=[]
    for i,n in enumerate((8,16,32)):
        history.append(dict(n_mu=n,estimates=copy.deepcopy(estimates),radiance=estimates[0][:],
            threshold=[1e-11+1e-9*x for x in estimates[0]],offset_change=[0.]*3,
            change=None if i==0 else [0.]*3,streak=i,evaluations=sum(4*k*k for k in (8,16,32)[:i+1])))
    ref=dict(status='converged',radiance=estimates[0][:],evaluations=5376,history=history)
    tool().validate_reference(ref)
    if change=='nan': ref['radiance'][0]=math.nan
    elif change=='threshold': ref['history'][-1]['threshold'][0]=100.
    elif change=='streak': ref['history'][-1]['streak']=1
    elif change=='count': ref['evaluations']=1
    elif change=='short': ref['history'][-1]['threshold']=[]
    else: ref['history'][-1]['estimates'][1][0]=2.
    with pytest.raises(ValueError):tool().validate_reference(ref)


def test_replay_rejects_unknown_model_and_incomplete_cohort(tmp_path):
    import hashlib
    from tools.environment_full_images import PROFILE
    row=dict(profile=PROFILE,status='working',tiles={},images={})
    raw=json.dumps(row).encode();(tmp_path/'cohort.json').write_bytes(raw)
    with pytest.raises(ValueError,match='complete'):
        tool().load(tmp_path,hashlib.sha256(raw).hexdigest(),torch.zeros((128,128),dtype=torch.bool))


def test_assembly_positive_and_pixel_tamper_controls():
    from tools.environment_full_images import assemble,tile_range
    mask=torch.zeros((128,128),dtype=torch.bool);mask.view(-1)[:6536]=True
    rows=[]
    for i in range(52):
        start,end=tile_range(i)
        rows.append(dict(index=i,start=start,end=end,status='passed tile; not a complete image',model='classic',rows=[
            dict(flat_index=j,reference=dict(status='converged',radiance=[.2,.4,.6],evaluations=5376,history=[dict(streak=2)]),
                 actual=dict(estimates=[[.2,.4,.6]]*2,directions=4096,calls=4)) for j in range(start,end)]))
    result=assemble(rows,mask)
    assert result['metrics']['passed'] and result['metrics']['covered_pixels']==6536
    assert torch.equal(result['actual'],result['reference'])
    rows[-1]['rows'][-1]['flat_index']=0
    with pytest.raises(ValueError,match='ownership'):assemble(rows,mask)


def test_saved_complete_environment_images_replay_all_pixels():
    import os
    from pathlib import Path
    from tools.environment_full_images import load_bundle, MANIFEST, MANIFEST_HASH
    root=os.environ.get('GFFX_FULL_ENV_REPORTS');digest=os.environ.get('GFFX_FULL_ENV_SHA256')
    corpus=os.environ.get('GFFX_FULL_ENV_CORPUS')
    if not all((root,digest,corpus)):pytest.skip('explicit complete cohort identity and corpus required')
    bundle=load_bundle(Path(corpus),path=MANIFEST,sha256=MANIFEST_HASH)
    result=tool().load(Path(root),digest,bundle['buffers']['covered'])
    assert set(result['images'])=={'classic','principled'}
    assert all(row['metrics']['passed'] and row['metrics']['covered_pixels']==6536 for row in result['images'].values())


def test_stopped_cohort_preserves_complete_images_without_promoting_partial_work():
    import os
    from pathlib import Path
    from tools.environment_full_images import load_bundle, MANIFEST, MANIFEST_HASH
    root=os.environ.get('GFFX_STOPPED_ENV_REPORTS');digest=os.environ.get('GFFX_STOPPED_ENV_SHA256')
    corpus=os.environ.get('GFFX_FULL_ENV_CORPUS')
    if not all((root,digest,corpus)):pytest.skip('explicit stopped cohort identity required')
    bundle=load_bundle(Path(corpus),path=MANIFEST,sha256=MANIFEST_HASH)
    with pytest.raises(ValueError,match='complete'):
        tool().load(Path(root),digest,bundle['buffers']['covered'])
    result=tool().load(Path(root),digest,bundle['buffers']['covered'],require_complete=False)
    assert result['cohort']['status']=='stopped'
    assert set(result['images'])=={'classic'}
    assert result['images']['classic']['metrics']['covered_pixels']==6536
    assert result['images']['classic']['metrics']['passed']
