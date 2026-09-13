"""Checkpointed completion of the fixed affine world-light image pair."""
import importlib
import os
from pathlib import Path
import time

import pytest


def tool():return importlib.import_module('tools.affine_images')


def test_remaining_schedule_covers_only_unmeasured_tiles():
    todo=tool().remaining()
    assert len(todo)==98 and len(set(todo))==98
    assert set(todo).isdisjoint(tool().tiles.schedule())
    assert set(todo+tool().tiles.schedule())=={(m,i) for m in ('classic','principled') for i in range(52)}
    assert sum(tool().tiles.tile_range(i)[1]-tool().tiles.tile_range(i)[0] for _,i in todo)==12544


def test_batches_have_finite_unique_ownership_and_watchdog_reserve():
    groups=tool().batches()
    assert len(groups)==17 and all(1<=len(x)<=6 for x in groups)
    assert [p for group in groups for p in group]==tool().remaining()
    assert tool().PROFILE['batch_seconds']==900
    assert tool().PROFILE['maximum_new_samples']==3835854848
    with pytest.raises(TimeoutError):tool().launch_guard(time.perf_counter()-721)


def test_parent_identity_is_required_before_reuse(tmp_path):
    (tmp_path/'cohort.json').write_text('{}',encoding='utf-8')
    with pytest.raises(ValueError,match='identity'):tool().parent(tmp_path,tmp_path)


def test_full_run_refuses_existing_output(tmp_path):
    with pytest.raises(FileExistsError):tool().run(tmp_path,tmp_path,tmp_path,tmp_path/'wheel')


@pytest.mark.parametrize('records',[[],[dict(index=1,model='classic',rows=[])]])
def test_image_assembly_rejects_missing_or_misordered_tiles(records):
    import torch
    mask=torch.zeros((128,128),dtype=torch.bool);mask.flatten()[:6536]=True
    with pytest.raises(ValueError):tool().assemble(iter(records),mask,'classic')


def test_batch_layout_rejects_missing_duplicate_or_reordered_descriptors():
    row=dict(status='passed',index=0,profile=tool().PROFILE,sources=tool().sources(),
             elapsed_seconds=1.,tiles=[dict(path=f'{m}-{i:02d}.json',sha256='a'*64) for m,i in tool().batches()[0]])
    tool().validate_batch(row,0)
    row['tiles'][1]=row['tiles'][0]
    with pytest.raises(ValueError):tool().validate_batch(row,0)
    row['tiles'].pop()
    with pytest.raises(ValueError):tool().validate_batch(row,0)


def test_saved_affine_images_replay_complete_pixel_coverage():
    root=os.environ.get('GFFX_AFFINE_IMAGES');digest=os.environ.get('GFFX_AFFINE_IMAGES_SHA256')
    corpus=os.environ.get('GFFX_FULL_ENV_CORPUS');parent=os.environ.get('GFFX_AFFINE_TILES')
    if not all((root,digest,corpus,parent)):pytest.skip('explicit completed affine image cohort required')
    result=tool().replay(Path(root),digest,Path(parent),Path(corpus))
    assert set(result['images'])=={'classic','principled'}
    assert result['record']['status']==tool().COMPLETE
    for image in result['images'].values():
        assert image['metrics']['passed'] and image['pixels']==6536


def test_assembly_reconstructs_every_pixel_and_rejects_invalid_estimates():
    import copy
    import torch
    mask=torch.zeros((128,128),dtype=torch.bool);mask.flatten()[:6536]=True
    records=[]
    for i in range(52):
        start,end=tool().tiles.tile_range(i)
        records.append(dict(index=i,model='classic',status='passed affine tile; not a complete image',
            counts=dict(reference_directions=0,candidate_directions=0,calls=0,moment_samples=0),
            rows=[dict(flat_index=j,reference=dict(radiance=[.2,.4,.6]),
                       actual=dict(estimates=[[.2,.4,.6],[.2,.4,.6]])) for j in range(start,end)]))
    result=tool().assemble(iter(records),mask,'classic')
    assert result['pixels']==6536 and result['metrics']['passed']
    assert torch.equal(result['actual'],result['reference'])
    assert torch.count_nonzero(result['actual'][~mask])==0
    bad=copy.deepcopy(records);bad[1]['rows'][0]['actual']['estimates'][1][0]=float('nan')
    with pytest.raises(ValueError,match='tolerance'):tool().assemble(iter(bad),mask,'classic')
    bad=copy.deepcopy(records);bad[1]['rows'][0]['flat_index']=0
    with pytest.raises(ValueError,match='coverage'):tool().assemble(iter(bad),mask,'classic')


def test_live_checkpoint_demo_marks_only_verified_progress():
    root=os.environ.get('GFFX_AFFINE_IMAGES_PROGRESS');parent=os.environ.get('GFFX_AFFINE_TILES')
    corpus=os.environ.get('GFFX_FULL_ENV_CORPUS')
    if not all((root,parent,corpus)):pytest.skip('explicit finite image progress snapshot required')
    demo=importlib.import_module('tools.testing_dashboard.affine_images_demo')
    result=demo.load(Path(root),Path(parent),Path(corpus))
    assert set(result['images'])=={'classic','principled'}
    for row in result['images'].values():
        assert 264<=row['pixels']<=6536
        if result['record']['status']!=tool().COMPLETE:
            assert int(row['known'].sum())==row['pixels']
        assert row['actual'].shape==row['reference'].shape==(128,128,3)
