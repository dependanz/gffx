"""New-pixel reference-only integration and checkpoint ownership contracts."""
import copy
import importlib
import json
import os
from pathlib import Path
import time
import pytest
import torch

def sampler():return importlib.import_module('tools.albedo_image_sampling')
def images():return importlib.import_module('tools.albedo_images')

@pytest.fixture(autouse=True)
def serial():
    before=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)

@pytest.fixture
def saved():
    from tools import albedo_environment as base
    from gffx.torch.material import PrincipledBudget
    budget=PrincipledBudget.bundled(base.old.RESOURCE,device='cpu',dtype=torch.float64)
    curve=base.old.PreparedCurve(base.old.LossCurve(budget.to_envelope()['payload']),roughness=1.)
    def make(model,env):
        deadline=time.perf_counter()+30
        prepared=base.Prepared(model,curve,env,deadline=deadline)
        n=[.1,0.,1.];v=[.2,.1,1.];a=[.2,.4,.6]
        ref=prepared.point(n,v,a,deadline=deadline)
        actual=base.candidate(n,v,a,model=model,environment=env,budget=budget,ref=ref,deadline=deadline)
        return dict(point=dict(flat_index=0,normal=n,view=v),albedo=a,reference=ref),actual,budget
    return make

@pytest.mark.parametrize('model,env',[('classic','uniform'),('classic','affine'),('principled','uniform'),('principled','affine')])
def test_new_pixels_need_reference_not_previous_public_outputs(saved,model,env):
    row,previous,budget=saved(model,env);rows=[copy.deepcopy(row) for _ in range(9)]
    result=sampler().sample(rows,model=model,environment=env,budget=budget,deadline=time.perf_counter()+30)
    sampler().validate(result,rows,model,env)
    assert result['width']==(1 if (model,env)==('principled','affine') else 8)
    assert result['max_pairs']<=1024 and result['max_points']<=8
    for values in result['estimates']:
        torch.testing.assert_close(torch.tensor(values,dtype=torch.float64),torch.tensor(previous['estimates'],dtype=torch.float64),atol=1e-12,rtol=1e-11)
    assert all('actual' not in r for r in rows)

def test_new_pixels_reject_mismatched_reference_before_public_call(saved,monkeypatch):
    row,_,budget=saved('classic','uniform')
    from gffx.torch import material
    def forbidden(*a,**kw):raise AssertionError('public call before validation')
    monkeypatch.setattr(material,'classic',forbidden)
    for key,value in [('albedo',[.3,.4,.6]),('model','principled'),('environment','affine')]:
        bad=copy.deepcopy(row);bad['reference'][key]=value
        with pytest.raises(ValueError):sampler().sample([bad],model='classic',environment='uniform',budget=budget,deadline=time.perf_counter()+30)
    with pytest.raises(TimeoutError):sampler().sample([row],model='classic',environment='uniform',budget=budget,deadline=time.perf_counter()-1)

def test_reference_only_result_rejects_forged_counts_routes_and_values(saved):
    row,_,budget=saved('classic','uniform')
    result=sampler().sample([row],model='classic',environment='uniform',budget=budget,deadline=time.perf_counter()+30)
    for field,value in [('calls',0),('directions',0),('max_pairs',1025),('width',1),('seconds',float('nan'))]:
        bad=copy.deepcopy(result);bad[field]=value
        with pytest.raises(ValueError):sampler().validate(bad,[row],'classic','uniform')
    for value in (float('nan'),float('inf'),99.):
        bad=copy.deepcopy(result);bad['estimates'][0][0][0]=value
        with pytest.raises(ValueError):sampler().validate(bad,[row],'classic','uniform')

def test_full_image_schedule_excludes_all_selected_tiles_and_is_create_only(tmp_path):
    from tools import albedo_tiles
    todo=images().schedule();parent=albedo_tiles.schedule()
    assert len(todo)==588 and len(set(todo))==588 and not set(todo)&set(parent)
    assert len(todo+parent)==624
    assert [k for b in range(98) for k in images().batch_schedule(b)]==todo
    assert all(len(images().batch_schedule(b))==6 for b in range(98))
    for b in (-1,98,True,1.5):
        with pytest.raises(ValueError):images().batch_schedule(b)
    with pytest.raises(FileExistsError):images().run(tmp_path,tmp_path,tmp_path,0)
    with pytest.raises(TimeoutError):images().launch_guard(time.perf_counter()-721)

def test_new_tile_inputs_match_raster_order_and_saved_albedo():
    root=os.environ.get('GFFX_FULL_ENV_CORPUS')
    if not root:pytest.skip('explicit frozen corpus required')
    from tools import albedo_environment as base
    fixed=base.old.fixed;buffers=fixed.load_bundle(Path(root),path=fixed.MANIFEST,sha256=fixed.MANIFEST_HASH)['buffers']
    _,ids,_=fixed.pixel_layout(buffers)
    for name in ('vertex','gradient','checker'):
        rows=images().inputs(buffers,name,1)
        assert [r['point']['flat_index'] for r in rows]==ids[128:256]
        assert all(r['albedo']==buffers['albedo/'+name].reshape(-1,3)[r['point']['flat_index']].tolist() for r in rows)
    for i in (-1,52,True):
        with pytest.raises(ValueError):images().inputs(buffers,'vertex',i)

def test_frozen_sampling_observations_match_reference_only_sampler():
    root=os.environ.get('GFFX_ALBEDO_SAMPLING')
    if not root:pytest.skip('explicit frozen screen required')
    from tools import albedo_sampling as frozen
    from tools import albedo_environment as base
    from gffx.torch.material import PrincipledBudget
    budget=PrincipledBudget.bundled(base.old.RESOURCE,device='cpu',dtype=torch.float64)
    for model,env in base.schedule():
        original=frozen.load_rows(Path(os.environ['GFFX_ALBEDO_TILES']),Path(os.environ['GFFX_FULL_ENV_CORPUS']),model,env)
        rows=[{k:v for k,v in r.items() if k!='actual'} for r in original]
        result=sampler().sample(rows,model=model,environment=env,budget=budget,deadline=time.perf_counter()+120)
        for values,old in zip(result['estimates'],original):
            torch.testing.assert_close(torch.tensor(values,dtype=torch.float64),torch.tensor(old['actual']['estimates'],dtype=torch.float64),atol=1e-12,rtol=1e-11)

def test_saved_new_tile_batch_replays_and_rejects_forgery(monkeypatch):
    root=os.environ.get('GFFX_ALBEDO_IMAGES');sha=os.environ.get('GFFX_ALBEDO_IMAGES_SHA256')
    if not root or not sha:pytest.skip('explicit completed new tile batch required')
    import hashlib
    from tools import albedo_environment as base
    corpus=Path(os.environ['GFFX_FULL_ENV_CORPUS'])
    result=images().replay(Path(root),sha,corpus)
    assert len(result['tiles'])==6 and sum(len(t['rows']) for t in result['tiles'])==768
    fixed=base.old.fixed;buffers=fixed.load_bundle(corpus,path=fixed.MANIFEST,sha256=fixed.MANIFEST_HASH)['buffers']
    row=result['tiles'][0]
    for mutation in ('order','colour','reference','counts','width','source','memory','failure'):
        bad=copy.deepcopy(row)
        if mutation=='order':bad['rows'].reverse()
        elif mutation=='colour':bad['rows'][0]['albedo'][0]=.999
        elif mutation=='reference':bad['rows'][0]['reference']['albedo'][0]=.999
        elif mutation=='counts':bad['counts']['calls']=0
        elif mutation=='width':bad['candidate']['width']=8 if row['candidate']['width']==1 else 1
        elif mutation=='source':bad['sources']['albedo_images.py']='0'*64
        elif mutation=='memory':bad['supervisor']['sampled_peak_working_set_bytes']=1024**3+1
        else:bad['supervisor']['failure']='injected'
        blob=json.dumps(bad).encode()
        with pytest.raises(ValueError):images().read_tile(Path('ignored'),hashlib.sha256(blob).hexdigest(),buffers,blob=blob)

def test_partial_progress_panels_show_only_computed_pixels():
    from tools.testing_dashboard.albedo_images_demo import panels
    tile=dict(rows=[dict(point=dict(flat_index=129),reference=dict(radiance=[.2,.4,.6]))],
              candidate=dict(estimates=[[[.2,.4,.6],[.2,.4,.6]]]))
    original=copy.deepcopy(tile);rgb=panels([tile])
    assert rgb.shape==(128,256,3) and rgb.dtype==torch.uint8
    assert (rgb[0]==96).all() and torch.equal(rgb[1,1],rgb[1,129])
    assert tile==original
    with pytest.raises(ValueError):panels([tile,tile])
