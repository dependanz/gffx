"""Selected image tiles are checkpoints, not complete-image acceptance."""
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import time
import pytest
import torch

def tool():return importlib.import_module('tools.albedo_tiles')

def test_tile_schedule_has_exact_unique_profile_ownership():
    expected=[(m,e,a,i) for m in ('classic','principled') for e in ('uniform','affine')
              for a in ('vertex','gradient','checker') for i in (0,26,51)]
    assert tool().schedule()==expected and len(set(expected))==36
    assert [x for b in range(6) for x in tool().batch_schedule(b)]==expected
    assert tool().PROFILE['maximum_samples']==968933376
    for bad in (-1,6,True,1.5):
        with pytest.raises(ValueError):tool().batch_schedule(bad)

def test_tile_launch_guard_and_destination_preserve_history(tmp_path):
    with pytest.raises(TimeoutError):tool().launch_guard(time.perf_counter()-721)
    with pytest.raises(FileExistsError):tool().run(tmp_path,tmp_path,tmp_path/'wheel',0)

def test_tile_reader_checks_hash_before_parsing(tmp_path):
    with pytest.raises(ValueError,match='hash'):
        tool().read_tile(tmp_path/'unused','bad',{},blob=b'{}')

@pytest.fixture
def buffers():
    corpus=os.environ.get('GFFX_FULL_ENV_CORPUS')
    if not corpus:pytest.skip('explicit immutable corpus required')
    fixed=tool().base.old.fixed
    return fixed.load_bundle(Path(corpus),path=fixed.MANIFEST,sha256=fixed.MANIFEST_HASH)['buffers']

def test_selected_tiles_use_exact_saved_fields_and_short_tail(buffers):
    for name in ('vertex','gradient','checker'):
        groups=[tool().inputs(buffers,name,i) for i in (0,26,51)]
        assert [len(g) for g in groups]==[128,128,8]
        assert len({r['point']['flat_index'] for g in groups for r in g})==264
        for g in groups:
            for r in g:
                assert r['albedo_name']==name
                assert r['albedo']==buffers['albedo/'+name].reshape(-1,3)[r['point']['flat_index']].tolist()

def test_tile_input_rejects_unknown_field_and_index(buffers):
    for field,index in [('constant',0),('checker',1),('vertex',True)]:
        with pytest.raises(ValueError):tool().inputs(buffers,field,index)

def test_aggregate_requires_all_six_batches_without_duplicates():
    with pytest.raises(ValueError):tool().combine([])
    with pytest.raises(ValueError):tool().combine([dict(record=dict(batch=0))]*6)

def test_partial_image_panels_keep_uncomputed_pixels_gray():
    from tools.testing_dashboard.albedo_tiles_demo import panels
    from tools.testing_dashboard.image_reference_demo import display_rgb
    row=dict(point=dict(flat_index=129),reference=dict(radiance=[.2,.4,.6]),actual=dict(estimates=[[.8,.6,.4]]*2))
    pixels=panels([dict(rows=[row])])
    assert pixels.shape==(128,256,3) and pixels.dtype==torch.uint8
    assert pixels[0,0].tolist()==[96]*3 and pixels[0,128].tolist()==[96]*3
    expected=display_rgb(torch.tensor([[.2,.4,.6],[.8,.6,.4]],dtype=torch.float64)/2)
    assert torch.equal(pixels[1,1],expected[0]) and torch.equal(pixels[1,129],expected[1])
    with pytest.raises(ValueError):panels([dict(rows=[row,row])])

def test_completed_tiles_replay_with_corruption_controls(monkeypatch):
    root=os.environ.get('GFFX_ALBEDO_TILES');corpus=os.environ.get('GFFX_FULL_ENV_CORPUS')
    if not root or not corpus:pytest.skip('explicit completed tile batches required')
    results=[]
    for batch in range(6):
        path=Path(root)/f'batch-{batch}';digest=hashlib.sha256((path/'batch.json').read_bytes()).hexdigest()
        results.append(tool().replay(path,digest,Path(corpus)))
    full=tool().combine(results)
    assert len(full['tiles'])==36 and sum(len(t['rows']) for t in full['tiles'])==3168
    bad=copy.deepcopy(results);bad[-1]=bad[0]
    with pytest.raises(ValueError):tool().combine(bad)
    row=copy.deepcopy(full['tiles'][0]);fixed=tool().base.old.fixed
    buf=fixed.load_bundle(Path(corpus),path=fixed.MANIFEST,sha256=fixed.MANIFEST_HASH)['buffers']
    for alteration in ('missing','reorder','colour','memory','work','environment'):
        changed=copy.deepcopy(row)
        if alteration=='missing':changed['rows'].pop()
        elif alteration=='reorder':changed['rows'][0],changed['rows'][1]=changed['rows'][1],changed['rows'][0]
        elif alteration=='colour':changed['rows'][0]['albedo']=[0.,0.,0.]
        elif alteration=='memory':changed['supervisor']['sampled_peak_working_set_bytes']=1024**3+1
        elif alteration=='work':changed['counts']['calls']=0
        else:changed['environment']='affine'
        blob=json.dumps(changed).encode()
        with pytest.raises(ValueError):tool().read_tile(Path('unused'),hashlib.sha256(blob).hexdigest(),buf,blob=blob)
