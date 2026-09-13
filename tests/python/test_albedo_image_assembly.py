"""Complete images require exact ownership, measured pixels and durable replay."""
import copy
import importlib
import os
from pathlib import Path
import pytest
import torch

KEY=('classic','uniform','vertex')
def tool():return importlib.import_module('tools.albedo_image_assembly')

@pytest.fixture
def fake():
    mask=torch.zeros((128,128),dtype=torch.bool);mask.flatten()[100:6636]=True
    ids=mask.flatten().nonzero().flatten().tolist();tiles=[]
    for i in range(52):
        rows=[dict(point=dict(flat_index=j),reference=dict(radiance=[.2,.4,.6]),actual=dict(estimates=[[.2,.4,.6]]*2)) for j in ids[i*128:(i+1)*128]]
        tile=dict(model=KEY[0],environment=KEY[1],albedo_name=KEY[2],index=i,rows=rows,
                  status='passed selected albedo tile; not complete image',counts=dict(reference_directions=1,candidate_directions=1,calls=1,moment_samples=0))
        if i not in (0,26,51):
            tile['status']='passed new albedo tile; not complete image'
            tile['candidate']=dict(estimates=[r.pop('actual')['estimates'] for r in rows])
        tiles.append(tile)
    return tiles,mask

def test_complete_assembly_preserves_all_pixels_and_exact_background(fake):
    tiles,mask=fake;original=copy.deepcopy(tiles)
    result=tool().assemble(iter(tiles),mask,KEY)
    assert result['pixels']==6536 and result['metrics']['passed']
    assert torch.equal(result['actual'],result['reference']) and not bool(result['actual'][~mask].any())
    assert tiles==original and result['counts']['calls']==52

@pytest.mark.parametrize('mutation',['missing','duplicate','reorder','wrong-profile','wrong-pixel','stopped','nan','wrong-value','tail'])
def test_complete_assembly_rejects_incomplete_or_wrong_evidence(fake,mutation):
    tiles,mask=fake
    if mutation=='missing':tiles.pop()
    elif mutation=='duplicate':tiles.append(tiles[-1])
    elif mutation=='reorder':tiles[0],tiles[1]=tiles[1],tiles[0]
    elif mutation=='wrong-profile':tiles[1]['environment']='affine'
    elif mutation=='wrong-pixel':tiles[1]['rows'][0]['point']['flat_index']=0
    elif mutation=='stopped':tiles[1]['status']='stopped'
    elif mutation=='tail':tiles[-1]['rows'].pop()
    else:tiles[1]['candidate']['estimates'][0][0][0]=float('nan') if mutation=='nan' else 9.
    with pytest.raises(ValueError):tool().assemble(iter(tiles),mask,KEY)

def test_assembly_requires_explicit_ordered_batch_hashes_and_preserves_output(tmp_path):
    with pytest.raises(ValueError):tool().descriptors(tmp_path,tmp_path,[],KEY)
    with pytest.raises(FileExistsError):tool().run(tmp_path,tmp_path,tmp_path,tmp_path,[],KEY)
    with pytest.raises(ValueError):tool().replay(tmp_path,'0'*64,tmp_path,tmp_path,tmp_path,blob=b'{}')

def test_saved_complete_textured_image_reconstructs_and_rejects_changed_bytes(monkeypatch):
    root=os.environ.get('GFFX_ALBEDO_ASSEMBLY');sha=os.environ.get('GFFX_ALBEDO_ASSEMBLY_SHA256')
    if not root or not sha:pytest.skip('explicit complete image evidence required')
    args=(Path(root),sha,Path(os.environ['GFFX_ALBEDO_IMAGES_ROOT']),Path(os.environ['GFFX_ALBEDO_TILES']),Path(os.environ['GFFX_FULL_ENV_CORPUS']))
    result=tool().replay(*args)
    assert result['image']['pixels']==6536 and result['image']['metrics']['passed']
    original=tool()._read
    def corrupted(path,limit):
        raw=original(path,limit)
        return bytes([raw[0]^1])+raw[1:] if path.name=='actual.bin' else raw
    monkeypatch.setattr(tool(),'_read',corrupted)
    with pytest.raises(ValueError,match='image bytes'):tool().replay(*args)

def test_complete_image_panels_keep_background_and_separate_reference():
    from tools.testing_dashboard.albedo_images_demo import complete_panels
    from tools.testing_dashboard.image_reference_demo import display_rgb
    actual=torch.zeros((128,128,3),dtype=torch.float64);reference=actual.clone()
    actual[3,4]=torch.tensor([.8,.6,.4]);reference[3,4]=torch.tensor([.2,.4,.6])
    picture=complete_panels(dict(actual=actual,reference=reference))
    assert picture.shape==(128,256,3) and not bool(picture[0].any())
    assert torch.equal(picture[3,4],display_rgb(reference[3,4]/2))
    assert torch.equal(picture[3,132],display_rgb(actual[3,4]/2))

@pytest.mark.parametrize('completed',[0,2,12])
def test_matrix_dashboard_shows_each_profile_once_within_artifact_limit(monkeypatch,completed):
    from unittest.mock import MagicMock
    import gffx.dashboard
    from tools.testing_dashboard import albedo_images_demo as demo,reporter
    live_reporter_class=reporter.Reporter;live_send=live_reporter_class.send
    keys=[(m,e,a) for m in ('classic','principled') for e in ('uniform','affine') for a in ('vertex','gradient','checker')]
    tiles=[];complete=[];events=[]
    for number,key in enumerate(keys):
        tiles.append(dict(model=key[0],environment=key[1],albedo_name=key[2],index=0,
            rows=[dict(point=dict(flat_index=129),reference=dict(radiance=[.2,.4,.6]))],
            candidate=dict(estimates=[[[.2,.4,.6],[.2,.4,.6]]]),counts={},
            supervisor=dict(elapsed_seconds=1.,sampled_peak_working_set_bytes=1024)))
        if number<completed:
            image=torch.zeros((128,128,3),dtype=torch.float64)
            complete.append(dict(record=dict(profile=list(key),arrays={}),sha256='a'*64,
                image=dict(actual=image,reference=image,metrics={},counts={})))
    fake_reporter=MagicMock()
    fake_reporter.send.side_effect=lambda event,**values:events.append((event,values)) or True
    with monkeypatch.context() as publishing:
        publishing.setattr(gffx.dashboard,'connect',lambda *a,**k:MagicMock())
        publishing.setattr(reporter,'Reporter',lambda *a,**k:fake_reporter)
        assert live_reporter_class.send is live_send, 'publisher test must not intercept the active pytest reporter'
        demo.publish(dict(tiles=tiles,complete=complete,batches=[],seconds=1.,samples=1),'unused','unit-test')
    artifacts=next(values['artifacts'] for event,values in events if event=='demo')
    pictures=[a for a in artifacts if a['kind']=='image']
    assert len(pictures)==12
    assert len(artifacts)==17+completed<=32
    assert sum(a['label'].startswith('COMPLETE ') for a in pictures)==completed
    assert all(not any(a['path']=='albedo-images/'+'-'.join(key) for a in pictures) for key in keys[:completed])
    assert events[-1]==('finish',dict(exit_code=0))
