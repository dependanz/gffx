import importlib,json,hashlib
import pytest
from tools.testing_dashboard.presentation import AssetWriter,png_rgb

def fixture(tmp_path):
    image=AssetWriter(tmp_path/'presentation').put(png_rgb(16,16,bytes([100,120,130])*256))
    value=dict(image_sha256=image['sha256'],width=16,height=16,landmarks=[[4,5]]*68,skin_polygons=[[[1,1],[10,1],[10,10]]],crop_xyxy=[1,1,15,15],annotation_sha256='a'*64)
    directory=tmp_path/'reviews';directory.mkdir();(directory/'portrait1.json').write_text(json.dumps(value));return value

def load(root):return importlib.import_module('tools.testing_dashboard.portrait_review').load_review(root)

def test_portrait_review_returns_only_verified_data_and_draft_status(tmp_path):
    value=fixture(tmp_path);got=load(tmp_path)
    assert got['image_url']=='/testing/media/'+value['image_sha256']+'.png'
    assert len(got['landmarks'])==68 and got['status']=='draft'
    assert got['revision_sha256']==hashlib.sha256((tmp_path/'reviews/portrait1.json').read_bytes()).hexdigest()

@pytest.mark.parametrize('bad',['path','point','nan','crop','polygon','dimension'])
def test_portrait_review_refuses_invalid_source_or_coordinates(tmp_path,bad):
    v=fixture(tmp_path)
    if bad=='path':v['image_sha256']='../secrets'
    elif bad=='point':v['landmarks'][0]=[16,4]
    elif bad=='nan':v['landmarks'][0]=[float('nan'),4]
    elif bad=='crop':v['crop_xyxy']=[10,10,2,2]
    elif bad=='polygon':v['skin_polygons']=[[[1,1],[2,2]]]
    else:v['width']=15
    (tmp_path/'reviews/portrait1.json').write_text(json.dumps(v))
    with pytest.raises(ValueError):load(tmp_path)

def test_portrait_review_refuses_changed_image_bytes(tmp_path):
    v=fixture(tmp_path);(tmp_path/'presentation'/(v['image_sha256']+'.png')).write_bytes(png_rgb(16,16,bytes([1,2,3])*256))
    with pytest.raises(FileNotFoundError):load(tmp_path)
