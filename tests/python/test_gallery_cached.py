"""Saved gallery identity checks never render or execute archived source."""
import hashlib,json
import pytest
from tools.testing_dashboard.presentation import png_rgb


def fixture(tmp_path):
    report=tmp_path/'report.json';report.write_text('saved producer report')
    data=png_rgb(1,1,b'\x01\x02\x03');sha=hashlib.sha256(data).hexdigest();image=tmp_path/(sha+'.png');image.write_bytes(data)
    row=dict(version=1,title='Saved result',page='image/chroma-demo',scope='Exact preserved pixels',provenance=dict(producer_report=str(report),producer_report_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),wheel_sha256='a'*64),images=[dict(label='One pixel',caption='Original fixture',path=str(image),asset=dict(url='/testing/media/'+sha+'.png',sha256=sha,bytes=len(data)))])
    path=tmp_path/'gallery.json';path.write_text(json.dumps(row));return path,hashlib.sha256(path.read_bytes()).hexdigest(),image,report


def test_gallery_cache_preserves_exact_saved_pixels(tmp_path):
    from tools.m1_gallery_cached import load
    path,sha,image,report=fixture(tmp_path);row,images=load(path,sha)
    assert images==[image.read_bytes()] and row['images'][0]['label']=='One pixel'


@pytest.mark.parametrize('damage',['manifest','image','report','scope'])
def test_gallery_cache_rejects_corrupted_evidence_before_publication(tmp_path,damage):
    from tools.m1_gallery_cached import load
    path,sha,image,report=fixture(tmp_path)
    if damage=='scope':
        row=json.loads(path.read_text());row.pop('scope');path.write_text(json.dumps(row));sha=hashlib.sha256(path.read_bytes()).hexdigest()
    else:{'manifest':path,'image':image,'report':report}[damage].write_bytes(b'changed')
    with pytest.raises(ValueError):load(path,sha)
