"""Cached demo publication verifies saved evidence before sending any messages."""
import hashlib
import json

import pytest


def _fixture(tmp_path):
    root=tmp_path/'source'; root.mkdir()
    source=root/'demo.py'; source.write_text('original',encoding='utf-8')
    wheel=tmp_path/'demo.whl'; wheel.write_bytes(b'wheel')
    digest=lambda value:hashlib.sha256(value).hexdigest()
    images=tmp_path/'images'; images.mkdir()
    pixels=b'\x01\x02\x03'; raw_digest=digest(pixels)
    raw=images/(raw_digest+'.rgb'); raw.write_bytes(pixels)
    report=tmp_path/'full.json'
    row={'status':'passed','mode':'render','supervisor':{'exit_code':0,'failure':None},
         'wheel':{'sha256':digest(b'wheel')},'sources':{'demo.py':digest(b'original')},
         'images':[{'label':label,'shape':[1,1,3],'raw_path':str(raw),
                    'raw_sha256':raw_digest} for label in ('hero','transparency','sampling')],
         'logical_output_bytes':9}
    report.write_text(json.dumps(row),encoding='utf-8')
    return report,wheel,digest(b'wheel'),root,digest(report.read_bytes()),raw


def test_cached_publication_loads_three_verified_images_without_renderer(tmp_path):
    from tools.image_showcase_cached import load_cached
    report,wheel,expected,root,report_digest,_=_fixture(tmp_path)
    row,images=load_cached(report,wheel,expected,root,report_digest)
    assert len(images)==3 and all(image.shape==(1,1,3) for image in images)
    assert all(image.data==b'\x01\x02\x03' for image in images)
    assert row['logical_output_bytes']==9


@pytest.mark.parametrize('damage',['report','wheel','source','raw'])
def test_cached_publication_rejects_changed_evidence(tmp_path,damage):
    from tools.image_showcase_cached import load_cached
    report,wheel,expected,root,report_digest,raw=_fixture(tmp_path)
    target={'report':report,'wheel':wheel,'source':root/'demo.py','raw':raw}[damage]
    target.write_bytes(b'changed')
    with pytest.raises(ValueError,match='identity'):
        load_cached(report,wheel,expected,root,report_digest)


def test_cached_publication_sends_only_three_images_and_honest_guide():
    from tools.image_showcase_cached import send_images
    calls=[]
    class Sink:
        def image(self,*args,**kwargs):calls.append(('image',args,kwargs))
        def text(self,*args,**kwargs):calls.append(('text',args,kwargs))
    result=send_images(Sink(),[object(),object(),object()])
    assert len(result)==4 and [call[0] for call in calls]==['image']*3+['text']
    assert 'not test acceptance' in calls[-1][1][1]


def test_cached_publication_cli_obeys_real_provenance_guard(tmp_path,monkeypatch):
    from tools import image_showcase_cached as cached
    from tools.testing_dashboard import reporter
    from gffx.dashboard._server import check_provenance
    import gffx.dashboard
    import sys
    report,wheel,expected,root,report_digest,_=_fixture(tmp_path)
    events=[];text=[];pictures=[]
    class Reporter:
        def __init__(self,*args):pass
        def send(self,event,**values):
            if event=='start':check_provenance(values['provenance'])
            events.append((event,values));return True
    class Dash:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def image(self,*args,**kwargs):pictures.append(args)
        def text(self,*args,**kwargs):text.append(args)
    def connect(*args,**kwargs):
        check_provenance(kwargs.get('provenance'))
        return Dash()
    monkeypatch.setattr(reporter,'Reporter',Reporter)
    monkeypatch.setattr(gffx.dashboard,'connect',connect)
    output=tmp_path/'outcome.json'
    monkeypatch.setattr(sys,'argv',['cached','--report',str(report),'--wheel',str(wheel),
                       '--sha256',expected,'--report-sha256',report_digest,'--source-root',str(root),
                       '--output',str(output),'--address','http://unused','--run','test'])
    assert cached.main()==0
    assert events[0][1]['provenance']=={'artifact_sha256':report_digest}
    assert events[-1]==('finish',{'exit_code':0}) and len(pictures)==3
    assert expected in text[0][1] and report_digest in text[0][1]
    assert json.loads(output.read_text())['status']=='passed'
