"""Finite reference generation and presentation, never public shader acceptance."""
import importlib
import struct
import zipfile
from pathlib import Path

import pytest
import torch


def test_display_conversion_is_srgb_and_does_not_change_linear_evidence():
    m=importlib.import_module('tools.testing_dashboard.image_reference_demo')
    x=torch.tensor([[-1.,0.,.0031308],[.18,1.,2.]],dtype=torch.float64)
    before=x.clone();rgb=m.display_rgb(x)
    assert rgb.dtype==torch.uint8 and rgb.tolist()==[[0,0,10],[118,255,255]]
    assert torch.equal(x,before)
    with pytest.raises(ValueError):m.display_rgb(torch.tensor([[float('nan'),0.,0.]],dtype=torch.float64))


def test_reference_contact_sheet_is_finite_labelled_and_not_a_residual():
    m=importlib.import_module('tools.testing_dashboard.image_reference_demo')
    names=('constant','vertex','gradient','checker')
    buffers={'albedo/'+name:torch.zeros((128,128,3),dtype=torch.float64) for name in names}
    images={model+'/'+name:torch.full((128,128,3),.18,dtype=torch.float64)
            for model in ('diffuse','classic','environment_diffuse') for name in names}
    result=m.presentation(dict(buffers=buffers,images=images))
    assert len(result['panels'])==16
    assert result['width']==1024 and result['height']==1080
    assert struct.unpack('!II',result['png'][16:24])==(1024,1080)
    assert 'reference' in result['limits'].lower() and 'candidate' in result['limits'].lower()
    assert all('residual' not in x['name'] and 'candidate' not in x['name'] for x in result['panels'])
    assert all('explanation' in x for x in result['panels'])


def test_probe_refuses_a_mismatched_installed_wheel_before_generation(tmp_path):
    m=importlib.import_module('tools.shading_image_probe')
    with pytest.raises(ValueError,match='wheel'):
        m.producer_identity(tmp_path/'missing.whl')


@pytest.mark.parametrize('kind',['changed_file','omitted_files','duplicate_member'])
def test_probe_checks_wheel_bytes_not_just_the_version_string(tmp_path,kind):
    m=importlib.import_module('tools.shading_image_probe')
    import gffx
    wheel=tmp_path/'fixture.whl'
    data=b'wrong bytes' if kind=='changed_file' else Path(gffx.__file__).read_bytes()
    with zipfile.ZipFile(wheel,'w') as archive:
        archive.writestr('gffx/__init__.py',data)
        if kind=='duplicate_member':
            with pytest.warns(UserWarning):archive.writestr('gffx/__init__.py',data)
    with pytest.raises(ValueError,match='differ|extra executable|duplicates'):m.producer_identity(wheel)


def test_probe_declares_existing_bounded_supervisor_and_no_automatic_services():
    m=importlib.import_module('tools.shading_image_probe')
    assert m.PROFILE['cooperative_seconds']==120
    assert m.PROFILE['watchdog_seconds']==180
    assert m.PROFILE['sampled_working_set_limit_bytes']==1024**3
    assert m.PROFILE['threads']==1 and m.PROFILE['width']==m.PROFILE['height']==128
    source=Path(m.__file__).read_text()
    assert "worker_module='tools.shading_image_probe'" in source
    assert 'start_server' not in source


def test_comparison_refuses_underflow_instead_of_claiming_zero_error():
    from tools.shading_image_reference import compare_images
    zero=torch.zeros((1,1,3),dtype=torch.float64)
    with pytest.raises(ValueError,match='underflow'):
        compare_images(zero+1e-200,zero,torch.ones((1,1),dtype=torch.bool))


def test_image_demo_publication_uses_supported_provenance_and_retains_full_identity(monkeypatch):
    from tools.testing_dashboard import image_reference_demo as demo,reporter
    import gffx.dashboard
    from gffx.dashboard._server import check_provenance
    events=[];records={};pictures=[]
    class Reporter:
        def __init__(self,*args):pass
        def send(self,event,**values):
            if event=='start':check_provenance(values['provenance'])
            events.append((event,values));return True
    class Dash:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def text(self,*args,**kwargs):pass
        def image(self,path,array,**kwargs):pictures.append(path)
        def record(self,path,value,**kwargs):records[path]=value
    monkeypatch.setattr(reporter,'Reporter',Reporter)
    monkeypatch.setattr(gffx.dashboard,'connect',lambda *a,**kw:Dash())
    colours=('constant','vertex','gradient','checker')
    image=torch.zeros((128,128,3),dtype=torch.float64)
    identity={'kind':'test-storage-only','versions':{'gffx':'test'}}
    bundle=dict(manifest=dict(producer=identity,details={'test':True}),
                buffers={'albedo/'+c:image for c in colours},
                images={m+'/'+c:image for m in ('diffuse','classic','environment_diffuse') for c in colours})
    reference=dict(path='explicit/manifest.json',sha256='a'*64)
    demo.publish('unused','test',bundle,reference)
    assert events[0][1]['provenance']=={'artifact_sha256':'a'*64}
    assert len(pictures)==16 and len(events[-2][1]['artifacts'])==18
    assert records['image-reference/evidence']['producer']==identity
    assert records['image-reference/evidence']['bundle']==reference
    assert events[-1]==('finish',{'exit_code':0})
