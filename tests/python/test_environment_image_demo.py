"""Partial numerical evidence must remain visually distinguishable from a render."""
import importlib
import pytest
import torch


def module():return importlib.import_module('tools.testing_dashboard.environment_image_demo')


def fixture():
    return dict(indices=[0,1],target=torch.tensor([[.3,.4,.5]]*2),
                values=[torch.tensor([[.3,.4,.5],[0.,0.,0.]]),torch.zeros((2,3))],
                validated_pixels=[1,0])


def test_environment_image_preview_marks_uncomputed_pixels_explicitly():
    panels=module().panels(fixture())
    assert set(panels)=={'target','before','after'}
    assert all(x.shape==(128,128,3) and x.dtype==torch.uint8 for x in panels.values())
    assert panels['before'][0,1].tolist()==[96,96,96]
    assert panels['after'][0,0].tolist()==[96,96,96]
    assert panels['before'][1,0].tolist()==[24,24,28]
    assert panels['before'][0,0].tolist()==panels['target'][0,0].tolist()


@pytest.mark.parametrize('bad',['duplicate','nonfinite','overclaim'])
def test_environment_image_preview_rejects_invalid_ownership_or_values(bad):
    value=fixture()
    if bad=='duplicate':value['indices']=[0,0]
    elif bad=='nonfinite':value['values'][0][0,0]=float('nan')
    else:value['validated_pixels']=[3,0]
    with pytest.raises(ValueError):module().panels(value)


def test_environment_image_publisher_uses_supported_provenance_and_finite_demo(tmp_path):
    from tools.testing_dashboard.server import TestingServer
    value=fixture()
    value.update(model='classic',environment='uniform',complete=False,workers=1,total_workers=104,
        pixels_per_image=2,cohort_sha256='a'*64,calls=2,directions=512,peak_host_bytes=1000,
        worker_seconds=1.,losses=[])
    with TestingServer(tmp_path,port=0,allow_loopback_for_tests=True) as server:
        module().publish(value,'http://'+server.address,'image-demo-control')
        summary=server.index.summary('image-demo-control')
        assert summary['state']=='finished' and summary['kind']=='demo'
        assert summary['counts']=={} and summary['diagnostic'] is None
        page=server.index.page('shading/environment-image-adjoints-demo')
        assert len(page['artifacts'])==5


def completed_fixture(model,environment):
    value=fixture();value['validated_pixels']=[2,2]
    value.update(model=model,environment=environment,complete=True,workers=104,total_workers=104,
        pixels_per_image=2,cohort_sha256='b'*64,calls=8,directions=2048,peak_host_bytes=1000,
        worker_seconds=1.,losses=[[.2,.2],[.1,.1]])
    return value


@pytest.mark.parametrize('bad',['missing','duplicate','partial'])
def test_environment_image_gallery_requires_all_four_complete_profiles(bad):
    values=[completed_fixture(m,e) for m in ('classic','principled') for e in ('uniform','affine')]
    if bad=='missing':values.pop()
    elif bad=='duplicate':values[-1]=values[0]
    else:values[-1]['complete']=False
    with pytest.raises(ValueError,match='four complete'):
        module().publish_gallery(values,'http://127.0.0.1:1','must-not-connect')


def test_environment_image_gallery_publishes_every_comparison_without_test_claims(tmp_path):
    from tools.testing_dashboard.server import TestingServer
    values=[completed_fixture(m,e) for m in ('classic','principled') for e in ('uniform','affine')]
    with TestingServer(tmp_path,port=0,allow_loopback_for_tests=True) as server:
        module().publish_gallery(values,'http://'+server.address,'gallery-control')
        summary=server.index.summary('gallery-control')
        assert summary['state']=='finished' and summary['counts']=={} and summary['diagnostic'] is None
        artifacts=server.index.page('shading/environment-image-adjoints-demo')['artifacts']
        assert len(artifacts)==21
        assert sum(x['kind']=='image' for x in artifacts)==12
        assert len({x['path'] for x in artifacts})==21
