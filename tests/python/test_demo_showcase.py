"""Finite demonstration behavior, not public shading or animation acceptance."""
import sys
import pytest
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

def test_showcase_provenance_obeys_dashboard_contract():
    from tools.testing_dashboard.showcase import provenance
    from gffx.dashboard._server import check_provenance
    value=provenance()
    assert check_provenance(value)==value
    assert len(value['artifact_sha256'])==64


def test_invalid_demo_provenance_returns_a_json_client_error(tmp_path):
    import json
    from urllib.request import Request,urlopen
    from urllib.error import HTTPError
    from tools.testing_dashboard.server import TestingServer
    with TestingServer(tmp_path,port=0,allow_loopback_for_tests=True) as server:
        body=json.dumps(dict(event='start',run='bad',kind='demo',provenance={'unknown':'bad'})).encode()
        request=Request('http://'+server.address+'/testing/api/events',data=body,headers={'Content-Type':'application/json'})
        with pytest.raises(HTTPError) as caught:urlopen(request)
        assert caught.value.code==400
        assert 'provenance' in json.load(caught.value)['error']
def test_archived_demo_can_be_reopened_after_idle_capacity(tmp_path):
    from tools.testing_dashboard.server import TestingServer
    from tools.testing_dashboard.reporter import Reporter
    from gffx.dashboard import connect
    from gffx.dashboard._page import PageConnection
    from gffx.dashboard._limits import ResourceLimits
    with TestingServer(tmp_path, port=0, allow_loopback_for_tests=True,
                       limits=ResourceLimits(runs=2)) as server:
        address='http://'+server.address
        for name in ('one','two','three','one'):
            reporter=Reporter(address,name)
            reporter.send('start',kind='demo',provenance={})
            with connect(address,run=name,wait_acks=True) as dash:
                dash.scalar('value',1.,step=0)
            reporter.send('finish',exit_code=0)
            page=PageConnection(address,run=name)
            assert 'value' in page.open()['paths']
            page.close()
        assert len(server.state.runs())<=2
        assert len(list((tmp_path/'viewer').glob('*/log.bin')))==3


@pytest.mark.parametrize('protection',['active','ephemeral'])
def test_archive_capacity_does_not_discard_active_or_unsaved_state(tmp_path,protection):
    from tools.testing_dashboard.server import TestingServer
    from gffx.dashboard import connect
    from gffx.dashboard._limits import ResourceLimits
    with TestingServer(tmp_path,port=0,allow_loopback_for_tests=True,limits=ResourceLimits(runs=1)) as server:
        address='http://'+server.address
        server.index.accept(dict(event='start',run='protected',kind='demo',provenance={}))
        with connect(address,run='protected',wait_acks=True) as dash:
            dash.queue('value','scalar',snapshot_every=0).push(2.,step=0)
            server.index.accept(dict(event='finish',run='protected',exit_code=0))
            if protection=='active':
                with pytest.raises(ConnectionError,match='limit'):
                    connect(address,run='other',wait_acks=True)
        if protection=='ephemeral':
            with pytest.raises(ConnectionError,match='limit'):
                connect(address,run='other',wait_acks=True)
        assert server.state.get('value',run='protected')['value']==2.


def test_showcase_scene_uses_offline_meshes_and_valid_topology():
    from tools.testing_dashboard.showcase import scene
    import torch
    s=scene()
    assert 3000 < len(s['vertices']) < 8000
    assert s['faces'].dtype==torch.int32
    assert int(s['faces'].max())<len(s['vertices'])
    assert torch.isfinite(s['vertices']).all()
    assert s['colors'].shape==s['vertices'].shape


@pytest.mark.parametrize('mode',['orbit','pan','dolly','zoom','dolly-zoom'])
def test_camera_paths_are_finite_and_distinct(mode):
    from tools.testing_dashboard.showcase import camera_path
    import torch
    a=camera_path(mode,0);b=camera_path(mode,1)
    assert torch.isfinite(a.world_to_view()).all() and torch.isfinite(b.world_to_view()).all()
    assert not torch.equal(a.world_to_view(),b.world_to_view()) or not torch.equal(a.projection(),b.projection())


def test_dolly_zoom_keeps_reference_plane_scale_not_background():
    from tools.testing_dashboard.showcase import camera_path
    import torch
    points=torch.tensor([[-.5,1.,0.],[.5,1.,0.],[-.5,1.,-1.5],[.5,1.,-1.5]])
    a=camera_path('dolly-zoom',0).project(points).pixels
    b=camera_path('dolly-zoom',1).project(points).pixels
    assert torch.allclose(a[:2],b[:2],atol=1e-4)
    assert not torch.allclose(a[2:],b[2:],atol=1e-3)


def test_showcase_render_changes_with_light_and_has_depth_normals():
    from tools.testing_dashboard.showcase import scene,render_frame,camera_path
    import torch
    s=scene();camera=camera_path('dolly',.5,height=48,width=64)
    a=render_frame(s,camera,0);b=render_frame(s,camera,.4)
    assert a['color'].shape==(48,64,3) and a['color'].dtype==torch.uint8
    assert not torch.equal(a['color'],b['color'])
    assert a['mask'].sum()>100
    assert torch.isfinite(a['depth'][a['mask']]).all()
    assert a['normals'].shape==a['color'].shape
    assert torch.equal(a['color'],render_frame(s,camera,0)['color'])


def test_bunny_fit_has_recognizable_subject_and_bounded_convergence():
    from tools.testing_dashboard.camera_fit_demo import solve,fit_frames
    result=solve(subject='bunny')
    frames=fit_frames(result)
    assert len(frames)==61 and frames[0][0]==0 and frames[-1][0]==300
    assert frames[0][1].data!=frames[-1][1].data
    assert result['history'][-1][1]<1e-6


def test_depth_reconstruction_reprojects_only_visible_samples():
    from tools.testing_dashboard.showcase import depth_samples,scene,camera_path,render_frame
    import torch
    c=camera_path('orbit',.5,height=48,width=64);r=render_frame(scene(),c,.2)
    points,colors,pixels=depth_samples(c,r,stride=4)
    assert 0<len(points)<=192 and points.shape==colors.shape
    assert torch.allclose(c.project(points).pixels,pixels,atol=1e-4)


def test_ball_energy_breakdown_sums_to_the_reported_mechanical_energy():
    from tools.testing_dashboard.classic_demo import simulate
    for s in simulate():
        assert abs(s['potential']+s['kinetic']+s['spring']-s['energy'])<1e-10
        assert abs(s['energy']+s['energy_difference']-19.62)<1e-10


def test_classic_inspection_uses_native_face_measurements():
    from tools.testing_dashboard.classic_demo import inspection
    info=inspection('teapot')
    assert len(info['areas'])==2256 and len(info['face_normals'])==2256
    assert len(info['normal_colors'])==1568
    assert all(a>=0 for a in info['areas'])


def test_depth_demo_publishes_points_through_shared_mesh_panel(tmp_path):
    from tools.testing_dashboard.showcase import publish_depth
    from tools.testing_dashboard.server import TestingServer
    with TestingServer(tmp_path,port=0,allow_loopback_for_tests=True) as server:
        publish_depth('http://'+server.address,'depth',height=48,width=64)
        page=server.index.page('camera/depth-demo')
        assert len(page['artifacts'])==4
        assert next(a for a in page['artifacts'] if a['path']=='depth/surface')['kind']=='mesh'
