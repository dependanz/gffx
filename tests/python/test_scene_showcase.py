"""Visual-presentation contracts; not physical-rendering acceptance."""
import torch


def test_bevels_have_outward_flat_normals_and_no_corner_averaging():
    from tools.testing_dashboard.scene_showcase import bevel_box
    v, f, n = bevel_box((.5, .5, .5), .08)
    assert v.shape == n.shape == (96, 3)
    assert f.shape == (44, 3)
    crosses = torch.linalg.cross(v[f[:, 1].long()]-v[f[:, 0].long()],
                                v[f[:, 2].long()]-v[f[:, 0].long()])
    assert bool(((crosses*n[f[:, 0].long()]).sum(-1) > 0).all())
    assert bool(((v*n).sum(-1) > 0).all())
    torch.testing.assert_close(n.norm(dim=-1), torch.ones(len(n)))


def test_public_scene_hierarchy_moves_only_the_owned_branch(monkeypatch):
    import gffx.scene as scene
    from tools.testing_dashboard.scene_showcase import frame
    calls = []
    original = scene.Scene.flatten
    def tracked(self, **kwargs):
        calls.append(True)
        return original(self, **kwargs)
    monkeypatch.setattr(scene.Scene, 'flatten', tracked)
    start, branch, whole = frame(0, 0), frame(0, .8), frame(.8, 0)
    assert len(calls) == 3
    assert start['vertices'].shape == start['normals'].shape == start['colors'].shape
    assert len(start['faces']) < 3000
    for name in ('plinth', 'turquoise', 'mast'):
        torch.testing.assert_close(start['world'][name], branch['world'][name])
    assert not torch.equal(start['world']['coral'], branch['world']['coral'])
    assert not torch.equal(start['world']['turquoise'], whole['world']['turquoise'])
    torch.testing.assert_close(start['world']['plinth'], whole['world']['plinth'])
    assert torch.unique(start['colors'], dim=0).shape[0] >= 5
    torch.testing.assert_close(start['normals'].norm(dim=-1), torch.ones(len(start['normals'])))


def test_view_transports_contiguous_explicit_types():
    from tools.testing_dashboard.scene_showcase import view
    value = torch.tensor([[.1, .2, .3]], dtype=torch.float64)
    actual = view(value)
    assert actual.dtype == 'float32'
    assert actual.shape == (1, 3)


def test_publication_has_colours_normals_and_finite_checkpoint_windows():
    from tools.testing_dashboard.scene_showcase import publish_data
    calls, queues = [], []
    class Sink:
        def mesh(self, *args, **kwargs): calls.append(('mesh', kwargs))
        def image(self, *args, **kwargs): calls.append(('image', kwargs))
        def text(self, *args, **kwargs): calls.append(('text', kwargs))
        def queue(self, *args, **kwargs): queues.append(kwargs); return self
        def push(self, *args, **kwargs): calls.append(('push', kwargs))
    triangle = dict(vertices=torch.zeros(3, 3), faces=torch.tensor([[0,1,2]], dtype=torch.int32),
                    normals=torch.ones(3,3), colors=torch.ones(3,3))
    data = dict(hero=triangle, clips=[[triangle]*24, [triangle]*24],
                images=[torch.zeros(4,4,3,dtype=torch.uint8)]*3)
    artifacts = publish_data(Sink(), data)
    assert len(artifacts) == 7
    assert len(queues) == 2
    assert all(q == dict(recent_window=24, snapshot_every=1) for q in queues)
    geometry = [kw for kind, kw in calls if kind in ('mesh', 'push')]
    assert len(geometry) == 49
    assert all('colors' in kw and 'normals' in kw for kw in geometry)


def test_all_showcase_checks_have_authored_explanations():
    from tools.testing_dashboard.explanations import explain
    for name, value in globals().copy().items():
        if name.startswith('test_') and callable(value):
            assert explain('tests/python/test_scene_showcase.py::'+name)['authored']


def test_tiled_render_matches_full_native_rasterization():
    from gffx.torch import render
    from tools.testing_dashboard.scene_showcase import tiled_samples
    vertices = torch.tensor([[-.81,-.7,.3],[.8,-.6,.3],[.1,.8,.3],
                             [-.4,-.3,.2],[.55,-.1,.2],[.1,.6,.2]])
    faces = torch.tensor([[0,1,2],[3,4,5]],dtype=torch.int32)
    attrs = torch.arange(18,dtype=torch.float32).reshape(2,3,3)/18
    fragments = render.rasterize(vertices,faces,31,47,cull_mode=render.CULL_NONE)
    expected = render.interpolate(fragments[0],fragments[1],attrs)[0,:,:,0]
    actual, valid = tiled_samples(vertices,faces,attrs,31,47,tile=16)
    assert torch.equal(valid,fragments[0][0,:,:,0]>=0)
    torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-6)
