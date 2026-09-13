"""Original demo geometry and a separately scheduled renderer smoke profile."""
import torch


def test_deferred_studio_geometry_is_nondegenerate_and_sphere_winding_is_outward():
    from tools.testing_dashboard.deferred_demo import sphere,ribbon
    v,f,n=sphere();rv,rf,rn,rc=ribbon()
    assert v.shape==(114,3) and f.shape==(224,3)
    assert rv.shape==(34,3) and rf.shape==(32,3) and rc.shape==(34,3)
    for vertices,faces in ((v,f),(rv,rf)):
        corner=vertices[faces.long()]
        area=torch.linalg.cross(corner[:,1]-corner[:,0],corner[:,2]-corner[:,0])
        assert torch.isfinite(area).all() and bool((torch.linalg.vector_norm(area,dim=-1)>0).all())
    sphere_faces=v[f.long()]
    area=torch.linalg.cross(sphere_faces[:,1]-sphere_faces[:,0],sphere_faces[:,2]-sphere_faces[:,0])
    assert bool(((area*n[f.long()].mean(1)).sum(-1)>0).all())
    torch.testing.assert_close(torch.linalg.vector_norm(n,dim=-1),torch.ones(len(n),dtype=n.dtype))
    torch.testing.assert_close(torch.linalg.vector_norm(rn,dim=-1),torch.ones(len(rn),dtype=rn.dtype))


def test_deferred_studio_32_smoke_uses_actual_public_renderer_frames():
    from tools.testing_dashboard.deferred_demo import payload
    result=payload(32)
    assert result['faces']==258 and len(result['frames'])==3
    for frame in result['frames']:
        assert frame['colour'].shape==(1,32,32,3) and torch.isfinite(frame['colour']).all()
        assert bool((frame['mask']>0).any()) and torch.isfinite(frame['normal']).all()
    assert not torch.equal(result['frames'][0]['colour'],result['frames'][2]['colour'])
