"""Minimal scene contract: independent layouts, ownership and error fixtures."""
import importlib
import pytest
import torch


def api():
    return importlib.import_module('gffx.torch.scene')


def triangle(dtype=torch.float64, device='cpu'):
    return (torch.tensor([[-.7,-.6,.2],[.6,-.5,.3],[0.,.7,.4]],dtype=dtype,device=device),
            torch.tensor([[0,1,2]],dtype=torch.int32,device=device))


def affine(x=0., dtype=torch.float64, device='cpu'):
    t=torch.eye(4,dtype=dtype,device=device);t[0,3]=x;return t


def test_scene_facade_and_empty():
    s=api();import gffx
    assert gffx.scene.Scene is s.Scene
    p=s.Scene([]).flatten(batch_size=2)
    assert p.vertices.shape==(0,3) and p.faces.shape==(0,3)
    assert p.vertex_offsets.tolist()==[0,0,0]
    assert p.object_vertex_offsets.shape==(2,1)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_scene_hierarchy_packing_and_existing_validation(dtype):
    s=api();v,f=triangle(dtype);root=affine(2,dtype);root[1,1]=2
    child=affine(3,dtype);leaf=affine(4,dtype)
    p=s.Scene([s.Rig('root',transform=root),s.Rig('child',parent='root',transform=child),
               s.Mesh('a',v,f,parent='child',transform=leaf),s.Mesh('b',v,f)]).flatten()
    expected=v.clone();expected[:,0]+=9;expected[:,1]*=2
    torch.testing.assert_close(p.vertices,torch.cat((expected,v)),rtol=0,atol=0)
    assert p.faces.tolist()==[[0,1,2],[3,4,5]]
    assert p.object_vertex_offsets.tolist()==[[0,3,6]]
    from gffx.torch import mesh
    assert mesh.validate(p.vertices,p.faces,vertex_offsets=p.vertex_offsets,face_offsets=p.face_offsets).is_clean


@pytest.mark.parametrize('domain',['constant','uniform','vertex','face_varying'])
def test_scene_attribute_domains_and_missing_mask(domain):
    s=api();v,f=triangle();shapes={'constant':(2,),'uniform':(1,2),'vertex':(3,2),'face_varying':(1,3,2)}
    a=torch.arange(torch.tensor(shapes[domain]).prod().item(),dtype=v.dtype).reshape(shapes[domain]).requires_grad_()
    p=s.Scene([s.Mesh('a',v,f,attributes={'uv':s.Attribute(a,domain)}),s.Mesh('b',v,f)]).flatten()
    expected=a.expand(3,2) if domain in ('constant','uniform') else a.reshape(3,2)
    torch.testing.assert_close(p.attributes['uv'].values[0],expected)
    assert p.attributes['uv'].present.tolist()==[True,False]
    assert p.attributes['uv'].domains==(domain,None)
    assert torch.equal(p.attributes['uv'].values[1],torch.zeros(3,2,dtype=v.dtype))
    assert torch.autograd.grad(p.attributes['uv'].values.sum(),a)[0].abs().sum()>0


def test_scene_frames_and_batched_attributes():
    s=api();v,f=triangle();t=torch.stack([affine(0),affine(.1)])
    colours=torch.tensor([[1.,0,0],[0,1.,0]],dtype=v.dtype)
    p=s.Scene([s.Mesh('a',v,f,transform=t,attributes={'colour':s.Attribute(colours,'constant')})]).flatten()
    assert p.vertex_offsets.tolist()==[0,3,6] and p.face_offsets.tolist()==[0,1,2]
    assert p.faces.tolist()==[[0,1,2],[3,4,5]]
    torch.testing.assert_close(p.vertices[3:],v+torch.tensor([.1,0,0],dtype=v.dtype))
    torch.testing.assert_close(p.attributes['colour'].values[:,0],colours)


def test_scene_bindings_snapshot_and_tensor_borrowing():
    s=api();v,f=triangle();from gffx.torch import material,light,camera
    params={'roughness':.5};attrs={};prims=[s.Material('m',material.classic,params),s.Light('l',light.point,{}),
       s.Camera('c',camera.Camera.from_intrinsics(fx=1.,fy=1.,cx=0.,cy=0.,image_width=8,image_height=8,near=.01,far=10.,pose=affine().unsqueeze(0))),
       s.Mesh('a',v,f,material='m',attributes=attrs)]
    scene=s.Scene(prims);prims.clear();params['roughness']=.8;attrs['bad']=None
    p=scene.flatten();assert p.face_material.tolist()==[0]
    assert p.materials[0].parameters['roughness']==.5 and len(p.lights)==len(p.cameras)==1
    v[0,0]=.1;assert scene.flatten().vertices[0,0]==.1
    p.vertices.zero_();assert not torch.equal(v,p.vertices)


def test_scene_normals_nonuniform_scale_and_reflection():
    s=api();v,f=triangle();n=torch.tensor([1.,1.,1.],dtype=v.dtype);t=affine();t[0,0]=-2
    p=s.Scene([s.Mesh('a',v,f,transform=t,attributes={'normals':s.Attribute(n,'constant')})]).flatten()
    expected=torch.tensor([.5,-1.,-1.],dtype=v.dtype);expected/=expected.norm()
    torch.testing.assert_close(p.attributes['normals'].values[0],expected.expand(3,3))


def test_scene_strides_defaults_repeat_and_no_alias():
    s=api();v,f=triangle();wide=torch.zeros(3,6,dtype=v.dtype);wide[:,::2]=v;v=wide[:,::2]
    scene=s.Scene([s.Mesh('a',v,f)])
    with torch.device('meta'):p=scene.flatten()
    assert p.vertices.device.type=='cpu' and p.faces.device.type=='cpu'
    assert torch.equal(p.vertices,scene.flatten().vertices)
    p.faces.fill_(2);assert f.tolist()==[[0,1,2]]


@pytest.mark.parametrize('case',['duplicate','missing_parent','cycle','bad_material','dtype','faces','index',
 'nan','singular','projective','batch','domain','channels','attribute_shape','zero_normal','device'])
def test_scene_rejects_invalid_input(case):
    s=api();v,f=triangle();kw={};prims=[]
    if case=='duplicate':prims=[s.Rig('a')]
    if case=='missing_parent':kw['parent']='absent'
    if case=='cycle':kw['parent']='a'
    if case=='bad_material':kw['material']='absent'
    if case=='dtype':v=v.half()
    if case=='faces':f=f.long()
    if case=='index':f=f+10
    if case=='nan':v[0,0]=float('nan')
    if case in ('singular','projective'):
        t=affine();t[0,0]=0 if case=='singular' else 1;t[3,0]=1 if case=='projective' else 0;kw['transform']=t
    if case=='batch':v=v.unsqueeze(0).expand(2,-1,-1);kw['transform']=affine().expand(3,-1,-1)
    if case=='domain':kw['attributes']={'uv':s.Attribute(torch.ones(2,dtype=v.dtype),'wrong')}
    if case=='channels':kw['attributes']={'uv':s.Attribute(torch.ones(3,dtype=v.dtype),'constant')}
    if case=='attribute_shape':kw['attributes']={'uv':s.Attribute(torch.ones(4,2,dtype=v.dtype),'vertex')}
    if case=='zero_normal':kw['attributes']={'normals':s.Attribute(torch.zeros(3,dtype=v.dtype),'constant')}
    if case=='device':kw['transform']=affine().to('meta')
    with pytest.raises((TypeError,ValueError)):
        s.Scene(prims+[s.Mesh('a',v,f,**kw)]).flatten()


def test_scene_mutation_revalidated_and_unknown_options_rejected():
    s=api();v,f=triangle();scene=s.Scene([s.Mesh('a',v,f)])
    with pytest.raises(ValueError):scene.flatten(batch_size=0)
    with pytest.raises((TypeError,ValueError)):scene.flatten(dtype=torch.float32)
    f[0,0]=-1
    with pytest.raises(ValueError):scene.flatten()


def test_scene_empty_mesh_and_shared_gradients():
    s=api();v,f=triangle();v.requires_grad_();t=affine().requires_grad_()
    p=s.Scene([s.Mesh('empty',v[:0],f[:0]),s.Mesh('a',v,f,transform=t),s.Mesh('b',v,f,transform=t)]).flatten()
    assert p.object_face_offsets.tolist()==[[0,0,1,2]]
    torch.testing.assert_close(torch.autograd.grad(p.vertices.sum(),v)[0],torch.full_like(v,2))


def test_scene_composed_singular_transform_rejected():
    s=api();v,f=triangle(torch.float32);t=affine(dtype=v.dtype);t[:3,:3]*=1e-10
    with pytest.raises(ValueError,match='world'):
        s.Scene([s.Rig('r',transform=t),s.Mesh('a',v,f,parent='r',transform=t)]).flatten()


def test_scene_vertex_sequence_seams_and_deep_hierarchy():
    s=api();v,f=triangle();faces=torch.cat((f,f.flip(1)))
    uv=torch.arange(12,dtype=v.dtype).reshape(2,3,2)
    frames=torch.stack((v,v+.1));prims=[s.Rig('r0')]
    prims.extend(s.Rig('r'+str(i),parent='r'+str(i-1)) for i in range(1,1100))
    p=s.Scene(prims+[s.Mesh('m',frames,faces,parent='r1099',attributes={'uv':s.Attribute(uv,'face_varying')})]).flatten()
    torch.testing.assert_close(p.vertices,frames.reshape(-1,3),rtol=0,atol=0)
    torch.testing.assert_close(p.attributes['uv'].values,uv.repeat(2,1,1),rtol=0,atol=0)


def test_scene_cross_mesh_channel_mismatch_and_constructor_types():
    s=api();v,f=triangle()
    with pytest.raises(ValueError,match='channel'):
        s.Scene([s.Mesh('a',v,f,attributes={'custom':s.Attribute(v[0],'constant')}),
                 s.Mesh('b',v,f,attributes={'custom':s.Attribute(v[0,:2],'constant')})]).flatten()
    with pytest.raises(TypeError):s.Material('bad',3)
    with pytest.raises(TypeError):s.Camera('bad',object())
    with pytest.raises(TypeError):s.Scene([object()])
