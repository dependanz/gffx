"""Installed-CPU image fixture controls; independent lighting, shared rasterization."""
import importlib
import math

import pytest
import torch


def module():
    return importlib.import_module('tools.shading_image_reference')


@pytest.fixture(scope='module')
def scene():
    from tools.synthetic_face_fixture import generate
    return generate()


@pytest.fixture(scope='module')
def buffers(scene):
    old=torch.get_num_threads(); torch.set_num_threads(1)
    try: yield module().build_buffers(scene)
    finally: torch.set_num_threads(old)


@pytest.fixture(scope='module')
def images(buffers,scene):
    return module().reference_images(buffers,scene)


def test_image_geometry_fields_are_finite_unit_and_masked(buffers):
    mask=buffers['covered']; assert mask.shape==(128,128) and mask.dtype==torch.bool
    assert 0<int(mask.sum())<128*128
    assert buffers['face'].dtype==torch.int32
    assert torch.equal(buffers['face']>=0,mask)
    for name in ('world','normal','view','uv','barycentric','affine','depth'):
        field=buffers[name]
        assert field.device.type=='cpu' and field.dtype==torch.float64
        assert torch.isfinite(field).all()
        assert (field[~mask]==0).all()
    for name in ('normal','view'):
        torch.testing.assert_close(torch.linalg.vector_norm(buffers[name][mask],dim=-1),torch.ones(int(mask.sum()),dtype=torch.float64),atol=1e-12,rtol=0)
    assert buffers['face'][~mask].eq(-1).all()


def test_camera_projection_and_world_roundtrip_have_independent_pinhole_answers(buffers,scene):
    mask=buffers['covered']; world=buffers['world'][mask]
    focal=64/math.tan(math.radians(20))
    projected=torch.stack((64+focal*world[:,0]/(3-world[:,2]),64-focal*world[:,1]/(3-world[:,2])),dim=-1)
    y,x=torch.where(mask)
    expected=torch.stack((x+.5,y+.5),dim=-1).to(torch.float64)
    torch.testing.assert_close(projected,expected,atol=1e-10,rtol=0)
    v=torch.tensor(scene['vertices'],dtype=torch.float64)
    analytic=torch.stack((64+focal*v[:,0]/(3-v[:,2]),64-focal*v[:,1]/(3-v[:,2])),dim=-1)
    torch.testing.assert_close(buffers['vertex_pixels'],analytic,atol=1e-11,rtol=0)


def test_perspective_weights_differ_from_affine_and_reconstruct_attributes(buffers,scene):
    mask=buffers['covered']; q=buffers['barycentric'][mask]
    affine=buffers['affine'][mask]
    assert (q>=0).all() and (q-affine).abs().max()>1e-4
    torch.testing.assert_close(q.sum(-1),torch.ones(len(q),dtype=torch.float64),atol=1e-12,rtol=0)
    faces=torch.tensor(scene['triangles'],dtype=torch.int64)[buffers['face'][mask].long()]
    vertices=torch.tensor(scene['vertices'],dtype=torch.float64)[faces]
    expected=(vertices*q[...,None]).sum(1)
    torch.testing.assert_close(buffers['world'][mask],expected,atol=1e-12,rtol=0)


def test_constant_vertex_and_texture_sources_equal_the_explicit_field(buffers):
    assert buffers['controls']['constant_vertex_max_error']<1e-12
    assert buffers['controls']['constant_texture_max_error']<1e-12
    expected=torch.tensor([.2,.4,.6],dtype=torch.float64).expand(int(buffers['covered'].sum()),3)
    torch.testing.assert_close(buffers['albedos']['constant'][buffers['covered']],expected,atol=1e-12,rtol=0)


@pytest.mark.parametrize('name',['gradient','checker'])
def test_every_visible_texture_sample_matches_independent_bilinear(buffers,scene,name):
    from tools.shading_independent_evaluator import bilinear
    mask=buffers['covered']
    expected=torch.tensor([bilinear(scene['textures'][name],uv) for uv in buffers['uv'][mask].tolist()],dtype=torch.float64)
    torch.testing.assert_close(buffers['albedos'][name][mask],expected,atol=1e-12,rtol=1e-12)
    assert buffers['uv'][mask][:,0].min()<.2 and buffers['uv'][mask][:,0].max()>.8


@pytest.mark.parametrize('specular',[0.,.1])
@pytest.mark.parametrize('albedo',['constant','vertex','gradient','checker'])
def test_direct_images_match_independent_vector_closed_form(buffers,images,albedo,specular):
    mask=buffers['covered']; n=buffers['normal']; v=buffers['view']
    light=torch.tensor([1.,1.,2.],dtype=torch.float64); light/=torch.linalg.vector_norm(light)
    half=v+light; half/=torch.linalg.vector_norm(half,dim=-1,keepdim=True)
    nl=(n*light).sum(-1).clamp_min(0); nh=(n*half).sum(-1).clamp_min(0)
    valid=mask & ((n*v).sum(-1)>0)
    expected=(buffers['albedos'][albedo]/math.pi+specular*10/(8*math.pi)*nh[...,None]**2)*nl[...,None]
    expected=torch.where(valid[...,None],expected,torch.zeros_like(expected))
    name=('diffuse' if specular==0 else 'classic')+'/'+albedo
    result=module().compare_images(images[name],expected,mask)
    assert result['passed'] and result['max_absolute_error']<1e-12


def test_unit_environment_lambert_images_preserve_albedo_and_background(buffers,images):
    valid=buffers['covered'] & ((buffers['normal']*buffers['view']).sum(-1)>0)
    for name,albedo in buffers['albedos'].items():
        torch.testing.assert_close(images['environment_diffuse/'+name],torch.where(valid[...,None],albedo,torch.zeros_like(albedo)),atol=0,rtol=0)
    assert len(images)==12


def test_image_environment_controls_are_bounded_independent_points(buffers):
    report=module().environment_controls(buffers)
    assert len(report)==8
    assert len({row['pixel_index'] for row in report})==8
    assert sum(row['evaluations'] for row in report)<=8*87296
    assert all(row['status']=='converged' and row['max_error']<1e-9 for row in report)


def test_image_comparison_reports_denominators_errors_and_exact_match():
    m=module(); reference=torch.ones((2,2,3),dtype=torch.float64)
    reference[1]=0; mask=torch.tensor([[True,True],[False,False]])
    exact=m.compare_images(reference,reference,mask)
    assert exact['passed'] and exact['exact_match'] and exact['psnr_db'] is None
    candidate=reference.clone();candidate[mask]+=.1
    row=m.compare_images(candidate,reference,mask)
    assert not row['passed'] and row['covered_pixels']==2 and row['covered_channels']==6
    assert row['mae']==pytest.approx(.1) and row['signed_mean_error']==pytest.approx(.1)
    assert row['mse']==pytest.approx(.01) and row['psnr_db']==pytest.approx(20.)


@pytest.mark.parametrize('kind',['background','channel','light'])
def test_wrong_image_negative_controls_cannot_pass(buffers,images,kind):
    ref=images['classic/gradient']; wrong=ref.clone(); mask=buffers['covered']
    if kind=='background':wrong[~mask]=.01
    elif kind=='channel':wrong=wrong.flip(-1)
    else:wrong*=1.1
    assert not module().compare_images(wrong,ref,mask)['passed']


@pytest.mark.parametrize('kind',['empty','nan','shape','dtype','mask'])
def test_invalid_comparison_inputs_fail_before_metrics(kind):
    m=module(); ref=torch.zeros((2,2,3),dtype=torch.float64); other=ref.clone(); mask=torch.ones((2,2),dtype=torch.bool)
    if kind=='empty':mask[:]=False
    elif kind=='nan':other[0,0,0]=float('nan')
    elif kind=='shape':other=other[:1]
    elif kind=='dtype':other=other.float()
    else:mask=mask.to(torch.int32)
    with pytest.raises((ValueError,TypeError)):m.compare_images(other,ref,mask)


def test_repeated_buffer_build_is_identical_and_profile_limits_are_fixed(buffers,scene):
    again=module().build_buffers(scene)
    for name in ('world','normal','view','uv','barycentric','affine','depth','face','covered'):
        assert torch.equal(buffers[name],again[name])
    broken=dict(scene,scene=dict(scene['scene'],camera=dict(scene['scene']['camera'],width=256)))
    with pytest.raises(ValueError,match='profile'):module().build_buffers(broken)
