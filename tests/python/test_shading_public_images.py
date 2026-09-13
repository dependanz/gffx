"""Public image comparisons stay red while the declared shading modules are absent."""
import importlib
import pytest
import torch


@pytest.mark.parametrize('specular',[0.,.1])
@pytest.mark.parametrize('albedo',['constant','vertex','gradient','checker'])
def test_public_classic_head_image_matches_independent_reference(albedo,specular):
    material=importlib.import_module('gffx.torch.material')
    light=importlib.import_module('gffx.torch.light')
    from tools.synthetic_face_fixture import generate
    from tools.shading_image_reference import build_buffers,reference_images,compare_images
    old=torch.get_num_threads(); torch.set_num_threads(1)
    try:
        scene=generate(); buffers=build_buffers(scene); mask=buffers['covered']
        n=buffers['normal'][mask]; v=buffers['view'][mask]
        direction=torch.tensor([[1.,1.,2.]],dtype=torch.float64); direction/=direction.norm(dim=-1,keepdim=True)
        brdf=material.classic(n,v,direction,albedo=buffers['albedos'][albedo][mask],
                              specular=torch.full((3,),specular,dtype=torch.float64),exponent=torch.tensor(2.,dtype=torch.float64))
        packet=dict(measure='delta',direction=direction.expand(len(n),1,3),
                    irradiance=torch.ones((len(n),1,3),dtype=torch.float64),valid=torch.ones((len(n),1),dtype=torch.bool))
        actual=torch.zeros((128,128,3),dtype=torch.float64)
        actual[mask]=light.integrate(n,brdf,packet)
        name=('diffuse' if specular==0 else 'classic')+'/'+albedo
        assert compare_images(actual,reference_images(buffers,scene)[name],mask)['passed']
    finally:torch.set_num_threads(old)
