"""Matched-operand diagnostic, not acceptance of differing CPU/CUDA bytes."""
import json
import math
import pytest
import torch
from gffx.torch import shade, material, light, render
from test_deferred_cuda import fixture, cuda_required


def bits(value):
    flat=value.detach().cpu().contiguous().reshape(-1)
    return bytes(flat.view(torch.uint8).tolist()).hex()


def division(device,dtype,*,tensor_denominator):
    albedo=torch.tensor([[.25,.5,.75]],dtype=dtype,device=device,requires_grad=True)
    cotangent=albedo.new_tensor([[.25,.5,.75]])
    denominator=albedo.new_tensor(math.pi)
    value=albedo/(denominator if tensor_denominator else math.pi)
    gradient,=torch.autograd.grad(value,albedo,cotangent)
    assert torch.isfinite(value).all() and torch.isfinite(gradient).all()
    return dict(value=bits(value),albedo_vjp=bits(gradient),
        values=value.detach().cpu().reshape(-1).tolist(),
        gradients=gradient.detach().cpu().reshape(-1).tolist(),
        operands=dict(albedo=bits(albedo),cotangent=bits(cotangent)),
        explicit_denominator=bits(denominator),python_scalar_hex=math.pi.hex())


def stage(device,dtype,name):
    def t(value):return torch.tensor(value,dtype=dtype,device=device)
    albedo=t([.25,.5,.75]).requires_grad_()
    normal=t([[0.,0.,1.]])
    view=t([[0.,0.,2.]])
    direction=t([[[0.,0.,1.]]])
    irradiance=t([[[1.,1.,1.]]])
    if name=='lambert':
        value=shade.lambert(albedo[None])
    elif name in ('classic','integrate'):
        value=material.classic(normal,view,direction,albedo=albedo,
                               specular=t(0.),exponent=t(4.))
        if name=='integrate':
            packet=dict(measure='delta',direction=direction,irradiance=irradiance,
                        valid=torch.ones((1,1),dtype=torch.bool,device=device))
            value=light.integrate(normal,value,packet)
    elif name=='render':
        model,camera,parameters=fixture(device,dtype)
        albedo=parameters[0]
        output=render(model,camera,outputs=('colour','normal','world_position','mask'))
        assert bool((output['mask']==1).all())
        # Verify actual geometric operands, not only the intended fixture.
        assert bits(output['normal'])==bits(normal)
        actual_view=camera.pose[0,:3,3]-output['world_position'].reshape(1,3)
        assert bits(actual_view)==bits(view)
        value=output['colour']
    else:raise ValueError('undeclared attribution stage')
    cotangent=t([.25,.5,.75]).reshape_as(value)
    gradient,=torch.autograd.grad(value,albedo,cotangent)
    assert torch.isfinite(value).all() and torch.isfinite(gradient).all()
    return dict(shape=list(value.shape),value=bits(value),albedo_vjp=bits(gradient),
                values=value.detach().cpu().reshape(-1).tolist(),
                gradients=gradient.detach().cpu().tolist(),
                operands={key:bits(tensor) for key,tensor in dict(albedo=albedo,
                    normal=normal,view=view,direction=direction,irradiance=irradiance,
                    cotangent=cotangent).items()})


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_deferred_classic_matched_operand_attribution(cuda_required,dtype,record_property):
    rows=[]
    for name in ('lambert','classic','integrate','render'):
        cpu=stage('cpu',dtype,name);cuda=stage('cuda',dtype,name)
        assert cpu['shape']==cuda['shape']
        assert cpu['operands']==cuda['operands'],'attribution requires identical operand bytes'
        rows.append(dict(stage=name,cpu=cpu,cuda=cuda,
            value_equal=cpu['value']==cuda['value'],
            albedo_vjp_equal=cpu['albedo_vjp']==cuda['albedo_vjp']))
    divisions={}
    for name,tensor_denominator in (('legacy_python_scalar',False),('explicit_device_tensor',True)):
        cpu=division('cpu',dtype,tensor_denominator=tensor_denominator)
        cuda=division('cuda',dtype,tensor_denominator=tensor_denominator)
        assert cpu['operands']==cuda['operands']
        assert cpu['explicit_denominator']==cuda['explicit_denominator']
        divisions[name]=dict(cpu=cpu,cuda=cuda,value_equal=cpu['value']==cuda['value'],
            albedo_vjp_equal=cpu['albedo_vjp']==cuda['albedo_vjp'])
    public_matches={lane:{variant:all(divisions[variant][lane][key]==rows[0][lane][key]
                      for key in ('value','albedo_vjp')) for variant in divisions}
                    for lane in ('cpu','cuda')}
    evidence=dict(dtype=str(dtype),diagnostic_only=True,rows=rows,division_microprobe=divisions,
        public_lambert_matches_variant=public_matches,
        first_value_difference=next((row['stage'] for row in rows if not row['value_equal']),None),
        first_vjp_difference=next((row['stage'] for row in rows if not row['albedo_vjp_equal']),None))
    record_property('deferred_classic_attribution',json.dumps(evidence,sort_keys=True))
    # A passing diagnostic means its operand/propagation controls held. It does
    # not turn any recorded CPU/CUDA mismatch into an accepted numerical result.
    for lane in ('cpu','cuda'):
        assert any(public_matches[lane].values()),lane+' has a different division boundary'
        for row in rows[1:]:
            assert row[lane]['value']==rows[0][lane]['value'],lane+' adds a downstream value difference'
            assert row[lane]['albedo_vjp']==rows[0][lane]['albedo_vjp'],lane+' adds a downstream VJP difference'
