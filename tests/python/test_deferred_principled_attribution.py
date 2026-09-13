"""One-point Principled VJP attribution, not relaxed device qualification."""
import hashlib
import json
import pytest
import torch
from gffx.torch import material, scene, render
from gffx.torch._deferred import _principled_recompute
from test_deferred_cuda import fixture, cuda_required


def raw(value):
    return bytes(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).tolist())


def tensor_record(value):
    return dict(dtype=str(value.dtype),shape=list(value.shape),hex=raw(value).hex(),
                values=value.detach().cpu().reshape(-1).tolist())


def evaluate(device,mode):
    model,camera,parameters=fixture(device,torch.float64,principled=True)
    mat=next(p for p in model.prims if type(p)==scene.Material)
    budget=mat.parameters['budget']
    tables={key:dict(dtype=str(value.dtype),shape=list(value.shape),
                    sha256=hashlib.sha256(raw(value)).hexdigest())
            for key,value in budget._data.items() if isinstance(value,torch.Tensor)}
    def t(value):return parameters[0].new_tensor(value)
    normal=t([[0.,0.,1.]]);view=t([[0.,0.,2.]]);direction=t([[[0.,0.,1.]]])
    if mode=='direct':value=material.principled(normal,view,direction,**mat.parameters)
    elif mode=='recompute':value=_principled_recompute(normal,view,direction,mat.parameters)
    elif mode=='render':
        output=render(model,camera,outputs=('colour','normal','world_position','mask'))
        assert bool((output['mask']==1).all())
        assert raw(output['normal'])==raw(normal)
        assert raw(camera.pose[0,:3,3]-output['world_position'].reshape(1,3))==raw(view)
        value=output['colour']
    else:raise ValueError('unknown attribution mode')
    cotangent=t([.25,.5,.75]).reshape_as(value)
    names=('albedo','roughness','f0')
    operands={name:tensor_record(parameter) for name,parameter in zip(names,parameters)}
    operands.update(normal=tensor_record(normal),view=tensor_record(view),direction=tensor_record(direction))
    joint=torch.autograd.grad(value,parameters,cotangent,retain_graph=True)
    per_channel=[];roughness_parts=[]
    for channel in range(3):
        incoming=torch.zeros_like(cotangent)
        incoming.reshape(-1)[channel]=cotangent.reshape(-1)[channel]
        gradients=torch.autograd.grad(value,parameters,incoming,retain_graph=channel<2)
        roughness_parts.append(gradients[1].detach())
        per_channel.append(dict(channel=channel,cotangent=tensor_record(incoming),
            gradients={name:tensor_record(g) for name,g in zip(names,gradients)}))
    # This explicit RGB order is a diagnostic, not a claim that autograd's
    # existing shared-graph reduction is required to use this regrouping.
    ordered=(roughness_parts[0]+roughness_parts[1])+roughness_parts[2]
    return dict(mode=mode,tables=tables,operands=operands,value=tensor_record(value),
                cotangent=tensor_record(cotangent),
                joint={name:tensor_record(g) for name,g in zip(names,joint)},
                channel_vjps=per_channel,ordered_channel_roughness_sum=tensor_record(ordered))


def test_deferred_principled_matched_vjp_attribution(cuda_required,record_property):
    rows=[]
    for mode in ('direct','recompute','render'):
        cpu=evaluate('cpu',mode);cuda=evaluate('cuda',mode)
        rows.append(dict(mode=mode,cpu=cpu,cuda=cuda,
            value_equal=cpu['value']==cuda['value'],
            joint_equal={name:cpu['joint'][name]==cuda['joint'][name] for name in ('albedo','roughness','f0')},
            channel_equal=[{name:cpu['channel_vjps'][channel]['gradients'][name]==cuda['channel_vjps'][channel]['gradients'][name]
                            for name in ('albedo','roughness','f0')} for channel in range(3)]))
        # Preserve each completed mode even if a later control fails.
        record_property('principled_attribution_'+mode,json.dumps(rows[-1],sort_keys=True))
        assert cpu['tables']==cuda['tables'],'all admitted table tensors must match bytes'
        assert cpu['operands']==cuda['operands'],'fixed geometry/material operands must match bytes'
        assert cpu['cotangent']==cuda['cotangent'],'identical incoming RGB cotangent required'
    for lane in ('cpu','cuda'):
        for row in rows[1:]:
            assert row[lane]['value']['hex']==rows[0][lane]['value']['hex']
            for name in ('albedo','roughness','f0'):
                assert row[lane]['joint'][name]==rows[0][lane]['joint'][name],lane+' wrapper changes '+name+' VJP'
    # CPU/CUDA differences remain explicitly recorded, never tolerance-qualified.
