"""Installed native Gaussian operations; independent accuracy and device gates."""
import pytest
import torch

@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_native_gaussian_registered_radius_zero(dtype):
    import gffx.torch.image
    sigma=torch.tensor(.83,dtype=dtype)
    actual=torch.ops.gffx.image_gaussian_weights(sigma,0)
    assert actual.shape==(1,) and actual.dtype==dtype and actual.item()==1.
    gradient=torch.ops.gffx.image_gaussian_weights_backward(sigma,torch.tensor([-3.],dtype=dtype),0)
    assert gradient.shape==() and gradient.item()==0. and not torch.signbit(gradient)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_native_gaussian_matches_independent_weights_and_gradient(dtype):
    import gffx.torch.image as image
    from decimal import Decimal
    from tools.gaussian_weights_oracle import bits_from_float,reference,decode_bits
    name='float32' if dtype==torch.float32 else 'float64'
    wb=Decimal(512)*Decimal(2)**-53+Decimal(2)**-990
    db=Decimal(8192)*Decimal(2)**-53+Decimal(2)**-990
    if dtype==torch.float32:wb+=Decimal(2)**-24;db+=Decimal(2)**-21
    for radius in (0,1,2,7,32):
        sigma=torch.tensor([.1,.83,32.],dtype=dtype)
        actual=torch.ops.gffx.image_gaussian_weights(sigma,radius)
        assert actual.shape==(3,2*radius+1)
        for batch in range(3):
            row=reference(bits_from_float(sigma[batch].item(),name),name,radius,precision=120)
            for tap in range(2*radius+1):
                got=decode_bits(bits_from_float(actual[batch,tap].item(),name),name)
                assert abs(got-row['weights'][tap])<=wb
                cot=torch.zeros_like(actual);cot[batch,tap]=1.
                derivative=torch.ops.gffx.image_gaussian_weights_backward(sigma,cot,radius)
                expected=row['derivatives'][tap]
                assert abs(decode_bits(bits_from_float(derivative[batch].item(),name),name)-expected)<=db
                assert all(derivative[j].item()==0. for j in range(3) if j!=batch)
    value=torch.tensor(.83,dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(lambda x:image._GaussianWeights.apply(x,2),(value,),eps=1e-6,atol=2e-5,rtol=2e-4)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_native_gaussian_empty_domains_and_first_order(dtype):
    import gffx.torch.image as image
    sigma=torch.empty(0,dtype=dtype,requires_grad=True)
    out=image._GaussianWeights.apply(sigma,2)
    assert out.shape==(0,5)
    assert torch.autograd.grad(out.sum(),sigma)[0].shape==(0,)
    for bad in (.01,33.,float('nan'),float('inf')):
        with pytest.raises(RuntimeError,match='Gaussian sigma'):
            torch.ops.gffx.image_gaussian_weights(torch.tensor(bad,dtype=dtype),2)
    with pytest.raises(RuntimeError):torch.ops.gffx.image_gaussian_weights(torch.tensor(.83,dtype=dtype),33)
    with pytest.raises(RuntimeError):torch.ops.gffx.image_gaussian_weights_backward(torch.tensor(.83,dtype=dtype),torch.ones(3,dtype=dtype),2)
    x=torch.tensor(.83,dtype=dtype,requires_grad=True);cot=torch.ones(5,dtype=dtype,requires_grad=True)
    y=image._GaussianWeights.apply(x,2);first=torch.autograd.grad(y,x,grad_outputs=cot,create_graph=True)[0]
    with pytest.raises(RuntimeError):torch.autograd.grad(first,cot)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_native_gaussian_cuda_exact_fresh_graphs_and_stream(dtype):
    import gffx.torch.image as image
    if not torch.cuda.is_available():pytest.fail('explicit Gaussian CUDA qualification requires a device')
    def raw(x):
        y=torch.empty(x.numel(),dtype=x.dtype);y.copy_(x.detach().cpu().reshape(-1));return y.view(torch.uint8)
    stream=torch.cuda.Stream()
    for radius in (0,1,2,7,32):
        for shape in ((),(3,),(0,)):
            host=torch.tensor(.83,dtype=dtype) if shape==() else torch.tensor([.1,.83,32.],dtype=dtype) if shape==(3,) else torch.empty(0,dtype=dtype)
            cot=torch.arange(host.numel()*(2*radius+1),dtype=dtype).reshape((*shape,2*radius+1))*.125-.75
            for repeat in range(2):
                cpu=host.clone().requires_grad_();expected=image._GaussianWeights.apply(cpu,radius)
                eg=torch.autograd.grad(expected,cpu,grad_outputs=cot)[0]
                with torch.cuda.stream(stream):
                    device=host.to('cuda').requires_grad_();cg=cot.to('cuda')
                    actual=image._GaussianWeights.apply(device,radius)
                    ag=torch.autograd.grad(actual,device,grad_outputs=cg)[0]
                stream.synchronize()
                assert actual.shape==expected.shape and ag.shape==eg.shape
                assert torch.equal(raw(actual),raw(expected));assert torch.equal(raw(ag),raw(eg))


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_native_gaussian_cuda_all_radii_and_transition_bytes(dtype,record_property):
    import json,random
    import gffx.torch.image
    from tools.gaussian_weights_oracle import bits_from_float,decode_bits,reference,round_bits
    name='float32' if dtype==torch.float32 else 'float64'
    rng=random.Random(913)
    inputs={bits_from_float(.1,name),bits_from_float(.1,name)+1,bits_from_float(32.,name)-1,bits_from_float(32.,name),bits_from_float(.83,name),bits_from_float(.83,name)+1}
    inputs.update(bits_from_float(.1*320**(i/16),name) for i in range(1,16))
    inputs.update(bits_from_float(.1+31.9*rng.random(),name) for _ in range(12))
    if dtype==torch.float64:
        # Preserve the independently attributed minimum-subnormal disagreements.
        for center in (4592130761096662846,4596590174398369896,4601097691735874510):inputs.update(range(center-2,center+3))
    else:inputs.add(bits_from_float(.138,name))
    sigma=torch.tensor([float(decode_bits(b,name)) for b in sorted(inputs)],dtype=dtype)
    def raw(x):
        v=torch.empty(x.numel(),dtype=x.dtype);v.copy_(x.detach().cpu().reshape(-1));return v.view(torch.uint8)
    cases=0
    for radius in range(33):
        cot=(torch.arange(sigma.numel()*(2*radius+1),dtype=dtype).reshape(sigma.numel(),2*radius+1)%17-8)*.125
        cpu=torch.ops.gffx.image_gaussian_weights(sigma,radius)
        cg=torch.ops.gffx.image_gaussian_weights_backward(sigma,cot,radius)
        device=sigma.cuda();dc=cot.cuda();gpu=torch.ops.gffx.image_gaussian_weights(device,radius)
        gg=torch.ops.gffx.image_gaussian_weights_backward(device,dc,radius)
        assert torch.equal(raw(cpu),raw(gpu)) and torch.equal(raw(cg),raw(gg))
        cases+=sigma.numel()
    # Independent underflow check through the actual native backward operation.
    width,radius=(.138,2) if dtype==torch.float32 else (.1033,4)
    width=torch.tensor(width,dtype=dtype);row=reference(bits_from_float(width.item(),name),name,radius,precision=180)
    g=torch.zeros(2*radius+1,dtype=dtype,device='cuda');g[0]=1.
    w=torch.ops.gffx.image_gaussian_weights(width.cuda(),radius)
    d=torch.ops.gffx.image_gaussian_weights_backward(width.cuda(),g,radius)
    assert bits_from_float(w[0].item(),name)==round_bits(row['weights'][0],name)==0
    assert bits_from_float(d.item(),name)==round_bits(row['derivatives'][0],name)!=0
    record_property('gaussian_cuda_domain',json.dumps(dict(dtype=name,cases=cases,sigma_count=sigma.numel(),all_radii=33)))
