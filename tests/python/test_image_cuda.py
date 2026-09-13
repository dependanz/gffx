"""Tiny CPU/CUDA byte-identity diagnostics, never allclose qualification.

Run only under the explicit CUDA worker lease and existing host/GPU allocation caps.
No skip hides an unavailable CUDA lane; numerical tolerances cannot satisfy this gate.
"""
import math

import pytest
import torch


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('direction',['decode','encode'])
def test_transfer_preprocessing_power_and_vjp_localization(dtype,direction,record_property):
    """Separate colour preprocessing from noninteger power using identical power inputs/cotangents."""
    assert torch.cuda.is_available()
    source=torch.tensor([.1,.2,.3,.7,.8,.9,1.,1.3],dtype=dtype)
    cotangent=torch.tensor([.125,.25,.375,.5,.625,.75,.875,1.],dtype=dtype)
    exponent=2.4 if direction=='decode' else 1/2.4
    base=((source+.055)/1.055) if direction=='decode' else source.clone()
    failures=[]
    for stage in ('preprocessing','power'):
        rows=[]
        for device in ('cpu','cuda','cuda'):
            x=(source if stage=='preprocessing' else base).to(device).clone().requires_grad_()
            weights=cotangent.to(device)
            y=((x+.055)/1.055 if direction=='decode' else x*1) if stage=='preprocessing' else x.pow(exponent)
            gx,=torch.autograd.grad(y,x,grad_outputs=weights)
            rows.append((y.detach().cpu(),gx.detach().cpu()))
        for index,(cpu,gpu,repeat) in enumerate(zip(*rows)):
            a,b,c=[v.contiguous().reshape(-1).view(torch.uint8) for v in (cpu,gpu,repeat)]
            record_property(f'{stage}_{index}_max_abs',float((cpu-gpu).abs().max()))
            record_property(f'{stage}_{index}_unequal_bytes',int((a!=b).sum()))
            record_property(f'{stage}_{index}_cpu',cpu.tolist())
            record_property(f'{stage}_{index}_cuda',gpu.tolist())
            if not torch.equal(a,b): failures.append(f'{stage}/{index}')
            if not torch.equal(b,c): failures.append(f'{stage}/{index}/repeat')
    assert not failures, f'{direction}/{dtype}: {failures}; attribution only, no tolerance waiver'


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_transfer_scalar_vs_tensor_division_localization(dtype,record_property):
    """Distinguish wrapped scalar division from explicit tensor division before choosing an arithmetic repair."""
    assert torch.cuda.is_available()
    source=torch.tensor([.1,.2,.3,.7,.8,.9,1.,1.3],dtype=dtype)
    summed=source+torch.tensor(.055,dtype=dtype)
    cotangent=torch.tensor([.125,.25,.375,.5,.625,.75,.875,1.],dtype=dtype)
    denominator=torch.tensor(1.055,dtype=dtype)
    reciprocal=denominator.reciprocal()
    failures=[]
    for stage in ('addition','scalar_division','tensor_division','fixed_reciprocal'):
        rows=[]
        for device in ('cpu','cuda','cuda'):
            x=(source if stage=='addition' else summed).to(device).clone().requires_grad_()
            if stage=='addition': y=x+torch.tensor(.055,dtype=dtype).to(device)
            elif stage=='scalar_division': y=x/1.055
            elif stage=='tensor_division': y=x/denominator.to(device)
            else: y=x*reciprocal.to(device)
            gx,=torch.autograd.grad(y,x,grad_outputs=cotangent.to(device))
            rows.append((y.detach().cpu(),gx.detach().cpu()))
        for index,(cpu,gpu,repeat) in enumerate(zip(*rows)):
            a,b,c=[v.contiguous().reshape(-1).view(torch.uint8) for v in (cpu,gpu,repeat)]
            record_property(f'{stage}_{index}_max_abs',float((cpu-gpu).abs().max()))
            record_property(f'{stage}_{index}_unequal_bytes',int((a!=b).sum()))
            if not torch.equal(a,b): failures.append(f'{stage}/{index}')
            if not torch.equal(b,c): failures.append(f'{stage}/{index}/repeat')
    assert not failures, f'{dtype}: {failures}; original failing scalar stage retained as attribution'


def _evaluate(image, name, dtype, device):
    x=torch.tensor([[[.1,.2,.3,.7],[.3,.1,.2,.8]],
                    [[.2,.3,.1,.6],[.4,.2,.1,.9]]],dtype=dtype,device=device,requires_grad=True)
    p=torch.tensor(.23,dtype=dtype,device=device,requires_grad=True)
    if name=='alpha': out=image.convert_alpha(x); args=(x,)
    elif name=='mask': out=image.apply_mask(x,opacity=p); args=(x,p)
    elif name=='over': out=image.composite([x],x*.5); args=(x,)
    elif name=='reconstruct': out=image.reconstruct(x,2); args=(x,)
    elif name=='adjust': out=image.adjust(x,exposure=p,brightness=.1); args=(x,p)
    elif name=='blur': out=image.gaussian_blur(x,p+.6,radius=1); args=(x,p)
    elif name=='decode': out=image.srgb_to_linear(x); args=(x,)
    elif name=='encode': out=image.linear_to_srgb(x); args=(x,)
    else:
        translation=torch.stack((p,p*.7)); scale=torch.stack((p*0+1,p*0+1.1))
        pivot=torch.tensor([.5,.5],dtype=dtype,device=device)
        matrix=image.affine_matrix(translation,p,scale,pivot)
        if name=='affine': out=matrix; args=(p,)
        else: out=image.warp_affine(x,matrix,(2,2)); args=(x,p)
    # A parity fixture must supply identical incoming cotangents. Independently
    # dividing on CPU/CUDA produced different inputs for non-power-of-two sizes.
    weights=(torch.arange(1,out.numel()+1,dtype=dtype).reshape(out.shape)/out.numel()).to(device)
    grads=torch.autograd.grad((out*weights).sum(),args)
    return (out.detach().cpu(),*(v.detach().cpu() for v in grads))


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('name',['alpha','mask','over','reconstruct','affine','warp','adjust','blur','decode','encode'])
def test_image_cpu_cuda_reference_bytes_and_repeat(name,dtype,record_property):
    """Compare forward/backward bytes and fixed-device repeats; report mismatch diagnostics without waiving exactness."""
    from gffx.torch import image
    assert torch.cuda.is_available(), 'explicit CUDA qualification requires an available GPU'
    cpu=_evaluate(image,name,dtype,'cpu')
    gpu=_evaluate(image,name,dtype,'cuda')
    repeat=_evaluate(image,name,dtype,'cuda')
    differences=[]
    for index,(a,b,c) in enumerate(zip(cpu,gpu,repeat)):
        aa,bb,cc=(v.contiguous().reshape(-1).view(torch.uint8) for v in (a,b,c))
        record_property(f'output_{index}_max_abs',float((a-b).abs().max()))
        record_property(f'output_{index}_unequal_bytes',int((aa!=bb).sum()))
        if not torch.equal(aa,bb): differences.append(f'CPU/CUDA item {index}')
        if not torch.equal(bb,cc): differences.append(f'CUDA repeat item {index}')
    assert not differences, f'{name}/{dtype}: byte identity failed: {differences}; see numerical diagnostics, not tolerance acceptance'


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_affine_independent_argument_vjps_cpu_cuda(dtype,record_property):
    """Isolate the operation's own input cotangents from caller-side shared-parameter accumulation."""
    from gffx.torch import image
    assert torch.cuda.is_available()
    def evaluate(device):
        tensor=lambda value: torch.tensor(value,dtype=dtype,device=device,requires_grad=True)
        args=(tensor([.23,.161]),tensor(.23),tensor([1.,1.1]),tensor([.5,.5]))
        out=image.affine_matrix(*args)
        cotangent=torch.tensor([[.125,.25,.375],[.5,.625,.75]],dtype=dtype,device=device)
        gradients=torch.autograd.grad(out,args,grad_outputs=cotangent)
        return tuple(value.detach().cpu() for value in (out,*gradients))
    cpu,gpu=evaluate('cpu'),evaluate('cuda')
    differences=[]
    for name,a,b in zip(('matrix','translation','rotation','scale','pivot'),cpu,gpu):
        record_property(name+'_max_abs',float((a-b).abs().max()))
        different=int((a.contiguous().reshape(-1).view(torch.uint8)!=b.contiguous().reshape(-1).view(torch.uint8)).sum())
        record_property(name+'_unequal_bytes',different)
        if different: differences.append(name)
    assert not differences, f'independent affine VJP bytes differ for {differences}'


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_affine_matched_nondyadic_cotangent_localization(dtype,record_property):
    """Transfer identical input/cotangent bytes before comparing the public VJP and a shared caller graph."""
    from gffx.torch import image
    assert torch.cuda.is_available()
    def arguments(p):
        return (torch.stack((p,p*.7)),p,torch.stack((p*0+1,p*0+1.1)),
                torch.tensor([.5,.5],dtype=dtype,device=p.device))
    host_p=torch.tensor(.23,dtype=dtype)
    host_args=tuple(v.detach() for v in arguments(host_p))
    host_cot=torch.arange(1,7,dtype=dtype).reshape(2,3)/6
    device_cot=(torch.arange(1,7,dtype=dtype,device='cuda').reshape(2,3)/6).cpu()
    def diffbytes(a,b):
        return int((a.contiguous().reshape(-1).view(torch.uint8)!=b.contiguous().reshape(-1).view(torch.uint8)).sum())
    record_property('independently_generated_original_cotangent_unequal_bytes',diffbytes(host_cot,device_cot))
    record_property('independently_generated_original_cotangent_max_abs',float((host_cot-device_cot).abs().max()))
    device_args=tuple(v.detach().cpu() for v in arguments(host_p.to('cuda')))
    for label,a,b in zip(('translation','rotation','scale','pivot'),host_args,device_args):
        record_property('original_'+label+'_input_unequal_bytes',diffbytes(a,b))
    def independent(device):
        args=tuple(value.to(device).clone().requires_grad_() for value in host_args)
        result=image.affine_matrix(*args)
        return tuple(v.detach().cpu() for v in (result,*torch.autograd.grad(result,args,grad_outputs=host_cot.to(device))))
    cpu,gpu=independent('cpu'),independent('cuda')
    differences=[]
    for label,a,b in zip(('matrix','translation','rotation','scale','pivot'),cpu,gpu):
        count=diffbytes(a,b)
        record_property('matched_'+label+'_unequal_bytes',count)
        record_property('matched_'+label+'_max_abs',float((a-b).abs().max()))
        if count: differences.append(label)
    def shared(device):
        p=host_p.to(device).clone().requires_grad_()
        result=image.affine_matrix(*arguments(p))
        return torch.autograd.grad(result,p,grad_outputs=host_cot.to(device))[0].detach().cpu()
    a,b=shared('cpu'),shared('cuda')
    count=diffbytes(a,b)
    record_property('matched_shared_caller_unequal_bytes',count)
    record_property('matched_shared_caller_max_abs',float((a-b).abs().max()))
    if count: differences.append('matched shared caller')
    assert not differences, f'matched input/cotangent bytes still differ for {differences}'


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_warp_independent_matrix_vjp_localization(dtype,record_property):
    """Use identical leaf matrix/image and incoming derivative bytes to locate warp's own VJP differences."""
    from gffx.torch import image
    assert torch.cuda.is_available()
    host_image=torch.tensor([[[.1,.2,.3,.7],[.3,.1,.2,.8]],
                             [[.2,.3,.1,.6],[.4,.2,.1,.9]]],dtype=dtype)
    p=torch.tensor(.23,dtype=dtype)
    host_matrix=image.affine_matrix(torch.stack((p,p*.7)),p,torch.stack((p*0+1,p*0+1.1)),
                                   torch.tensor([.5,.5],dtype=dtype)).detach()
    cotangent=torch.arange(1,17,dtype=dtype).reshape(2,2,4)/16
    def evaluate(device):
        x=host_image.to(device).clone().requires_grad_()
        matrix=host_matrix.to(device).clone().requires_grad_()
        out=image.warp_affine(x,matrix,(2,2))
        gradients=torch.autograd.grad(out,(x,matrix),grad_outputs=cotangent.to(device))
        return tuple(v.detach().cpu() for v in (out,*gradients))
    cpu,gpu=evaluate('cpu'),evaluate('cuda')
    differences=[]
    for name,a,b in zip(('pixels','source_gradient','matrix_gradient'),cpu,gpu):
        count=int((a.contiguous().reshape(-1).view(torch.uint8)!=b.contiguous().reshape(-1).view(torch.uint8)).sum())
        record_property(name+'_unequal_bytes',count)
        record_property(name+'_max_abs',float((a-b).abs().max()))
        if count: differences.append(name)
    record_property('cpu_matrix_vjp',str(cpu[2].tolist()))
    record_property('cuda_matrix_vjp',str(gpu[2].tolist()))
    assert not differences, f'warp independent VJP bytes differ for {differences}'


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_warp_sampler_and_inverse_grid_vjp_localization(dtype,record_property):
    """Run this node alone in a fresh worker: its first CUDA backward must establish its own context."""
    _warp_component_vjps(dtype,record_property,prime_context=False)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_warp_context_primed_attribution_only(dtype,record_property):
    """An explicit identity kernel diagnoses arithmetic after the separately failing context boundary."""
    _warp_component_vjps(dtype,record_property,prime_context=True)


def _warp_component_vjps(dtype,record_property,*,prime_context):
    from gffx.torch import image,render
    assert torch.cuda.is_available()
    record_property('context_priming_identity_kernel',prime_context)
    record_property('scope','attribution only; does not qualify the direct-call context boundary' if prime_context else 'direct-call context regression and arithmetic attribution')
    host_image=torch.tensor([[[.1,.2,.3,.7],[.3,.1,.2,.8]],
                             [[.2,.3,.1,.6],[.4,.2,.1,.9]]],dtype=dtype)
    p=torch.tensor(.23,dtype=dtype)
    host_matrix=image.affine_matrix(torch.stack((p,p*.7)),p,torch.stack((p*0+1,p*0+1.1)),
                                   torch.tensor([.5,.5],dtype=dtype)).detach()
    def grid(matrix):
        yy,xx=torch.meshgrid(torch.arange(2,dtype=dtype,device=matrix.device),
                            torch.arange(2,dtype=dtype,device=matrix.device),indexing='ij')
        a,b,tx=matrix[0].unbind(); d,e,ty=matrix[1].unbind()
        det=a*e-b*d; qx,qy=xx-tx,yy-ty
        return torch.stack((((e*qx-b*qy)/det+.5)/2,((a*qy-d*qx)/det+.5)/2),-1).reshape(-1,2)
    host_uv=grid(host_matrix).detach()
    host_cot=torch.arange(1,17,dtype=dtype).reshape(4,4)/16
    def sample(device):
        pixels=host_image.to(device).clone().requires_grad_()
        uv=host_uv.to(device).clone().requires_grad_()
        out=render.texture(pixels.reshape(-1),torch.tensor([0,16],dtype=torch.int32,device=device),
                           2,2,uv,channel_count=4,wrap_u=render.WRAP_BORDER,wrap_v=render.WRAP_BORDER,
                           border=torch.zeros(4,dtype=dtype,device=device))
        if prime_context:
            # Multiplication backward runs a framework CUDA kernel before the native backward.
            # This is deliberately NOT a production repair or direct-call qualification.
            out=out*1
        grads=torch.autograd.grad(out,(pixels,uv),grad_outputs=host_cot.to(device))
        return tuple(v.detach().cpu() for v in (out,*grads))
    cpu,gpu=sample('cpu'),sample('cuda')
    def count(a,b):
        return int((a.contiguous().reshape(-1).view(torch.uint8)!=b.contiguous().reshape(-1).view(torch.uint8)).sum())
    differences=[]
    for name,a,b in zip(('native_pixels','native_source_vjp','native_coordinate_vjp'),cpu,gpu):
        n=count(a,b);record_property(name+'_unequal_bytes',n)
        record_property(name+'_max_abs',float((a-b).abs().max()))
        if n: differences.append(name)
    record_property('inverse_grid_forward_unequal_bytes',count(host_uv,grid(host_matrix.to('cuda')).cpu()))
    def grid_vjp(device):
        matrix=host_matrix.to(device).clone().requires_grad_()
        # Exercise the production inverse-grid VJP after its fixed-tree repair, while
        # retaining the independent forward formula above as the matching input oracle.
        yy,xx=torch.meshgrid(torch.arange(2,dtype=dtype,device=device),
                            torch.arange(2,dtype=dtype,device=device),indexing='ij')
        operation=getattr(image,'_InverseGrid',None)
        result=grid(matrix) if operation is None else operation.apply(matrix,xx,yy,2,2)
        return torch.autograd.grad(result,matrix,grad_outputs=cpu[2].to(device))[0].detach().cpu()
    a,b=grid_vjp('cpu'),grid_vjp('cuda')
    n=count(a,b);record_property('inverse_grid_vjp_unequal_bytes',n)
    record_property('inverse_grid_vjp_max_abs',float((a-b).abs().max()))
    record_property('inverse_grid_cpu_vjp',str(a.tolist()));record_property('inverse_grid_cuda_vjp',str(b.tolist()))
    if n: differences.append('inverse_grid_vjp')
    assert not differences, f'warp component VJP bytes differ for {differences}'


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_texture_nondefault_stream_preserves_caller_and_exact_vjps(dtype,record_property):
    """A native device guard must preserve the caller's stream and ordered texture work."""
    from gffx.torch import render
    assert torch.cuda.is_available()
    host_pixels=torch.tensor([.1,.2,.3,.7,.3,.1,.2,.8,.2,.3,.1,.6,.4,.2,.1,.9],dtype=dtype)
    host_uv=torch.tensor([[.15,.2],[.7,.25],[.35,.75],[.8,.8]],dtype=dtype)
    cotangent=torch.arange(1,17,dtype=dtype).reshape(4,4)/16
    def evaluate(device):
        # Clone and arithmetic queue producer work on the same chosen stream as sampling.
        pixels=(host_pixels.to(device).clone()*1).requires_grad_()
        uv=host_uv.to(device).clone().requires_grad_()
        out=render.texture(pixels,torch.tensor([0,16],dtype=torch.int32,device=device),2,2,uv,
                           channel_count=4,wrap_u=render.WRAP_BORDER,wrap_v=render.WRAP_BORDER,
                           border=torch.zeros(4,dtype=dtype,device=device))
        gradients=torch.autograd.grad(out,(pixels,uv),grad_outputs=cotangent.to(device))
        return out,*gradients
    cpu=tuple(value.detach() for value in evaluate('cpu'))
    previous=torch.cuda.current_stream()
    stream=torch.cuda.Stream()
    assert stream.cuda_stream!=torch.cuda.default_stream().cuda_stream
    with torch.cuda.stream(stream):
        gpu=evaluate('cuda')
        assert torch.cuda.current_stream()==stream
    assert torch.cuda.current_stream()==previous
    stream.synchronize()
    differences=[]
    for label,a,value in zip(('pixels','source_vjp','coordinate_vjp'),cpu,gpu):
        b=value.detach().cpu()
        count=int((a.contiguous().reshape(-1).view(torch.uint8)!=b.contiguous().reshape(-1).view(torch.uint8)).sum())
        record_property(label+'_unequal_bytes',count)
        if count:differences.append(label)
    record_property('scope','one device, one nondefault stream; not a multi-device or concurrency stress qualification')
    assert not differences,f'nondefault-stream texture bytes differ: {differences}'


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('shape',[(1,5),(3,5)])
@pytest.mark.parametrize('batched',[False,True])
def test_warp_odd_grid_and_batch_cpu_cuda_exact(dtype,shape,batched,record_property):
    """Fixed reduction ordering must cover odd pixel counts and independent batch members."""
    from gffx.torch import image
    source=torch.tensor([[[.03*(1+x),.04*(1+y),.02*(1+x+y),.9]
                          for x in range(4)] for y in range(3)],dtype=dtype)
    matrix=torch.tensor([[.9,.14,.23],[.07,1.13,-.19]],dtype=dtype)
    if batched:
        source=torch.stack((source,source*.7))
        matrix=torch.stack((matrix,torch.tensor([[-1.,.1,2.65],[.02,.93,.17]],dtype=dtype)))
    cot_shape=(2,*shape,4) if batched else (*shape,4)
    cot=torch.arange(1,1+math.prod(cot_shape),dtype=dtype).reshape(cot_shape)/64
    def evaluate(device):
        pixels=source.to(device).clone().requires_grad_()
        transform=matrix.to(device).clone().requires_grad_()
        out=image.warp_affine(pixels,transform,shape)
        gradients=torch.autograd.grad(out,(pixels,transform),grad_outputs=cot.to(device))
        return tuple(value.detach().cpu() for value in (out,*gradients))
    cpu,gpu,repeat=evaluate('cpu'),evaluate('cuda'),evaluate('cuda')
    differences=[]
    for label,a,b,c in zip(('pixels','source_vjp','matrix_vjp'),cpu,gpu,repeat):
        aa,bb,cc=(v.contiguous().reshape(-1).view(torch.uint8) for v in (a,b,c))
        count=int((aa!=bb).sum());record_property(label+'_unequal_bytes',count)
        record_property(label+'_max_abs',float((a-b).abs().max()))
        if count or not torch.equal(bb,cc):differences.append(label)
    assert not differences,f'odd/batched warp bytes differ: {differences}'


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_warp_odd_extent_normalization_and_matched_sampler_localization(dtype,record_property):
    """Separate division by an odd extent from native sampling using identical coordinate bytes."""
    from gffx.torch import image,render
    assert torch.cuda.is_available()
    host_matrix=torch.tensor([[.9,.14,.23],[.07,1.13,-.19]],dtype=dtype)
    host_pixels=torch.tensor([[[.03*(1+x),.04*(1+y),.02*(1+x+y),.9]
                               for x in range(4)] for y in range(3)],dtype=dtype)
    host_yy,host_xx=torch.meshgrid(torch.arange(1,dtype=dtype),torch.arange(5,dtype=dtype),indexing='ij')
    host_grid_cot=torch.arange(1,11,dtype=dtype).reshape(5,2)/64
    host_pixel_cot=torch.arange(1,21,dtype=dtype).reshape(5,4)/64
    def same_bits(a,b):
        return torch.equal(a.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
                           b.detach().cpu().contiguous().reshape(-1).view(torch.uint8))
    rows=[]
    for device in ('cpu','cuda','cuda'):
        matrix=host_matrix.to(device).clone().requires_grad_()
        yy,xx=host_yy.to(device),host_xx.to(device)
        cotangent=host_grid_cot.to(device)
        assert same_bits(matrix,host_matrix) and same_bits(xx,host_xx) and same_bits(yy,host_yy)
        assert same_bits(cotangent,host_grid_cot)
        a,b,tx=matrix[0].unbind();d,e,ty=matrix[1].unbind()
        det=a*e-b*d;qx,qy=xx-tx,yy-ty
        sx,sy=(e*qx-b*qy)/det,(a*qy-d*qx)/det
        xy=torch.stack((sx,sy),-1).reshape(5,2)
        legacy=torch.stack(((sx+.5)/4,(sy+.5)/3),-1).reshape(5,2)
        explicit=(xy+.5)/xy.new_tensor([4,3])
        production=image._InverseGrid.apply(matrix,xx,yy,3,4)
        vjp,=torch.autograd.grad(production,matrix,grad_outputs=cotangent)
        legacy_cot=torch.stack((cotangent[:,0]/4,cotangent[:,1]/3),-1)
        explicit_cot=cotangent/cotangent.new_tensor([4,3])
        def fixed_matrix_vjp(coordinate_cotangent):
            gu=coordinate_cotangent[:,0].reshape(sx.shape)
            gv=coordinate_cotangent[:,1].reshape(sy.shape)
            rx=(d*gv-e*gu)/det;ry=(b*gu-a*gv)/det
            return torch.stack((torch.stack((image._fixed_sum(rx*sx),image._fixed_sum(rx*sy),image._fixed_sum(rx))),
                                torch.stack((image._fixed_sum(ry*sx),image._fixed_sum(ry*sy),image._fixed_sum(ry)))))
        legacy_matrix=fixed_matrix_vjp(legacy_cot)
        explicit_matrix=fixed_matrix_vjp(explicit_cot)
        record_property(device+'_production_matches_manual_scalar_matrix_vjp',same_bits(vjp,legacy_matrix))
        rows.append(tuple(v.detach().cpu() for v in (xy,legacy,explicit,production,vjp,
                                                     legacy_cot,explicit_cot,legacy_matrix,explicit_matrix)))
    differences=[]
    def compare(label,values,required=True):
        a,b,c=values
        aa,bb,cc=(v.contiguous().reshape(-1).view(torch.uint8) for v in values)
        count=int((aa!=bb).sum());record_property(label+'_unequal_bytes',count)
        record_property(label+'_max_abs',float((a-b).abs().max()))
        record_property(label+'_repeat_unequal_bytes',int((bb!=cc).sum()))
        if required and (count or not torch.equal(bb,cc)):differences.append(label)
    labels=('inverse_xy','legacy_scalar_uv','explicit_tensor_uv','production_uv','production_matrix_vjp',
            'legacy_scalar_coordinate_cotangent','explicit_tensor_coordinate_cotangent',
            'legacy_scalar_fixed_matrix_vjp','explicit_tensor_fixed_matrix_vjp')
    for index,label in enumerate(labels):
        compare(label,[row[index] for row in rows],required=not label.startswith('legacy_scalar'))
    host_uv=rows[0][3]
    native=[]
    for device in ('cpu','cuda','cuda'):
        pixels=host_pixels.to(device).clone().requires_grad_()
        uv=host_uv.to(device).clone().requires_grad_();cotangent=host_pixel_cot.to(device)
        assert same_bits(pixels,host_pixels) and same_bits(uv,host_uv) and same_bits(cotangent,host_pixel_cot)
        output=render.texture(pixels.reshape(-1),torch.tensor([0,48],dtype=torch.int32,device=device),
                              3,4,uv,channel_count=4,wrap_u=render.WRAP_BORDER,wrap_v=render.WRAP_BORDER,
                              border=pixels.new_zeros(4))
        gradient,coordinate_gradient=torch.autograd.grad(output,(pixels,uv),grad_outputs=cotangent)
        native.append(tuple(v.detach().cpu() for v in (output,gradient,coordinate_gradient)))
    for index,label in enumerate(('matched_native_pixels','matched_native_source_vjp','matched_native_coordinate_vjp')):
        compare(label,[row[index] for row in native])
    record_property('inputs_transforms_coordinates_cotangents_transferred_identically',True)
    assert not differences,f'odd extent attribution/current API failures: {differences}'


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_blur_weight_generation_vs_spatial_reduction_localization(dtype,record_property):
    """Separate Gaussian weight derivatives from spatial broadcast/reduction derivatives."""
    from gffx.torch import image
    source=torch.tensor([[[.1,.2,.3,.7],[.3,.1,.2,.8]],
                         [[.2,.3,.1,.6],[.4,.2,.1,.9]]],dtype=dtype)
    sigma=(torch.tensor(.23,dtype=dtype)+.6).detach()
    cot=torch.arange(1,17,dtype=dtype).reshape(2,2,4)/16
    def weights(s):
        raw=[torch.exp(-.5*(i/s).square()) for i in range(-1,2)]
        total=raw[0]
        for value in raw[1:]:total=total+value
        return torch.stack([value/total for value in raw])
    host_weights=weights(sigma).detach()
    def convolution(pixels,w):
        result=pixels
        for axis in (-2,-3):
            indices=torch.arange(result.shape[axis],device=result.device)
            value=torch.zeros_like(result)
            for offset,weight in zip(range(-1,2),w.unbind()):
                raw=indices+offset
                shifted=torch.index_select(result,axis,raw.clamp(0,result.shape[axis]-1))
                shape=[1]*result.ndim;shape[axis]=len(indices)
                shifted=shifted*((raw>=0)&(raw<result.shape[axis])).reshape(shape)
                multiply=getattr(image,'_WeightedMultiply',None)
                value=value+(shifted*weight if multiply is None else multiply.apply(shifted,weight))
            result=value
        return result
    def evaluate(device):
        pixels=source.to(device).clone().requires_grad_()
        w=host_weights.to(device).clone().requires_grad_()
        out=convolution(pixels,w)
        grads=torch.autograd.grad(out,(pixels,w),grad_outputs=cot.to(device))
        return tuple(value.detach().cpu() for value in (out,*grads))
    cpu,gpu=evaluate('cpu'),evaluate('cuda')
    def count(a,b):return int((a.contiguous().reshape(-1).view(torch.uint8)!=b.contiguous().reshape(-1).view(torch.uint8)).sum())
    differences=[]
    for label,a,b in zip(('convolution_pixels','convolution_source_vjp','convolution_weight_vjp'),cpu,gpu):
        n=count(a,b);record_property(label+'_unequal_bytes',n)
        record_property(label+'_max_abs',float((a-b).abs().max()))
        if n:differences.append(label)
    record_property('cpu_weight_vjp',str(cpu[2].tolist()))
    record_property('cuda_weight_vjp',str(gpu[2].tolist()))
    def weights_vjp(device):
        s=sigma.to(device).clone().requires_grad_()
        value=weights(s)
        grad=torch.autograd.grad(value,s,grad_outputs=cpu[2].to(device))[0]
        return value.detach().cpu(),grad.detach().cpu()
    a,b=weights_vjp('cpu'),weights_vjp('cuda')
    for label,left,right in zip(('gaussian_weights','matched_gaussian_sigma_vjp'),a,b):
        n=count(left,right);record_property(label+'_unequal_bytes',n)
        record_property(label+'_max_abs',float((left-right).abs().max()))
        if n:differences.append(label)
    assert not differences,f'blur component byte differences: {differences}'


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('per_batch',[False,True])
@pytest.mark.parametrize('border',['transparent','clamp'])
def test_blur_odd_batch_cpu_cuda_exact(dtype,per_batch,border,record_property):
    """Ordered weight reductions must cover odd images and shared or independent batch sigmas."""
    from gffx.torch import image
    source=torch.arange(120,dtype=dtype).reshape(2,3,5,4)/256
    source[...,3]=.8
    sigma=torch.tensor([.83,1.17] if per_batch else .83,dtype=dtype)
    cot=torch.arange(1,121,dtype=dtype).reshape(source.shape)/128
    def evaluate(device):
        pixels=source.to(device).clone().requires_grad_();s=sigma.to(device).clone().requires_grad_()
        out=image.gaussian_blur(pixels,s,radius=1,border=border)
        grads=torch.autograd.grad(out,(pixels,s),grad_outputs=cot.to(device))
        return tuple(value.detach().cpu() for value in (out,*grads))
    cpu,gpu,repeat=evaluate('cpu'),evaluate('cuda'),evaluate('cuda')
    differences=[]
    for label,a,b,c in zip(('pixels','source_vjp','sigma_vjp'),cpu,gpu,repeat):
        aa,bb,cc=(v.contiguous().reshape(-1).view(torch.uint8) for v in (a,b,c))
        n=int((aa!=bb).sum());record_property(label+'_unequal_bytes',n)
        record_property(label+'_max_abs',float((a-b).abs().max()))
        if n or not torch.equal(bb,cc):differences.append(label)
    assert not differences,f'odd/batched blur bytes differ: {differences}'


def test_blur_exact_sigma_staged_operand_localization(record_property):
    """Localize the actual .83f32 blur using matched operands at every arithmetic stage."""
    from gffx.torch import image
    dtype = torch.float32
    host_sigma = torch.tensor(.83, dtype=dtype)
    old_sigma = torch.tensor(.23, dtype=dtype)+.6
    def bits(value):
        return value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    record_property('exact_sigma_bits', str(bits(host_sigma).tolist()))
    record_property('historical_sigma_bits', str(bits(old_sigma).tolist()))
    record_property('historical_sigma_matches', torch.equal(bits(host_sigma), bits(old_sigma)))
    failures = []
    def compare(label, values):
        a,b,c = values
        count = int((bits(a)!=bits(b)).sum())
        repeat = int((bits(b)!=bits(c)).sum())
        record_property(label+'_unequal_bytes', count)
        record_property(label+'_repeat_unequal_bytes', repeat)
        record_property(label+'_max_abs', float((a-b).abs().max()))
        if count or repeat: failures.append(label)
    def staged(label, host, operation, cotangent):
        results = []
        for device in ('cpu','cuda','cuda'):
            value = host.to(device).clone().requires_grad_()
            cot = cotangent.to(device)
            assert torch.equal(bits(value), bits(host))
            assert torch.equal(bits(cot), bits(cotangent))
            out = operation(value)
            gradient, = torch.autograd.grad(out, value, grad_outputs=cot)
            results.append((out.detach().cpu(), gradient.detach().cpu()))
        compare(label+'_value', [v[0] for v in results])
        compare(label+'_vjp', [v[1] for v in results])
        return results[0][0]
    cot3 = torch.tensor([.125,.25,.375], dtype=dtype)
    def divide(s):
        return torch.stack([i/s for i in (-1,0,1)])
    host_ratio = staged('tap_division', host_sigma, divide, cot3)
    def exponent(q):
        return torch.stack([-.5*v.square() for v in q.unbind()])
    def exponential(q):
        return torch.stack([torch.exp(v) for v in q.unbind()])
    host_exponent = staged('exponent', host_ratio, exponent, cot3)
    host_raw = staged('exp', host_exponent, exponential, cot3)
    def normalize(raw):
        total = raw[0]+raw[1]+raw[2]
        return torch.stack([v/total for v in raw.unbind()])
    host_weights = staged('normalization', host_raw, normalize, cot3)
    source = torch.arange(120,dtype=dtype).reshape(2,3,5,4)/256
    source[...,3] = .8
    cot = torch.arange(1,121,dtype=dtype).reshape(source.shape)/128
    def convolution(pixels, weights):
        result = pixels
        for axis in (-2,-3):
            indices = torch.arange(result.shape[axis],device=result.device)
            value = torch.zeros_like(result)
            for offset, weight in zip(range(-1,2),weights.unbind()):
                raw = indices+offset
                shifted = torch.index_select(result,axis,raw.clamp(0,result.shape[axis]-1))
                shape = [1]*result.ndim; shape[axis] = len(indices)
                shifted = shifted*((raw>=0)&(raw<result.shape[axis])).reshape(shape)
                value = value+image._WeightedMultiply.apply(shifted,weight)
            result = value
        return result
    matched = []
    actual = []
    for device in ('cpu','cuda','cuda'):
        pixels = source.to(device).clone().requires_grad_()
        weights = host_weights.to(device).clone().requires_grad_()
        incoming = cot.to(device)
        assert torch.equal(bits(pixels),bits(source)) and torch.equal(bits(weights),bits(host_weights))
        assert torch.equal(bits(incoming),bits(cot))
        out = convolution(pixels,weights)
        gradients = torch.autograd.grad(out,(pixels,weights),grad_outputs=incoming)
        matched.append(tuple(v.detach().cpu() for v in (out,*gradients)))
        pixels = source.to(device).clone().requires_grad_()
        sigma = host_sigma.to(device).clone().requires_grad_()
        out = image.gaussian_blur(pixels,sigma,radius=1,border='transparent')
        gradients = torch.autograd.grad(out,(pixels,sigma),grad_outputs=incoming)
        actual.append(tuple(v.detach().cpu() for v in (out,*gradients)))
    for index, label in enumerate(('pixels','source_vjp','weight_vjp')):
        compare('matched_convolution_'+label,[v[index] for v in matched])
    for index, label in enumerate(('pixels','source_vjp','sigma_vjp')):
        compare('production_'+label,[v[index] for v in actual])
    def full_weights(s):
        return normalize(exponential(exponent(divide(s))))
    staged('weights_matched_cotangent',host_sigma,full_weights,matched[0][2])
    record_property('matched_operand_and_cotangent_bits_checked',True)
    assert not failures, f'exact sigma stage differences: {failures}'


def _compare_device_rows(cpu,gpu,repeat,labels,record_property):
    differences=[]
    for label,a,b,c in zip(labels,cpu,gpu,repeat):
        aa,bb,cc=(v.contiguous().reshape(-1).view(torch.uint8) for v in (a,b,c))
        n=int((aa!=bb).sum());record_property(label+'_unequal_bytes',n)
        record_property(label+'_max_abs',float((a-b).abs().max()))
        if n or not torch.equal(bb,cc):differences.append(label)
    assert not differences,f'CPU/CUDA or CUDA-repeat byte differences: {differences}'


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('per_batch',[False,True])
@pytest.mark.parametrize('operation',['mask','adjust'])
def test_odd_batch_mask_and_adjust_shared_parameter_vjps(dtype,per_batch,operation,record_property):
    """Shared/per-image scalar reductions and per-pixel mask derivatives require their own evidence."""
    from gffx.torch import image
    source=torch.arange(120,dtype=dtype).reshape(2,3,5,4)/256
    source[...,3]=.8
    mask=(.2+torch.arange(30,dtype=dtype)/64).reshape(2,3,5,1)
    opacity=torch.tensor([.63,.41] if per_batch else .63,dtype=dtype)
    exposure=torch.tensor([.21,-.13] if per_batch else .21,dtype=dtype)
    brightness=torch.tensor([.017,-.023] if per_batch else .017,dtype=dtype)
    cot=torch.arange(1,121,dtype=dtype).reshape(source.shape)/128
    def evaluate(device):
        if operation=='mask':
            args=tuple(v.to(device).clone().requires_grad_() for v in (source,mask,opacity))
            out=image.apply_mask(args[0],args[1],opacity=args[2])
        else:
            args=tuple(v.to(device).clone().requires_grad_() for v in (source,exposure,brightness))
            out=image.adjust(args[0],exposure=args[1],brightness=args[2])
        grads=torch.autograd.grad(out,args,grad_outputs=cot.to(device))
        return tuple(v.detach().cpu() for v in (out,*grads))
    labels=('pixels','source_vjp','mask_vjp','opacity_vjp') if operation=='mask' else ('pixels','source_vjp','exposure_vjp','brightness_vjp')
    _compare_device_rows(evaluate('cpu'),evaluate('cuda'),evaluate('cuda'),labels,record_property)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('alpha',['straight','premultiplied'])
def test_odd_batch_multilayer_independent_leaf_vjps(dtype,alpha,record_property):
    """Three independent layers must preserve exact colour/alpha derivatives across representations."""
    from gffx.torch import image
    source=torch.arange(120,dtype=dtype).reshape(2,3,5,4)/256
    source[...,3]=.8;source[0,0,0,3]=0
    background=source.clone();background[...,3]=1
    layers=(source*.73,source*.41,source*.29)
    cot=torch.arange(1,121,dtype=dtype).reshape(source.shape)/128
    def evaluate(device):
        args=tuple(v.to(device).clone().requires_grad_() for v in (*layers,background))
        out=image.composite(args[:-1],args[-1],alpha=alpha)
        grads=torch.autograd.grad(out,args,grad_outputs=cot.to(device))
        return tuple(v.detach().cpu() for v in (out,*grads))
    _compare_device_rows(evaluate('cpu'),evaluate('cuda'),evaluate('cuda'),
                         ('pixels','bottom_vjp','middle_vjp','top_vjp','background_vjp'),record_property)
