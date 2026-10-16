"""Independent image contracts: tiny deterministic fixtures, not artistic acceptance.

Every test explains the question and oracle in its docstring. Central differences avoid
sampling knots and branch singularities; dedicated boundary tests assert their conventions.
"""
import importlib
import math

import pytest
import torch


@pytest.fixture
def image():
    return importlib.import_module('gffx.torch.image')


def rgba(values, dtype=torch.float64):
    return torch.tensor(values, dtype=dtype).reshape(1, 1, 4)


def close(actual, expected):
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)


def test_public_surface_and_facade(image):
    """Every promised operation is available through the same lazy public facade."""
    facade = importlib.import_module('gffx.image')
    names = ('convert_alpha', 'srgb_to_linear', 'linear_to_srgb', 'apply_mask',
             'composite', 'reconstruct', 'affine_matrix', 'warp_affine', 'adjust',
             'gaussian_blur', 'capabilities')
    for name in names:
        assert callable(getattr(image, name))
        assert callable(getattr(facade, name))
    x = rgba([.2, .3, .4, .5])
    assert torch.equal(facade.convert_alpha(x), image.convert_alpha(x))


@pytest.mark.parametrize('shape',[(1,5),(3,5)])
@pytest.mark.parametrize('border',['transparent','clamp'])
@pytest.mark.parametrize('reflection',[False,True])
def test_warp_matrix_vjp_odd_images_matches_independent_scalar_difference(image,shape,border,reflection):
    """Odd reductions, shear and reflection must retain six correct matrix derivatives."""
    values=[[[.03*(1+x),.04*(1+y),.02*(1+x+y),.9] for x in range(4)] for y in range(3)]
    source=torch.tensor(values,dtype=torch.float64)
    rows=[[-1.,.1,2.65],[.02,.93,.17]] if reflection else [[.9,.14,.23],[.07,1.13,-.19]]
    matrix=torch.tensor(rows,dtype=source.dtype,requires_grad=True)
    cot=(torch.arange(shape[0]*shape[1]*4,dtype=source.dtype)+1).reshape(*shape,4)/64
    output=image.warp_affine(source,matrix,shape,border=border)
    actual=torch.autograd.grad(output,matrix,grad_outputs=cot)[0]
    weights=cot.tolist()
    def objective(m):
        (a,b,tx),(d,e,ty)=m; det=a*e-b*d; total=0.
        for y in range(shape[0]):
            for x in range(shape[1]):
                sx=(e*(x-tx)-b*(y-ty))/det; sy=(a*(y-ty)-d*(x-tx))/det
                ix,iy=math.floor(sx),math.floor(sy); fx,fy=sx-ix,sy-iy
                for oy,wy in ((0,1-fy),(1,fy)):
                    for ox,wx in ((0,1-fx),(1,fx)):
                        px,py=ix+ox,iy+oy
                        if border=='clamp':px,py=min(3,max(0,px)),min(2,max(0,py))
                        if 0<=px<4 and 0<=py<3:
                            total+=sum(values[py][px][c]*weights[y][x][c] for c in range(4))*wx*wy
        return total
    expected=torch.empty_like(matrix)
    for row in range(2):
        for col in range(3):
            plus=[v[:] for v in rows]; minus=[v[:] for v in rows]
            plus[row][col]+=1e-6;minus[row][col]-=1e-6
            expected[row,col]=(objective(plus)-objective(minus))/2e-6
    torch.testing.assert_close(actual,expected,atol=2e-5,rtol=2e-4)


@pytest.mark.parametrize('extents', [(3, 5), (1, 7)])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_inverse_grid_odd_extent_values_and_vjp_independent(image, extents, dtype):
    """Odd and singleton extents preserve inverse-map values and all six derivatives."""
    height, width = extents
    rows = [[.9, .14, .23], [.07, 1.13, -.19]]
    matrix = torch.tensor(rows, dtype=dtype, requires_grad=True)
    yy, xx = torch.meshgrid(torch.arange(2, dtype=dtype),
                            torch.arange(3, dtype=dtype), indexing='ij')
    cot = torch.arange(1, 13, dtype=dtype).reshape(6, 2)/64
    actual = image._InverseGrid.apply(matrix, xx, yy, height, width)
    derivative = torch.autograd.grad(actual, matrix, grad_outputs=cot)[0]

    def reference(m):
        (a, b, tx), (d, e, ty) = m
        det = a*e-b*d
        return [[((e*(x-tx)-b*(y-ty))/det+.5)/width,
                 ((a*(y-ty)-d*(x-tx))/det+.5)/height]
                for y in range(2) for x in range(3)]

    represented = matrix.detach().tolist()
    weights = cot.tolist()
    def objective(m):
        return sum(value*weight for pair, pair_weights in zip(reference(m), weights)
                   for value, weight in zip(pair, pair_weights))
    expected_derivative = torch.empty_like(matrix)
    for row in range(2):
        for col in range(3):
            plus = [v[:] for v in represented]; minus = [v[:] for v in represented]
            plus[row][col] += 1e-6; minus[row][col] -= 1e-6
            expected_derivative[row, col] = (objective(plus)-objective(minus))/2e-6
    tolerance = 2e-6 if dtype == torch.float32 else 2e-12
    torch.testing.assert_close(actual, torch.tensor(reference(represented), dtype=dtype),
                               atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(derivative, expected_derivative, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_half_red_over_blue_closed_form(image, dtype):
    """Half red over opaque blue is half red/half blue with opaque output alpha."""
    bottom, top = rgba([0, 0, 1, 1], dtype), rgba([.5, 0, 0, .5], dtype)
    actual = image.composite([top], bottom)
    assert torch.equal(actual, rgba([.5, 0, .5, 1], dtype))
    assert torch.equal(bottom, rgba([0, 0, 1, 1], dtype))
    assert actual.data_ptr() != bottom.data_ptr()


def test_layer_order_straight_agreement_and_empty_clone(image):
    """List order is bottom-to-top; straight conversion preserves the same physical over."""
    bg = rgba([0, 0, 1, .5]); r = rgba([1, 0, 0, .5]); g = rgba([0, 1, 0, .5])
    straight = image.composite([r, g], bg, alpha='straight')
    close(straight, rgba([2/7, 4/7, 1/7, 7/8]))
    pm = [image.convert_alpha(v) for v in (bg, r, g)]
    close(image.convert_alpha(straight), image.composite(pm[1:], pm[0]))
    empty = image.composite([], pm[0])
    assert torch.equal(empty, pm[0]) and empty.data_ptr() != pm[0].data_ptr()


def test_zero_alpha_conversion_has_finite_defined_derivatives(image):
    """Hidden colour at zero alpha is removed; safe unpremultiply has no NaN backward."""
    x = rgba([1, -2, 3, 0]).requires_grad_()
    y = image.convert_alpha(x, source='premultiplied', target='straight')
    assert torch.equal(y, torch.zeros_like(y))
    y.sum().backward()
    assert torch.equal(x.grad, rgba([0, 0, 0, 1]))


def test_mask_opacity_existing_alpha_multiply(image):
    """Mask, layer opacity and existing alpha are multiplicative, not alternatives."""
    x = rgba([.2, .1, .3, .5])
    mask = torch.full((1, 1, 1), .5, dtype=x.dtype)
    assert torch.equal(image.apply_mask(x, mask, opacity=.5), x*.25)
    assert torch.equal(image.apply_mask(x, mask*0), torch.zeros_like(x))


@pytest.mark.parametrize('factor', [1, 2, 3])
def test_reconstruction_independent_block_sums_and_backward(image, factor):
    """Box reconstruction agrees with Python scalar loops, with exactly uniform cotangents."""
    x = torch.arange(2*factor*3*factor*2, dtype=torch.float64).reshape(2*factor, 3*factor, 2)
    x.requires_grad_()
    expected = torch.empty(2, 3, 2, dtype=x.dtype)
    for y in range(2):
        for z in range(3):
            for c in range(2):
                expected[y,z,c] = sum(float(x[y*factor+i,z*factor+j,c].detach())
                                       for i in range(factor) for j in range(factor))/(factor*factor)
    actual = image.reconstruct(x, factor)
    assert torch.equal(actual, expected) and actual.data_ptr() != x.data_ptr()
    actual.sum().backward()
    assert torch.equal(x.grad, torch.full_like(x, 1/(factor*factor)))


def test_layout_batch_noncontiguous_and_empty(image):
    """Explicit HWC/BHWC supports strided tensors and empty batches without aliasing."""
    x = torch.ones(2, 4, 6, 4, dtype=torch.float64).transpose(1, 2)
    y = image.reconstruct(x, 2)
    assert y.shape == (2, 3, 2, 4) and y.is_contiguous()
    empty = x[:0]
    assert image.reconstruct(empty, 2).shape == (0, 3, 2, 4)
    assert image.composite([], empty).shape == empty.shape


def test_srgb_scalar_reference_signed_hdr_and_thresholds(image):
    """Transfer functions use explicit piecewise scalar equations without clipping alpha/RGB."""
    v = [-.1, 0, .0031308, .04045, .5, 1, 2]
    x = torch.tensor(v, dtype=torch.float64)
    dec = [a/12.92 if a <= .04045 else ((a+.055)/1.055)**2.4 for a in v]
    enc = [12.92*a if a <= .0031308 else 1.055*a**(1/2.4)-.055 for a in v]
    close(image.srgb_to_linear(x), torch.tensor(dec, dtype=x.dtype))
    close(image.linear_to_srgb(x), torch.tensor(enc, dtype=x.dtype))
    for name, threshold, derivative in [('srgb_to_linear', .04045, 1/12.92),
                                        ('linear_to_srgb', .0031308, 12.92)]:
        t = torch.tensor(threshold, dtype=x.dtype, requires_grad=True)
        getattr(image, name)(t).backward()
        assert t.grad.item() == derivative


def test_affine_matrix_clockwise_pivot_and_translation(image):
    """The matrix rotates around the supplied pixel pivot then translates; no implicit centre."""
    t = torch.tensor([2., 3.], dtype=torch.float64)
    r = torch.tensor(math.pi/2, dtype=t.dtype)
    s = torch.tensor([1., 1.], dtype=t.dtype); p = torch.tensor([4., 5.], dtype=t.dtype)
    a = image.affine_matrix(t, r, s, p)
    close(a, torch.tensor([[0., -1., 11.], [1., 0., 4.]], dtype=t.dtype))


def test_warp_identity_integer_translation_and_clamp(image):
    """Inverse mapping uses pixel centres; translating right leaves a transparent left edge."""
    x = torch.tensor([[[1.,0,0,1], [0,1.,0,1], [0,0,1.,1]]], dtype=torch.float64)
    a = torch.tensor([[1.,0,0], [0,1,0]], dtype=x.dtype)
    assert torch.equal(image.warp_affine(x, a, (1,3)), x)
    a[0,2] = 1
    actual = image.warp_affine(x, a, (1,3))
    assert torch.equal(actual, torch.cat((torch.zeros_like(x[:,:1]), x[:,:2]), dim=1))
    actual = image.warp_affine(x, a, (1,3), border='clamp')
    assert torch.equal(actual, torch.cat((x[:,:1], x[:,:2]), dim=1))


def test_bilinear_half_pixel_and_nearest_tie(image):
    """Half-pixel interpolation mixes two centres; nearest's half tie picks increasing index."""
    x = torch.tensor([[[1.,0,0,1], [0,0,1.,1]]], dtype=torch.float64)
    a = torch.tensor([[1.,0,-.5], [0,1,0]], dtype=x.dtype)
    close(image.warp_affine(x, a, (1,1)), rgba([.5,0,.5,1]))
    assert torch.equal(image.warp_affine(x, a, (1,1), filter='nearest'), x[:,1:2])


def test_warp_straight_transparent_colours_do_not_fringe(image):
    """Transparent red beside opaque blue must interpolate blue rather than purple."""
    x = torch.tensor([[[1.,0,0,0], [0,0,1.,1]]], dtype=torch.float64)
    a = torch.tensor([[1.,0,-.5], [0,1,0]], dtype=x.dtype)
    close(image.warp_affine(x, a, (1,1), alpha='straight'), rgba([0,0,1,.5]))


def test_exposure_brightness_preserve_alpha_and_hdr(image):
    """One stop doubles linear RGB; brightness is multiplied by alpha in premultiplied form."""
    x = rgba([.2, .3, .4, .5])
    close(image.adjust(x, exposure=1, brightness=.1), rgba([.45,.65,.85,.5]))
    close(image.adjust(x, exposure=-1), rgba([.1,.15,.2,.5]))


def test_gaussian_against_independent_scalar_convolution(image):
    """Finite Gaussian filtering agrees with scalar two-dimensional convolution and zero padding."""
    x = torch.zeros(3, 4, 4, dtype=torch.float64); x[1,1] = torch.tensor([.2,.3,.4,1.])
    sigma = .8; raw = [math.exp(-.5*(i/sigma)**2) for i in (-1,0,1)]
    w = [v/sum(raw) for v in raw]
    expected = torch.zeros_like(x)
    for y in range(3):
        for z in range(4):
            for dy in range(-1,2):
                for dx in range(-1,2):
                    if 0 <= y+dy < 3 and 0 <= z+dx < 4:
                        expected[y,z] += x[y+dy,z+dx]*w[dy+1]*w[dx+1]
    close(image.gaussian_blur(x, sigma, radius=1), expected)
    close(image.gaussian_blur(torch.ones_like(x), sigma, radius=1, border='clamp'), torch.ones_like(x))


def test_gaussian_zero_radius_fresh_and_zero_sigma_gradient(image):
    """Radius-zero blur retains a connected but exactly zero sigma derivative."""
    x = rgba([.2,.3,.4,1.]).requires_grad_(); sigma = torch.tensor(.8, requires_grad=True, dtype=x.dtype)
    y = image.gaussian_blur(x, sigma, radius=0)
    assert y.data_ptr() != x.data_ptr() and torch.equal(y,x)
    y.sum().backward()
    assert sigma.grad is not None and sigma.grad.item() == 0


@pytest.mark.parametrize('border',['transparent','clamp'])
@pytest.mark.parametrize('per_batch',[False,True])
def test_gaussian_odd_batch_sigma_vjp_independent_difference(image,border,per_batch):
    """Scalar and per-image widths retain correct derivatives after ordered spatial reduction."""
    values=[[[[.02*(1+b+x),.03*(1+y),.01*(1+x+y),.8] for x in range(5)]
              for y in range(3)] for b in range(2)]
    pixels=torch.tensor(values,dtype=torch.float64)
    widths=[.83,1.17] if per_batch else [.83]
    sigma=torch.tensor(widths if per_batch else widths[0],dtype=pixels.dtype,requires_grad=True)
    cot=torch.arange(1,121,dtype=pixels.dtype).reshape(2,3,5,4)/128
    result=image.gaussian_blur(pixels,sigma,radius=1,border=border)
    actual=torch.autograd.grad(result,sigma,grad_outputs=cot)[0].reshape(-1)
    def objective(params):
        total=0.;weights=cot.tolist()
        for batch in range(2):
            s=params[batch] if per_batch else params[0]
            raw=[math.exp(-.5*(i/s)**2) for i in (-1,0,1)]
            w=[v/sum(raw) for v in raw]
            for y in range(3):
                for x in range(5):
                    for dy in (-1,0,1):
                        for dx in (-1,0,1):
                            px,py=x+dx,y+dy
                            if border=='clamp':px,py=min(4,max(0,px)),min(2,max(0,py))
                            if 0<=px<5 and 0<=py<3:
                                total+=sum(values[batch][py][px][c]*weights[batch][y][x][c]
                                           for c in range(4))*w[dy+1]*w[dx+1]
        return total
    expected=[]
    for index in range(len(widths)):
        plus=widths[:];minus=widths[:];plus[index]+=1e-6;minus[index]-=1e-6
        expected.append((objective(plus)-objective(minus))/2e-6)
    torch.testing.assert_close(actual,torch.tensor(expected,dtype=actual.dtype),atol=2e-5,rtol=2e-4)


@pytest.mark.parametrize('operation', ['alpha', 'mask', 'over', 'transfer', 'adjust', 'blur', 'affine', 'warp'])
def test_smooth_region_gradients(image, operation):
    """Central differences independently check every first-order continuous parameter on tiny inputs."""
    x = rgba([.2,.3,.4,.7]).requires_grad_()
    scalar = lambda x: torch.tensor(x, dtype=torch.float64, requires_grad=True)
    if operation == 'alpha': fn, args = image.convert_alpha, (x,)
    elif operation == 'mask':
        mask = torch.full((1,1,1), .6, dtype=x.dtype, requires_grad=True)
        fn, args = lambda a,b,c: image.apply_mask(a,b,opacity=c), (x,mask,scalar(.8))
    elif operation == 'over': fn, args = lambda a,b: image.composite([a],b), (x,x.detach().clone().requires_grad_())
    elif operation == 'transfer': fn, args = image.linear_to_srgb, (x,)
    elif operation == 'adjust': fn, args = lambda a,b,c: image.adjust(a,exposure=b,brightness=c), (x,scalar(.2),scalar(.1))
    elif operation == 'blur': fn, args = lambda a,b: image.gaussian_blur(a,b,radius=1), (x,scalar(.8))
    elif operation == 'affine':
        vec = lambda v: torch.tensor(v,dtype=x.dtype,requires_grad=True)
        fn, args = image.affine_matrix, (vec([.2,.3]),scalar(.17),vec([1.1,.9]),vec([.1,.2]))
    else:
        pixels = torch.tensor([[[.1,.2,.3,.7],[.4,.2,.1,.8]],[[.3,.1,.2,.6],[.2,.3,.1,.9]]],dtype=x.dtype,requires_grad=True)
        a = torch.tensor([[1.,.1,-.2],[.1,1.,-.3]],dtype=x.dtype,requires_grad=True)
        fn, args = lambda v,m: image.warp_affine(v,m,(1,1)), (pixels,a)
    assert torch.autograd.gradcheck(fn,args,eps=1e-6,atol=2e-5,rtol=2e-4)


@pytest.mark.parametrize('case', ['dtype','nan','alpha','factor','divisibility','mask','opacity','singular','radius','sigma','enum','batch'])
def test_invalid_inputs_fail_without_mutation(image, case):
    """Invalid domains and ambiguous layouts fail explicitly rather than silently correcting the request."""
    x = torch.ones(2,3,4,dtype=torch.float64); before=x.clone()
    cases = {
        'dtype': lambda: image.reconstruct(x.int(),1),
        'nan': lambda: image.reconstruct(x*float('nan'),1),
        'alpha': lambda: image.convert_alpha(x*2),
        'factor': lambda: image.reconstruct(x,True),
        'divisibility': lambda: image.reconstruct(x,2),
        'mask': lambda: image.apply_mask(x,torch.ones(2,3,dtype=x.dtype)),
        'opacity': lambda: image.apply_mask(x,opacity=2),
        'singular': lambda: image.warp_affine(x,torch.zeros(2,3,dtype=x.dtype),(2,3)),
        'radius': lambda: image.gaussian_blur(x,1,radius=33),
        'sigma': lambda: image.gaussian_blur(x,0,radius=1),
        'enum': lambda: image.composite([],x,alpha='auto'),
        'batch': lambda: image.composite([x.unsqueeze(0)],x),
    }
    with pytest.raises((ValueError,TypeError)):
        cases[case]()
    assert torch.equal(x,before)


def test_repeat_fresh_graph_and_static_capabilities(image):
    """Repeated evaluation retains no stale tensor graph; capabilities disclose unqualified profiles."""
    original=rgba([.2,.3,.4,.8]); outputs=[]
    for _ in range(3):
        x=original.clone().requires_grad_(); out=image.adjust(x,exposure=.2)
        out.sum().backward(); outputs.append((out.detach(),x.grad))
    assert all(torch.equal(a,outputs[0][0]) and torch.equal(b,outputs[0][1]) for a,b in outputs)
    cap=image.capabilities()
    assert cap['schema_version']==1 and cap['gpu_probed'] is False
    assert cap['cross_device_reference_qualified'] is False
    assert cap['compilation_export_qualified'] is False
    assert set(cap['operations']) >= {'composite','warp_affine','gaussian_blur'}


def test_batched_transforms_masks_effects_match_separate_calls(image):
    """Per-batch parameters must affect only their own image and agree with scalar calls."""
    x = torch.tensor([[[.1,.2,.3,.7],[.3,.2,.1,.8]],[[.2,.1,.2,.6],[.1,.3,.2,.9]]],dtype=torch.float64)
    batch = torch.stack((x,x*.75))
    mask = torch.full((2,2,2,1), .8,dtype=x.dtype)
    opacity = torch.tensor([.5,.7],dtype=x.dtype)
    exposure = torch.tensor([.1,.3],dtype=x.dtype)
    sigma = torch.tensor([.7,.9],dtype=x.dtype)
    matrix = torch.tensor([[[1.,0,.2],[0,1.,.3]],[[1.,.1,-.2],[0,1.,.1]]],dtype=x.dtype)
    def recipe(values,m,o,e,s,a):
        return image.warp_affine(image.gaussian_blur(image.apply_mask(image.adjust(values,exposure=e),m,opacity=o),s,radius=1),a,(2,2))
    actual = recipe(batch,mask,opacity,exposure,sigma,matrix)
    expected = torch.stack([recipe(batch[i],mask[i],opacity[i],exposure[i],sigma[i],matrix[i]) for i in range(2)])
    assert torch.equal(actual,expected)


def test_nearest_matrix_derivative_is_connected_zero(image):
    """Nearest sampling has a documented zero coordinate derivative, not a missing gradient."""
    x = torch.ones(2,2,4,dtype=torch.float64,requires_grad=True)
    a = torch.tensor([[1.,0.,.1],[0.,1.,.2]],dtype=x.dtype,requires_grad=True)
    y = image.warp_affine(x,a,(1,1),filter='nearest')
    gx,ga = torch.autograd.grad(y.sum(),(x,a))
    assert torch.isfinite(gx).all() and torch.equal(ga,torch.zeros_like(ga))


def test_full_layer_chain_parameter_gradients_and_inputs_preserved(image):
    """All eight fitting parameters remain differentiable through the complete layer recipe."""
    source = torch.tensor([[[.2,.1,.3,.7],[.3,.2,.1,.8],[.1,.3,.2,.6]],
                           [[.1,.2,.2,.7],[.2,.3,.1,.8],[.3,.1,.1,.6]],
                           [[.3,.1,.2,.7],[.1,.2,.3,.8],[.2,.2,.1,.6]]],dtype=torch.float64)
    before=source.clone(); mask=torch.full((3,3,1),.8,dtype=source.dtype)
    background=torch.full((2,2,4),.1,dtype=source.dtype); background[...,3]=1
    pivot=torch.tensor([1.,1.],dtype=source.dtype)
    def recipe(params):
        a=image.affine_matrix(params[:2],params[2],params[3:5],pivot)
        layer=image.adjust(source,exposure=params[6])
        layer=image.apply_mask(layer,mask,opacity=params[5])
        layer=image.gaussian_blur(layer,params[7],radius=1)
        layer=image.warp_affine(layer,a,(2,2))
        return image.reconstruct(image.composite([layer],background),2)
    params=torch.tensor([.21,.32,.13,1.1,.9,.7,.2,.8],dtype=source.dtype,requires_grad=True)
    assert torch.autograd.gradcheck(recipe,(params,),eps=1e-6,atol=5e-5,rtol=5e-4)
    assert torch.equal(source,before)


@pytest.mark.parametrize('operation', ['convert','mask','warp','adjust','blur'])
def test_empty_batch_keeps_connected_parameter_gradients(image, operation):
    """Zero images are a valid batch; every applicable input keeps a correctly shaped zero gradient."""
    x=torch.empty((0,2,3,4),dtype=torch.float64,requires_grad=True)
    p=torch.tensor(.7,dtype=x.dtype,requires_grad=True)
    if operation=='convert': y=image.convert_alpha(x); args=(x,)
    elif operation=='mask': y=image.apply_mask(x,opacity=p); args=(x,p)
    elif operation=='adjust': y=image.adjust(x,exposure=p); args=(x,p)
    elif operation=='blur': y=image.gaussian_blur(x,p,radius=1); args=(x,p)
    else:
        a=torch.empty((0,2,3),dtype=x.dtype,requires_grad=True)
        y=image.warp_affine(x,a,(2,3)); args=(x,a)
    gradients=torch.autograd.grad(y.sum(),args)
    assert all(torch.equal(g,torch.zeros_like(g)) for g in gradients)


def test_empty_batch_blur_keeps_per_image_sigma_gradient_shape(image):
    """An empty explicit per-image width tensor must return an empty connected derivative."""
    pixels=torch.empty((0,3,5,4),dtype=torch.float64,requires_grad=True)
    sigma=torch.empty((0,),dtype=pixels.dtype,requires_grad=True)
    result=image.gaussian_blur(pixels,sigma,radius=1)
    gp,gs=torch.autograd.grad(result.sum(),(pixels,sigma))
    assert result.shape==pixels.shape and gp.shape==pixels.shape and gs.shape==sigma.shape
    assert gp.numel()==gs.numel()==0


def test_finite_affine_input_cannot_hide_overflowed_inverse_arithmetic(image):
    """Finite matrix entries can still overflow a determinant; do not render a bogus collapsed map."""
    x=torch.ones(1,1,4,dtype=torch.float64)
    a=torch.tensor([[1e200,0,0],[0,1e200,0]],dtype=x.dtype)
    with pytest.raises(RuntimeError,match='nonfinite'):
        image.warp_affine(x,a,(1,1))


def test_capability_parameters_are_actionable_and_explicit(image):
    """A consumer needs parameter domains, units and discrete choices, not just operation names."""
    ops=image.capabilities()['operations']
    assert ops['gaussian_blur']['parameters']['sigma']['range']==[.1,32]
    assert ops['gaussian_blur']['parameters']['radius']['differentiable'] is False
    assert ops['adjust']['parameters']['exposure']['units']=='stops'
    assert ops['warp_affine']['parameters']['border']['choices']==['transparent','clamp']
    assert ops['reconstruct']['parameters']['factor']['differentiable'] is False
    assert ops['composite']['parameters']['layers']['order']=='bottom-to-top'


def test_inverse_grid_rejects_second_derivative_instead_of_partial_graph(image):
    """A first-order custom VJP must refuse a Hessian rather than differentiate detached geometry."""
    matrix=torch.tensor([[1.1,.13,.21],[-.07,.93,-.18]],dtype=torch.float64,requires_grad=True)
    yy,xx=torch.meshgrid(torch.arange(2,dtype=matrix.dtype),torch.arange(3,dtype=matrix.dtype),indexing='ij')
    coordinates=image._InverseGrid.apply(matrix,xx,yy,2,3)
    cotangent=torch.linspace(.1,.9,coordinates.numel(),dtype=matrix.dtype).reshape_as(coordinates).requires_grad_()
    first,=torch.autograd.grad(coordinates,matrix,grad_outputs=cotangent,create_graph=True)
    assert torch.isfinite(first).all()
    with pytest.raises(RuntimeError,match='once_differentiable'):
        first.sum().backward()


@pytest.mark.parametrize('name',['srgb_to_linear','linear_to_srgb'])
def test_native_transfer_discloses_and_rejects_second_derivatives(image,name):
    """The new shared arithmetic has a declared first VJP, not an accidental incomplete Hessian."""
    values=torch.tensor([.02,.2,.7,1.4],dtype=torch.float64,requires_grad=True)
    cotangent=torch.ones_like(values,requires_grad=True)
    result=getattr(image,name)(values)
    first,=torch.autograd.grad(result,values,grad_outputs=cotangent,create_graph=True)
    assert torch.isfinite(first).all()
    with pytest.raises(RuntimeError,match='once_differentiable'):
        first.sum().backward()
    contract=image.capabilities()['operations'][name]
    assert contract['implementation']=='native C11/CUDA transfer'
    assert contract['higher_derivatives'].startswith('unsupported:')


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('name,mode',[('srgb_to_linear',0),('linear_to_srgb',1)])
def test_native_transfer_wrapper_matches_registered_values_and_vjp(image,dtype,name,mode):
    """Both public wrappers must reach the same registered native forward and first VJP."""
    values=torch.tensor([-.1,.0031308,.04045,.5,2.],dtype=dtype,requires_grad=True)
    weights=torch.tensor([.125,.25,.5,.75,1.],dtype=dtype)
    expected=torch.ops.gffx.image_transfer(values.detach(),mode)
    expected_gradient=torch.ops.gffx.image_transfer_backward(values.detach(),weights,mode)
    actual=getattr(image,name)(values)
    gradient,=torch.autograd.grad(actual,values,grad_outputs=weights)
    assert torch.equal(actual.detach(),expected)
    assert torch.equal(gradient,expected_gradient)


def _isolate_transfer_exception_boundary(monkeypatch):
    """Attribution only: source11's separate namespace defect must not hide error typing.

    The unaliased dispatcher test above remains mandatory and fails source11.
    These temporary aliases do not alter a wheel or its numerical implementation.
    """
    for name in ('image_transfer','image_transfer_backward'):
        if not hasattr(torch.ops.gffx_internal,name):
            monkeypatch.setattr(torch.ops.gffx_internal,name,getattr(torch.ops.gffx,name),raising=False)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('name',['srgb_to_linear','linear_to_srgb'])
@pytest.mark.parametrize('phase',['forward','backward'])
def test_native_transfer_finite_overflow_is_typed_and_preserves_inputs(image,monkeypatch,dtype,name,phase):
    """Finite input/cotangent arithmetic overflow must translate to the adapter's OverflowError."""
    _isolate_transfer_exception_boundary(monkeypatch)
    maximum=torch.finfo(dtype).max
    value=(maximum if name=='srgb_to_linear' else -maximum) if phase=='forward' else (1. if name=='srgb_to_linear' else 0.)
    values=torch.full((2,),value,dtype=dtype,requires_grad=True)
    original=values.detach().clone()
    with pytest.raises(OverflowError,match='OVERFLOW.*image transfer'):
        result=getattr(image,name)(values)
        if phase=='backward':
            torch.autograd.grad(result,values,grad_outputs=torch.full_like(values,maximum))
    assert torch.equal(values.detach(),original)
    assert values.grad is None


@pytest.mark.parametrize('name',['srgb_to_linear','linear_to_srgb'])
def test_native_transfer_singleton_expanded_input_and_sum_cotangent(image,monkeypatch,name):
    """A size-one zero stride remains contiguous in torch but must be materialized for native ABI."""
    _isolate_transfer_exception_boundary(monkeypatch)
    base=torch.tensor([.5],dtype=torch.float64,requires_grad=True)
    expanded=torch.as_strided(base,(1,),(0,))
    assert expanded.is_contiguous() and expanded.stride()==(0,)
    result=getattr(image,name)(expanded)
    gradient,=torch.autograd.grad(result.sum(),base)
    assert result.shape==base.shape and torch.isfinite(result).all() and torch.isfinite(gradient).all()


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('name,mode',[('srgb_to_linear',0),('linear_to_srgb',1)])
def test_native_transfer_rank_zero_descriptor_preserves_scalar_forward_and_vjp(image,dtype,name,mode):
    """A scalar is rank zero, not a hidden length-one vector; ABI metadata must reflect that."""
    scalar=torch.tensor(.2,dtype=dtype,requires_grad=True)
    reference=scalar.detach().reshape(1)
    expected=torch.ops.gffx.image_transfer(reference,mode).reshape(())
    expected_gradient=torch.ops.gffx.image_transfer_backward(reference,torch.ones_like(reference),mode).reshape(())
    result=getattr(image,name)(scalar)
    gradient,=torch.autograd.grad(result,scalar)
    assert result.shape==gradient.shape==torch.Size([])
    assert torch.equal(result.detach(),expected) and torch.equal(gradient,expected_gradient)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('per_batch',[False,True])
@pytest.mark.parametrize('masked',[False,True])
def test_mask_odd_shared_opacity_gradient_independent_sum(image,dtype,per_batch,masked):
    """An explicit reduction must still equal the independently summed opacity derivative."""
    pixels=torch.arange(120,dtype=dtype).reshape(2,3,5,4)/256;pixels[...,3]=.8
    mask=(.2+torch.arange(30,dtype=dtype)/64).reshape(2,3,5,1) if masked else None
    opacity=torch.tensor([.63,.41] if per_batch else .63,dtype=dtype,requires_grad=True)
    cot=torch.arange(1,121,dtype=dtype).reshape(pixels.shape)/128
    actual=torch.autograd.grad(image.apply_mask(pixels,mask,opacity=opacity),opacity,grad_outputs=cot)[0]
    values=[]
    for b in range(2):
        values.append(math.fsum(float(pixels[b,y,x,c])*float(cot[b,y,x,c])*(float(mask[b,y,x,0]) if masked else 1.) for y in range(3) for x in range(5) for c in range(4)))
    expected=torch.tensor(values if per_batch else math.fsum(values),dtype=dtype)
    torch.testing.assert_close(actual,expected,atol=2e-5 if dtype==torch.float32 else 2e-12,rtol=2e-4 if dtype==torch.float32 else 2e-12)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_alpha_over_independent_derivatives(image,dtype):
    """The fixed channel reduction must retain the analytic alpha-over derivative."""
    import math
    source=torch.tensor([[[.13,.21,.17,.63],[.23,.11,.07,.41]]],dtype=dtype,requires_grad=True)
    destination=torch.tensor([[[.19,.29,.31,.83],[.03,.09,.17,.71]]],dtype=dtype,requires_grad=True)
    cot=torch.tensor([[[.37,-.23,.71,.13],[.19,.43,-.17,.61]]],dtype=dtype)
    out=image.composite([source],destination)
    gs,gd=torch.autograd.grad(out,(source,destination),grad_outputs=cot)
    expected=cot.clone()
    for j in range(2):
        expected[0,j,3]=float(cot[0,j,3])-math.fsum(float(cot[0,j,k])*float(destination[0,j,k]) for k in range(4))
    eps=torch.finfo(dtype).eps
    torch.testing.assert_close(gs,expected,rtol=8*eps,atol=8*eps)
    torch.testing.assert_close(gd,cot*(1-source[...,3:4]),rtol=0,atol=0)
    torch.testing.assert_close(out,source+(1-source[...,3:4])*destination,rtol=0,atol=0)


def test_adjust_independent_alpha_and_parameter_derivatives():
    from gffx.torch import image
    import math
    pixels=torch.tensor([[[.12,.24,.31,.7],[.2,.1,.05,.8]]],dtype=torch.float64,requires_grad=True)
    ev=torch.tensor(-.04,dtype=torch.float64,requires_grad=True);lift=torch.tensor(-.006,dtype=torch.float64,requires_grad=True)
    cot=torch.tensor([[[.17,-.29,.41,.53],[-.67,.79,-.83,.97]]],dtype=torch.float64)
    result=image.adjust(pixels,exposure=ev,brightness=lift)
    gp,ge,gb=torch.autograd.grad(result,(pixels,ev,lift),cot)
    gain=2**float(ev.detach())
    expected_rgb=cot[...,:3]*gain
    expected_alpha=cot[...,3:4]+(cot[...,:3]*lift.detach()).sum(-1,keepdim=True)
    torch.testing.assert_close(gp,torch.cat((expected_rgb,expected_alpha),-1),atol=1e-15,rtol=1e-15)
    torch.testing.assert_close(ge,(cot[...,:3]*pixels.detach()[...,:3]).sum()*gain*math.log(2),atol=1e-15,rtol=1e-15)
    torch.testing.assert_close(gb,(cot[...,:3]*pixels.detach()[...,3:4]).sum(),atol=1e-15,rtol=1e-15)
    fn=lambda a,b,c:image.adjust(a,exposure=b,brightness=c)
    assert torch.autograd.gradcheck(fn,(pixels,ev,lift))
    assert torch.autograd.gradgradcheck(fn,(pixels,ev,lift))


def test_masked_multiply_independent_local_mask_derivative():
    from gffx.torch import image
    pixels=torch.tensor([[[.12,.24,.31,.7],[.2,.1,.05,.8]]],dtype=torch.float64,requires_grad=True)
    mask=torch.tensor([[[.37],[.61]]],dtype=torch.float64,requires_grad=True)
    opacity=torch.tensor(.68,dtype=torch.float64,requires_grad=True)
    cot=torch.tensor([[[.17,-.29,.41,.53],[-.67,.79,-.83,.97]]],dtype=torch.float64)
    result=image.apply_mask(pixels,mask,opacity=opacity)
    a,b,c=torch.autograd.grad(result,(pixels,mask,opacity),cot)
    terms=(pixels.detach()*cot).sum(-1,keepdim=True)
    torch.testing.assert_close(a,cot*mask.detach()*opacity.detach(),atol=1e-15,rtol=1e-15)
    torch.testing.assert_close(b,terms*opacity.detach(),atol=1e-15,rtol=1e-15)
    torch.testing.assert_close(c,(terms*mask.detach()).sum(),atol=1e-15,rtol=1e-15)
    fn=lambda x,y,z:image.apply_mask(x,y,opacity=z)
    assert torch.autograd.gradcheck(fn,(pixels,mask,opacity))
    assert torch.autograd.gradgradcheck(fn,(pixels,mask,opacity))
