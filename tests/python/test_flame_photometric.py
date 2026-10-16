"""Independent synthetic controls for the gffx-only FLAME fitting composition."""
import importlib
import math
import os
import pytest
import torch

def api():
    return importlib.import_module('tools.flame_photometric')

def fixture():
    dev=os.environ.get('GFFX_GROUP4_DEVICE','cpu')
    def t(x):return torch.tensor(x,dtype=torch.float64,device=dev)
    v=t([[0.,0.,0.],[1.,0.,.1],[0.,1.,.2],[.2,.3,1.]])
    basis=torch.zeros((2,4,3),dtype=v.dtype,device=dev)
    basis[0,1,0]=.5;basis[1,3,2]=.4
    reg=t([[1.,0.,0.,0.],[0.,.5,.5,0.]])
    weights=t([[1.,0.],[.75,.25],[.25,.75],[0.,1.]])
    corrective=torch.zeros((9,4,3),dtype=v.dtype,device=dev)
    corrective[1,3,0]=.08;corrective[0,2,2]=.03
    return api().FlameData(v,basis,corrective,reg,torch.tensor([-1,0],device=dev,dtype=torch.int32),weights,torch.tensor([[0,1,2],[0,1,3]],device=dev,dtype=torch.int32))

def independent(data,coeff,angle):
    # Two-joint closed geometry; no gffx/model forward or reference repository code.
    shaped=data.template+coeff[0]*data.blend_basis[0]+coeff[1]*data.blend_basis[1]
    c=torch.cos(angle);s=torch.sin(angle);z=angle*0;o=z+1
    R=torch.stack([c,-s,z,s,c,z,z,z,o]).reshape(3,3)
    pivot=(shaped[1]+shaped[2])*.5
    corrected=shaped.clone();corrected[3,0]=corrected[3,0]-.08*s;corrected[2,2]=corrected[2,2]+.03*(c-1)
    child=(corrected-pivot)@R.T+pivot
    return corrected*data.skin_weights[:,:1]+child*data.skin_weights[:,1:]

def test_model_joint_regression_pose_correctives_bind_and_landmarks():
    d=fixture();coeff=d.template.new_tensor([.12,-.08]);angle=coeff.new_tensor(.3)
    pose=coeff.new_zeros((2,3));pose[1,2]=angle
    before=[x.clone() for x in (d.template,d.blend_basis,d.pose_basis,d.joint_regressor,d.skin_weights)]
    got=api().deform(d,coeff,pose)
    torch.testing.assert_close(got,independent(d,coeff,angle),atol=1e-12,rtol=1e-11)
    indices=d.faces.new_tensor([0,1]);bary=d.template.new_tensor([[.2,.3,.5],[.1,.2,.7]])
    lmk=api().landmarks(d,got,indices,bary)
    expected=torch.stack([.2*got[0]+.3*got[1]+.5*got[2],.1*got[0]+.2*got[1]+.7*got[3]])
    torch.testing.assert_close(lmk,expected,atol=1e-12,rtol=1e-11)
    assert all(torch.equal(x,y) for x,y in zip(before,(d.template,d.blend_basis,d.pose_basis,d.joint_regressor,d.skin_weights)))

def test_model_shape_and_joint_gradients_independent_and_two_scale():
    d=fixture();p=d.template.new_tensor([.12,-.08,.3],requires_grad=True)
    weight=torch.arange(1,13,device=p.device,dtype=p.dtype).reshape(4,3)/13
    def function(q):
        zero=q[0]*0;pose=torch.stack([zero,zero,zero,zero,zero,q[2]]).reshape(2,3)
        return (api().deform(d,q[:2],pose)*weight).sum()
    got=torch.autograd.grad(function(p),p)[0]
    want=torch.autograd.grad((independent(d,p[:2],p[2])*weight).sum(),p)[0]
    torch.testing.assert_close(got,want,atol=1e-11,rtol=1e-10)
    for h in (1e-5,5e-6):
        for i in range(3):
            delta=torch.zeros_like(p);delta[i]=h
            fd=(function(p.detach()+delta)-function(p.detach()-delta))/(2*h)
            torch.testing.assert_close(got[i],fd,atol=1e-10,rtol=1e-8)

def test_camera_matches_reference_scale_translation_flip_and_gradients():
    d=fixture();v=d.template.detach().clone().requires_grad_();c=v.new_tensor([1.3,.2,-.1],requires_grad=True)
    got=api().project(v,c)
    want=torch.stack([c[0]*(v[:,0]+c[1]),-c[0]*(v[:,1]+c[2]),-c[0]*v[:,2]],-1)
    torch.testing.assert_close(got,want,atol=1e-13,rtol=1e-12)
    for a,b in zip(torch.autograd.grad(got.square().sum(),(v,c)),torch.autograd.grad(want.square().sum(),(v,c))):
        torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-11)

def test_texture_basis_channels_and_coefficient_derivative():
    d=fixture();mean=d.template.new_tensor([[[.2,.3,.4],[.4,.2,.5]],[[.5,.3,.1],[.6,.7,.8]]]);basis=torch.stack([mean*.1,mean.flip(0)*-.2]);p=mean.new_tensor([.3,-.2],requires_grad=True)
    got=api().texture(mean,basis,p)
    torch.testing.assert_close(got,mean+p[0]*basis[0]+p[1]*basis[1],atol=1e-14,rtol=1e-13)
    torch.testing.assert_close(torch.autograd.grad(got.sum(),p)[0],basis.sum((1,2,3)),atol=1e-13,rtol=1e-12)

def test_sh_coefficient_conversion_independent_unit_normal_basis():
    d=fixture();n=d.template.new_tensor([[0.,0.,1.],[1.,0.,0.],[.3,.4,math.sqrt(.75)]])
    coeff=torch.arange(27,device=n.device,dtype=n.dtype).reshape(9,3)/100;coeff.requires_grad_()
    x,y,z=n.unbind(-1);pi=math.pi
    basis=torch.stack([x*0+1,x,y,z,x*y,x*z,y*z,x*x-y*y,3*z*z-1],-1)
    factors=n.new_tensor([1/math.sqrt(4*pi),*( [2*pi/3*math.sqrt(3/(4*pi))]*3),*( [pi/4*3*math.sqrt(5/(12*pi))]*3),pi/4*1.5*math.sqrt(5/(12*pi)),pi/8*math.sqrt(5/(4*pi))])
    want=(basis*factors)@coeff;got=api().shade_sh(n,coeff)
    torch.testing.assert_close(got,want,atol=1e-12,rtol=1e-11)
    torch.testing.assert_close(torch.autograd.grad(got.sum(),coeff)[0],torch.autograd.grad(want.sum(),coeff)[0],atol=1e-12,rtol=1e-11)

def test_losses_mask_denominator_landmark_norm_and_regularizers():
    d=fixture();pred=d.template.new_tensor([[[.2,.4,.6],[.1,.2,.3]]]);target=torch.zeros_like(pred);mask=pred.new_tensor([[1.,0.]])
    lmk=pred.new_tensor([[3.,4.],[0.,2.]]);zero=torch.zeros_like(lmk)
    p={'shape':pred.new_tensor([2.]),'expression':pred.new_tensor([3.]),'pose':pred.new_zeros(6)}
    loss=api().objective(pred,target,mask,lmk,zero,p)
    assert float(loss['photometric'])==pytest.approx(8*(.2+.4+.6)/6)
    assert float(loss['landmark'])==pytest.approx(3.5)
    assert float(loss['prior'])==pytest.approx(.5e-4*(4+9))

@pytest.mark.parametrize('bad',['nonfinite','shape','parents'])
def test_model_refuses_invalid_inputs(bad):
    d=fixture();coeff=d.template.new_zeros(2);pose=d.template.new_zeros((2,3))
    if bad=='nonfinite':coeff[0]=float('nan')
    elif bad=='shape':coeff=coeff[:1]
    else:d.parents[1]=1
    with pytest.raises((ValueError,RuntimeError)):api().deform(d,coeff,pose)

def image_fixture():
    d=fixture();v=d.template.new_tensor([[-1.,-1.,0.],[1.,-1.,0.],[1.,1.,0.],[-1.,1.,0.]])
    f=d.faces.new_tensor([[0,1,2],[0,2,3]])
    uv=v.new_tensor([[.25,.75],[.75,.75],[.75,.25],[.25,.25]])
    yy,xx=torch.meshgrid(torch.arange(4,device=v.device,dtype=v.dtype),torch.arange(4,device=v.device,dtype=v.dtype),indexing='ij')
    tex=torch.stack([.2+.03*xx,.3+.04*yy,.4+.02*(xx+yy)],-1)
    light=v.new_zeros((9,3));light[0]=math.sqrt(4*math.pi)
    return v,f,uv[f.long()].contiguous(),tex,light

def test_image_matches_independent_affine_texture_and_camera_gradient():
    v,f,uv,tex,lights=image_fixture();cam=v.new_tensor([1.,.031,-.027],requires_grad=True)
    got=api().render_image(v,f,uv,tex,lights,cam,4)
    y,x=torch.meshgrid(torch.arange(4,device=v.device,dtype=v.dtype),torch.arange(4,device=v.device,dtype=v.dtype),indexing='ij')
    world_x=(2*(x+.5)/4-1)/cam[0]-cam[1]
    world_y=(1-2*(y+.5)/4)/cam[0]-cam[2]
    u=.5+.25*world_x;vv=.5-.25*world_y
    xx=4*u-.5;yy=4*vv-.5
    want=torch.stack([.2+.03*xx,.3+.04*yy,.4+.02*(xx+yy)],-1)
    assert bool(got['mask'].all())
    torch.testing.assert_close(got['colour'],want,atol=1e-12,rtol=1e-11)
    actual=torch.autograd.grad(got['colour'].square().sum(),cam)[0]
    expected=torch.autograd.grad(want.square().sum(),cam)[0]
    torch.testing.assert_close(actual,expected,atol=1e-11,rtol=1e-10)

def test_image_texture_light_geometry_vjp_two_scale():
    v,f,uv,tex,lights=image_fixture();p=v.new_tensor([.05,.08,.03],requires_grad=True)
    def fun(q):
        moved=v+torch.stack([q[0],q[0]*0,q[0]*0])
        appearance=tex+q[1];lamp=lights+torch.nn.functional.pad(q[2].expand(1,3),(0,0,0,8))
        image=api().render_image(moved,f,uv,appearance,lamp,v.new_tensor([1.,0.,0.]),4)['colour']
        return image.square().mean()
    gradient=torch.autograd.grad(fun(p),p)[0]
    assert bool((gradient.abs()>1e-6).all())
    for h in (1e-5,5e-6):
        for i in range(3):
            delta=torch.zeros_like(p);delta[i]=h
            fd=(fun(p.detach()+delta)-fun(p.detach()-delta))/(2*h)
            torch.testing.assert_close(gradient[i],fd,atol=1e-10,rtol=1e-8)

def test_bounded_optimizer_recovers_independent_texture_target(record_property):
    v,f,uv,tex,lights=image_fixture();coeff=v.new_tensor([-.12,.10,-.08],requires_grad=True)
    # The four by four rays sample affine texel fields at x/y = .75,1.25,1.75,2.25.
    y,x=torch.meshgrid(torch.arange(4,device=v.device,dtype=v.dtype),torch.arange(4,device=v.device,dtype=v.dtype),indexing='ij')
    truth=v.new_tensor([.04,-.03,.02]);xx=.75+.5*x;yy=.75+.5*y
    target=torch.stack([.2+.03*xx,.3+.04*yy,.4+.02*(xx+yy)],-1)+truth
    fixed=tex.clone()
    def loss():
        prediction=api().render_image(v,f,uv,tex+coeff,lights,v.new_tensor([1.,0.,0.]),4)['colour']
        return (prediction-target).square().mean()
    result=api().optimize_stage({'texture':coeff},loss,iterations=64,learning_rate=.02,seconds=90)
    assert result['final_loss']<result['initial_loss']*.002
    assert float((coeff.detach()-truth).abs().max())<.008
    assert torch.equal(tex,fixed)
    record_property('recovery',str(result))

@pytest.mark.parametrize('failure',['cancel','deadline','nonfinite','iterations'])
def test_optimizer_refuses_cancel_exhaustion_and_nonfinite(failure):
    d=fixture();p=d.template.new_tensor([1.],requires_grad=True);before=p.detach().clone()
    kwargs=dict(iterations=2,learning_rate=.01,seconds=90)
    loss=lambda:p.square().sum()
    if failure=='cancel':kwargs['cancelled']=lambda:True
    elif failure=='deadline':kwargs['seconds']=0
    elif failure=='iterations':kwargs['iterations']=65
    else:loss=lambda:(p*float('nan')).sum()
    with pytest.raises((ValueError,TimeoutError,RuntimeError)):
        api().optimize_stage({'p':p},loss,**kwargs)
    assert torch.equal(p.detach(),before)

def test_image_profile_refuses_invalid_extent_and_zero_area():
    v,f,uv,tex,lights=image_fixture();cam=v.new_tensor([1.,0.,0.])
    with pytest.raises(ValueError):api().render_image(v,f,uv,tex,lights,cam,257)
    with pytest.raises(ValueError):api().render_image(v*0,f,uv,tex,lights,cam,4)

def test_five_joint_pose_layout_and_dynamic_contour_bins():
    d=fixture();p=d.template.new_tensor([0.,.1,0.,.2,0.,0.])
    full=api().flame_pose(p)
    assert full.shape==(5,3)
    assert torch.equal(full[0],p[:3]) and torch.equal(full[2],p[3:])
    assert not bool(full[[1,3,4]].any())
    for degrees,want in [(0,0),(10,10),(-10,49),(40,39),(-40,78)]:
        pose=p*0;pose[1]=degrees*math.pi/180
        assert api().contour_bin(api().flame_pose(pose))==want

def chain_fixture():
    v,f,uv,tex,lights=image_fixture();a=api();dev=v.device
    basis=v.new_zeros((3,4,3));basis[0,:,0]=v[:,0]*.05;basis[1,:,1]=v[:,1]*.04;basis[2,0,2]=.02
    posebasis=v.new_zeros((36,4,3));posebasis[10,1,2]=.01
    reg=v.new_tensor([[.25,.25,.25,.25],[.5,0.,0.,.5],[.5,.5,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]])
    parents=torch.tensor([-1,0,1,1,1],dtype=torch.int32,device=dev)
    weights=v.new_zeros((4,5));weights[:,0]=.7;weights[:,2]=.3
    model=a.FlameData(v,basis,posebasis,reg,parents,weights,f)
    texbasis=torch.stack([tex*.08,tex.flip(0)*.06])
    indices=f.new_tensor([0,0,1]);bary=v.new_tensor([[.2,.3,.5],[.3,.6,.1],[.1,.4,.5]])
    inputs=a.FittingInputs(model,uv,tex,texbasis,indices,bary)
    params=dict(shape=v.new_tensor([.1,-.08]),expression=v.new_tensor([.12]),pose=v.new_tensor([.02,.015,-.01,.025,.01,.005]),camera=v.new_tensor([1.,.01,-.01]),texture=v.new_tensor([.1,-.05]),lighting=lights.clone())
    params['lighting'][1:]=.025
    return inputs,{k:x.requires_grad_() for k,x in params.items()}

def test_complete_parameter_chain_loss_derivatives_before_optimization():
    inputs,p=chain_fixture();a=api();base=a.evaluate(inputs,p,size=4)
    target=base['colour'].detach()+.013
    target_lmk=base['landmarks'].detach()+base['landmarks'].new_tensor([.02,-.03])
    mask=target.new_ones((4,4))
    def loss(q):
        out=a.evaluate(inputs,q,size=4)
        return a.objective(out['colour'],target,mask,out['landmarks'],target_lmk,q)['total']
    grads=torch.autograd.grad(loss(p),tuple(p.values()))
    for (name,value),grad in zip(p.items(),grads):
        assert bool(torch.isfinite(grad).all()) and float(grad.abs().max())>1e-8,name
        direction=torch.arange(1,value.numel()+1,device=value.device,dtype=value.dtype).reshape(value.shape)/value.numel()
        expected=(grad*direction).sum()
        for h in (1e-5,5e-6):
            plus={k:v.detach() for k,v in p.items()};minus=dict(plus)
            plus[name]=value.detach()+h*direction;minus[name]=value.detach()-h*direction
            observed=(loss(plus)-loss(minus))/(2*h)
            torch.testing.assert_close(observed,expected,atol=1e-9,rtol=1e-7,msg=name)

def test_complete_chain_fresh_step_decreases_fixed_objective():
    inputs,p=chain_fixture();a=api();base=a.evaluate(inputs,p,size=4)
    target=base['colour'].detach()+.013;target_lmk=base['landmarks'].detach()+.01
    mask=target.new_ones((4,4))
    def loss():
        out=a.evaluate(inputs,p,size=4)
        return a.objective(out['colour'],target,mask,out['landmarks'],target_lmk,p)['total']
    report=a.optimize_stage(p,loss,iterations=8,learning_rate=.001,seconds=90)
    assert report['final_loss']<report['initial_loss']


def test_complete_chain_exact_cpu_cuda_output_and_gradient_bytes():
    import os
    if not torch.cuda.is_available():pytest.skip('CPU/CUDA conformance requires CUDA')
    old=os.environ.get('GFFX_GROUP4_DEVICE');saved=[]
    try:
        for device in ('cpu','cuda'):
            os.environ['GFFX_GROUP4_DEVICE']=device
            inputs,p=chain_fixture();out=api().evaluate(inputs,p,size=4)
            loss=out['colour'].square().sum()+out['landmarks'].square().sum()
            grads=torch.autograd.grad(loss,tuple(p.values()))
            tensors=[v for v in out.values() if isinstance(v,torch.Tensor)]+list(grads)
            saved.append([v.detach().cpu().contiguous().numpy().tobytes() for v in tensors])
        assert saved[0]==saved[1]
    finally:
        if old is None:os.environ.pop('GFFX_GROUP4_DEVICE',None)
        else:os.environ['GFFX_GROUP4_DEVICE']=old

@pytest.mark.parametrize('shapes',[((5,5023),(5023,3)),((1,5,4,4),(5,4,4))])
def test_fixed_model_matrix_products_and_vjps_real_extents(shapes):
    if not torch.cuda.is_available():pytest.skip('CUDA needed for cross-device control')
    a=torch.sin(torch.arange(math.prod(shapes[0]),dtype=torch.float64)).reshape(shapes[0])*.1
    b=torch.cos(torch.arange(math.prod(shapes[1]),dtype=torch.float64)).reshape(shapes[1])*.1
    outputs=[]
    for dev in ('cpu','cuda'):
        x=a.to(dev).detach().requires_grad_();y=b.to(dev).detach().requires_grad_();z=api()._model_matmul(x,y)
        cot=torch.linspace(-.2,.3,z.numel(),dtype=torch.float64).reshape(z.shape).to(dev)
        gx,gy=torch.autograd.grad((z*cot).sum(),(x,y));outputs.append(tuple(t.detach().cpu().numpy() for t in (z,gx,gy)))
    for cpu,gpu in zip(*outputs):assert cpu.tobytes()==gpu.tobytes()
    x=a.requires_grad_();y=b.requires_grad_();z=x@y;cot=torch.linspace(-.2,.3,z.numel(),dtype=torch.float64).reshape(z.shape)
    grads=torch.autograd.grad((z*cot).sum(),(x,y))
    for got,want in zip(outputs[0],(z,*grads)):torch.testing.assert_close(torch.from_numpy(got),want,atol=2e-13,rtol=2e-13)
    # Separate high-accuracy scalar sum, not another tensor contraction.
    aa=a.reshape(-1,*a.shape[-2:])[0];bb=b.reshape(-1,*b.shape[-2:])[0]
    exact=math.fsum(float(aa[0,k])*float(bb[k,0]) for k in range(aa.shape[-1]))
    assert abs(outputs[0][0].reshape(-1)[0]-exact)<2e-13
