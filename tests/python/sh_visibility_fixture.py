import math,torch
from gffx.torch import scene,material,light,render
from gffx.torch.camera import Camera
def exact(a,b,label):
 b=b.detach().cpu();a=a.detach().cpu()
 assert a.dtype==b.dtype and a.shape==b.shape,label
 x=torch.empty(a.numel(),dtype=a.dtype);y=torch.empty(b.numel(),dtype=b.dtype)
 x.copy_(a.reshape(-1));y.copy_(b.reshape(-1))
 assert torch.equal(x.view(torch.uint8),y.view(torch.uint8)),label
def sample(device,z,back=False,two=False,cull=None,scale=1.):
 t=lambda x:torch.tensor(x,dtype=torch.float64,device=device)
 zz=t(z).requires_grad_();albedo=t([.2,.4,.6]).requires_grad_()
 n=torch.stack((zz*0+1,zz*0,zz));n=n*(-1 if back else 1)
 v=t([[-2.,-2.,-2.],[2.,-2.,-2.],[0.,2.,-2.]])
 v[:,:2]*=scale
 f=torch.tensor([[0,2,1] if back else [0,1,2]],dtype=torch.int32,device=device)
 c=t([[math.sqrt(4*math.pi)]*3]+[[0.,0.,0.]]*8);c[2]=.2;c.requires_grad_()
 m=scene.Material('m',material.classic,dict(albedo=albedo,specular=t(0.),exponent=t(4.),two_sided=two))
 model=scene.Scene([scene.Mesh('t',v,f,material='m',attributes={'normals':scene.Attribute(n.expand(3,3).contiguous(),'vertex')}),m,scene.Light('sh',light.spherical_harmonics,dict(coefficients=c))])
 cam=Camera.from_intrinsics(t(1./scale),t(1./scale),t(.5),t(.5),1,1,.1,10.)
 result=render(model,cam,cull_mode=render.CULL_NONE if cull is None else cull,outputs=('colour','mask','face_index'))
 if back and cull==render.CULL_BACK:
  assert result['mask'].item()==0 and not bool(result['colour'].any())
  return result,()
 grad=torch.autograd.grad(result['colour'].sum(),(zz,albedo,c))
 # Coefficient2 is Y10: the analytic irradiance divided by pi is
 # 1 + .2 * sqrt(3/(4pi)) * (2/3) * unit_normal_z.
 k=.2*math.sqrt(3/(4*math.pi))*2/3
 expected=albedo.detach()*(1+k*z/math.sqrt(1+z*z))
 dz=albedo.detach().sum()*k/(1+z*z)**1.5
 active=(not back or two) and not (back and cull==render.CULL_BACK)
 torch.testing.assert_close(result['colour'].reshape(3),expected if active else expected*0,atol=1e-13,rtol=1e-13)
 torch.testing.assert_close(grad[0],dz if active else dz*0,atol=1e-12,rtol=1e-12)
 assert result['mask'].item()==(0 if back and cull==render.CULL_BACK else 1)
 return result,grad
