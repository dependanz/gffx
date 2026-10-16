"""Grouped renderer equivalence, shared first gradients and retained-input memory."""
import pytest,torch
from test_principled_native_operation import raw


def case(points,lights,device):
 from gffx.torch.material import PrincipledBudget
 budget=PrincipledBudget.bundled('ggx-smith-loss-cpu-f64-v1',device='cpu',dtype=torch.float64)
 if device=='cuda':budget=budget.to(device='cuda',dtype=torch.float64,allow_unverified=True)
 def t(x):return torch.tensor(x,dtype=torch.float64,device=device,requires_grad=True)
 n=t([[.1+(i%3)*.04,.2,1.] for i in range(points)]);v=t([[.2,-.1+(i%5)*.03,1.] for i in range(points)])
 l=t([[.1+(j%7)*.03,-.2+(j%5)*.07,1.] for j in range(lights)])
 return budget,(n,v,l,t([.23,.47,.61]),t(.63),t([.04,.15,.7]))


def evaluate(budget,args,grouped):
 from gffx.torch import _deferred as d
 n,v,l,a,r,f=args;directions=l[None].expand(len(n),-1,-1)
 params=dict(albedo=a,roughness=r,f0=f,budget=budget,allow_experimental_budget=n.device.type=='cuda')
 if grouped:return d._principled_blocks(n,v,directions,params)
 return torch.cat([d._principled_recompute(n[i:i+4],v[i:i+4],directions[i:i+4],params) for i in range(0,len(n),4)])


def test_renderer_dispatches_grouped_native_batch(monkeypatch):
 budget,args=case(9,3,'cpu');calls=[];original=torch.ops.gffx.shading_principled_shared_groups
 def observed(*values):calls.append(values[6].shape[0]);return original(*values)
 monkeypatch.setattr(torch.ops.gffx,'shading_principled_shared_groups',observed)
 y=evaluate(budget,args,True);torch.autograd.grad(y.sum(),args)
 assert any(p>4 for p in calls),calls


def ordered_reference(budget,args,cot):
 from gffx.torch import _deferred as d
 n,v,directions,a,r,f=args;points=len(n);ys=[];partials=[]
 for i in range(0,points,4):
  local=[x[i:i+4].detach().requires_grad_() for x in (n,v,directions)]+[x.detach().requires_grad_() for x in (a,r,f)]
  params=dict(albedo=local[3],roughness=local[4],f0=local[5],budget=budget,allow_experimental_budget=n.device.type=='cuda')
  y=d._principled_recompute(*local[:3],params);ys.append(y.detach());partials.append(torch.autograd.grad(y,local,cot[i:i+4]))
 gradients=[torch.zeros_like(x) for x in args]
 for index in reversed(range(len(partials))):
  for field,part in enumerate(partials[index]):
   if field<3:gradients[field][index*4:index*4+len(part)].add_(part)
   else:gradients[field].add_(part)
 return (torch.cat(ys),*gradients)


@pytest.mark.parametrize('points,lights',[(5,3),(9,5),(129,3),(4,256)])
@pytest.mark.parametrize('device',['cpu','cuda'])
def test_grouped_renderer_preserves_values_and_shared_first_gradients(points,lights,device):
 from gffx.torch import _deferred as d
 rows=[]
 for current in (('cpu','cuda') if device=='cuda' else ('cpu',)):
  budget,base=case(points,lights,current);n,v,l,a,r,f=base
  directions=l.detach()[None].expand(points,-1,-1).clone().requires_grad_();args=(n,v,directions,a,r,f)
  cot=(torch.arange(points*lights*3,dtype=torch.float64,device=current).reshape(points,lights,3)%17-8)/19
  expected=ordered_reference(budget,args,cot)
  y=d._principled_blocks(n,v,directions,dict(albedo=a,roughness=r,f0=f,budget=budget,allow_experimental_budget=current=='cuda'))
  actual=(y,*torch.autograd.grad(y,args,cot))
  for x,z in zip(actual,expected):assert torch.equal(raw(x),raw(z)),float((x-z).detach().abs().max())
  rows.append(actual)
 if len(rows)==2:
  for x,z in zip(*rows):assert torch.equal(raw(x),raw(z)),float((x.cpu()-z.cpu()).detach().abs().max())



def test_grouped_renderer_saves_only_original_inputs_and_refuses_before_prepare(monkeypatch):
 budget,args=case(129,3,'cpu');saved=[]
 roots={x.untyped_storage().data_ptr() for x in args}
 with torch.autograd.graph.saved_tensors_hooks(lambda t:(saved.append(t),t)[1],lambda t:t):
  y=evaluate(budget,args,True)
 assert saved and all(t.untyped_storage().data_ptr() in roots for t in saved)
 torch.autograd.grad(y.sum(),args)
 from gffx.torch import _deferred as d
 calls=[];original=budget._prepare_curve
 def observed(*a,**kw):calls.append(True);return original(*a,**kw)
 monkeypatch.setattr(type(budget),'_prepare_curve',lambda self,*a,**kw:observed(*a,**kw))
 n,v,l,a,r,f=args
 with pytest.raises(ValueError,match='workspace'):
  d._principled_blocks(n,v,l[None].expand(len(n),-1,-1),dict(albedo=a,roughness=r,f0=f,budget=budget,workspace_limit_bytes=0))
 assert not calls


def test_grouped_renderer_cuda_fresh_graphs_keep_bounded_live_storage(record_property):
 import gc,json
 budget,base=case(128,8,'cuda');live=[];peaks=[]
 for repeat in range(21):
  args=tuple(x.detach().clone().requires_grad_() for x in base)
  torch.cuda.reset_peak_memory_stats();before=torch.cuda.memory_allocated()
  y=evaluate(budget,args,True);g=torch.autograd.grad(y.sum(),args)
  assert all(torch.isfinite(x).all() for x in (y,*g))
  torch.cuda.synchronize();peaks.append(torch.cuda.max_memory_allocated())
  del y,g,args;gc.collect();torch.cuda.synchronize();live.append(torch.cuda.memory_allocated())
 assert len(set(live[1:]))==1 and max(peaks)<=64*1024**2
 record_property('grouped_repeated_memory',json.dumps(dict(fresh_graphs=20,postwarm_live_bytes=live[1:],peak_allocated_bytes=max(peaks))))



def test_grouped_shared_directional_derivative_two_steps():
 from gffx.torch import _deferred as d
 budget,base=case(5,3,'cpu');n,v,l,a,r,f=base;directions=l.detach()[None].expand(5,-1,-1).clone().requires_grad_();args=(n,v,directions,a,r,f)
 vectors=tuple(torch.linspace(-.13,.17,x.numel(),dtype=x.dtype).reshape(x.shape) for x in args)
 cot=(torch.arange(45,dtype=torch.float64).reshape(5,3,3)%11-5)/13
 def value(values):
  n,v,l,a,r,f=values
  return d._principled_blocks(n,v,l,dict(albedo=a,roughness=r,f0=f,budget=budget))
 y=value(args);grads=torch.autograd.grad(y,args,cot);analytic=sum((g*v).sum() for g,v in zip(grads,vectors))
 for step in (1e-5,5e-6):
  with torch.no_grad():
   plus=value(tuple(x+step*v for x,v in zip(args,vectors)));minus=value(tuple(x-step*v for x,v in zip(args,vectors)))
   numerical=((plus-minus)*cot).sum()/(2*step)
  torch.testing.assert_close(analytic,numerical,atol=1e-10,rtol=1e-9)


@pytest.mark.parametrize('shape',[(),(4,),(1,4)])
def test_native_material_shared_roughness_uses_fixed_point_reduction(shape):
 from gffx.torch import material
 rows=[]
 for device in ('cpu','cuda'):
  budget,base=case(4,5,device);n,v,l,a,r,f=base
  value=r.detach().expand(shape).clone().requires_grad_()
  point_r=r.detach().expand(4).clone().requires_grad_()
  cot=(torch.arange(60,dtype=torch.float64,device=device).reshape(4,5,3)%17-8)/19
  def evaluate(rough):
   return material.principled(n,v,l,albedo=a,roughness=rough,f0=f,budget=budget,allow_experimental_budget=device=='cuda')
  point_y=evaluate(point_r);point_grad,=torch.autograd.grad(point_y,point_r,cot)
  y=evaluate(value);gradient,=torch.autograd.grad(y,value,cot)
  expected=((point_grad[0]+point_grad[1])+(point_grad[2]+point_grad[3])) if not shape else point_grad.reshape(shape)
  assert torch.equal(raw(y),raw(point_y))
  assert torch.equal(raw(gradient),raw(expected))
  rows.append((y,gradient))
 for x,z in zip(*rows):assert torch.equal(raw(x),raw(z))


@pytest.mark.parametrize('soft',[0.,.2])
def test_grouped_complete_render_shared_material_and_light_vjps(soft):
 from gffx.torch import scene,material,light,render
 from gffx.torch.camera import Camera
 rows=[]
 for device in ('cpu','cuda'):
  budget,_=case(1,1,device)
  def t(x):return torch.tensor(x,dtype=torch.float64,device=device)
  values=tuple(t(x).requires_grad_() for x in ([.23,.47,.61],.63,[.04,.15,.7],[[.2,-.3,1.],[.4,.2,1.]],[[1.,.7,.4],[.2,.4,.9]]))
  a,r,f,directions,irradiance=values
  mat=scene.Material('p',material.principled,dict(albedo=a,roughness=r,f0=f,budget=budget,allow_experimental_budget=device=='cuda'))
  lamp=scene.Light('key',light.directional,dict(directions=directions,irradiance=irradiance))
  vertices=t([[-2.,-2.,-2.],[2.,-2.,-2.],[2.,2.,-2.],[-2.,2.,-2.]])
  faces=torch.tensor([[0,1,2],[0,2,3]],dtype=torch.int32,device=device)
  model=scene.Scene([scene.Mesh('quad',vertices,faces,material='p'),mat,lamp])
  camera=Camera.from_intrinsics(t(2.),t(2.),t(1.5),t(1.5),3,3,.1,10.)
  y=render(model,camera,soft=soft)['colour'];cot=(torch.arange(y.numel(),device=device,dtype=y.dtype).reshape_as(y)%11-5)/13
  rows.append((y,*torch.autograd.grad(y,values,cot)))
 for field,(x,z) in enumerate(zip(*rows)):
  assert torch.equal(raw(x),raw(z)),(field,float((x.cpu()-z.cpu()).detach().abs().max()))
