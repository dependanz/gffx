"""Installed complete material-pair family and its bounded first VJP."""
import pytest,torch
from test_principled_material_scalar import inputs,prepared,reference
from test_principled_native_operation import raw

def fixture():
 from gffx.torch.material import PrincipledBudget
 budget=PrincipledBudget.bundled('ggx-smith-loss-cpu-f64-v1',device='cpu',dtype=torch.float64)
 rows=list(inputs())[:2];parameters=torch.tensor([x for x,s in rows],dtype=torch.float64)
 curves=[prepared(budget,p) for p in parameters];cn,logs=curves[0][:2]
 v=torch.stack([c[2] for c in curves]);d=torch.stack([c[3] for c in curves]);m=torch.stack([c[4] for c in curves]);h=torch.tensor([c[5] for c in curves],dtype=torch.float64)
 n=parameters[:,:3].contiguous();view=parameters[:,3:6].contiguous();l=parameters[:,6:9,None].transpose(1,2).contiguous();a=parameters[:,9:12].contiguous();roughness=parameters[:,12].contiguous();f=parameters[:,13:].contiguous()
 return budget,parameters,(cn,logs,v,d,m,h,n,view,l,a,roughness,f)

def test_native_principled_pairs_values_all_first_inputs():
 budget,p,args=fixture();y,jac=torch.ops.gffx.shading_principled_pairs(*args,False,True)
 g=torch.tensor([[[.125,-.75,.3]]]*2,dtype=torch.float64);pg,lg=torch.ops.gffx.shading_principled_pairs_backward(jac,g)
 for row in range(len(p)):
  q=p[row].clone().requires_grad_();expected=reference(budget,q,False);assert torch.equal(raw(y[row].flatten()),raw(expected))
  want,=torch.autograd.grad(expected,q,g[row].flatten());actual=pg[row].clone();actual[6:9]=lg[row,0]
  torch.testing.assert_close(actual,want,atol=1e-10,rtol=1e-9)
 primal,empty=torch.ops.gffx.shading_principled_pairs(*args,False,False);assert empty.shape==(0,) and torch.equal(raw(primal),raw(y))

def test_native_principled_pairs_cpu_cuda_exact():
 budget,p,args=fixture();g=torch.tensor([[[.125,-.75,.3]]]*2,dtype=torch.float64)
 y,j=torch.ops.gffx.shading_principled_pairs(*args,False,True);pg,lg=torch.ops.gffx.shading_principled_pairs_backward(j,g)
 stream=torch.cuda.Stream()
 with torch.cuda.stream(stream):
  cy,cj=torch.ops.gffx.shading_principled_pairs(*(t.cuda() for t in args),False,True);cp,cl=torch.ops.gffx.shading_principled_pairs_backward(cj,g.cuda())
 stream.synchronize()
 for a,b in ((y,cy),(j,cj),(pg,cp),(lg,cl)):assert torch.equal(raw(a),raw(b))

def test_native_principled_pairs_limits_domains_empty():
 budget,p,args=fixture()
 for slot,bad in ((6,torch.zeros_like(args[6])),(10,args[10]*0),(9,args[9]+2),(11,args[11]+2)):
  candidate=list(args);candidate[slot]=bad
  with pytest.raises(RuntimeError):torch.ops.gffx.shading_principled_pairs(*candidate,False,False)
 def sized(points,lights):
  c=list(args)
  for k in range(2,12):c[k]=args[k][:1].expand((points,)+args[k].shape[1:]).contiguous()
  c[8]=c[8].expand(points,lights,3).contiguous()
  if points==0:
   # Empty expand may preserve zero strides even after contiguous(); the raw C
   # ABI requires positive dense strides. Materialize an actual empty owner.
   for k in range(2,12):c[k]=c[k].new_empty(c[k].shape)
  return c
 y,j=torch.ops.gffx.shading_principled_pairs(*sized(8,128),False,False);assert y.shape==(8,128,3) and j.numel()==0
 for points,lights,grad in ((9,1,False),(5,1,True),(4,257,True),(8,129,False)):
  with pytest.raises(RuntimeError):torch.ops.gffx.shading_principled_pairs(*sized(points,lights),False,grad)
 y,j=torch.ops.gffx.shading_principled_pairs(*sized(0,1),False,True);assert y.shape==(0,1,3) and j.shape==(0,1,3,16)
 pg,lg=torch.ops.gffx.shading_principled_pairs_backward(j,y);assert pg.shape==(0,16) and lg.shape==(0,1,3)


def test_public_material_dispatches_complete_native_pairs(monkeypatch):
 from gffx.torch.material import principled
 budget,p,args=fixture();calls=[];original=torch.ops.gffx.shading_principled_pairs
 def observed(*args):calls.append((args[6].shape[0],args[8].shape[1],args[-1]));return original(*args)
 monkeypatch.setattr(torch.ops.gffx,'shading_principled_pairs',observed)
 n=args[6].clone().requires_grad_();a=args[9].clone().requires_grad_();r=args[10].clone().requires_grad_()
 result=principled(n,args[7],args[8],albedo=a,roughness=r,f0=args[11],budget=budget)
 assert calls==[(2,1,True)]
 g=torch.autograd.grad(result.sum(),(n,a,r));assert all(torch.isfinite(x).all() for x in g)
 calls.clear()
 with torch.no_grad():principled(n,args[7],args[8],albedo=a,roughness=r,f0=args[11],budget=budget)
 assert calls==[(2,1,False)]


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('lights', [3, 256])
def test_public_native_shared_inputs_match_reference(device, lights, record_property):
 from gffx.torch.material import principled, _principled_reference
 budget,_,_=fixture()
 if device=='cuda':budget=budget.to(device='cuda',dtype=torch.float64,allow_unverified=True)
 def tensor(x):return torch.tensor(x,dtype=torch.float64,device=device)
 # Four owners share albedo/roughness/F0 and light vectors. Signed, unequal
 # cotangents exercise both RGB contraction and odd/even point/light reduction.
 n=tensor([[.1,.2,1.],[-.2,.1,.9],[.3,-.1,1.2],[.2,.2,1.]])
 v=tensor([[.2,.1,1.],[.1,-.3,1.],[.2,.1,1.1],[-.1,.2,1.]])
 l=tensor([[.1+(j%7)*.03,-.2+(j%5)*.07,1.] for j in range(lights)])
 host=(n,v,l,tensor([.23,.47,.61]),tensor(.63),tensor([.04,.15,.7]))
 cot=(torch.arange(4*lights*3,dtype=torch.float64,device=device).reshape(4,lights,3)%13-6)/17
 rows=[]
 for call in (principled,_principled_reference):
  args=[x.clone().requires_grad_() for x in host]
  y=call(*args[:3],albedo=args[3],roughness=args[4],f0=args[5],budget=budget,allow_experimental_budget=device=='cuda')
  rows.append((y,torch.autograd.grad(y,args,cot)))
 difference=(rows[0][0]-rows[1][0]).detach()
 record_property('reference_value_max_absolute',float(difference.abs().max()))
 record_property('reference_nonzero_mismatch',int(((rows[0][0]==0)!=(rows[1][0]==0)).sum()))
 record_property('reference_gradient_maxima',str([float((a-b).abs().max()) for a,b in zip(rows[0][1],rows[1][1])]))
 indices=(difference!=0).nonzero()
 if len(indices):
  i,j,k=indices[0].tolist()
  a=[x.detach().clone() for x in host]
  single=[]
  for call in (principled,_principled_reference):
   y=call(a[0][i:i+1],a[1][i:i+1],a[2][j:j+1],albedo=a[3],roughness=a[4],f0=a[5],budget=budget,allow_experimental_budget=device=='cuda')
   single.append(y.flatten().tolist())
  record_property('first_difference',str(dict(index=[i,j,k],native=rows[0][0][i,j].detach().tolist(),reference=rows[1][0][i,j].detach().tolist(),single=single)))
 # Exact device identity is checked separately below. The original eager CPU
 # Smith sqrt differs from correctly rounded sqrt for a captured operand. This
 # new reference-bridge assumption is preserved as a failed diagnostic; physical
 # algebra tolerances and exact zero support remain unchanged.
 assert torch.equal(rows[0][0]==0,rows[1][0]==0)
 torch.testing.assert_close(rows[0][0],rows[1][0],atol=1e-10,rtol=1e-9)
 if device=='cuda':
  cpu=budget.to(device='cpu',dtype=torch.float64,allow_unverified=True)
  args=[x.detach().cpu().requires_grad_() for x in host]
  y=principled(*args[:3],albedo=args[3],roughness=args[4],f0=args[5],budget=cpu,allow_experimental_budget=True)
  gradients=torch.autograd.grad(y,args,cot.cpu())
  for a,b in zip((y,*gradients),(rows[0][0],*rows[0][1])):
   assert torch.equal(raw(a),raw(b))
 for actual,expected in zip(rows[0][1],rows[1][1]):
  assert torch.isfinite(actual).all() and bool((actual!=0).any())
  torch.testing.assert_close(actual,expected,atol=1e-10,rtol=1e-9)


def test_native_material_stage_diagnostic(record_property):
 import os,ctypes as C,json,hashlib
 from pathlib import Path
 from gffx.torch import material,shade
 from test_principled_material_scalar import prepared
 name=os.environ.get('GFFX_PRINCIPLED_STAGE_PROBE')
 if not name:pytest.skip('explicit standalone trace probe required')
 budget,_,_=fixture()
 p=torch.tensor([.1,.2,1.,.2,.1,1.,.1+(12%7)*.03,-.2+(12%5)*.07,1.,.23,.47,.61,.63,.04,.15,.7],dtype=torch.float64)
 data=prepared(budget,p);lib=C.CDLL(name);ptr=C.POINTER(C.c_double)
 lib.principled_test_material_point.argtypes=[C.c_int]+[ptr]*5+[C.c_double,ptr,C.c_int,ptr]
 arrays=[(C.c_double*t.numel())(*t.tolist()) for t in data[:5]];inp=(C.c_double*16)(*p.tolist());out=(C.c_double*99)()
 assert lib.principled_test_material_point(len(data[0]),*arrays,data[5],inp,0,out)==0
 n,v,l=[material._unit(x.reshape(1,3),'diagnostic') for x in (p[:3],p[3:6],p[6:9])];h=material._unit(v+l,'half')
 nv,nl,nh,vh=[material._dot(a,b) for a,b in ((n,v),(n,l),(n,h),(v,h))]
 distribution=shade.ggx_distribution(nh,p[12]);visibility=shade.smith_visibility(nv,nl,p[12]);fresnel=shade.fresnel_schlick(vh,p[13:])
 vl,mean,ll=budget._material_losses(nv,nl,p[12:13],p[13:].reshape(1,3),torch.tensor([0]))
 specular=(distribution*visibility)[:,None]*fresnel;matte=p[9:12]*vl*ll/(torch.pi*mean)
 groups=[('normal',51,n),('view',54,v),('light',57,l),('half',60,h),('cosines',63,torch.cat((nv,nl,nh,vh))),('lobes',67,torch.cat((distribution,visibility))),('view_loss',81,vl),('light_loss',84,ll),('mean',87,mean),('fresnel',90,fresnel),('specular',93,specular),('matte',96,matte)]
 rows=[]
 for name,start,t in groups:
  a=torch.tensor(list(out)[start:start+t.numel()],dtype=torch.float64);b=t.flatten()
  rows.append(dict(stage=name,native=a.tolist(),reference=b.tolist(),max_abs=float((a-b).abs().max()),exact=torch.equal(raw(a),raw(b))))
 r4=p[12].square().square();a=nv.square()*(1-r4)+r4;b=nl.square()*(1-r4)+r4
 denominator=nl*torch.sqrt(a)+nv*torch.sqrt(b)
 import math
 scalar_den=float(nl)*math.sqrt(float(a))+float(nv)*math.sqrt(float(b))
 record_property('smith_division',json.dumps(dict(inputs=[float(a),float(b)],torch_roots=[float(torch.sqrt(a)),float(torch.sqrt(b))],scalar_roots=[math.sqrt(float(a)),math.sqrt(float(b))],denominator=float(denominator),scalar_denominator=scalar_den,scalar_division=.5/scalar_den,reverse_division=float(.5/denominator),tensor_division=float(p.new_tensor(.5)/denominator))))
 record_property('trace_rows',json.dumps(rows));record_property('trace_probe_sha256',hashlib.sha256(Path(os.environ['GFFX_PRINCIPLED_STAGE_PROBE']).read_bytes()).hexdigest())
 assert all(torch.isfinite(t).all() for name,start,t in groups)


def test_native_material_captured_sqrt_independent_oracle(record_property):
 import math,json
 from decimal import Decimal,localcontext
 from tools.principled_material_oracle import material_pair
 from gffx.torch.material import principled
 budget,_,_=fixture();D=Decimal.from_float
 p=torch.tensor([.1,.2,1.,.2,.1,1.,.1+(12%7)*.03,-.2+(12%5)*.07,1.,.23,.47,.61,.63,.04,.15,.7],dtype=torch.float64,requires_grad=True)
 y=principled(p[:3].reshape(1,3),p[3:6].reshape(1,3),p[6:9].reshape(1,3),albedo=p[9:12],roughness=p[12],f0=p[13:],budget=budget).flatten()
 direction=torch.linspace(-.17,.23,16,dtype=torch.float64)
 g=torch.stack([(torch.autograd.grad(y[j],p,retain_graph=True)[0]*direction).sum() for j in range(3)])
 with localcontext() as context:
  context.prec=90
  operand=.9298300710805834;exact=D(operand).sqrt();native=math.sqrt(operand);legacy=float(torch.sqrt(torch.tensor(operand,dtype=torch.float64)))
  assert D(native)-D(math.nextafter(native,-math.inf))>0
  # Nearest float and its two neighbours establish correct rounding here.
  assert all(abs(D(native)-exact)<abs(D(math.nextafter(native,z))-exact) for z in (-math.inf,math.inf))
  record_property('captured_sqrt',json.dumps(dict(operand=operand,native=native,legacy=legacy,exact=str(exact))))
  data={k:v.tolist() for k,v in budget._data.items()};point=[D(v) for v in p.detach().tolist()];vector=[D(v) for v in direction.tolist()]
  expected=material_pair(data,point,False)
  torch.testing.assert_close(y,torch.tensor([float(v) for v in expected],dtype=torch.float64),atol=1e-10,rtol=1e-9)
  rows=[]
  for step in (Decimal('1e-5'),Decimal('5e-6')):
   plus=material_pair(data,[v+step*d for v,d in zip(point,vector)],False);minus=material_pair(data,[v-step*d for v,d in zip(point,vector)],False)
   fd=torch.tensor([float((a-b)/(2*step)) for a,b in zip(plus,minus)],dtype=torch.float64)
   torch.testing.assert_close(g,fd,atol=1e-10,rtol=1e-9)
   rows.append(dict(step=str(step),native=g.tolist(),independent=fd.tolist()))
  record_property('captured_pair_independent',json.dumps(rows))
