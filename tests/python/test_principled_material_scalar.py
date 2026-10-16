"""Standalone complete Principled pair: preserved forward and first-input Jacobian."""
import ctypes as C,os,json,hashlib
from pathlib import Path
import pytest,torch
from test_principled_native_operation import raw

@pytest.fixture
def point_probe(record_property):
 name=os.environ.get('GFFX_PRINCIPLED_MATERIAL_PROBE')
 if not name:pytest.skip('explicit standalone material library required')
 path=Path(name);lib=C.CDLL(str(path));ptr=C.POINTER(C.c_double)
 lib.principled_test_material_point.argtypes=[C.c_int]+[ptr]*5+[C.c_double,ptr,C.c_int,ptr]
 lib.principled_test_material_point.restype=C.c_int
 record_property('material_probe',json.dumps(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())))
 return lib

def inputs():
 base=[.1,.2,1., .2,.1,1., .3,.4,1., .2,.4,.6, .427, .07,.47,.93]
 yield base,False
 for r in (.00179,.213,.827,1.):
  p=base.copy();p[12]=r;yield p,False
 for s in (1e-30,1e30):
  p=base.copy();p[:9]=[v*s for v in p[:9]];yield p,False
 for n,v,l,side in (([0.,0.,1.],[0.,0.,1.],[0.,0.,1.],False),([0.,0.,-2.],[0.,0.,1.],[.1,.2,1.],True),([0.,0.,1.],[0.,0.,-1.],[0.,0.,1.],False),([0.,0.,1.],[0.,0.,1.],[0.,0.,-1.],False),([0.,0.,1.],[1.,0.,.013],[0.,1.,.019],False)):
  p=base.copy();p[:9]=n+v+l;yield p,side

def prepared(budget,p):
 r=p[12:13].detach();cn,coeff,u=budget._prepare_curve(r);x=budget._data['_clog'];v,d=torch.ops.gffx.shading_loss_curve(x,coeff,u)
 q,m=torch.ops.gffx.shading_loss_query(cn[1:],x,budget._data['_weights'],v,d,r.new_zeros((1,)),torch.zeros(1,dtype=torch.int32))
 i=(torch.searchsorted(budget._data['roughness_nodes'],r,right=True)-1).clamp(max=len(budget._data['_rlog'])-2)
 h=budget._data['_rlog'][i+1]-budget._data['_rlog'][i]
 return cn[1:],x,v.flatten(),d.flatten(),m.flatten(),float(h)

def evaluate(lib,budget,p,side):
 args=prepared(budget,p);arrays=[(C.c_double*t.numel())(*t.tolist()) for t in args[:5]];x=(C.c_double*16)(*p.detach().tolist());out=(C.c_double*51)()
 assert lib.principled_test_material_point(len(args[0]),*arrays,args[5],x,int(side),out)==0
 return torch.tensor(list(out),dtype=torch.float64)

def reference(budget,p,side):
 from gffx.torch.material import _principled_reference as principled
 return principled(p[:3].reshape(1,3),p[3:6].reshape(1,3),p[6:9].reshape(1,3),albedo=p[9:12],roughness=p[12],f0=p[13:16],budget=budget,two_sided=side).flatten()

def test_native_material_pair_values_and_all_input_first_jacobian(point_probe,record_property):
 from gffx.torch.material import PrincipledBudget
 budget=PrincipledBudget.bundled('ggx-smith-loss-cpu-f64-v1',device='cpu',dtype=torch.float64);count=0
 for values,side in inputs():
  p=torch.tensor(values,dtype=torch.float64,requires_grad=True);out=evaluate(point_probe,budget,p,side);expected=reference(budget,p,side)
  assert torch.equal(raw(out[:3]),raw(expected)),(values,side,out[:3],expected)
  jac=out[3:].reshape(3,16)
  for cot in (p.new_tensor([1.,1.,1.]),p.new_tensor([.125,-.75,.3])):
   g,=torch.autograd.grad(expected,p,cot,retain_graph=True);actual=(jac[0]*cot[0]+jac[1]*cot[1])+jac[2]*cot[2]
   torch.testing.assert_close(actual,g,atol=1e-10,rtol=1e-9)
  count+=1
 record_property('native_material_pair_rows',str(count))

def test_native_material_pair_two_step_differences(point_probe,record_property):
 from decimal import Decimal,localcontext
 from tools.principled_material_oracle import material_pair
 from gffx.torch.material import PrincipledBudget
 budget=PrincipledBudget.bundled('ggx-smith-loss-cpu-f64-v1',device='cpu',dtype=torch.float64)
 values,side=next(inputs());p=torch.tensor(values,dtype=torch.float64);out=evaluate(point_probe,budget,p,side);jac=out[3:].reshape(3,16);direction=torch.linspace(-.17,.23,16,dtype=torch.float64);want=(jac*direction).sum(1)
 data={k:v.tolist() for k,v in budget._data.items()};rows=[]
 with localcontext() as ctx:
  ctx.prec=90;D=Decimal.from_float;point=list(map(D,p.tolist()));vector=list(map(D,direction.tolist()))
  primal=material_pair(data,point,side);torch.testing.assert_close(torch.tensor([float(v) for v in primal],dtype=torch.float64),out[:3],atol=1e-10,rtol=1e-9)
  for step in (Decimal('1e-5'),Decimal('5e-6')):
   a=material_pair(data,[x+step*d for x,d in zip(point,vector)],side);b=material_pair(data,[x-step*d for x,d in zip(point,vector)],side)
   difference=torch.tensor([float((x-y)/(2*step)) for x,y in zip(a,b)],dtype=torch.float64)
   torch.testing.assert_close(difference,want,atol=1e-10,rtol=1e-9)
   rows.append(dict(step=str(step),difference=difference.tolist(),native=want.tolist(),absolute_error=(difference-want).abs().tolist()))
 record_property('decimal_two_step_rows',json.dumps(rows))


def test_native_material_difference_rounding_diagnostic(point_probe,record_property):
 from gffx.torch.material import PrincipledBudget
 budget=PrincipledBudget.bundled('ggx-smith-loss-cpu-f64-v1',device='cpu',dtype=torch.float64)
 values,side=next(inputs());p=torch.tensor(values,dtype=torch.float64,requires_grad=True);out=evaluate(point_probe,budget,p,side);direction=torch.linspace(-.17,.23,16,dtype=torch.float64);native=(out[3:].reshape(3,16)*direction).sum(1)
 value=reference(budget,p,side);analytic=torch.stack([(torch.autograd.grad(value[j],p,retain_graph=True)[0]*direction).sum() for j in range(3)])
 torch.testing.assert_close(native,analytic,atol=1e-10,rtol=1e-9)
 rows=[]
 for step in (1e-5,5e-6,2.5e-6):
  difference=(reference(budget,p.detach()+step*direction,side)-reference(budget,p.detach()-step*direction,side))/(2*step)
  rows.append(dict(step=step,difference=difference.tolist(),native=native.tolist(),installed_analytic=analytic.tolist(),native_absolute_error=(difference-native).abs().tolist(),installed_absolute_error=(difference-analytic).abs().tolist()))
 path=os.environ.get('GFFX_MATERIAL_DIFFERENCE_ROWS')
 if path:
  path=Path(path);assert not path.exists();path.write_text(json.dumps(rows,indent=2))
 record_property('difference_diagnostic',json.dumps(rows))


def test_native_material_pair_cuda_exact_jacobian(point_probe,record_property):
 from gffx.torch.material import PrincipledBudget
 name=os.environ.get('GFFX_PRINCIPLED_MATERIAL_CUDA_PROBE')
 if not name:pytest.skip('explicit standalone CUDA material library required')
 path=Path(name);lib=C.CDLL(str(path));lib.principled_test_material_point_cuda.argtypes=[C.c_int]+[C.c_void_p]*5+[C.c_double,C.c_void_p,C.c_int]+[C.c_void_p]*3;lib.principled_test_material_point_cuda.restype=C.c_int
 budget=PrincipledBudget.bundled('ggx-smith-loss-cpu-f64-v1',device='cpu',dtype=torch.float64);stream=torch.cuda.Stream();count=0
 for values,side in inputs():
  p=torch.tensor(values,dtype=torch.float64);expected=evaluate(point_probe,budget,p,side);args=prepared(budget,p)
  with torch.cuda.stream(stream):
   tensors=[x.cuda() for x in args[:5]];parameters=p.cuda();out=torch.empty(51,dtype=torch.float64,device='cuda');status=torch.full((1,),99,dtype=torch.int32,device='cuda')
   assert lib.principled_test_material_point_cuda(len(args[0]),*[x.data_ptr() for x in tensors],args[5],parameters.data_ptr(),int(side),out.data_ptr(),status.data_ptr(),stream.cuda_stream)==0
  stream.synchronize();assert status.item()==0
  assert torch.equal(raw(out),raw(expected)),(values,side,(out.cpu()-expected).abs().max())
  count+=1
 record_property('native_material_cuda_rows',str(count));record_property('material_cuda_probe',json.dumps(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())))
