"""Independent values and strict device first-VJP controls for soft coverage."""
from decimal import Decimal,localcontext
import math,pytest,torch
from test_deferred_cuda import exact

@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_soft_sigmoid_independent_values_and_device_gradients(dtype,record_property):
 from gffx.torch._reference_sigmoid import sigmoid
 values=torch.cat((torch.linspace(-40,40,513,dtype=dtype),torch.tensor([-math.inf,-1000.,-746.,-745.,-100.,0.,100.,745.,1000.,math.inf],dtype=dtype)))
 rows=[]
 for device in ('cpu','cuda'):
  x=values.to(device).requires_grad_();y=sigmoid(x);cot=(torch.arange(len(x),dtype=dtype,device=device)%13-6)/8
  g,=torch.autograd.grad(y,x,cot);rows.append((y.detach().cpu(),g.detach().cpu()))
 exact(rows[0][0],rows[1][0],'soft sigmoid');exact(rows[0][1],rows[1][1],'soft sigmoid first VJP')
 bound=2**-23 if dtype==torch.float32 else 32*2**-53
 with localcontext() as ctx:
  ctx.prec=100
  for i,(x,y) in enumerate(zip(values.tolist(),rows[0][0].tolist())):
   reference=Decimal(0) if x==-math.inf else Decimal(1) if x==math.inf else 1/(1+(-Decimal.from_float(x)).exp())
   assert abs(Decimal.from_float(y)-reference)<=Decimal.from_float(bound)
   gradient=reference*(1-reference)*Decimal.from_float(float((i%13-6)/8))
   assert abs(Decimal.from_float(float(rows[0][1][i]))-gradient)<=Decimal.from_float(bound)
 assert rows[0][0][0]>=0 and rows[0][0].min()==0 and rows[0][0].max()==1
 x=torch.tensor([-.7,.1,2.3],dtype=dtype,requires_grad=True)
 if dtype==torch.float64:assert torch.autograd.gradcheck(sigmoid,(x,),eps=1e-6,atol=1e-10,rtol=1e-9)
 record_property('sigmoid_samples',len(values))
