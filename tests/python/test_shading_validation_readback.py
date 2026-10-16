"""Collect CUDA validation flags together while preserving ordered errors."""
import pytest,torch
from gffx.torch import shade,light

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_validation_retains_first_nonfinite_field_and_type_precedence(device):
 t=lambda v:torch.tensor(v,dtype=torch.float64,device=device)
 for index in range(6):
  fields={f'f{i}':t([1.,2.]) for i in range(6)}
  fields[f'f{index}'][0]=float('nan');fields['f5'][1]=float('inf')
  with pytest.raises(ValueError,match=f'f{index} must be finite'):shade._inputs(**fields)
 with pytest.raises(TypeError,match='last must be a tensor'):shade._inputs(first=t([float('nan')]),last=1.)
 shade._inputs(first=t([]),second=t([-0.,1e-300]))


def test_cuda_validation_uses_one_readback_for_all_finite_fields(monkeypatch):
 fields={f'f{i}':torch.ones(128,dtype=torch.float64,device='cuda') for i in range(6)}
 original_bool=torch.Tensor.__bool__;original_list=torch.Tensor.tolist;reads=[]
 def truth(x):
  if x.device.type=='cuda':reads.append('bool')
  return original_bool(x)
 def listing(x):
  if x.device.type=='cuda':reads.append('list')
  return original_list(x)
 monkeypatch.setattr(torch.Tensor,'__bool__',truth);monkeypatch.setattr(torch.Tensor,'tolist',listing)
 shade._inputs(**fields)
 assert len(reads)==1, 'finite fields require one CUDA host readback per validation call'

@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('field',['direction','irradiance'])
def test_light_packet_refuses_every_nonzero_invalid_placeholder(device,field):
 t=lambda v:torch.tensor(v,dtype=torch.float64,device=device)
 n=t([[0.,0.,1.]]);packet=dict(measure='delta',direction=t([[[0.,0.,1.],[0.,0.,0.]]]),irradiance=t([[[1.,1.,1.],[0.,0.,0.]]]),valid=torch.tensor([[True,False]],device=device))
 light._packet(n,packet)
 for axis in range(3):
  packet[field][0,1,axis]=1e-300
  with pytest.raises(ValueError,match='invalid sample placeholders must be zero'):light._packet(n,packet)
  packet[field][0,1,axis]=-0.
 light._packet(n,packet)

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_material_domain_flags_keep_ordered_errors_and_empty_geometry(device):
 from test_deferred_native_storage import scale
 from gffx.torch.material import _principled_values
 model,_,_=scale(device,16);params=next(p for p in model.prims if p.name=='skin').parameters
 t=lambda v:torch.tensor(v,dtype=torch.float64,device=device)
 n=t([]).reshape(0,3);v=n.clone();d=t([]).reshape(0,1,3)
 lo,hi=params['budget']._metadata['domain']['roughness']
 for albedo,f0,roughness,message in [(-1.,-1.,lo/2,'albedo outside'),(.2,-1.,lo/2,'f0 outside'),(.2,.04,lo/2,'roughness outside')]:
  with pytest.raises(ValueError,match=message):_principled_values(n,v,d,t([albedo]*3),t(roughness),t(f0),params['budget'])
 _principled_values(n,v,d,t([0.,-0.,1.]),t(hi),t(.04),params['budget'])
