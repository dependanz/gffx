"""Frozen original raster bytes across cooperative blocks and face chunks."""
import os,json,hashlib
from pathlib import Path
import pytest,torch

@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_cooperative_raster_preserves_faces_blocks_and_partial_lanes(device,dtype):
 import gffx.torch
 capture=os.environ.get('GFFX_COOPERATIVE_CAPTURE');root=Path(capture or os.environ.get('GFFX_COOPERATIVE_BASELINE') or pytest.skip('requires pre-change byte fixture'))
 gen=torch.Generator().manual_seed(202102)
 vertices=(torch.rand((3*513,3),generator=gen,dtype=torch.float64)*2-1).to(dtype);vertices[:,2]*=.1
 faces=torch.arange(len(vertices),dtype=torch.int32).reshape(-1,3)
 faces[256]=faces[0];faces[512]=faces[0] # ties across consecutive face chunks
 rows=[]
 for count in (0,1,255,256,257,513):
  v=vertices.to(device);f=faces[:count].to(device);vo=torch.tensor([0,len(v)],dtype=torch.int32,device=device);fo=torch.tensor([0,count],dtype=torch.int32,device=device)
  for region in ((0,0,17,31),(2,3,11,23),(16,30,1,1)):
   for blur,cull in ((0.,1),(.6,2),(2.,3)):
    try:out=torch.ops.gffx.rasterize_window(v,f,vo,fo,17,31,*region,3,blur,cull,1e-12)
    except RuntimeError as error:
     if count!=0 or device!='cuda':raise
     rows.append(dict(count=count,region=region,blur=blur,cull=cull,refusal=str(error)));continue
    rows.append(dict(count=count,region=region,blur=blur,cull=cull,sha256=[hashlib.sha256(x.cpu().contiguous().numpy().tobytes()).hexdigest() for x in out]))
 name=device+'-'+str(dtype).split('.')[-1]+'.json'
 if capture:
  root.mkdir(exist_ok=True)
  with (root/name).open('x') as h:json.dump(rows,h)
 else:assert json.loads(json.dumps(rows))==json.loads((root/name).read_bytes())

@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_cooperative_raster_awkward_spatial_rectangles(device,dtype):
 import gffx.torch
 capture=os.environ.get('GFFX_SPATIAL_CAPTURE');root=Path(capture or os.environ.get('GFFX_SPATIAL_BASELINE') or pytest.skip('requires pre-change spatial fixture'))
 gen=torch.Generator().manual_seed(205);v=(torch.rand((3*96,3),generator=gen,dtype=torch.float64)*2-1).to(dtype=dtype,device=device)
 f=torch.arange(len(v),dtype=torch.int32,device=device).reshape(-1,3)
 vo=torch.tensor([0,len(v)],dtype=torch.int32,device=device);fo=torch.tensor([0,len(f)],dtype=torch.int32,device=device);rows=[]
 for h,w in ((1,513),(17,257),(257,1),(31,17)):
  for region in ((0,0,h,w),(h-1,w-1,1,1)):
   for blur,cull in ((0.,1),(.2,2)):
    out=torch.ops.gffx.rasterize_window(v,f,vo,fo,h,w,*region,3,blur,cull,1e-12)
    rows.append(dict(h=h,w=w,region=region,blur=blur,cull=cull,sha256=[hashlib.sha256(x.cpu().contiguous().numpy().tobytes()).hexdigest() for x in out]))
 name=device+'-'+str(dtype).split('.')[-1]+'.json'
 if capture:
  root.mkdir(exist_ok=True)
  with (root/name).open('x') as out:json.dump(rows,out)
 else:assert json.loads(json.dumps(rows))==json.loads((root/name).read_bytes())
