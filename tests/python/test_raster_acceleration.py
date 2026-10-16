"""Original installed window40 raw outputs are frozen before acceleration."""
import os,json,hashlib
from pathlib import Path
import pytest,torch

def cases(dtype):
 g=torch.Generator().manual_seed(19414)
 for seed in range(8):
  v=torch.rand((9,3),generator=g,dtype=torch.float64)*3-1.5;v[:,2]*=.1
  yield v.to(dtype),torch.tensor([[0,1,2],[0,1,2],[3,4,5],[6,7,8]],dtype=torch.int32)
 for magnitude in (0.,1e-12,1e6,1e15):
  v=torch.tensor([[-.8,-.7,.3],[.9,-.6,.3],[.1,.9,.3]],dtype=dtype);v[:,:2]+=magnitude
  yield v,torch.tensor([[0,1,2],[2,1,0],[0,0,0]],dtype=torch.int32)

@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_accelerated_raster_preserves_frozen_scan(device,dtype):
 import gffx.torch
 capture=os.environ.get('GFFX_RASTER_CAPTURE');root=Path(capture or os.environ.get('GFFX_RASTER_BASELINE') or pytest.skip('requires frozen original raster baseline'))
 name=device+'-'+str(dtype).split('.')[-1]+'.json';rows=[]
 for case,(v,f) in enumerate(cases(dtype)):
  v=v.to(device);f=f.to(device);vo=torch.tensor([0,len(v)],dtype=torch.int32,device=device);fo=torch.tensor([0,len(f)],dtype=torch.int32,device=device)
  for blur in (0.,.6):
   for cull in (1,2,3):
    for region in ((0,0,13,17),(2,3,9,11)):
     outputs=torch.ops.gffx.rasterize_window(v,f,vo,fo,13,17,*region,2,blur,cull,1e-12)
     raw=[x.detach().cpu().contiguous().numpy().tobytes().hex() for x in outputs]
     rows.append(dict(case=case,blur=blur,cull=cull,region=list(region),raw=raw))
 if capture:
  root.mkdir(exist_ok=True);p=root/name
  with p.open('x',encoding='utf-8') as h:json.dump(rows,h)
 else:
  expected=json.loads((root/name).read_text());assert rows==expected

@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_accelerated_raster_preserves_packed_block_boundaries(device,dtype):
 import gffx.torch
 capture=os.environ.get('GFFX_RASTER_CAPTURE')
 root=Path(capture or os.environ.get('GFFX_RASTER_BASELINE') or pytest.skip('requires frozen original raster baseline'))
 # Two packed meshes, ties, empty third mesh, full and offset windows crossing32.
 v,f=next(cases(dtype));v=torch.cat((v,v*.3)).to(device)
 f=torch.cat((f,f+9)).to(device)
 vo=torch.tensor([0,9,18,18],dtype=torch.int32,device=device)
 fo=torch.tensor([0,4,8,8],dtype=torch.int32,device=device)
 rows=[]
 for blur in (0.,.6,64.):
  for cull in (1,2,3):
   for region in ((0,0,65,67),(1,3,64,63),(32,32,33,35)):
    out=torch.ops.gffx.rasterize_window(v,f,vo,fo,65,67,*region,2,blur,cull,1e-12)
    # Hash every byte while keeping the fixture compact; no tolerance or image compression.
    rows.append(dict(blur=blur,cull=cull,region=list(region),sha256=[hashlib.sha256(x.cpu().contiguous().numpy().tobytes()).hexdigest() for x in out]))
 name='packed-'+device+'-'+str(dtype).split('.')[-1]+'.json'
 if capture:
  root.mkdir(exist_ok=True)
  with (root/name).open('x',encoding='utf-8') as h:json.dump(rows,h)
 else:assert rows==json.loads((root/name).read_text())
