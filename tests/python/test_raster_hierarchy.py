"""Capture the installed pre-hierarchy raster before changing traversal."""
import os,json,hashlib
from pathlib import Path
import pytest,torch
from test_raster_acceleration import cases
@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_hierarchical_raster_preserves_wide_and_thin_windows(device,dtype):
 import gffx.torch
 capture=os.environ.get('GFFX_RASTER_CAPTURE');root=Path(capture or os.environ['GFFX_RASTER_HIERARCHY_BASELINE']);rows=[]
 for case,(v,f) in enumerate(cases(dtype)):
  v=v.to(device);f=f.to(device);vo=torch.tensor([0,len(v)],dtype=torch.int32,device=device);fo=torch.tensor([0,len(f)],dtype=torch.int32,device=device)
  for blur in (0.,.6,64.):
   for region in ((0,0,1,1920),(511,0,3,1920),(530,701,19,67),(0,955,1080,1)):
    out=torch.ops.gffx.rasterize_window(v,f,vo,fo,1080,1920,*region,2,blur,2,1e-12)
    rows.append(dict(case=case,blur=blur,region=list(region),sha256=[hashlib.sha256(x.cpu().contiguous().numpy().tobytes()).hexdigest() for x in out]))
 name='thin-'+device+'-'+str(dtype).split('.')[-1]+'.json'
 if capture:
  root.mkdir(exist_ok=True)
  with (root/name).open('x') as h:json.dump(rows,h)
 else:assert rows==json.loads((root/name).read_text())
