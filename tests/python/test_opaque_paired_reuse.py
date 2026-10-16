"""Saved FLAME bottleneck: geometry-work admission."""
import os,hashlib,numpy as np,pytest,torch
import gffx.torch

def test_saved_paired_geometry_work():
 from gffx.torch._common import materialize
 from pathlib import Path
 path=Path('D:/devspace/gffit/qualification-20261003/boundary-sh8/prefix-capture1-batch02.npz')
 if not path.exists():pytest.skip('local historical FLAME fixture unavailable')
 assert hashlib.sha256(path.read_bytes()).hexdigest()=='5d2c2122798e80dac2ed52e588ed54eeed3904318725b012f6af1a7343ac13b2'
 dev=os.environ.get('GFFX_BATCH_DEVICE','cpu')
 with np.load(path) as z:
  v,f,c=[materialize(torch.tensor(z[k],device=dev)) for k in ('vertices','faces','cuts')];ox,oy=z['origin'].tolist()
 g=torch.ones((1,800,4),dtype=v.dtype,device=dev)
 gv,gc,receipt,prefix=torch.ops.gffx.opaque_material_prefix_backward(v,f,c,g,ox,oy,256,8,6400,256,67108864,1,69,255)
 assert receipt[7]<200000,receipt[7]
 assert torch.isfinite(gv).all() and torch.count_nonzero(gc)==0
