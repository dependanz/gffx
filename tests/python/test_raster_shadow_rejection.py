"""Depth-only rejection must retain independent native coverage/depth rules."""
import pytest
import torch
from gffx.torch import _raster_accel as accel
from gffx.torch import render as native

@pytest.mark.parametrize('height,width',[(31,47),(64,64),(17,73),(129,193)])
def test_depth_rejection_preserves_thin_edge_and_overlapping_triangle_coverage(height,width):
    gen=torch.Generator().manual_seed(246)
    triangles=torch.rand((80,3,3),generator=gen)*2-1
    # Deliberately tiny, reversed, exactly on pixel centres, and flat-depth cases.
    triangles[:20,:,0]*=.02;triangles[20:40,:,1]*=.01
    triangles[40:50]=triangles[50:60][:,[0,2,1]]
    triangles[60:70,:,2]=-.3
    triangles[70]=torch.tensor([[-1,-1,-.7],[1,-1,-.7],[-1,1,-.7]])
    triangles[71]=torch.tensor([[1,1,-.7],[1,-1,-.7],[-1,1,-.7]])
    triangles[72,:,0]=2*(torch.tensor([10.5,20.5,10.5])/width)-1
    triangles[72,:,1]=1-2*(torch.tensor([10.5,10.5,20.5])/height)
    ndc=triangles.reshape(-1,3).cuda();faces=torch.arange(len(ndc),dtype=torch.int32,device='cuda').reshape(-1,3)
    actual=accel.shadow(ndc,faces,height,width)
    index,_,expected,_=native.rasterize(ndc,faces,height,width,cull_mode=native.CULL_NONE)
    expected=torch.where(index>=0,expected,float('inf')).reshape(height,width)
    assert torch.equal(torch.isfinite(actual),torch.isfinite(expected))
    torch.testing.assert_close(actual,expected,atol=2e-7,rtol=2e-7)
