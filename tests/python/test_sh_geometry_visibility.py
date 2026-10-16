import pytest,torch
from gffx.torch import render
from sh_visibility_fixture import sample,exact

@pytest.mark.parametrize('z',[-1e-4,0.,1e-4])
@pytest.mark.parametrize('back,two',[(False,False),(False,True),(True,False),(True,True)])
def test_sh_geometric_sidedness_and_smooth_normal_crossing(z,back,two):sample('cpu',z,back,two)

def test_sh_raster_culling_still_owns_coverage():sample('cpu',-.01,True,True,render.CULL_BACK)

@pytest.mark.parametrize('back,two',[(False,False),(True,False),(True,True)])
def test_sh_cpu_cuda_bytes_and_gradients(back,two):
 if not torch.cuda.is_available():pytest.skip('CUDA required')
 for z in (-1e-4,0.,1e-4):
  a,ag=sample('cpu',z,back,two);b,bg=sample('cuda',z,back,two)
  for k in a:exact(a[k],b[k],k)
  for i,(x,y) in enumerate(zip(ag,bg)):exact(x,y,'gradient'+str(i))

def test_sh_small_visible_triangle_retains_radiance():sample('cpu',-.01,scale=1e-4)
