"""Closed-form controls for bounded independent reference partitioning."""
from fractions import Fraction as F
import pytest
from tools.opaque_shading_intervals import I,Jet
from tools.opaque_sh_texture_reference import certify_partition

@pytest.mark.parametrize('power',[0,9,10,12])
def test_partition_encloses_polynomial_and_first_differential(power):
 fn=lambda u,v:[Jet(2,order=u.order,tangent=3)*(u**power)*(v**10)]
 for n in (1,2):
  c=certify_partition(fn,n)
  for key,coefficient in [('value',2),('tangent',3)]:
   exact=I(F(coefficient,(power+1)*11));box=c[key][0]
   assert box.lo<=exact.lo<=exact.hi<=box.hi


def test_partition_reduces_nonzero_tenth_derivative_remainder():
 fn=lambda u,v:[u**12*v**10]
 a=certify_partition(fn,1);b=certify_partition(fn,2)
 assert a['remainder'][0][0].hi>0
 assert b['remainder'][0][0].hi<a['remainder'][0][0].hi/100

@pytest.mark.parametrize('n',[0,-1,3,True,2.0])
def test_partition_refuses_unbounded_or_ambiguous_grid(n):
 with pytest.raises(ValueError):certify_partition(lambda u,v:[u+v],n)
