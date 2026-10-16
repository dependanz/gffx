"""Independent controls for finite immutable Peano polynomial coefficient reuse."""
from fractions import Fraction as Q
import pytest
from tools.opaque_shading_intervals import I
from tools import opaque_peano as p

@pytest.mark.parametrize('order',[1,2,3])
def test_kernel_power_constants_reused_across_domains_and_safe_to_mutate_results(order):
 p._kernel_power_cached.cache_clear();nodes=[I('.4'),I('.7')];weights=[I('.3'),I('.2')]
 a=p._kernel_segment_bernstein(I('.1'),I('.2'),nodes,weights,order);info=p._kernel_power_cached.cache_info();assert info.misses==1
 b=p._kernel_segment_bernstein(I('.11'),I('.21'),nodes,weights,order);assert p._kernel_power_cached.cache_info().hits==1
 before=[(v.lo,v.hi) for v in a];a[0].lo=I(-100).lo
 again=p._kernel_segment_bernstein(I('.1'),I('.2'),nodes,weights,order);assert [(v.lo,v.hi) for v in again]==before
 # Full endpoint identity in the key: changed weights must not hit old constants.
 weights[0].hi=I('.31').hi;p._kernel_segment_bernstein(I('.1'),I('.2'),nodes,weights,order);assert p._kernel_power_cached.cache_info().misses==2


def test_kernel_power_cache_has_fixed_eighteen_entry_bound():
 p._kernel_power_cached.cache_clear()
 for i in range(40):p._kernel_segment_bernstein(I('.1'),I('.2'),[I(Q(i+1,50))],[I('.3')],3)
 assert p._kernel_power_cached.cache_info().maxsize==18 and p._kernel_power_cached.cache_info().currsize==18

@pytest.mark.parametrize('order',[1,2,3])
def test_cached_power_coefficients_contain_independent_exact_moments(order):
 import math
 nodes=[Q(2,5),Q(7,10)];weights=[Q(3,10),Q(1,5)];key=tuple((I(x).lo,I(x).hi,I(w).lo,I(w).hi) for x,w in zip(nodes,weights));got=p._kernel_power_cached(order,key)
 for j,(lo,hi) in enumerate(got):
  v=sum((w*Q(math.comb(order-1,j),math.factorial(order-1))*x**(order-1-j)*(-1)**j for x,w in zip(nodes,weights)),Q(0)) if j<order else Q(0)
  v-=Q(math.comb(order,j),math.factorial(order))*(-1)**j
  assert Q(lo)<=v<=Q(hi)
