"""Preserve live angular tables while avoiding repeated proxy lookup per knot."""
from types import SimpleNamespace
from fractions import Fraction as Q
import pytest
from tools.opaque_shading_intervals import I,Jet
from tools.opaque_full_material_reference import HorizonExtension
from tools.opaque_peano_material import segmented
from tests.packaging.test_opaque_localized_peano import material_fixture

class Observed(HorizonExtension):
 def __init__(self,curve):super().__init__(curve);self.reads=0
 def __getattr__(self,name):
  if name=='cn':self.reads+=1
  return super().__getattr__(name)

@pytest.mark.parametrize('count',[16,128,512])
def test_segment_table_is_resolved_once_per_call_and_not_retained(count):
 cn=[I(Q(i,count)) for i in range(count+1)];e=Observed(SimpleNamespace(cn=cn));index=count//2;q=Jet(Q(2*index+1,2*count))
 got=segmented(e,q,index);assert all(not v.c for v in got);assert e.reads==1
 # A replacement table is observed on the next call: no hidden stale cache.
 e.curve.cn=[I(0),I(1)];e.reads=0;got=segmented(e,q,0);assert all(not v.c for v in got);assert e.reads==1

@pytest.mark.parametrize('order',[0,1,3])
@pytest.mark.parametrize('value',[I('.3'),I('.49','.51'),I('.5'),I('.7'),I('.3','.9')])
def test_segment_enclosures_keep_all_legacy_coefficients(order,value):
 e,_,_=material_fixture();q=Jet.variable(value,order=order) if order else Jet(value)
 from tools.opaque_peano_material import difference
 v=q[0,0];chosen=[i for i in range(len(e.cn)-1) if v.hi>=e.cn[i].lo and v.lo<=e.cn[i+1].hi];rows=[]
 for i in chosen:
  c=dict(q.c);c[0,0]=I(max(v.lo,e.cn[i].lo),min(v.hi,e.cn[i+1].hi));rows.append(difference(e,Jet(order=q.order,coeff=c),i,0))
 expected=[Jet(order=order,coeff={(i,0):I(min(row[k][i,0].lo for row in rows),max(row[k][i,0].hi for row in rows)) for i in range(order+1)}) for k in range(2)]
 got=segmented(e,q,0);assert [[(key,v.lo,v.hi) for key,v in sorted(z.c.items())] for z in got]==[[(key,v.lo,v.hi) for key,v in sorted(z.c.items())] for z in expected]
