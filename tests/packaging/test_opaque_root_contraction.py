"""Independent root brackets retain exact roots, clipped absence and tiny terms."""
from decimal import Decimal as D,localcontext
import pytest
from tools.opaque_shading_intervals import I,Unresolved

@pytest.mark.parametrize('sign',[-1,1])
@pytest.mark.parametrize('tiny',[False,True])
def test_contracted_crossing_contains_independent_quadratic_root(sign,tiny):
 from tools.opaque_root_contraction import contract_root
 scale=I('1e-40') if tiny else I(1)
 def evaluate(x):return (x+x*x/4)*scale*sign,(1+x/2)*scale*sign
 target=I('0.7')*scale*sign
 got=contract_root(I(0,1),evaluate,target)
 with localcontext() as ctx:
  ctx.prec=100;root=-2+(D('6.8')).sqrt()
  assert got.lo<=root<=got.hi and got.width<D('.002')

@pytest.mark.parametrize('target',[-2,0,.25,1,3])
@pytest.mark.parametrize('sign',[-1,1])
def test_contracted_clipped_crossing_retains_absence_and_endpoints(target,sign):
 from tools.opaque_root_contraction import contract_root
 got=contract_root(I(0,1),lambda x:(x*sign,I(sign)),I(target)*sign,clipped=True)
 expected=I(min(1,max(0,target)));assert got.lo<=expected.lo<=expected.hi<=got.hi

@pytest.mark.parametrize('sign',[-1,1])
def test_contracted_crossing_keeps_uncertain_knots(sign):
 from tools.opaque_root_contraction import contract_root
 got=contract_root(I(0,1),lambda x:(x*sign,I(sign)),I('.2','.8')*sign)
 assert got.lo<=D('.2') and got.hi>=D('.8')

def test_contracted_crossing_refuses_invalid_contract_and_propagates_cancel():
 from tools.opaque_root_contraction import contract_root
 f=lambda x:(x,I(1))
 for kwargs in [dict(clipped=1),dict(iterations=0),dict(iterations=5)]:
  with pytest.raises(ValueError):contract_root(I(0,1),f,I('.3'),**kwargs)
 with pytest.raises(ValueError):contract_root(I(1,0),f,I('.3'))
 with pytest.raises(Unresolved):contract_root(I(0,1),f,I('.3'),checkpoint=lambda:(_ for _ in ()).throw(Unresolved('cancel')))
 assert contract_root(I(0,1),f,I(2)) is None

@pytest.mark.parametrize('sx,sy',[(-1,-1),(-1,1),(1,-1),(1,1)])
def test_contracted_material_contains_independent_log_integral(sx,sy):
 from tools.opaque_contracted_correction import ContractedMaterialCorrection
 from tests.packaging.test_opaque_localized_peano import material_fixture
 from tests.packaging.test_opaque_transverse_peano import _log_rectangle
 e,eps,slope=material_fixture();b=I('.001')*sx;c=I('.001')*sy
 a=I('.4988')+(I('.002') if sx<0 else I(0))+(I('.001') if sy<0 else I(0))
 def inputs(x,y):return (x+x*x)*b+y*c+a,x.lift('.75'),x.lift(1),1+2*x
 proof=ContractedMaterialCorrection(e,inputs,curved=True,transverse=True,smooth_partition=True,segmented_conditioned=True)
 got=proof((I(0,1),I(0,1)))[1];expected=_log_rectangle(e,eps,slope,a,b,c,I(2))
 assert got.lo<=expected.lo<=expected.hi<=got.hi and proof.inputs is inputs
 assert proof.contractions>0
