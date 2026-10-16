"""Closed logarithmic integrals qualify ordinary tangent residuals across knots."""
from fractions import Fraction as Q
from decimal import Decimal as D,localcontext
from types import SimpleNamespace
import pytest
from tools.opaque_shading_intervals import I,Jet
from tools.opaque_loss_extension import LossExtension

@pytest.mark.parametrize('degree',[1,2,3])
@pytest.mark.parametrize('axis',[0,1])
@pytest.mark.parametrize('reverse',[False,True])
def test_tangent_peano_contains_closed_piecewise_log_integral(degree,axis,reverse):
    from tools.opaque_tangent_peano import TangentCorrection
    h=I(2).ln();controls={1:[I(0),h/3,2*h/3,h],2:[I(0),I(0),h*h/3,h*h],3:[I(0),I(0),I(0),h*h*h]}[degree]
    curve=SimpleNamespace(cn=list(map(I,[Q(1,4),Q(1,2),1])),x=[I(Q(1,4)).ln(),I(Q(1,2)).ln(),I(0)],controls=[[[I(0)]*4]*2,[controls]*2],mean=[I(1),I(1)],pi=I(1))
    e=LossExtension(curve);e.indices=[0,1]
    def inputs(x,y):
        variable=x if axis==0 else y
        q=((1-variable) if reverse else variable)/2+x.lift(Q(1,4),tangent=1)
        return q,x.lift(1),x.lift(1),x.lift(1)
    result=TangentCorrection(e,inputs)((I(0,1),I(0,1)))
    with localcontext() as c:
        c.prec=90;expected=2*(D('1.5').ln()*D(2).ln())**degree
        assert result[1].lo<=expected<=result[1].hi
        import math
        def primitive(q):
            z=(q/D('.5')).ln()
            return q*sum(D((-1)**j*math.factorial(degree))/D(math.factorial(degree-j))*z**(degree-j) if degree!=j else D((-1)**j*math.factorial(degree)) for j in range(degree+1))
        expected_value=2*(primitive(D('.75'))-primitive(D('.5')))*D(2).ln()**degree
        assert result[0].lo<=expected_value<=result[0].hi
    assert result[1].lo>0


def test_tangent_peano_preserves_zero_parameter_direction():
    from tools.opaque_tangent_peano import TangentCorrection
    from tests.packaging.test_opaque_loss_extension import Curve
    curve=Curve();curve.mean=[I(1),I(1)];curve.pi=I(1);e=LossExtension(curve);e.indices=[1,1]
    result=TangentCorrection(e,lambda x,y:(x/2+Q(1,4),x.lift(Q(3,4)),x.lift(1),x.lift(1)))((I(0,1),I(0,1)))
    assert result[1].zero


@pytest.mark.parametrize('scale',[Q(1,10**14),-Q(1,10**14),Q(1,10**40)])
def test_conditioned_range_selection_retains_complete_small_integral(scale):
    from tools.opaque_tangent_peano import TangentCorrection
    proof=TangentCorrection(SimpleNamespace(mean=[I(1),I(1)]),None,order=4,conditioned=True)
    proof.ordinary=lambda x,y:[x*scale,y*scale]
    got=proof((I(0,1),I(0,1)))
    assert proof.last_proof=='range'
    exact=I(scale/2)
    for v in got:
        assert v.lo<=exact.lo<=exact.hi<=v.hi and not v.zero
        expected=I(I(min(Q(0),scale)).lo,I(max(Q(0),scale)).hi)
        assert (v.lo,v.hi)==(expected.lo,expected.hi)
