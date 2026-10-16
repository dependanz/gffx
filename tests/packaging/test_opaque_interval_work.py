"""Arithmetic work reductions retain outward endpoints and every live tangent."""
from decimal import Decimal as D,localcontext,ROUND_FLOOR,ROUND_CEILING
from fractions import Fraction as Q
import pytest
from tools.opaque_shading_intervals import I,Jet,ZERO,PRECISION

DOMAINS=[('-3','-1'),('1','4'),('-2','5'),('0','3'),('-2','0'),('0','0'),('1e-70','3e-70'),('-1e70','2e70')]

@pytest.mark.parametrize('a',DOMAINS)
@pytest.mark.parametrize('b',DOMAINS)
def test_sign_selected_products_match_all_four_directed_endpoints(a,b):
    x,y=I(*a),I(*b);got=x*y
    if x.zero or y.zero:expected=I(0)
    else:
        with localcontext() as c:
            c.prec=PRECISION;c.rounding=ROUND_FLOOR;lo=min(u*v for u in (x.lo,x.hi) for v in (y.lo,y.hi))
            c.rounding=ROUND_CEILING;hi=max(u*v for u in (x.lo,x.hi) for v in (y.lo,y.hi))
        expected=I(lo,hi)
    assert (got.lo,got.hi)==(expected.lo,expected.hi)
    assert got is not x and got is not y

@pytest.mark.parametrize('a',[('-3','-1'),('1','4'),('-2','5')])
@pytest.mark.parametrize('b',[('-3','-1'),('1','4'),('-2','5')])
def test_strict_signs_select_only_extremal_products(a,b):
    from tools.opaque_shading_intervals import _product_endpoints
    lo,hi=_product_endpoints(I(*a),I(*b))
    assert len(lo)+len(hi)==(4 if a==('-2','5') and b==('-2','5') else 2)

def legacy(q,kind):
    out={(0,0):I(1)/q[0,0] if kind=='reciprocal' else q[0,0].sqrt()}
    for i,j in sorted(((i,j) for i in range(q.order+1) for j in range(2) if i+j),key=lambda k:(sum(k),k)):
        total=ZERO
        if kind=='reciprocal':
            for (k,l),a in q.c.items():
                if k+l and k<=i and l<=j:total=total+a*out.get((i-k,j-l),ZERO)
            out[i,j]=-total/q[0,0]
        else:
            for (k,l),a in list(out.items()):
                if k+l and k<=i and l<=j:total=total+a*out.get((i-k,j-l),ZERO)
            out[i,j]=(q[i,j]-total)/(2*out[0,0])
    return Jet(order=q.order,coeff=out)

@pytest.mark.parametrize('kind',['reciprocal','sqrt'])
@pytest.mark.parametrize('order',[1,3,6,10])
@pytest.mark.parametrize('tangent',[0,1,'1e-80'])
def test_jet_recurrence_matches_full_two_axis_reference(kind,order,tangent):
    q=Jet.variable(I('1','1.00001'),order=order)
    if tangent:q.c[0,1]=I(tangent)
    actual=getattr(q,kind)();expected=legacy(q,kind)
    assert actual.c.keys()==expected.c.keys()
    assert all((actual[k].lo,actual[k].hi)==(expected[k].lo,expected[k].hi) for k in expected.c)
    if tangent:assert any(j==1 and not v.zero for (i,j),v in actual.c.items())

@pytest.mark.parametrize('kind',['reciprocal','sqrt'])
@pytest.mark.parametrize('order',[3,10])
@pytest.mark.parametrize('tangent',[0,1])
def test_jet_recurrence_contains_independent_binomial_coefficients(kind,order,tangent):
    q=Jet.variable(I(1),order=order)
    if tangent:q.c[0,1]=I(tangent)
    got=getattr(q,kind)();power=Q(-1) if kind=='reciprocal' else Q(1,2);coefficient=Q(1)
    for k in range(order+1):
        if k:coefficient*= (power-k+1)/k
        for j,exact in [(0,coefficient),(1,coefficient*(power-k)*tangent)]:
            value=I(exact);assert got[k,j].lo<=value.lo<=value.hi<=got[k,j].hi

@pytest.mark.parametrize('kind',['reciprocal','sqrt'])
def test_fixed_geometry_does_not_evaluate_absent_tangent_recurrence(kind,monkeypatch):
    q=Jet.variable(I(1),order=10);original=I.__mul__;calls=[]
    def count(a,b):calls.append(1);return original(a,b)
    monkeypatch.setattr(I,'__mul__',count)
    getattr(q,kind)();fixed=len(calls);calls.clear()
    q.c[0,1]=I('1e-80');getattr(q,kind)();assert len(calls)>fixed
