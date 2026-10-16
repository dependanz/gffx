"""Repeated denominators may be reused only with exact outward endpoints."""
from decimal import Decimal as D,localcontext,ROUND_FLOOR,ROUND_CEILING
import pytest
from tools.opaque_shading_intervals import I,Jet

@pytest.mark.parametrize('a',[('-3','-1'),('1','4'),('-2','5'),('0','0'),('-0','1e-80'),('-1e70','2e70')])
@pytest.mark.parametrize('b',[('-3','-1'),('1','4'),('1e-70','3e-70'),('-1e70','-2e69')])
def test_quotient_matches_independent_directed_endpoint_operations(a,b):
    x,y=I(*a),I(*b)
    with localcontext() as c:
        c.prec=60;c.rounding=ROUND_FLOOR;lo=1/y.hi
        c.rounding=ROUND_CEILING;hi=1/y.lo
        c.rounding=ROUND_FLOOR;low=min(u*v for u in (x.lo,x.hi) for v in (lo,hi))
        c.rounding=ROUND_CEILING;high=max(u*v for u in (x.lo,x.hi) for v in (lo,hi))
    with localcontext() as c:
        c.prec=7;got=x/y
    assert (got.lo,got.hi)==(low,high)

@pytest.mark.parametrize('kind',['reciprocal','sqrt'])
@pytest.mark.parametrize('order',[3,10])
@pytest.mark.parametrize('tangent',[0,1,'1e-80'])
def test_recurrence_uses_one_denominator_without_coefficient_change(kind,order,tangent,monkeypatch):
    from tests.packaging.test_opaque_interval_work import legacy
    q=Jet.variable(I('1','1.00001'),order=order)
    if tangent:q.c[0,1]=I(tangent)
    expected=legacy(q,kind);original=I.__truediv__;calls=[]
    def observed(a,b):calls.append(1);return original(a,b)
    monkeypatch.setattr(I,'__truediv__',observed);actual=getattr(q,kind)()
    assert len(calls)==1
    assert actual.c.keys()==expected.c.keys()
    assert all((actual[k].lo,actual[k].hi)==(expected[k].lo,expected[k].hi) for k in expected.c)


@pytest.mark.parametrize('kind,expected',[('add',1),('mul',1),('truediv',2)])
def test_existing_interval_operands_do_not_need_temporary_copies(kind,expected,monkeypatch):
    import operator
    a,b=I(1,2),I(3,4);initial=(a.lo,a.hi,b.lo,b.hi);original=I.__init__;calls=[]
    def observed(self,*args,**kwargs):calls.append(1);original(self,*args,**kwargs)
    monkeypatch.setattr(I,'__init__',observed);result=getattr(operator,kind)(a,b)
    assert len(calls)==expected
    assert result is not a and result is not b and initial==(a.lo,a.hi,b.lo,b.hi)


def test_zero_addition_keeps_its_fresh_return_contract():
    a,b=I(0),I(1,2);q=a+b;assert q is not a and q is not b
    q.lo=I(-9).lo;assert (b.lo,b.hi)==(I(1).lo,I(2).hi)
