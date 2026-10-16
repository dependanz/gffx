"""Convolution keeps every product/add rounding; one local context owns a call."""
from decimal import Decimal as D,localcontext
from fractions import Fraction as Q
import pytest
from tools.opaque_shading_intervals import I,Jet,ZERO

def original_mul(self,b):
    b=self.other(b);out={}
    for (i,j),a in self.c.items():
        for (k,l),v in b.c.items():
            if i+k<=self.order and j+l<=1:
                key=(i+k,j+l);out[key]=out.get(key,ZERO)+a*v
    return Jet._from_owned(self.order,out)

def encoded(q):return {k:(v.lo.as_tuple(),v.hi.as_tuple()) for k,v in q.c.items()}

def pair(order,tangent,pattern):
    rows=[]
    for side in (0,1):
        coeff={}
        for i in range(order+1):
            for j in range(2 if tangent else 1):
                if pattern=='sparse' and (i+j+side)%3:continue
                n=(-1)**(i+j+side)*(2*i+3*j+side+1)
                v=I(Q(n,7+i))
                if pattern=='wide':v=I(v.lo-D('0.01'),v.hi+D('0.02'))
                if j:v=v*I('1e-80')
                coeff[i,j]=v
        rows.append(Jet(order=order,coeff=coeff))
    return rows

@pytest.mark.parametrize('order',[0,1,3,6,10])
@pytest.mark.parametrize('tangent',[False,True])
@pytest.mark.parametrize('pattern',['dense','sparse','wide'])
@pytest.mark.parametrize('precision',[6,28,90])
def test_fused_convolution_keeps_every_directed_endpoint(order,tangent,pattern,precision):
    a,b=pair(order,tangent,pattern);before=[encoded(q) for q in (a,b)]
    with localcontext() as c:
        c.prec=precision;expected=original_mul(a,b);actual=a*b
    assert encoded(actual)==encoded(expected)
    assert [encoded(q) for q in (a,b)]==before
    for v in actual.c.values():v.lo=D(-999);v.hi=D(999)
    assert [encoded(q) for q in (a,b)]==before

@pytest.mark.parametrize('order',[0,1,3,6,10])
@pytest.mark.parametrize('tangent',[False,True])
def test_convolution_uses_one_local_rounding_context(order,tangent,monkeypatch):
    import tools.opaque_shading_intervals as m
    a,b=pair(order,tangent,'dense');old=m.localcontext;calls=[]
    def observed(*args,**kwargs):calls.append(1);return old(*args,**kwargs)
    monkeypatch.setattr(m,'localcontext',observed)
    a*b
    assert len(calls)==1

@pytest.mark.parametrize('order',[0,1,3,6,10])
@pytest.mark.parametrize('tangent',[False,True])
def test_coefficients_enclose_independent_rational_polynomial_product(order,tangent):
    aa={(i,j):Q((-1)**i*(i+j+1),7+i+j) for i in range(order+1) for j in range(2 if tangent else 1)}
    bb={(i,j):Q((-1)**j*(i+2*j+1),11+i+j) for i in range(order+1) for j in range(2 if tangent else 1)}
    a=Jet(order=order,coeff={k:I(v) for k,v in aa.items()});b=Jet(order=order,coeff={k:I(v) for k,v in bb.items()})
    result=a*b
    for i in range(order+1):
        for j in range(2):
            exact=sum((aa.get((k,l),Q(0))*bb.get((i-k,j-l),Q(0)) for k in range(i+1) for l in range(j+1)),Q(0))
            assert Q(result[i,j].lo)<=exact<=Q(result[i,j].hi)

@pytest.mark.parametrize('left,right',[(0,2),(2,0),(0,0),(-1,1),(1,-1)])
def test_explicit_zero_coefficients_and_cancellation_preserve_zero_behavior(left,right):
    a=Jet(1,order=1);b=Jet(-1,order=1);a.c[1,0]=I(left);b.c[1,0]=I(right)
    assert encoded(a*b)==encoded(original_mul(a,b))

def test_order_mismatch_and_caller_context_are_preserved():
    with localcontext() as c:
        c.prec=7;before=(c.prec,c.rounding,dict(c.flags));Jet(2,order=3)*Jet(3,order=3)
        assert (c.prec,c.rounding,dict(c.flags))==before
        with pytest.raises(ValueError,match='order mismatch'):Jet(order=1)*Jet(order=2)

def test_synthetic_material_expression_exact_bounds_and_controlled_timing(monkeypatch,record_property):
    import time,json
    new=Jet.__mul__;rows=[]
    def evaluate():
        q=Jet.variable(I('0.7','0.701'),order=10)
        return ((q*q+3).sqrt().reciprocal().ln()+q**3)/(q+2)
    for implementation in [original_mul,new,new,original_mul]:
        monkeypatch.setattr(Jet,'__mul__',implementation);monkeypatch.setattr(Jet,'__rmul__',implementation)
        start=time.perf_counter();values=[encoded(evaluate()) for _ in range(8)]
        rows.append(dict(implementation='old' if implementation is original_mul else 'new',seconds=time.perf_counter()-start,values=values))
    assert all(r['values']==rows[0]['values'] for r in rows)
    record_property('controlled_synthetic_timing',json.dumps([{k:v for k,v in r.items() if k!='values'} for r in rows]))
