"""Every higher jump must remain in a localized full integral enclosure."""
import pytest
@pytest.mark.parametrize('parts',[2,4])
@pytest.mark.parametrize('axis',[0,1])
@pytest.mark.parametrize('sx,sy',[(1,1),(-1,1),(1,-1),(-1,-1)])
def test_localized_higher_jumps_keep_independent_closed_integral(monkeypatch,axis,sx,sy,parts):
    from tools import opaque_boundary_material as m
    from tests.packaging.test_opaque_boundary_material import test_transport_keeps_independent_log_squared_rectangle
    original=m.BoundaryMaterial
    monkeypatch.setattr(m,'BoundaryMaterial',lambda *a,**k:original(*a,**k,localize_higher=True))
    test_transport_keeps_independent_log_squared_rectangle(axis,sx,sy,parts)
@pytest.mark.parametrize('case',['steps','simultaneous','refusal'])
def test_localized_higher_jumps_keep_discontinuities_and_refusals(monkeypatch,case):
    from tools import opaque_boundary_material as m
    from tests.packaging.test_opaque_boundary_material import test_transport_preserves_discontinuous_and_refusal_controls
    original=m.BoundaryMaterial
    monkeypatch.setattr(m,'BoundaryMaterial',lambda *a,**k:original(*a,**k,localize_higher=True))
    test_transport_preserves_discontinuous_and_refusal_controls(case,monkeypatch)
def test_higher_localization_is_explicit():
    from tools.opaque_boundary_material import BoundaryMaterial
    from tests.packaging.test_opaque_aggregate_transport import fixture
    assert BoundaryMaterial(fixture()).localize_higher is False
    with pytest.raises(ValueError):BoundaryMaterial(fixture(),localize_higher=1)

@pytest.mark.parametrize('axis',[0,1])
def test_curved_higher_jump_integral_matches_independent_polynomial(axis):
    from tools.opaque_boundary_material import BoundaryMaterial
    from tools.opaque_shading_intervals import I
    from tests.packaging.test_opaque_aggregate_transport import fixture
    from decimal import Decimal as D,localcontext
    import math
    c=fixture()
    def inputs(u,v):
        x,y=(u,v) if axis==0 else (v,u)
        return x*x*I('.2')+y*I('.4')+x.lift('.3'),x.lift(1),x.lift(1),x.lift(1)
    c.inputs=inputs;p=BoundaryMaterial(c,localize_higher=True)
    got=p.localized_higher(dict(x=I('.2','.4'),y=I('.42','.48')),axis,0,1,I('.5'),I(1),1,I('.2'),I('.4'))
    with localcontext() as ctx:
        ctx.prec=90;a,b=D('.2'),D('.4');v=(D(10)/7).sqrt();x=(5+2*v).sqrt()/6;y=(5-2*v).sqrt()/6
        nodes=[D('.5')-x,D('.5')-y,D('.5'),D('.5')+y,D('.5')+x];u=(322-13*D(70).sqrt())/1800;v=(322+13*D(70).sqrt())/1800;weights=[u,v,D(64)/225,v,u]
        polynomials={4:{2:D('3.84'),4:D('-3.072')},5:{1:D('3.84'),3:D('-18.432'),5:D('9.0112')},6:{2:D('-46.08'),4:D('90.112'),6:D('-32.768')}}
        expected=D(0);edges=sorted({a,b,*[v for v in nodes if a<v<b]})
        for order,poly in polynomials.items():
            for lo,hi in zip(edges,edges[1:]):
                kernel=[D(0)]*(order+1)
                for x,w in zip(nodes,weights):
                    if x>(lo+hi)/2:
                        for j in range(order):kernel[j]+=w*math.comb(order-1,j)*x**(order-1-j)*(-1)**j/D(math.factorial(order-1))
                for j in range(order+1):kernel[j]-=D(math.comb(order,j)*(-1)**j)/math.factorial(order)
                for i,c in poly.items():
                    for j,k in enumerate(kernel):expected+=c*k*(hi**(i+j+1)-lo**(i+j+1))/(i+j+1)
        expected*=1+D(2).ln()**2
        assert got.lo<=expected<=got.hi

@pytest.mark.parametrize('kind',['fallback','disjoint','stop'])
def test_localized_jump_failure_boundaries(monkeypatch,kind):
    from tools.opaque_boundary_material import BoundaryMaterial
    from tools.opaque_shading_intervals import I,Unresolved
    from tests.packaging.test_opaque_aggregate_transport import fixture
    box=(I(0,1),I(0,1));expected=BoundaryMaterial(fixture(),parts=4)(box)[1];calls=[]
    def replacement(*args,**kwargs):
        calls.append(1)
        if kind=='fallback':raise ValueError('localized jump monotonicity unavailable')
        if kind=='stop':raise Unresolved('intentional stop')
        return I(100)
    monkeypatch.setattr(BoundaryMaterial,'localized_higher',replacement)
    proof=BoundaryMaterial(fixture(),parts=4,localize_higher=True)
    if kind=='fallback':
        got=proof(box)[1];assert (got.lo,got.hi)==(expected.lo,expected.hi)
    else:
        with pytest.raises(Unresolved if kind=='stop' else ValueError):proof(box)
    assert calls
