"""Closed piecewise-polynomial integrals for ordinary-tangent jump transport."""
from fractions import Fraction as Q
import math
import pytest
from tools.opaque_shading_intervals import I,Jet

@pytest.mark.parametrize('integrated_axis',[0,1])
@pytest.mark.parametrize('flip',[False,True])
@pytest.mark.parametrize('degree',[1,2,3,4])
@pytest.mark.parametrize('scale',[Q(1),Q(-1,2**60)])
def test_transverse_tangent_preserves_closed_integrals(degree,scale,flip,integrated_axis):
    from tools.opaque_tangent_transverse import refine
    from tools.opaque_fourth_tangent import integral
    from tools.opaque_arb_jets import backend
    with backend(profile='ball',fast_conversion=True):
        def field(x,y):
            z=((1-x) if flip else x)+y-Q(1,2)
            if z[0,0].hi<0:return [x.lift(0)]*2
            assert z[0,0].lo>0
            return [z**degree,scale*degree*z**(degree-1)]
        def inputs(x,y):return ((1-x) if flip else x)+y,x.lift(1),x.lift(1),x.lift(1)
        count=[]
        def bounds(box,axis):
            other=box[1-axis]
            if not flip:root=I('.5')-other
            elif axis==0:root=I('.5')+other
            else:root=other-I('.5')
            lo,hi=max(box[axis].lo,root.lo),min(box[axis].hi,root.hi);position=None if lo>hi else I(lo,hi);rows=[]
            slope=-1 if flip and axis==0 else 1
            for d,factor in [(degree,Q(1)),(degree-1,scale*degree)]:
                values=[I(factor*math.factorial(d)*slope**(d+1)) if j==d else I(0) for j in range(4)]
                rows.append(dict(order=4,derivative=I(abs(factor)*24 if d==4 else 0),jumps=[] if position is None else [dict(position=position,values=values,required=False,role=0,knot=0)]))
            refine(rows,inputs,box,axis,[I('.5')]);count.extend(w for r in rows for w in r['jumps'] if 'transverse_integral' in w);return rows
        got=integral(field,(I(0,1),I(0,1)),bounds,conditioned=True,direct_conditioned=True,condition_parts=4,integrated_axis=integrated_axis)
        exact=[(Q(3,2)**(degree+2)-2*Q(1,2)**(degree+2))/((degree+1)*(degree+2)),scale*(Q(3,2)**(degree+1)-2*Q(1,2)**(degree+1))/(degree+1)]
        for actual,value in zip(got,exact):
            ref=I(value);assert actual.lo<=ref.lo<=ref.hi<=actual.hi
        assert count


def test_transverse_tangent_retains_unproved_crossings():
    from tools.opaque_tangent_transverse import refine
    rows=[dict(jumps=[dict(position=I(0,1),values=[I(1)]*4,required=False,role=0,knot=0)])]
    refine(rows,lambda x,y:((x-Q(1,2))**2+y,x.lift(1),x.lift(1),x.lift(1)),(I(0,1),I(0,1)),0,[I('.5')])
    assert 'transverse_integral' not in rows[0]['jumps'][0]


@pytest.mark.parametrize('axis',[0,1,2])
def test_exact_cosine_partials_enclose_independent_closed_derivatives(axis):
    from tools.opaque_tangent_transverse import CosinePartials
    from tools.opaque_moving_reference import P
    from tools.opaque_arb_jets import backend
    coefficients={'n':[[P(0),P([1,Q(1,2**60)]),P(1)],[P(0)]*3,[P(1),P(0),P(0)]],'v':[[P(1),P(0),P(0)],[P(0)]*3,[P(0)]*3]}
    with backend(profile='ball',fast_conversion=True):
        proof=CosinePartials(coefficients)
        # Rational square-root brackets from integer arithmetic are independent
        # of both the interval backend and the polynomial slope implementation.
        def root(q):
            scale=10**100;lo=math.isqrt(q.numerator*scale*scale//q.denominator)
            assert Q(lo*lo,scale*scale)<=q<Q((lo+1)**2,scale*scale)
            return I(I(Q(lo,scale)).lo,I(Q(lo+1,scale)).hi)
        for u,s in [(I('.25'),I('.5')),(I('.2','.3'),I('.4','.6'))]:
            for role in (0,1):
                actual=proof.bound(role,(u,s),axis)
                for x in (Q(u.lo),(Q(u.lo)+Q(u.hi))/2,Q(u.hi)):
                    for y in (Q(s.lo),(Q(s.lo)+Q(s.hi))/2,Q(s.hi)):
                        a=x*(1+y);chain=1+y if axis==0 else x if axis==1 else x*Q(1,2**60)
                        expected=I(chain)/(I(1+a*a)*root(1+a*a)) if role==0 else I(chain*(1-2*a))/(I(1+a*a)*root(6*(1+a*a)))
                        assert actual.lo<=expected.lo<=expected.hi<=actual.hi
        with pytest.raises(ValueError):proof.bound(3,(I(0,1),I(0,1)),axis)


def test_exact_partial_refinement_activates_uncertain_crossings_and_refuses_disjoint_proof():
    from tools.opaque_tangent_transverse import refine
    def inputs(x,y):
        q=x+y
        if q.order:q.c[1,0]=I(-1,3)
        return q,x.lift(1),x.lift(1),x.lift(1)
    def rows():return [dict(jumps=[dict(position=I(0,1),values=[I(1),I(0),I(0),I(0)],required=False,role=0,knot=0)])]
    plain=rows();refine(plain,inputs,(I(0,1),I(0,1)),0,[I('.5')]);assert 'transverse_integral' not in plain[0]['jumps'][0]
    called=[]
    def bound(role,box,axis):called.append((role,box,axis));return I(1)
    selected=rows();refine(selected,inputs,(I(0,1),I(0,1)),0,[I('.5')],partial_bound=bound)
    assert 'transverse_integral' in selected[0]['jumps'][0] and len(called)==2
    with pytest.raises(ValueError):refine(rows(),inputs,(I(0,1),I(0,1)),0,[I('.5')],partial_bound=lambda *a:I(4))


def test_exact_partial_controls_crossing_count_before_localization(monkeypatch):
    from types import SimpleNamespace
    import tools.opaque_tangent_peano as tp
    from tools.opaque_loss_extension import LossExtension
    h=I(2).ln();curve=SimpleNamespace(cn=list(map(I,[Q(1,4),Q(1,2),1])),x=[I(Q(1,4)).ln(),I(Q(1,2)).ln(),I(0)],controls=[[[I(0)]*4]*2,[[I(0),h/3,2*h/3,h]]*2],mean=[I(1),I(1)],pi=I(1));e=LossExtension(curve);e.indices=[0,1]
    def inputs(x,y):
        q=x/2+y/8+x.lift(Q(1,4),tangent=1)
        if q.order:q.c[1,0]=I(-1,1)
        return q,x.lift(1),x.lift(1),x.lift(1)
    locations=[];old=tp.crossing_location
    def locate(domain,center,derivative,knot):locations.append(derivative);return old(domain,center,derivative,knot)
    monkeypatch.setattr(tp,'crossing_location',locate)
    correction=tp.TangentCorrection(e,inputs,order=4,transverse=True,transverse_partial=lambda role,box,axis:I(Q(1,2) if axis==0 else Q(1,8)))
    correction.axis_bounds((I(0,1),I(0,1)),0)
    assert locations and all(v.lo==v.hi==I('.5').lo for v in locations)


@pytest.mark.parametrize('sign',[1,-1])
def test_exact_cosine_motion_keeps_intrinsic_and_coordinate_chain(sign):
    from tools.opaque_tangent_transverse import CosinePartials
    from tools.opaque_moving_reference import P
    from tools.opaque_arb_jets import backend
    tiny=Q(sign,2**60);coefficients={'n':[[P(0),P([1,tiny]),P(1)],[P(0)]*3,[P(1),P(0),P(0)]],'v':[[P(1),P(0),P(0)],[P(0)]*3,[P(0)]*3]}
    with backend(profile='ball',fast_conversion=True):
        proof=CosinePartials(coefficients);u=Jet(I('.2','.3'),tangent=Q(sign,100));s=Jet(I('.4','.6'),tangent=tiny);a=u*(1+s)+u*u.lift(0,tangent=tiny)
        expected=[a/(1+a*a).sqrt(),(a+2)/(6*(1+a*a)).sqrt()]
        for role,q in enumerate(expected):
            actual=proof.motion_bound(role,u,s);assert max(actual.lo,q[0,1].lo)<=min(actual.hi,q[0,1].hi)
            for x in (Q(1,5),Q(1,4),Q(3,10)):
                for y in (Q(2,5),Q(1,2),Q(3,5)):
                    z=I(x*(1+y));chain=I(Q(sign,100)*(1+y)+x*tiny+x*tiny);exact=chain/(1+z*z)/(1+z*z).sqrt() if role==0 else chain*(1-2*z)/(1+z*z)/(6*(1+z*z)).sqrt()
                    assert actual.lo<=exact.lo<=exact.hi<=actual.hi


@pytest.mark.parametrize('degree',[1,2,4])
@pytest.mark.parametrize('axis',[0,1])
@pytest.mark.parametrize('flip',[False,True])
@pytest.mark.parametrize('scale',[Q(1),Q(-1,2**60)])
def test_curved_crossing_transverse_integral_has_closed_polynomial_reference(degree,axis,flip,scale):
    from tools.opaque_tangent_transverse import refine
    from tools.opaque_fourth_tangent import integral
    from tools.opaque_arb_jets import backend
    a,b=Q(1,4),Q(1,2);used=[]
    with backend(profile='ball',fast_conversion=True):
        def positive(z,power):
            if z[0,0].hi<=0:return z.lift(0)
            q=z**power
            if z[0,0].lo>0:return q
            return Jet(order=q.order,coeff={key:I(min(I(0).lo,v.lo),max(I(0).hi,v.hi)) for key,v in q.c.items()})
        def field(x,y):
            z=((1-x) if flip else x)+y*y-Q(1,2)
            return [positive(z,degree),scale*degree*positive(z,degree-1)]
        def inputs(x,y):return ((1-x) if flip else x)+y*y,x.lift(1),x.lift(1),x.lift(1)
        def bounds(box,chosen):
            args=[Jet(v,order=4) for v in box];args[chosen]=Jet.variable(box[chosen],order=4)
            z=inputs(*args)[0]-Q(1,2);rows=[]
            active=z[0,0].lo<=0<=z[0,0].hi
            for power,factor in [(degree,Q(1)),(degree-1,scale*degree)]:
                smooth=z**power;values=[]
                for j in range(4):
                    if chosen==0:value=I(factor*math.factorial(power)*(-1 if flip else 1)**(power+1)) if j==power else I(0)
                    elif power<=j<=2*power:value=I(factor*math.factorial(j)*math.comb(power,j-power))*(2*box[1])**(2*power-j)
                    else:value=I(0)
                    values.append(value)
                rows.append(dict(order=4,derivative=I((factor*smooth[4,0]*24).bound),jumps=[] if not active else [dict(position=box[chosen],values=values,required=False,role=0,knot=0)]))
            refine(rows,inputs,box,chosen,[I('.5')]);used.extend(w for row in rows for w in row['jumps'] if 'transverse_integral' in w);return rows
        got=integral(field,(I(0,1),I(I(a).lo,I(b).hi)),bounds,conditioned=True,direct_conditioned=True,condition_parts=4,integrated_axis=axis)
        def exact(power):return sum(Q(math.comb(power+1,k))*Q(1,2)**(power+1-k)*(b**(2*k+1)-a**(2*k+1))/((power+1)*(2*k+1)) for k in range(power+2))
        for value,target in zip(got,[exact(degree),scale*degree*exact(degree-1)]):
            expected=I(target);assert value.lo<=expected.lo<=expected.hi<=value.hi
        assert used


def test_curved_transverse_profile_is_explicit_and_preserves_monotonicity_refusal():
    from types import SimpleNamespace
    from tools.opaque_tangent_peano import TangentCorrection
    from tools.opaque_loss_extension import LossExtension
    h=I(2).ln();curve=SimpleNamespace(cn=list(map(I,[Q(1,4),Q(1,2),1])),x=[I(Q(1,4)).ln(),I(Q(1,2)).ln(),I(0)],controls=[[[I(0)]*4]*2,[[I(0),h/3,2*h/3,h]]*2],mean=[I(1),I(1)],pi=I(1));e=LossExtension(curve);e.indices=[0,1]
    def inputs(x,y):return (x-Q(1,2))**2+y/8+Q(1,2),x.lift(1),x.lift(1),x.lift(1)
    c=TangentCorrection(e,inputs,order=4,curved=True,transverse=True,curved_transverse=True)
    with pytest.raises(ValueError,match='certified monotonicity'):c.axis_bounds((I(0,1),I(0,1)),0)
    for kw in [dict(curved_transverse=True),dict(curved_transverse=1,curved=True,transverse=True,order=4),dict(curved_transverse=True,curved=True,transverse=True),dict(curved=True,transverse=True,order=4)]:
        with pytest.raises(ValueError):TangentCorrection(e,inputs,**kw)


def test_curved_quadrature_lines_keep_crossings_without_optional_newton_work(monkeypatch):
    from types import SimpleNamespace
    from tools.opaque_tangent_peano import TangentCorrection
    from tools.opaque_loss_extension import LossExtension
    import tools.opaque_tangent_transverse as tr
    h=I(2).ln();curve=SimpleNamespace(cn=list(map(I,[Q(1,4),Q(1,2),1])),x=[I(Q(1,4)).ln(),I(Q(1,2)).ln(),I(0)],controls=[[[I(0)]*4]*2,[[I(0),h/3,2*h/3,h]]*2],mean=[I(1),I(1)],pi=I(1));e=LossExtension(curve);e.indices=[0,1]
    def inputs(x,y):return x/2+y/8+x.lift(Q(1,4),tangent=Q(-1,2**60)),x.lift(1),x.lift(1),x.lift(1)
    def forbidden(*a,**kw):pytest.fail('optional line Newton calculation repeated')
    monkeypatch.setattr(tr,'contract_crossing',forbidden)
    c=TangentCorrection(e,inputs,order=4,curved=True,transverse=True,curved_transverse=True)
    rows=c.axis_bounds((I(0,1),I('.5')),0,transverse=False)
    assert len(rows)==2 and all(r['order']==4 and len(r['jumps'])==1 and len(r['jumps'][0]['values'])==4 for r in rows)
    assert any(not v.zero for v in rows[1]['jumps'][0]['values'])


def test_transverse_empty_jump_list_reuses_complete_smooth_proof_without_field_work():
    from tools.opaque_tangent_transverse import refine
    rows=[dict(order=4,derivative=I(Q(1,2**60)),jumps=[])]
    def forbidden(*a,**kw):pytest.fail('field calculation cannot improve absent jump proof')
    assert refine(rows,forbidden,(I(0,1),I(0,1)),0,[I('.5')]) is rows


def test_partitioned_transverse_skips_only_already_enclosed_small_jump():
    from copy import deepcopy
    from tools.opaque_tangent_transverse import refine
    from tools.opaque_shading_intervals import I,Jet
    from fractions import Fraction as Q
    for coefficient in (Q(1,10**40),-Q(1,10**40)):
        def inputs(x,y):return (x+y)/4+Q(1,4),x.lift(1)
        rows=[dict(order=4,jumps=[dict(role=0,knot=0,values=[I(coefficient)]+[I(0)]*3)]) for _ in range(2)]
        box=(I(0,1),I(0,1));knots=[I('.5')]
        expected=refine(deepcopy(rows),inputs,box,0,knots)
        def unnecessary(*a):raise AssertionError('complete tiny jump enclosure needs no partition')
        actual=refine(rows,inputs,box,0,knots,root_parts=2,jump_evaluator=unnecessary)
        for a,b in zip(actual,expected):
            av,bv=a['jumps'][0]['transverse_integral'],b['jumps'][0]['transverse_integral']
            assert (av.lo,av.hi)==(bv.lo,bv.hi) and not av.zero


@pytest.mark.parametrize('axis',[0,1])
@pytest.mark.parametrize('sign',[1,-1])
def test_crossing_contraction_stops_after_sufficient_enclosed_reduction(axis,sign):
    from tools.opaque_tangent_transverse import contract_crossing
    from decimal import Decimal as D,localcontext
    calls=[]
    def inputs(x,y):
        calls.append(1);q=(x,y)[axis]
        return (sign*(q*q+Q(1,10**40)),q.lift(1))
    box=[I(0,1),I(0,1)];box[axis]=I(1,2)
    got=contract_crossing(inputs,tuple(box),axis,0,I(2*sign),I(1,2))
    with localcontext() as c:
        c.prec=120;root=(D(2)-D('1e-40')).sqrt();assert got.lo<=root<=got.hi
    assert got.width<=D('1e-8') and len(calls)<=6
