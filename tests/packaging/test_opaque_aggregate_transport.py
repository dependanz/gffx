"""Exact swept-knot controls for the integrated aggregate jump remainder."""
from fractions import Fraction as Q
import pytest

def fixture(sx=1,sy=1,vary=False):
    from tools.opaque_shading_intervals import I
    from tools.opaque_full_material_reference import HorizonExtension
    from tools.opaque_peano_material import MaterialCorrection
    from types import SimpleNamespace
    cn=[I('.1'),I('.5'),I(1)];x=[q.ln() for q in cn]
    # Continuous value/slope with a curvature jump in log(cosine).
    h=x[2]-x[1];controls=[[[I(1)]*4]*2,[[I(1),I(1),1+h*h/3,1+h*h]]*2]
    e=HorizonExtension(SimpleNamespace(cn=cn,x=x,controls=controls,mean=[I(1)]*2,pi=I(1),horizon=[I(1)]*2,values=[[I(1)]*2,[I(1)]*2,[1+h*h]*2]));e.indices=[0,1]
    def inputs(u,v):
        a=u if sx>0 else 1-u;b=v if sy>0 else 1-v
        return a*I('.01')+b*I('.4')+I('.3'),u.lift(1),u.lift(1),1+u*I('.01') if vary else u.lift(1)
    return MaterialCorrection(e,inputs,curved=True)

@pytest.mark.parametrize('sx,sy',[(1,1),(-1,1),(1,-1),(-1,-1)])
def test_full_sweep_bound_retains_curvature_and_lower_jump_terms(sx,sy):
    from tools.opaque_shading_intervals import I
    from tools.opaque_aggregate_peano import AggregateMaterial
    from tools.opaque_aggregate_transport import integrated_jumps
    from tools.opaque_peano import kernel_bounds
    c=fixture(sx,sy);proof=AggregateMaterial(c);box=(I(0,1),I(0,1));raw=proof.axis_bounds(box,0)[1];k,_=kernel_bounds();old=sum((j*k[i] for i,j in enumerate(raw['jumps'])),I(0));new=integrated_jumps(proof,box,0)
    assert new.lo>=0 and new.hi<old.hi/10
    # Exact constant spatial coefficient times K3 has zero complete moment;
    # all outward resource uncertainty and tiny lower jumps remain nonzero.
    assert new.hi>0

def test_unknown_outer_sign_keeps_uniform_bound():
    from tools.opaque_shading_intervals import I
    from tools.opaque_aggregate_peano import AggregateMaterial
    from tools.opaque_aggregate_transport import integrated_jumps
    from tools.opaque_peano import kernel_bounds
    c=fixture();old=c.inputs
    def inputs(u,v):
        row=list(old(u,(v-I('.5'))**2));row[0]=row[0]+I('.15');return row
    c.inputs=inputs;proof=AggregateMaterial(c);box=(I(0,1),I(0,1));row=proof.axis_bounds(box,0)[1];k,_=kernel_bounds();expected=sum((j*k[i] for i,j in enumerate(row['jumps'])),I(0));got=integrated_jumps(proof,box,0)
    assert got.hi>=expected.lo and got.lo>=0


@pytest.mark.parametrize('sx,sy',[(1,1),(-1,1),(1,-1),(-1,-1)])
@pytest.mark.parametrize('axis',[0,1])
def test_transported_integral_contains_independent_log_squared_rectangle(sx,sy,axis):
    from decimal import Decimal as D,localcontext
    from tools.opaque_aggregate_peano import AggregateMaterial
    from tools.opaque_shading_intervals import I
    c=fixture(sx,sy);got=AggregateMaterial(c)((I(0,1),I(0,1)),integrated_axis=axis)[1]
    with localcontext() as context:
        context.prec=90;k=D('.5');a=D('.3');b=D('.01');d=D('.4')
        def primitive(q):
            if q<=k:return D(0)
            z=(q/k).ln();return q*q/2*(z*z-3*z+D('3.5'))-2*k*q+k*k/4
        expected=(primitive(a+b+d)-primitive(a+b)-primitive(a+d)+primitive(a))/(b*d)*(1+D(2).ln()**2)
        assert got.lo<=expected<=got.hi

def test_no_knots_do_not_require_inner_monotonicity():
    from tools.opaque_aggregate_peano import AggregateMaterial
    from tools.opaque_aggregate_transport import integrated_jumps
    from tools.opaque_shading_intervals import I
    c=fixture();old=c.inputs
    def inputs(u,v):return old((u-I('.5'))**2,v*0)
    c.inputs=inputs;assert integrated_jumps(AggregateMaterial(c),(I(0,1),I(0,1)),0).zero

def test_invalid_integrated_bound_is_rejected():
    from tools.opaque_transverse_peano import _error
    from tools.opaque_shading_intervals import I
    with pytest.raises(ValueError,match='nonnegative integrated jump'):_error(dict(third=I(0),jumps=[I(0)]*3,global_integral=I(-1)),I(0,1),I(1),True)
