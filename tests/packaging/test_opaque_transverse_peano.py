"""Exact moments and clipped-boundary controls for transverse error integration."""
from fractions import Fraction as Q
import math,json
import pytest
from tools.opaque_shading_intervals import I,Jet


@pytest.mark.parametrize('order',[1,2,3])
def test_kernel_primitive_preserves_exact_zero_moment(order):
    from tools.opaque_transverse_peano import kernel_integral
    v=kernel_integral(I(0),I(1),order)
    assert v.lo<=0<=v.hi and v.width<I('1e-50').lo


@pytest.mark.parametrize('order',[1,2,3])
def test_integrated_kernel_matches_independent_polynomial_antiderivative(order):
    from tools.opaque_transverse_peano import kernel_integral
    from tools.opaque_peano import gauss5
    a=I('.11');b=I('.17');nodes,weights=gauss5()
    # Neither endpoint is a Gauss node; integrate each active truncated power
    # directly in x rather than use the implementation's primitive.
    direct=I(0)
    for x,w in zip(nodes,weights):
        if x.hi<=a.lo:continue
        high=b if b.hi<x.lo else x
        direct=direct+w*((x-a)**order-(x-high)**order)/math.factorial(order)
    direct=direct-((1-a)**(order+1)-(1-b)**(order+1))/math.factorial(order+1)
    got=kernel_integral(a,b,order)
    assert got.lo<=direct.hi and direct.lo<=got.hi and got.width<I('1e-50').lo


@pytest.mark.parametrize('orientation',[-1,1])
@pytest.mark.parametrize('root',[Q(-1),Q(0),Q(3,10),Q(1),Q(2)])
def test_clamped_root_encloses_exact_affine_crossing(root,orientation):
    from tools.opaque_transverse_peano import clamped_crossing
    domain=I(2,4);actual=2+2*root;k=Q(7);slope=Q(3)*orientation
    q=lambda x:I(k+slope*(x-actual))
    v=clamped_crossing(domain,q(2),q(4),q(3),I(slope),I(k));target=min(Q(4),max(Q(2),actual))
    assert Q(v.lo)<=target<=Q(v.hi) and v.width<I('1e-50').lo


def test_clamped_root_refuses_nonmonotone_map():
    from tools.opaque_transverse_peano import clamped_crossing
    with pytest.raises(ValueError):clamped_crossing(I(0,1),I(0),I(1),I('.5'),I(-1,1),I('.5'))


def test_weighted_kernel_keeps_coefficient_variation_and_endpoint_uncertainty():
    from tools.opaque_transverse_peano import weighted_kernel_integral,kernel_integral
    coefficient=I('-1e-9','3e-9');a=I('.12','.13');b=I('.8','.81')
    got=weighted_kernel_integral(coefficient,a,b,3)
    # Constant coefficients and endpoint extremes are independent admissible cases.
    for c in [coefficient.lo,coefficient.hi]:
        for lo in [a.lo,a.hi]:
            for hi in [b.lo,b.hi]:
                target=I(c)*kernel_integral(I(lo),I(hi),3)
                assert got.lo<=target.lo<=target.hi<=got.hi
    assert got.width>0


@pytest.mark.parametrize('degree',[0,1,2])
@pytest.mark.parametrize('orientation',[-1,1])
def test_conditioned_error_recovers_diagonal_clipped_polynomial(degree,orientation,record_property):
    from tools.opaque_transverse_peano import conditioned_integral,weighted_kernel_integral
    from tools.opaque_peano import residual_integral
    scale=Q(1,10**9);box=(I(0,1),I(0,1))
    def field(x,y):
        z=(x[0,0]-y[0,0])*orientation
        if z.hi<0:v=I(0)
        elif z.lo>0:v=z**degree
        else:v=I(0,max(I(1).hi,(I(0,max(I(0).hi,z.hi))**degree).hi)) if degree==0 else I(0,(I(0,max(I(0).hi,z.hi))**degree).hi)
        return [x.lift(v*scale)]
    def bounds(box,axis):
        signed=[I(0)]*3;signed[degree]=I(scale*math.factorial(degree)*(orientation*(1 if axis==0 else -1))**(degree+1))
        w=dict(position=box[1-axis],jumps=[I(v.bound) for v in signed],signed_jumps=signed,required=True,max_crossings=1)
        return [dict(third=I(0),jumps=[I(0)]*3,localized_jumps=[w])]
    def integrated(box,axis):
        rows=bounds(box,axis);w=rows[0]['localized_jumps'][0];w['transverse_integral']=weighted_kernel_integral(w['signed_jumps'][degree],I(0),I(1),degree+1);return rows
    got=conditioned_integral(field,box,bounds,integrated)[0];old=residual_integral(field,box,bounds)[0]
    exact=scale/Q((degree+1)*(degree+2))
    assert Q(got.lo)<=exact<=Q(got.hi)
    # A value jump sampled exactly on x=y legitimately retains node ambiguity.
    if degree:assert got.width<I('1e-45').lo and got.width*1000<old.width
    record_property('transverse_exact',json.dumps(dict(degree=degree,orientation=orientation,width=str(got.width),previous_width=str(old.width))))


def test_conditioned_integral_rejects_tangents():
    from tools.opaque_transverse_peano import conditioned_integral
    with pytest.raises(ValueError,match='fixed'):conditioned_integral(lambda x,y:[Jet(1,tangent=1)],(I(0,1),I(0,1)),lambda b,a:[],lambda b,a:[])


def _log_rectangle(e,eps,slope,a,b,c,width=I(1)):
    # Independent twice-integrated log polynomial, anchored with zero value and
    # first derivative at the knot; its mixed rectangle difference is exact.
    k=I('.5')
    def h(q):
        if q.hi<=k.lo:return I(0)
        assert q.lo>=k.hi
        z=(q/k).ln()
        p=q*q/2*(z*z-3*z+I(Q(7,2))+slope*(z-I(Q(3,2)))+eps)
        p0=k*k/2*(I(Q(7,2))-slope*I(Q(3,2))+eps)
        d0=k*(2-slope+eps)
        return p-p0-d0*(q-k)
    answer=(h(a+b*width+c)-h(a+b*width)-h(a+c)+h(a))/(b*c)
    z=I('1.5').ln()
    return answer*I('.75')*(1+2*z*z+eps+slope*z)


@pytest.mark.parametrize('orientation_x',[-1,1])
@pytest.mark.parametrize('orientation_y',[-1,1])
@pytest.mark.parametrize('offset',['.4988','.4992','.4998'])
def test_material_transverse_producer_encloses_clipped_log_rectangle(orientation_x,orientation_y,offset,record_property):
    from tools.opaque_peano_material import MaterialCorrection
    from tests.packaging.test_opaque_localized_peano import material_fixture
    e,eps,slope=material_fixture();b=I('.001')*orientation_x;c=I('.001')*orientation_y
    a=I(offset)+(I('.001') if orientation_x<0 else I(0))+(I('.001') if orientation_y<0 else I(0))
    def inputs(x,y):return x*b+y*c+a,x.lift('.75'),x.lift(1),x.lift(1)
    correction=MaterialCorrection(e,inputs,curved=False,transverse=True);box=(I(0,1),I(0,1))
    w=correction.integrated_bounds(box,1)[1]['localized_jumps'][0]
    assert 'transverse_integral' in w and len(w['signed_jumps'])==3
    assert all(v.bound>0 for v in w['signed_jumps'])
    got=correction(box)[1];known=_log_rectangle(e,eps,slope,a,b,c)
    assert got.lo<=known.lo<=known.hi<=got.hi
    record_property('transverse_log',json.dumps(dict(offset=offset,orientation_x=orientation_x,orientation_y=orientation_y,width=str(got.width),known=str(known),integrated_jump=str(w['transverse_integral']))))


def test_material_transverse_curved_map_preserves_jacobian_and_exact_integral(record_property):
    from tools.opaque_peano_material import MaterialCorrection
    from tests.packaging.test_opaque_localized_peano import material_fixture
    e,eps,slope=material_fixture();a=I('.4988');b=I('.001');c=I('.001')
    def inputs(x,y):return (x+x*x)*b+y*c+a,x.lift('.75'),x.lift(1),1+2*x
    correction=MaterialCorrection(e,inputs,curved=True,transverse=True);box=(I(0,1),I(0,1))
    w=correction.integrated_bounds(box,1)[1]['localized_jumps'][0]
    assert 'transverse_integral' in w
    got=correction(box)[1];known=_log_rectangle(e,eps,slope,a,b,c,I(2))
    assert got.lo<=known.lo<=known.hi<=got.hi
    record_property('transverse_curved_log',json.dumps(dict(width=str(got.width),known=str(known))))


def test_material_transverse_zero_derivative_keeps_conservative_fallback():
    from tools.opaque_peano_material import MaterialCorrection
    from tests.packaging.test_opaque_localized_peano import material_fixture
    e,_,_=material_fixture()
    def inputs(x,y):return y*I('.001')+I('.4995'),x.lift('.75'),x.lift(1),x.lift(1)
    c=MaterialCorrection(e,inputs,curved=False,transverse=True)
    w=c.integrated_bounds((I(0,1),I(0,1)),1)[1]['localized_jumps'][0]
    assert 'transverse_integral' not in w
    assert c((I(0,1),I(0,1)))[1].width>0


@pytest.mark.parametrize('offset',['.4988','.4992'])
def test_material_transverse_work_uses_same_accuracy_and_cell_budget(offset,record_property):
    from tools.opaque_peano_material import MaterialCorrection
    from tools.opaque_square_reference import integrate_square
    from tests.packaging.test_opaque_localized_peano import material_fixture
    e,eps,slope=material_fixture();a=I(offset);b=I('.001');c=I('.001');calls=[0]
    def inputs(x,y):
        calls[0]+=1
        return x*b+y*c+a,x.lift('.75'),x.lift(1),x.lift(1)
    results=[]
    for transverse in (False,True):
        calls[0]=0
        correction=MaterialCorrection(e,inputs,curved=False,transverse=transverse)
        r=integrate_square(lambda x,y:[x.lift(0),x.lift(0)],correction_integrator=correction,tolerance='1e-12',max_cells=256,max_depth=10)
        r['input_calls']=calls[0];results.append(r)
    known=_log_rectangle(e,eps,slope,a,b,c)
    for r in results:
        v=r['value'][1];assert v.lo<=known.lo<=known.hi<=v.hi and v.width<=I('2e-12').lo
    record_property('transverse_material_work',json.dumps(dict(offset=offset,old_cells=results[0]['cells'],new_cells=results[1]['cells'],old_calls=results[0]['input_calls'],new_calls=results[1]['input_calls'],old_seconds=results[0]['seconds'],new_seconds=results[1]['seconds'],old_width=str(results[0]['value'][1].width),new_width=str(results[1]['value'][1].width))))


@pytest.mark.parametrize('orientation',[-1,1])
def test_material_transverse_nonunit_box_keeps_physical_scale(orientation):
    from tools.opaque_peano_material import MaterialCorrection
    from tests.packaging.test_opaque_localized_peano import material_fixture
    e,eps,slope=material_fixture();b=I('.001');c=I('.001')*orientation;a=I('.4992')+(I('.001') if orientation<0 else I(0))
    def inputs(x,y):return (x-2)*b/2+(y+3)*c/3+a,x.lift('.75'),x.lift(1),x.lift(I(Q(1,6)))
    correction=MaterialCorrection(e,inputs,curved=False,transverse=True)
    got=correction((I(2,4),I(-3,0)))[1];known=_log_rectangle(e,eps,slope,a,b,c)
    assert got.lo<=known.lo<=known.hi<=got.hi


def test_material_transverse_nonmonotone_outer_map_keeps_whole_domain_bound():
    from tools.opaque_peano_material import MaterialCorrection
    from tests.packaging.test_opaque_localized_peano import material_fixture
    e,_,_=material_fixture()
    def inputs(x,y):return (x-I('.5'))**2*I('.001')+y*I('.001')+I('.4995'),x.lift('.75'),x.lift(1),x.lift(1)
    correction=MaterialCorrection(e,inputs,curved=False,transverse=True)
    w=correction.integrated_bounds((I(0,1),I(0,1)),1)[1]['localized_jumps'][0]
    assert 'transverse_integral' not in w and w['max_crossings']==1
    with pytest.raises(ValueError,match='monotonicity'):MaterialCorrection(e,inputs,curved=True,transverse=True)((I(0,1),I(0,1)))


def test_transverse_integral_refuses_unsigned_witness():
    from tools.opaque_transverse_peano import conditioned_integral
    def bounds(box,axis):return [dict(third=I(0),jumps=[I(0)]*3,localized_jumps=[])]
    def integrated(box,axis):return [dict(third=I(0),jumps=[I(0)]*3,localized_jumps=[dict(position=I('.5'),jumps=[I(0)]*3,transverse_integral=I(0))])]
    with pytest.raises(ValueError,match='signed crossing'):conditioned_integral(lambda x,y:[x.lift(0)],(I(0,1),I(0,1)),bounds,integrated)
