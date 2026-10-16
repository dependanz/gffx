"""Closed integrals for spatially localized Peano jump bounds; no original rerun."""
from fractions import Fraction as Q
import json,math
import pytest
from tools.opaque_shading_intervals import I,Jet

@pytest.mark.parametrize('degree',[0,1,2])
@pytest.mark.parametrize('knot',[Q(1,100),Q(99,100),Q(1,2)])
def test_localized_jump_encloses_exact_truncated_power(degree,knot,record_property):
    from tools.opaque_peano import residual_integral,kernel_local_bounds
    scale=Q(1,10**9)
    def field(x,y):
        z=x[0,0]-I(knot)
        if z.hi<0:return [x.lift(0)]
        if z.lo>0:return [((x-knot)**degree)*scale]
        return [x.lift(I(0,1))*scale]
    jumps=[I(0)]*3;jumps[degree]=I(scale*math.factorial(degree))
    def bounds(box,axis):return [dict(third=I(0),jumps=[I(0)]*3,localized_jumps=[] if axis else [dict(position=I(knot),jumps=jumps)])]
    def old(box,axis):return [dict(third=I(0),jumps=[I(0)]*3 if axis else jumps)]
    got=residual_integral(field,(I(0,1),I(0,1)),bounds)[0];prior=residual_integral(field,(I(0,1),I(0,1)),old)[0]
    exact=scale*(1-knot)**(degree+1)/(degree+1)
    assert Q(got.lo)<=exact<=Q(got.hi)
    assert got.width<=prior.width
    if degree==2 and knot!=Q(1,2):assert got.width*100<prior.width and got.width<I('2e-12').lo
    record_property('localized_bound',json.dumps(dict(degree=degree,knot=str(knot),width=str(got.width),previous_width=str(prior.width))))


def test_oblique_crossing_retains_value_slope_and_curvature_jumps(record_property):
    from tools.opaque_peano import residual_integral,kernel_local_bounds
    a=Q(1,100);b=Q(1,100);scale=Q(1,10**9);coeff=[Q(1,10**6),Q(2,10**6),Q(1)]
    def field(x,y):
        z=x[0,0]-I(a)-I(b)*y[0,0]
        if z.hi<0:return [x.lift(0)]
        lo=max(Q(0),Q(z.lo));hi=max(Q(0),Q(z.hi));v=sum((I(c)*I(I(lo**j).lo,I(hi**j).hi) for j,c in enumerate(coeff)),I(0))*scale
        if z.lo<=0:v=I(0,v.hi)
        return [x.lift(v)]
    def bounds(box,axis,localized=True):
        jumps=[I(c*scale*math.factorial(j)*((-b if axis else Q(1))**j)).bound for j,c in enumerate(coeff)]
        position=I(0,1) if axis else I(I(a).lo,I(a+b).hi)
        return [dict(third=I(0),jumps=[I(0)]*3 if localized else jumps,localized_jumps=[dict(position=position,jumps=jumps)] if localized else [])]
    box=(I(0,1),I(0,1));got=residual_integral(field,box,bounds)[0];old=residual_integral(field,box,lambda box,axis:bounds(box,axis,False))[0]
    exact=scale*sum((c*((1-a)**(j+2)-(1-a-b)**(j+2))/(b*(j+1)*(j+2)) for j,c in enumerate(coeff)),Q(0))
    assert Q(got.lo)<=exact<=Q(got.hi) and got.width*50<old.width and got.width<I('2e-12').lo
    record_property('oblique_bound',json.dumps(dict(width=str(got.width),previous_width=str(old.width),exact=str(exact))))


def test_crossing_location_uses_certified_mean_value_inclusion():
    from tools.opaque_peano import crossing_location
    # q(x,y)=2*x+y, x in[2,4], y in[.1,.2], q(mid,y) in[6.1,6.2].
    root=crossing_location(I(2,4),I('6.1','6.2'),I(2),I(7))
    assert root.lo<=I('3.4').lo and root.hi>=I('3.45').hi and root.width<=I('.051').lo
    # Decreasing coordinate and interval-valued knot must retain every root.
    root=crossing_location(I(2,4),I('-5.9','-5.8'),I(-2),I('-7.01','-6.99'))
    for y in (Q(1,10),Q(1,5)):
        for k in (Q(-701,100),Q(-699,100)):
            assert Q(root.lo)<=(y-k)/2<=Q(root.hi)
    assert crossing_location(I(2,4),I(0,1),I(-1,1),I(9)).lo==2
    assert crossing_location(I(2,4),I(6),I(2),I(20)) is None


def test_local_kernel_hull_covers_gauss_node_uncertainty_and_global_fallback():
    from tools.opaque_peano import kernel_local_bounds,kernel_suprema,gauss5
    old=kernel_suprema();nodes,_=gauss5()
    assert all(a.hi<=b.hi for a,b in zip(kernel_local_bounds(I(0,1)),old))
    for x in nodes:
        q=kernel_local_bounds(x)
        # The discontinuous K1 uses both neighboring polynomial pieces.
        assert q[0].hi>0 and all(a.hi<=b.hi for a,b in zip(q,old))
    assert kernel_local_bounds(I('.009','.011'))[2].hi*100<old[2].lo


@pytest.mark.parametrize('entry',[dict(position=I(-1,0),jumps=[I(0)]*3),dict(position=I(0,2),jumps=[I(0)]*3),dict(position=I(0,1),jumps=[I(-1)]*3),dict(position=I(0,1),jumps=[I(1)])])
def test_localized_witness_refuses_invalid_domain_or_jump_bounds(entry):
    from tools.opaque_peano import residual_integral,kernel_local_bounds
    with pytest.raises(ValueError):residual_integral(lambda x,y:[x],(I(0,1),I(0,1)),lambda box,axis:[dict(third=I(0),jumps=[I(0)]*3,localized_jumps=[entry])])


def test_localized_witness_count_remains_bounded():
    from tools.opaque_peano import residual_integral,kernel_local_bounds
    entry=dict(position=I(0,1),jumps=[I(0)]*3)
    with pytest.raises(ValueError):residual_integral(lambda x,y:[x],(I(0,1),I(0,1)),lambda box,axis:[dict(third=I(0),jumps=[I(0)]*3,localized_jumps=[entry]*17)])


def material_fixture():
    from types import SimpleNamespace
    from tools.opaque_full_material_reference import HorizonExtension
    cn=[I('.25'),I('.5'),I(1)];x=[q.ln() for q in cn];eps=I('1e-17');slope=I('1e-12');controls=[]
    for i in range(2):
        h=x[i+1]-x[i];shift=x[i]-x[1];factor=I(i+1)
        a=1+factor*shift**2+(eps+slope*shift if i else I(0));b=2*factor*shift+(slope if i else I(0));c=factor
        controls.append([[a,a+h*b/3,a+2*h*b/3+h*h*c/3,a+h*b+h*h*c]]*2)
    extension=HorizonExtension(SimpleNamespace(cn=cn,x=x,controls=controls,mean=[I(1)]*2,pi=I(1)))
    extension.indices=[0,1]
    return extension,eps,slope


def test_material_producer_localizes_real_log_spline_jumps_and_preserves_integral(record_property):
    from tools.opaque_peano_material import MaterialCorrection
    from tools.opaque_peano import residual_integral
    e,eps,slope=material_fixture();offset=I('.4999');rate=I('.01')
    def inputs(x,y):return x*rate+offset,x.lift('.75'),x.lift(1),x.lift(1)
    correction=MaterialCorrection(e,inputs,curved=False);box=(I(0,1),I(0,1));axis=correction.axis_bounds(box,0)
    witnesses=axis[1]['localized_jumps'];assert len(witnesses)==1
    assert witnesses[0]['position'].lo<=I('.01').lo<=witnesses[0]['position'].hi
    assert witnesses[0]['position'].width<I('1e-50').lo
    def global_bounds(box,axis):
        rows=correction.axis_bounds(box,axis)
        for row in rows:
            for witness in row.pop('localized_jumps',[]):row['jumps']=[a+b for a,b in zip(row['jumps'],witness['jumps'])]
        return rows
    got=correction(box)[1];old=residual_integral(correction.ordinary,box,global_bounds)[1]
    def primitive(t):
        z=(t/I('.5')).ln();return t*(z*z-2*z+2+slope*(z-1)+eps)
    known=(primitive(offset+rate)-primitive(I('.5')))/rate
    z=I('1.5').ln();known=known*I('.75')*(1+2*z*z+eps+slope*z)
    assert got.lo<=known.lo<=known.hi<=got.hi
    assert got.width*2<old.width
    assert correction.error_breakdown[0][1]['localized_jumps'].hi>0
    assert correction.error_breakdown[0][1]['ordinary'].hi>0
    record_property('material_localization',json.dumps(dict(width=str(got.width),previous_width=str(old.width),exact_enclosure=str(known),crossing=str(witnesses[0]['position']),breakdown=correction.error_breakdown),default=str))


def test_material_nonmonotone_roots_keep_existing_count_and_curved_refusal():
    from tools.opaque_peano_material import MaterialCorrection
    e,_,_=material_fixture()
    def inputs(x,y):return (x-I('.5'))**2*I('.01')+I('.499'),x.lift('.75'),x.lift(1),x.lift(1)
    box=(I(0,1),I(0,1));rows=MaterialCorrection(e,inputs,curved=False).axis_bounds(box,0)
    witness=rows[1]['localized_jumps'][0]
    assert witness['position'].lo==0 and witness['position'].hi==1 and witness['max_crossings']==4
    with pytest.raises(ValueError,match='monotonicity'):MaterialCorrection(e,inputs,curved=True).axis_bounds(box,0)


def test_work_refusal_preserves_last_complete_enclosure_and_error_components():
    from tools.opaque_square_reference import integrate_square
    from tools.opaque_shading_intervals import Unresolved
    from tools.opaque_peano_material import MaterialCorrection
    # A single x^10 cell has a nonzero Gauss5 remainder; refining it removes
    # that parent from the active list. The last complete cover must survive.
    with pytest.raises(Unresolved) as error:integrate_square(lambda x,y:[x**10],tolerance='1e-12',max_cells=1)
    record=error.value.details['last_complete_enclosure'];lo,hi=record['value'][0]
    assert Q(lo)<=Q(1,11)<=Q(hi) and record['cells']==1 and record['leaves']==1
    assert record['complete_cover'] is True
    e,_,_=material_fixture()
    def inputs(x,y):return x*I('.01')+I('.4999'),x.lift('.75'),x.lift(1),x.lift(1)
    correction=MaterialCorrection(e,inputs,curved=False)
    with pytest.raises(Unresolved) as error:integrate_square(lambda x,y:[x.lift(0),x.lift(0)],correction_integrator=correction,tolerance='1e-12',max_cells=1)
    parts=error.value.details['last']['correction_error_components']
    assert len(parts)==2 and Q(parts[0][1]['ordinary'][1])>0 and Q(parts[0][1]['localized_jumps'][1])>0


@pytest.mark.parametrize('orientation',[-1,1])
def test_material_signed_crossing_has_correct_orientation_and_log_integral(orientation,record_property):
    from tools.opaque_peano_material import MaterialCorrection
    from tools.opaque_peano import residual_integral
    e,eps,slope=material_fixture();offset=I('.4999') if orientation==1 else I('.5099');rate=I('.01')*orientation
    def inputs(x,y):return x*rate+offset,x.lift('.75'),x.lift(1),x.lift(1)
    correction=MaterialCorrection(e,inputs,curved=False);box=(I(0,1),I(0,1));w=correction.axis_bounds(box,0)[1]['localized_jumps'][0]
    assert w['required'] is True and w['max_crossings']==1
    assert w['signed_jumps'][0].lo>0 if orientation==1 else w['signed_jumps'][0].hi<0
    def unsigned(box,axis):
        rows=correction.axis_bounds(box,axis)
        for row in rows:
            for witness in row.get('localized_jumps',[]):witness.pop('signed_jumps',None)
        return rows
    got=correction(box)[1];old=residual_integral(correction.ordinary,box,unsigned)[1]
    def primitive(t):
        z=(t/I('.5')).ln();return t*(z*z-2*z+2+slope*(z-1)+eps)
    known=(primitive(I('.5099'))-primitive(I('.5')))/I('.01');z=I('1.5').ln();known=known*I('.75')*(1+2*z*z+eps+slope*z)
    assert got.lo<=known.lo<=known.hi<=got.hi
    # This near-endpoint fixture is dominated by the retained ORDINARY remainder.
    # The initial whole-bound 2x prediction failed and remains in material1.
    # Test the actual signed-jump obligation separately; do not waive containment.
    assert got.width<old.width
    parts=[];residual_integral(correction.ordinary,box,unsigned,error_breakdown=parts.append)
    signed_part=correction.error_breakdown[0][1]['localized_jumps']
    unsigned_part=parts[0][0][1]['localized_jumps']
    assert signed_part.hi*100<unsigned_part.hi
    record_property('signed_material',json.dumps(dict(orientation=orientation,width=str(got.width),unsigned_width=str(old.width),signed_jump_radius=str(signed_part.hi),unsigned_jump_radius=str(unsigned_part.hi),known=str(known))))


def test_material_partial_crossing_keeps_absence_and_nonmonotone_fallback():
    from tools.opaque_peano_material import MaterialCorrection
    e,_,_=material_fixture();box=(I(0,1),I(0,1))
    def inputs(x,y):return (x+y)*I('.01')+I('.495'),x.lift('.75'),x.lift(1),x.lift(1)
    for axis in (0,1):
        w=MaterialCorrection(e,inputs,curved=False).axis_bounds(box,axis)[1]['localized_jumps'][0]
        assert w['required'] is False and len(w['signed_jumps'])==3
    def nonmonotone(x,y):return (x-I('.5'))**2*I('.01')+I('.499'),x.lift('.75'),x.lift(1),x.lift(1)
    w=MaterialCorrection(e,nonmonotone,curved=False).axis_bounds(box,0)[1]['localized_jumps'][0]
    assert w['max_crossings']==4 and 'signed_jumps' not in w


@pytest.mark.parametrize('knot',[Q(13,100),Q(87,100)])
def test_signed_material_refinement_reduces_work_with_same_acceptance(knot,record_property):
    from tools.opaque_peano_material import MaterialCorrection
    from tools.opaque_peano import residual_integral
    from tools.opaque_square_reference import integrate_square
    e,eps,slope=material_fixture();rate=I('.001');offset=I('.5')-I(knot)*rate
    def inputs(x,y):return x*rate+offset,x.lift('.75'),x.lift(1),x.lift(1)
    signed=MaterialCorrection(e,inputs,curved=False)
    class Unsigned(MaterialCorrection):
        def axis_bounds(self,box,axis):
            rows=super().axis_bounds(box,axis)
            for row in rows:
                for w in row.get('localized_jumps',[]):w.pop('signed_jumps',None)
            return rows
    unsigned=Unsigned(e,inputs,curved=False)
    def base(x,y):return [x.lift(0),x.lift(0)]
    results=[integrate_square(base,correction_integrator=c,tolerance='1e-12',max_cells=256,max_depth=10) for c in (signed,unsigned)]
    def primitive(t):
        z=(t/I('.5')).ln();return t*(z*z-2*z+2+slope*(z-1)+eps)
    known=(primitive(offset+rate)-primitive(I('.5')))/rate;z=I('1.5').ln();known=known*I('.75')*(1+2*z*z+eps+slope*z)
    for r in results:
        v=r['value'][1];assert v.lo<=known.lo<=known.hi<=v.hi and v.width<=I('2e-12').lo
    assert results[0]['cells']<results[1]['cells']
    record_property('signed_material_work',json.dumps(dict(knot=str(knot),signed_cells=results[0]['cells'],unsigned_cells=results[1]['cells'],signed_seconds=results[0]['seconds'],unsigned_seconds=results[1]['seconds'],signed_width=str(results[0]['value'][1].width),unsigned_width=str(results[1]['value'][1].width),known=str(known))))
