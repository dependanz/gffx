"""Closed first-segment tangents and strict curved-map proof obligations."""
from fractions import Fraction as Q
from decimal import Decimal as D,localcontext
from types import SimpleNamespace
import pytest
from tools.opaque_shading_intervals import I,Jet
from tools.opaque_full_material_reference import HorizonExtension

def curve():
    h=I(2).ln();row=[I('.5')+h*Q(k,3) for k in range(4)]
    return SimpleNamespace(cn=[I('.5'),I(1)],x=[I('.5').ln(),I(0)],controls=[[row,row]],horizon=[I(0)]*2,values=[[I('.5')]*2,[I('.5')+h]*2],mean=[I(1)]*2,pi=I(1))

@pytest.mark.parametrize('reverse',[False,True])
@pytest.mark.parametrize('jac_motion',[0,Q(3,10)])
def test_first_linear_log_join_keeps_ordinary_tangent_and_measure(reverse,jac_motion):
    from tools.opaque_tangent_peano import TangentCorrection
    e=HorizonExtension(curve());e.indices=[-1,0]
    def inputs(x,y):
        q=((1-x) if reverse else x)/2+x.lift(Q(1,4),tangent=1)
        return q,x.lift(1),x.lift(1),x.lift(1,tangent=jac_motion)
    result=TangentCorrection(e,inputs,horizon=True,curved=True)((I(0,1),I(0,1)))
    # Result is residual relative to extending L(q)=q across the entire domain.
    with localcontext() as ctx:
        ctx.prec=90;h=D(2).ln();L1=D('.5')+h
        primitive=lambda q:D('.5')*q+q*(2*q).ln()-q
        full=2*((D('.5')**2-D('.25')**2)/2+primitive(D('.75'))-primitive(D('.5')))
        residual=(full-D('.5'))*L1
        tangent=(D('.5')+2*D('1.5').ln()-1)*L1+D(jac_motion.numerator)/D(jac_motion.denominator)*residual if isinstance(jac_motion,Q) else (D('.5')+2*D('1.5').ln()-1)*L1
        for got,expected in zip(result,[residual,tangent]):assert got.lo<=expected<=got.hi


def test_curved_map_cannot_borrow_polynomial_crossing_count():
    from tools.opaque_tangent_peano import TangentCorrection
    e=HorizonExtension(curve());e.indices=[-1,0]
    def inputs(x,y):return (x-Q(1,2))**2+Q(3,8),x.lift(1),x.lift(1),x.lift(1)
    with pytest.raises(ValueError,match='monotonicity'):TangentCorrection(e,inputs,horizon=True,curved=True).axis_bounds((I(0,1),I(0,1)),0)


def test_first_segment_retains_tiny_tangent_at_zero_cosine():
    from tools.opaque_tangent_peano import segmented_mixed
    e=HorizonExtension(curve());q=Jet(I(0,D('.25')),tangent=I('1e-70'))
    got=segmented_mixed(e,q,-1,horizon=True)
    assert all(v[0,1].zero for v in got)  # exact same branch, no invented residual
    assert all(v[0,1].lo>0 for v in e.at(q,-1))



def test_first_join_parameter_flux_encloses_known_discontinuous_loss():
    from tools.opaque_knot_transport import material_transport
    c=curve();c.controls=[[[v+I('.2') for v in row] for row in c.controls[0]]]
    def angles(x,y):return [x/2+x.lift(Q(1,4),tangent=1),x.lift(1)]
    got=material_transport(c,angles,(I(0,1),I(0,1)),[1]*3,[.04]*3,horizon=True)
    expected=I('.4')*(I('.7')+I(2).ln())
    assert all(v.hi>=expected.hi and v.hi<expected.hi+ D('1e-45') for v in got['bound'])
    stationary=material_transport(c,lambda x,y:[x/2+Q(1,4),x.lift(1)],(I(0,1),I(0,1)),[1]*3,[.04]*3,horizon=True)
    assert all(v.zero for v in stationary['bound'])


def frozen():
    import os,json,hashlib
    from pathlib import Path
    p=Path(os.environ['GFFX_FULL_INPUT']);assert hashlib.sha256(p.read_bytes()).hexdigest()=='a15bea40cb5cdead7994d828b57f5dcfac9f44fab0292a06e7d0a7b08719f73a';return json.loads(p.read_bytes())

@pytest.mark.parametrize('motion',['normal_tilt','view_x','screen_x'])
def test_original_motion_fields_match_independent_vector_chain(motion):
    from tools.opaque_original_motion_reference import prepare,mapped_inputs
    p=prepare(frozen(),piece=0,motion=motion,anchor=1)
    for a,b in [(Q(1,4),Q(1,3)),(Q(3,4),Q(2,3))]:
        u,s=Jet(a),Jet(b);got=mapped_inputs(p,u,s);tri=p['patch'].triangle
        xy=[tri[0][k]+(tri[1][k]-tri[0][k]+(tri[2][k]-tri[1][k])*b)*a for k in range(2)]
        aa,bb,cc=p['triangle'];det=(bb[0]-aa[0])*(cc[1]-aa[1])-(bb[1]-aa[1])*(cc[0]-aa[0]);x,y=xy
        q=((x-aa[0])*(cc[1]-aa[1])-(y-aa[1])*(cc[0]-aa[0]))*Q(1,det);r=((bb[0]-aa[0])*(y-aa[1])-(bb[1]-aa[1])*(x-aa[0]))*Q(1,det)
        weights=[w*z for w,z in zip([1-q-r,q,r],p['inverse_w'])]
        from tools.opaque_moving_reference import P
        vectors=[[sum((w*P([Q(v[k]),Q(d[k])]) for w,v,d in zip(weights,rows,motions)),P(0)) for k in range(3)] for rows,motions in [(p['source']['normal'],p['normal_motion']),(p['source']['view'],p['view_motion'])]]
        with localcontext() as ctx:
            ctx.prec=90;cast=lambda q:D(q.numerator)/D(q.denominator);n,v=[[cast(x[0]) for x in row] for row in vectors];dn,dv=[[cast(x[1]) for x in row] for row in vectors];l=list(map(D,[1,1,2]));dot=lambda a,b:sum(x*y for x,y in zip(a,b));norm=lambda a:dot(a,a).sqrt()
            def cos(a,b,da,db):
                val=dot(a,b)/(norm(a)*norm(b));dval=(dot(da,b)+dot(a,db))/(norm(a)*norm(b))-val*(dot(a,da)/dot(a,a)+dot(b,db)/dot(b,b));return val,dval
            nv=cos(n,v,dn,dv);nl=cos(n,l,dn,[D(0)]*3);vl,dvl=cos(v,l,dv,[D(0)]*3);vh=((1+vl)/2).sqrt();expected=[nv,nl,(vh,dvl/(4*vh))]
            for actual,pair in zip(got,expected):
                for j,w in enumerate(pair):assert actual[0,j].lo<=w<=actual[0,j].hi

@pytest.mark.parametrize('piece',[8,9,11,12])
@pytest.mark.parametrize('motion',['normal_tilt','view_x','screen_x'])
def test_original_curved_map_keeps_moving_horizon_zero(motion,piece):
    from tools.opaque_original_motion_reference import prepare,mapped_inputs
    p=prepare(frozen(),piece=piece,motion=motion,anchor=0);assert p['curved']
    edge=1 if p['patch'].inside else 0
    for s in [Q(1,4),Q(3,4)]:
        nv,nl,vh,jac=mapped_inputs(p,Jet(edge),Jet(s));assert nv[0,0].zero and nv[0,1].zero
        assert nl[0,0].lo>0 and jac[0,0].zero


def test_selected_original_motion_reference(record_property):
    import json,os
    from pathlib import Path
    from tools.opaque_original_motion_reference import reference
    from tools.opaque_arb_jets import backend
    from tools.opaque_shading_intervals import Unresolved
    rows=[];p=Path(os.environ['GFFX_FULL_REFERENCE']);assert not p.exists();case=json.loads(os.environ['GFFX_MOTION_CASE'])
    try:
        with backend(profile='ball'):result=reference(frozen(),progress=rows.append,**case)
    except Unresolved as e:
        p.write_text(json.dumps(dict(status='unresolved',case=case,details=e.details,progress=rows),default=str,indent=2),encoding='utf8');raise
    p.write_text(json.dumps(dict(status='qualified',case=case,result=result,progress=rows),default=str,indent=2),encoding='utf8');record_property('motion_reference',p.read_text())


def test_curved_range_fallback_contains_closed_two_crossing_integrals():
    from tools.opaque_tangent_peano import TangentCorrection
    class Curve:
        cn=[I('.5'),I(1)];mean=[I(1)]*2;pi=I(1);indices=[-1,0]
        def at(self,q,k):return [q if k==-1 else q*q+Q(1,5)]*2
    e=Curve()
    def inputs(x,y):return (x-Q(1,2))**2+x.lift(Q(3,8),tangent=1),x.lift(1),x.lift(1),x.lift(1)
    calc=TangentCorrection(e,inputs,horizon=True,curved=True,range_fallback=True);result=calc((I(0,1),I(0,1)))
    assert calc.range_fallbacks==1
    with localcontext() as ctx:
        ctx.prec=90;r=D('.125').sqrt();fv=lambda z:z**5/5-z**3/12-D('.034375')*z;fd=lambda z:2*z**3/3-z/4
        for got,expected in zip(result,[D('2.4')*(fv(D('.5'))-fv(r)),D('2.4')*(fd(D('.5'))-fd(r))]):assert got.lo<=expected<=got.hi
    # Same uncertain crossing must still refuse without explicit fallback.
    with pytest.raises(ValueError,match='monotonicity'):TangentCorrection(e,inputs,horizon=True,curved=True)((I(0,1),I(0,1)))


@pytest.mark.parametrize('power',[4,8])
@pytest.mark.parametrize('motion',['normal_tilt','view_x','screen_x'])
def test_unflattened_original_chart_preserves_angles_tangents_and_measure(motion,power):
    from tools.opaque_original_motion_reference import prepare,mapped_inputs,unflattened_box
    p=prepare(frozen(),piece=8,motion=motion,anchor=0,horizon_power=power)
    for u,s in [(Q(1,4),Q(1,3)),(Q(3,4),Q(2,3))]:
        v=1-u if p['patch'].inside else u;z=1-v**power if p['patch'].inside else v**power;factor=power*v**(power-1)
        a=mapped_inputs(p,Jet(u),Jet(s));b=mapped_inputs(p,Jet(z),Jet(s),flatten=False)
        for x,y in zip(a[:3],b[:3]):
            for j in (0,1):assert max(x[0,j].lo,y[0,j].lo)<=min(x[0,j].hi,y[0,j].hi)
        for j in (0,1):
            q=b[3][0,j]*factor;assert max(a[3][0,j].lo,q.lo)<=min(a[3][0,j].hi,q.hi)
    box=unflattened_box((I(0,D('.25')),I(0,1)),inside=False);assert box[0].lo==0 and box[0].hi==D(1)/256


def test_unflattened_flux_removes_coordinate_zero_without_discarding_transport():
    from tools.opaque_knot_transport import transversal_flux
    from tools.opaque_original_motion_reference import unflattened_box
    box=(I(0,D('.75')),I(0,1));knot=Q(1,16)
    with pytest.raises(ValueError,match='transversal'):transversal_flux(lambda u,s:u**4+u.lift(0,tangent=1),box,[(knot,1)])
    # Physical measure 4u^3 du becomes dz exactly. Integral H(z-c+theta) dz
    # has derivative +1. No root or tiny positive term is removed.
    result=transversal_flux(lambda z,s:z+z.lift(0,tangent=1),unflattened_box(box,inside=False),[(knot,1)])
    assert result['bound'].hi==1


def test_saved_horizon_transport_chart_bounds(record_property):
    import json,os
    from pathlib import Path
    from tools.opaque_original_motion_reference import reference
    from tools.opaque_arb_jets import backend
    boxes=[(Q(0),Q(1,128),Q(0),Q(1,8))]
    with backend(profile='ball'):result=reference(frozen(),piece=8,motion='normal_tilt',anchor=0,diagnostic_boxes=boxes)
    p=Path(os.environ['GFFX_FULL_REFERENCE']);assert not p.exists();p.write_text(json.dumps(result,default=str,indent=2),encoding='utf8');record_property('saved_horizon_bounds',p.read_text())
    assert result['status']=='diagnostic_only' and result['rows'][0]['range_fallbacks']>0


def test_original_horizon_domain_preflight(record_property):
    import json,os
    from pathlib import Path
    from tools.opaque_original_motion_reference import reference
    from tools.opaque_arb_jets import backend
    from tools.opaque_shading_intervals import Unresolved
    case=json.loads(os.environ['GFFX_MOTION_CASE']);p=Path(os.environ['GFFX_FULL_REFERENCE']);assert not p.exists()
    try:
        with backend(profile='ball'):result=reference(frozen(),**case,domain_only=True)
    except Unresolved as e:
        p.write_text(json.dumps(dict(status='unresolved',case=case,details=e.details),default=str,indent=2),encoding='utf8');raise
    p.write_text(json.dumps(result,default=str,indent=2),encoding='utf8');record_property('horizon_domain_preflight',p.read_text())
