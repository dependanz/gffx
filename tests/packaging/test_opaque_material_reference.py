"""Closed-form controls for material branches and roughness jets."""
from fractions import Fraction as Q
from decimal import Decimal as D,localcontext
import math,pytest
from tools.opaque_shading_intervals import Jet,Interval as I

def api():
    from tools.opaque_material_reference import exp_jet,select,partition,integrate_partition,bilinear
    return exp_jet,select,partition,integrate_partition,bilinear

def has(a,x):
    with localcontext() as c:
        c.prec=90;q=Q(x);v=D(q.numerator)/D(q.denominator)
    assert a.lo<=v<=a.hi,(a,x)

def test_exponential_mixed_jet_includes_order_ten_and_tangent():
    exp,*_=api();x=Jet.variable(I('.1','.2'),order=10);x.c[0,1]=I(3);q=exp(x)
    with localcontext() as c:
        c.prec=90
        for i in range(11):
            for j in range(2):
                for v in (D('.1'),D('.2')):
                    wanted=v.exp()*3**j/math.factorial(i);assert q[i,j].lo<=wanted<=q[i,j].hi

def test_branch_selection_refuses_uncertainty_instead_of_picking_midpoint():
    _,choose,*_=api();a=Jet.variable(I(-1,1));b=a.lift(0)
    with pytest.raises(ValueError,match='branch'):choose(a,b,'min')
    assert choose(a+3,b,'min')[0,0].zero
    assert choose(b,b,'max')[0,0].zero

@pytest.mark.parametrize('jump',[False,True])
def test_material_boundary_pixel_includes_moving_interface_flux(jump):
    _,_,parts,integrate,_=api()
    regions=parts([[(1,0),(0,0),(-Q(1,4),-1)]])
    field=lambda signs:(lambda x,y:[x.lift(3 if signs[0]>0 else 1)]*3) if jump else (lambda x,y:[(x-x.lift(Q(1,4),tangent=1))*(2 if signs[0]>0 else 1)]*3)
    result=integrate(regions,field)
    for q in result['value']:has(q,Q(5,2) if jump else Q(17,32))
    for q in result['derivative']:has(q,-2 if jump else -Q(7,4))
    assert len(result['pieces'])==4

def test_bilinear_texture_integral_and_texel_uv_derivatives():
    *_,integrate,sample=api();_,_,parts,*_=api()
    texels=[[[0],[1]],[[2],[4]]]
    # Cell-space x,y; texel (1,1) has tangent1, U has tangent2.
    def f(signs):
        def field(x,y):
            x=Jet(order=x.order,coeff={**x.c,(0,1):x[0,1]+2})
            q=sample(texels,x,y,tangent=[[[0],[0]],[[0],[1]]])[0]
            return [q]*3
        return field
    result=integrate(parts([]),f)
    for q in result['value']:has(q,Q(7,4))
    for q in result['derivative']:has(q,Q(13,4))

def test_texture_kink_is_partitioned_even_when_point_samples_miss_it():
    _,_,parts,integrate,_=api();edge=Q(1,65536)
    result=integrate(parts([[(1,0),(0,0),(-edge,-1)]]),lambda signs:lambda x,y:[x.lift(1 if signs[0]<0 else 0)]*3)
    for q in result['value']:has(q,edge);assert q.lo>0
    for q in result['derivative']:has(q,1)

def test_partition_refuses_topology_events_and_work_exhaustion():
    _,_,parts,integrate,_=api()
    with pytest.raises(ValueError,match='topology'):parts([[(1,0),(0,0),(0,-1)]])
    with pytest.raises(ValueError,match='piece'):parts([[(1,0),(0,0),(-Q(i,10),0)] for i in range(1,10)]+[[(0,0),(1,0),(-Q(i,10),0)] for i in range(1,10)])
    with pytest.raises(ValueError):integrate(parts([]),lambda s:lambda x,y:[x.lift(1)]*3,max_cells=0)
    def cancel():raise RuntimeError('cancelled')
    with pytest.raises(RuntimeError,match='cancelled'):parts([],checkpoint=cancel)


def test_centred_jets_enclose_every_derivative_and_cancel_dependency():
    from tools.opaque_material_reference import RangeJet,exp_jet
    x=RangeJet(Jet.variable(I('.826','.828'),order=7),I('.827'),I('-.001','.001'))
    value=exp_jet(x).ln()-x
    assert value[0,0].lo<=0<=value[0,0].hi and value[0,0].width<D('1e-16')
    q=exp_jet(x)
    with localcontext() as c:
        c.prec=90
        for k in range(8):
            for v in (D('.826'),D('.827'),D('.828')):
                target=v.exp()/math.factorial(k);assert q[k,0].lo<=target<=q[k,0].hi

def test_taylor_envelope_remainder_cannot_drop_unseen_seventh_order_term():
    from tools.opaque_material_reference import CurveEnvelope,RoughPrincipled
    env=object.__new__(CurveEnvelope);env.lo=D(-1);env.hi=D(1);env.centre=I(0)
    def curve(q):
        c=object.__new__(RoughPrincipled);c.pi=I(3);c.cn=[I('.5'),I(1)];c.x=[I(-1),I(0)]
        c.values=[[q,q],[q,q]];c.horizon=[q,q];c.controls=[[[q]*4,[q]*4]];c.mean=[q,q];return c
    env.base=curve(Jet.variable(0,order=7)**7);env.bound=curve(Jet.variable(I(-1,1),order=7)**7)
    r=Jet.variable(Q(1,4),order=6);r.c[0,1]=I(3);q=env.at(r).mean[0]
    for k in range(7):
        has(q[k,0],math.comb(7,k)*Q(1,4)**(7-k));has(q[k,1],3*(7-k)*math.comb(7,k)*Q(1,4)**(6-k))
    assert q[0,0].lo>0 and q[0,1].lo>0


@pytest.mark.parametrize('mode',['clamp','repeat'])
def test_normalized_texture_cells_integrate_wrap_and_uv_motion(mode):
    from tools.opaque_material_reference import texture_cell
    _,_,parts,integrate,_=api()
    cuts=[[(1,0),(0,0),(-v,1)] for v in (Q(1,4),Q(3,4))]+[[(0,0),(1,0),(-v,0)] for v in (Q(1,4),Q(3,4))]
    def fn(sides):
        column=-1+sum(s>0 for s in sides[:2]);row=-1+sum(s>0 for s in sides[2:])
        def field(x,y):
            u=x+x.lift(0,tangent=1);return texture_cell([[[0],[1]],[[2],[4]]],u,y,column,row,wrap_u=mode,wrap_v=mode)*3
        return field
    result=integrate(parts(cuts),fn)
    for q in result['value']:has(q,Q(7,4))
    for q in result['derivative']:has(q,Q(3,2) if mode=='clamp' else 0)


def test_perspective_texture_boundary_motion_has_exact_quotient_gradient():
    from tools.opaque_material_reference import perspective_boundary,perspective_value
    _,_,parts,integrate,_=api();triangle=[(-2,-2),(6,-2),(-2,6)]
    # u=x/(1+x/4), threshold1/4 => x=4/15. Shifting u by theta moves
    # the boundary by -1/u'(4/15)=-256/225.
    iw=[1+Q(x,4) for x,y in triangle];values=[Q(x)/q for (x,y),q in zip(triangle,iw)]
    cut=perspective_boundary(triangle,iw,values,Q(1,4),value_motion=[1]*3)
    result=integrate(parts([cut]),lambda s:lambda x,y:[x.lift(1 if s[0]<0 else 0)]*3)
    for q in result['value']:has(q,Q(4,15))
    for q in result['derivative']:has(q,-Q(256,225))
    x=Jet(Q(4,15));y=Jet(Q(1,2));v=perspective_value(x,y,triangle,iw,values,value_motion=[1]*3)
    has(v[0,0],Q(1,4));has(v[0,1],1)
