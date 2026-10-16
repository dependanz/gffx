"""Independent high-order log-polynomial coefficients for the smooth base."""
from decimal import Decimal as D,localcontext
from fractions import Fraction as Q
import math,itertools,pytest
from tools.opaque_shading_intervals import I,Jet
from tests.packaging.test_opaque_smooth_branch import fixture

def oracle(coefficients,point,rate,order):
    with localcontext() as c:
        c.prec=110;v=D(point);rate=D(rate);log=[v.ln()]+[D((-1)**(k+1))*rate**k/(D(k)*v**k) for k in range(1,order+1)]
        power=[D(1)]+[D(0)]*order;out=[D(0)]*(order+1)
        for coefficient in coefficients:
            out=[a+D(coefficient)*b for a,b in zip(out,power)]
            power=[sum((power[j]*log[k-j] for j in range(k+1)),D(0)) for k in range(order+1)]
        return out

@pytest.mark.parametrize('coefficients',[(1,0,0,0),(1,-2,0,0),(1,-2,3,0),(1,-2,3,-4)])
@pytest.mark.parametrize('point',['.25','.5','1'])
@pytest.mark.parametrize('rate',[-1,1])
@pytest.mark.parametrize('order',[0,3,4,6,10])
def test_smooth_base_contains_independent_high_order_log_coefficients(coefficients,point,rate,order):
    from tools.opaque_smooth_branch import SmoothBaseBranches
    q=Jet(order=order,coeff={(0,0):I(point),**({(1,0):I(rate)} if order else {})})
    got=SmoothBaseBranches(fixture(coefficients)).at(q,0);expected=oracle(coefficients,point,rate,order)
    for row in got:
        for k,v in enumerate(expected):assert row[k,0].lo<=v<=row[k,0].hi

def test_high_order_base_retains_uncertain_controls_and_shared_two_entry_limit():
    from types import SimpleNamespace
    from tools.opaque_loss_extension import LossExtension
    from tools.opaque_smooth_branch import SmoothBaseBranches
    bs=[I(1,2),I(-2,-1),I(3,4),I(0,1)];e=LossExtension(SimpleNamespace(x=[I(-2),I(0)],controls=[[bs,bs]]));cache=SmoothBaseBranches(e)
    got=cache.at(Jet.variable(I('.3','.8'),order=10),0)
    # Convert each exact Bernstein-control vertex independently to a polynomial
    # in log(q), with exact rational arithmetic rather than interval transforms.
    for a,b,c,d in itertools.product(*[(Q(v.lo),Q(v.hi)) for v in bs]):
        u=[a,3*(b-a),3*(a-2*b+c),-a+3*b-3*c+d]
        coeff=[sum(u[j]*Q(math.comb(j,k),2**k) for j in range(k,4)) for k in range(4)]
        for point in ['.3','.47','.8']:
            with localcontext() as ctx:
                ctx.prec=110;dc=[D(v.numerator)/D(v.denominator) for v in coeff]
            for row in got:
                for k,v in enumerate(oracle(dc,point,1,10)):assert row[k,0].lo<=v<=row[k,0].hi
    got[0].c[0,0].lo=D(-9999);assert cache.at(Jet.variable(I('.3','.8'),order=10),0)[0][0,0].lo!=D(-9999)
    for n in range(4):
        e.curve.controls[0][0][0]=I(n);cache.at(Jet.variable(I('.5'),order=10),0);assert len(cache._cache)<=2

def test_high_order_base_keeps_horizon_and_refuses_moving_angles():
    from tools.opaque_smooth_branch import SmoothBaseBranches
    from tools.opaque_full_material_reference import HorizonExtension
    from types import SimpleNamespace
    e=HorizonExtension(SimpleNamespace(cn=[I('.25')],values=[[I(2),I(3)]],horizon=[I(1),I(0)]));q=Jet.variable(I('.1'),order=10)
    got=SmoothBaseBranches(e).at(q,-1)
    assert got[0][0,0].lo<=D('1.4')<=got[0][0,0].hi and got[0][1,0].lo<=4<=got[0][1,0].hi
    assert all(got[0][k,0].zero for k in range(2,11))
    with pytest.raises(ValueError):SmoothBaseBranches(e).at(Jet('.1',order=10,tangent=1),-1)


def test_original_base_and_correction_share_the_qualified_branch_cache(monkeypatch):
    import os,json,hashlib
    from pathlib import Path
    from tools import opaque_full_material_reference as module
    from tools.opaque_smooth_branch import SmoothBaseBranches
    source=Path(os.environ['GFFX_FULL_INPUT']);assert hashlib.sha256(source.read_bytes()).hexdigest()=='a15bea40cb5cdead7994d828b57f5dcfac9f44fab0292a06e7d0a7b08719f73a'
    class Stop(Exception):pass
    def capture(field,**kwargs):
        correction=kwargs['correction_integrator'];assert isinstance(correction.smooth,SmoothBaseBranches)
        box=(I('.625','.6875'),I('.75','1'));kwargs['prepare'](0,0,box);calls=[];original=correction.smooth.at
        def observed(q,index):calls.append(q.order);return original(q,index)
        monkeypatch.setattr(correction.smooth,'at',observed)
        field(Jet.variable(box[0],order=10),Jet(box[1],order=10));assert calls==[10,10] and len(correction.smooth._cache)<=2
        raise Stop
    monkeypatch.setattr(module,'integrate_square',capture)
    with pytest.raises(Stop):module.full_pixel(json.loads(source.read_bytes()),selected_piece=2)


def test_centered_nonlinear_base_integral_matches_independent_log_antiderivative():
    from tools.opaque_smooth_branch import SmoothBaseBranches
    from tools.opaque_square_reference import integrate_square
    coefficients=(1,-2,3,-4);branch=SmoothBaseBranches(fixture(coefficients))
    result=integrate_square(lambda u,v:branch.at(1/(2+u),0))
    with localcontext() as c:
        c.prec=110
        def primitive(x,k):
            z=x.ln();return x*sum((D((-1)**(k-j))*D(math.factorial(k))/D(math.factorial(j))*z**j for j in range(k+1)),D(0))
        expected=sum((D(a)*D((-1)**k)*(primitive(D(3),k)-primitive(D(2),k)) for k,a in enumerate(coefficients)),D(0))
    for row in result['value']:assert row.lo<=expected<=row.hi and row.width<=D('2e-12')
