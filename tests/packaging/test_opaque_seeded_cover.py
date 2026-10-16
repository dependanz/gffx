"""Independent cover, integral, budget and cancellation controls for fresh seeds."""
from fractions import Fraction as Q
from decimal import Decimal as D,localcontext
import pytest

@pytest.mark.parametrize('shape',[(1,1),(2,4),(8,2),(32,4),(64,2)])
def test_seeded_cover_is_complete_disjoint_and_fresh(shape):
    from tools.opaque_square_reference import integrate_square
    seen=[]
    def prepare(x,y,box):seen.append(tuple(Q(z) for a in box for z in (a.lo,a.hi)))
    result=integrate_square(lambda u,v:[(u+u.lift(1,tangent=1))*u*v**3],seed_shape=shape,prepare=prepare)
    assert len(seen)==shape[0]*shape[1]==result['cells']==result['accepted']
    assert sum((b-a)*(d-c) for a,b,c,d in seen)==1
    assert all(0<=a<b<=1 and 0<=c<d<=1 for a,b,c,d in seen)
    for i,(a,b,c,d) in enumerate(seen):
        for e,f,g,h in seen[i+1:]:assert min(b,f)<=max(a,e) or min(d,h)<=max(c,g)
    assert set(seen)==set(tuple(map(Q,row)) for row in result['leaves'])
    assert Q(result['value'][0].lo)<=Q(5,24)<=Q(result['value'][0].hi)
    assert Q(result['derivative'][0].lo)<=Q(1,8)<=Q(result['derivative'][0].hi)
    assert result['max_depth']==sum(n.bit_length()-1 for n in shape)

@pytest.mark.parametrize('shape',[(0,1),(3,2),(1,257),(True,2),[2,2],(2,),None,('2',2)])
def test_seeded_cover_refuses_invalid_shapes(shape):
    from tools.opaque_square_reference import integrate_square
    with pytest.raises(ValueError):integrate_square(lambda u,v:[u],seed_shape=shape)

def test_seeded_cover_checks_work_and_depth_before_field():
    from tools.opaque_square_reference import integrate_square
    calls=[]
    for kw in [dict(seed_shape=(8,4),max_cells=31),dict(seed_shape=(8,4),max_depth=4),dict(seed_shape=(8,4),max_cells=95,experimental_rule='gauss8')]:
        with pytest.raises(ValueError):integrate_square(lambda u,v:calls.append(1),**kw)
    assert not calls

def test_seeded_cover_has_independent_nonlinear_value_and_tangent():
    from tools.opaque_square_reference import integrate_square
    r=integrate_square(lambda u,v:[1/(u+v+u.lift(2,tangent=1))],seed_shape=(4,4))
    with localcontext() as c:
        c.prec=90;a=D(2);val=(a+2)*(a+2).ln()-2*(a+1)*(a+1).ln()+a*a.ln();der=((a+2)*a/(a+1)**2).ln()
    for key,want in [('value',val),('derivative',der)]:assert r[key][0].lo<=want<=r[key][0].hi and r[key][0].width<=D('2e-12')

def test_seeded_cover_never_accepts_partial_cover_or_ignores_cancellation():
    from tools.opaque_square_reference import integrate_square
    from tools.opaque_shading_intervals import Unresolved
    seen=[]
    def prepare(*a):
        seen.append(a)
        if len(seen)==3:raise Unresolved('cancel seed')
    with pytest.raises(Unresolved) as exc:integrate_square(lambda u,v:[u],seed_shape=(4,2),prepare=prepare)
    assert exc.value.details['last_complete_enclosure'] is None and exc.value.details['cells']==3

def test_seeded_cover_budget_failure_retains_only_complete_fresh_bounds():
    from tools.opaque_square_reference import integrate_square
    from tools.opaque_shading_intervals import I,Unresolved
    class Correction:
        def __call__(self,box):return [I('-1e-8','1e-8')]
    with pytest.raises(Unresolved) as exc:integrate_square(lambda u,v:[u.lift(0)],seed_shape=(4,2),max_cells=8,correction_integrator=Correction())
    c=exc.value.details['last_complete_enclosure'];assert c['complete_cover'] and c['leaves']==8 and c['cells']==8
    assert D(c['value'][0][0])<=D('-8e-8') and D(c['value'][0][1])>=D('8e-8')
