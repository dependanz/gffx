"""Independent adaptive Gauss8 integrals and conservative work limits."""
from fractions import Fraction as Q
import pytest

def test_nonunit_polynomial_values_and_tangents_are_enclosed():
    from tools.opaque_square_reference import integrate_square
    from tools.opaque_shading_intervals import Jet,I
    def f(x,y):
        p=(2*x+1)**7*(3*y+2)**8
        return [Jet(order=p.order,coeff={**p.c,**{(i,1):v*3 for (i,j),v in p.c.items() if j==0}})]
    r=integrate_square(f,experimental_rule='gauss8')
    q=I(Q(3**8-1,16)*Q(5**9-2**9,27))
    for got,expected in [(r['value'][0],q),(r['derivative'][0],3*q)]:assert got.lo<=expected.lo<=expected.hi<=got.hi
    assert r['cells']==3 and r['regions']==1 and r['work_charge']==3

def test_failed_domain_attempts_consume_full_charge_before_work():
    from tools.opaque_square_reference import integrate_square
    from tools.opaque_shading_intervals import Unresolved
    calls=[]
    def f(x,y):calls.append(1);raise ValueError('intentionally unresolved domain')
    with pytest.raises(Unresolved) as e:integrate_square(f,experimental_rule='gauss8',max_cells=5)
    assert e.value.details['cells']==3 and len(calls)==1

def test_too_small_allowance_never_calls_field():
    from tools.opaque_square_reference import integrate_square
    from tools.opaque_shading_intervals import Unresolved
    with pytest.raises(Unresolved) as e:integrate_square(lambda x,y:pytest.fail('uncharged work'),experimental_rule='gauss8',max_cells=2)
    assert e.value.details['cells']==0

def test_rule_cannot_silently_change_or_relax_depth_and_count():
    from tools.opaque_square_reference import integrate_square
    for args in [dict(experimental_rule='gauss12'),dict(experimental_rule='gauss8',max_cells=257),dict(experimental_rule='gauss8',max_depth=11)]:
        with pytest.raises(ValueError):integrate_square(lambda x,y:pytest.fail('invalid policy evaluated'),**args)

def test_tiny_tolerance_cannot_admit_incomplete_cover():
    from tools.opaque_square_reference import integrate_square
    from tools.opaque_shading_intervals import Unresolved
    with pytest.raises(Unresolved) as e:integrate_square(lambda x,y:[1/(x+1)/(y+1)],experimental_rule='gauss8',max_cells=3,tolerance='1e-50')
    assert e.value.details['cells']==3 and e.value.details['last_complete_enclosure']['complete_cover']
