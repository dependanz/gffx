"""Implicit recurrence only needs its current coefficient prefix."""
import pytest
from fractions import Fraction as Q
from tools.opaque_shading_intervals import I,Jet
from tools.opaque_boundary_taylor import curve_jet
@pytest.mark.parametrize('order',[1,2,3,4])
def test_implicit_equation_never_computes_unused_higher_coefficients(order):
    seen=[]
    def equation(x,y):
        seen.append(x.order);return x*(1+y)-Q(3,4)
    got=curve_jet(equation,I('.75'),I(0),order=order)
    assert seen==[0,*range(1,order+1)]
    for n in range(1,order+1):
        want=Q((-1)**n)*Q(3,4)/Q(3,4)**(n+1)
        assert Q(got[n,0].lo)<=want<=Q(got[n,0].hi)
