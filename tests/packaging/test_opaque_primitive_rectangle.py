"""Complete rectangle primitive bounds with separate product residual."""
import pytest
from fractions import Fraction as Q

@pytest.mark.parametrize('order',[3,6])
@pytest.mark.parametrize('axis',[0,1])
@pytest.mark.parametrize('sign',[-1,1])
def test_primitive_rectangle_contains_closed_one_coordinate_material(axis,sign,order):
    from tools.opaque_primitive_rectangle import Rectangle
    from tests.packaging.test_opaque_aggregate_transport import fixture
    from tools.opaque_shading_intervals import I
    from decimal import Decimal as D,localcontext
    c=fixture()
    def inputs(u,v):
        t=[u,v][axis];q=t*I('.4')+t.lift('.3') if sign==1 else t.lift('.7')-t*I('.4')
        return q,t.lift(1),t.lift(1),t.lift(1)
    c.inputs=inputs;got=Rectangle(c,order=order)((I(0,1),I(0,1)),axis=axis)[1]
    with localcontext() as ctx:
        ctx.prec=90;q=D('.7');k=D('.5');z=(q/k).ln();one=q*(z*z-2*z+2)-2*k
        expected=one/D('.4')*(1+D(2).ln()**2)
        assert got.lo<=expected<=got.hi

def test_product_residual_is_separate_and_never_discarded():
    from tools.opaque_primitive_rectangle import Rectangle
    from tests.packaging.test_opaque_aggregate_transport import fixture
    from tools.opaque_shading_intervals import I
    c=fixture();c.inputs=lambda u,v:(u*I('.4')+u.lift('.3'),v*I('.4')+v.lift('.3'),u.lift(1),u.lift(1))
    p=Rectangle(c)
    # q_x=0 in one role must be retained as a complete range, not omitted.
    got=p((I(0,1),I(0,1)),axis=0)
    assert p.last['cross'][1].hi>0 and got[1].width>0


@pytest.mark.parametrize('order',[3,6])
@pytest.mark.parametrize('axis',[0,1])
def test_line_requires_fixed_other_coordinate_and_keeps_independent_integral(axis,order):
    from tools.opaque_primitive_rectangle import Rectangle
    from tests.packaging.test_opaque_aggregate_transport import fixture
    from tools.opaque_shading_intervals import I
    c=fixture();c.inputs=lambda u,v:([u,v][axis]*I('.4')+u.lift('.3'),u.lift(1),u.lift(1),u.lift(1))
    p=Rectangle(c,order=order);box=[I(0,1),I(0,1)];box[1-axis]=I('.5')
    line=p.line(tuple(box),axis=axis)[1];rect=p((I(0,1),I(0,1)),axis=axis)[1]
    assert (line.lo,line.hi)==(rect.lo,rect.hi)
    with pytest.raises(ValueError):p.line((I(0,1),I(0,1)),axis=axis)
    with pytest.raises(ValueError):p(tuple(box),axis=axis)
