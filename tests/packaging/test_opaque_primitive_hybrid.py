"""Analytic inner lines plus independently bounded outer quadrature."""
from decimal import Decimal as D,localcontext
import pytest

@pytest.mark.parametrize('outer',['transport','boundary'])
@pytest.mark.parametrize('axis',[0,1])
@pytest.mark.parametrize('sx,sy',[(1,1),(-1,1),(1,-1),(-1,-1)])
def test_hybrid_keeps_exact_swept_log_cubic_integral(axis,sx,sy,outer):
    from tools.opaque_primitive_hybrid import Hybrid
    from tools.opaque_shading_intervals import I
    from tests.packaging.test_opaque_aggregate_transport import fixture
    c=fixture(sx,sy);p=Hybrid(c,order=3,outer=outer);got=p((I(0,1),I(0,1)),axis=axis)[1]
    with localcontext() as ctx:
        ctx.prec=90;a,b,d,k=D('.3'),D('.01'),D('.4'),D('.5')
        def primitive(q):
            if q<=k:return D(0)
            z=(q/k).ln();return q*q/2*(z*z-3*z+D('3.5'))-2*k*q+k*k/4
        exact=(primitive(a+b+d)-primitive(a+b)-primitive(a+d)+primitive(a))/(b*d)*(1+D(2).ln()**2)
        assert got.lo<=exact<=got.hi
        assert len(p.last['traces'])==5 and p.last['outer_error'][1].width>0

@pytest.mark.parametrize('axis',[0,1])
def test_trace_encloses_every_transverse_point_and_keeps_its_dimension(axis):
    from tools.opaque_primitive_rectangle import Rectangle
    from tools.opaque_shading_intervals import I
    from tests.packaging.test_opaque_aggregate_transport import fixture
    c=fixture();p=Rectangle(c);box=[I('.2','.4'),I('.2','.4')];box[1-axis]=I('.1','.9');wide=p.trace(tuple(box),axis=axis)[1]
    for point in ['.1','.5','.9']:
        line=list(box);line[1-axis]=I(point);got=p.line(tuple(line),axis=axis)[1]
        # The separate natural enclosures need only overlap, each enclosing the same exact line.
        assert wide.lo<=got.hi and got.lo<=wide.hi
    with pytest.raises(ValueError):p.line(tuple(box),axis=axis)

def test_hybrid_propagates_cancellation_and_refuses_invalid_axis():
    from tools.opaque_primitive_hybrid import Hybrid
    from tools.opaque_shading_intervals import I,Unresolved
    from tests.packaging.test_opaque_aggregate_transport import fixture
    p=Hybrid(fixture())
    with pytest.raises(ValueError):p((I(0,1),I(0,1)),axis=2)
    def stop():raise Unresolved('cancel hybrid')
    p.c.checkpoint=stop
    with pytest.raises(Unresolved):p((I(0,1),I(0,1)))
