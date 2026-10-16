"""Range-first correction with a complete optional high-order proof."""
from fractions import Fraction as Q
import pytest

@pytest.mark.parametrize('axis',[0,1])
@pytest.mark.parametrize('direction',[1,-1])
def test_range_first_contains_independent_material_integral(axis,direction):
    from tools.opaque_gauss8 import context
    from tools.opaque_gauss8_correction import Correction
    from tests.packaging.test_opaque_aggregate_transport import fixture
    from tools.opaque_shading_intervals import I
    from decimal import Decimal as D,localcontext
    with context():got=Correction(fixture(direction,1),tolerance='1e-30',integrated_axis=axis)((I(0,1),I(0,1)))[1]
    with localcontext() as ctx:
        ctx.prec=90;k=D('.5');a=D('.3');b=D('.01');d=D('.4')
        def f(q):
            if q<=k:return D(0)
            z=(q/k).ln();return q*q/2*(z*z-3*z+D('3.5'))-2*k*q+k*k/4
        expected=(f(a+b+d)-f(a+b)-f(a+d)+f(a))/(b*d)*(1+D(2).ln()**2)
        assert got.lo<=expected<=got.hi

def test_narrow_range_skips_optional_proof_without_numerical_threshold(monkeypatch):
    from tools.opaque_gauss8_correction import Correction
    from tests.packaging.test_opaque_aggregate_transport import fixture
    from tools.opaque_shading_intervals import I
    c=Correction(fixture(),tolerance='1e-12');monkeypatch.setattr(c.proof,'ordinary',lambda u,v:[u.lift(0),u.lift('1e-30')]);monkeypatch.setattr(c.proof,'__call__',lambda *a,**k:pytest.fail('unnecessary proof'))
    got=c((I(0,1),I(0,1)));assert got[1].lo==got[1].hi==I('1e-30').lo
    assert c.used_high==0 and c.error_breakdown is None

def test_monotonicity_refusal_keeps_complete_range_but_disjoint_evidence_fails(monkeypatch):
    from tools.opaque_gauss8_correction import Correction
    from tools.opaque_high_transport import TransportMaterial
    from tests.packaging.test_opaque_aggregate_transport import fixture
    from tools.opaque_shading_intervals import I,Unresolved
    c=Correction(fixture(),tolerance='1e-30')
    def refused(*a,**k):raise ValueError('aggregate angular crossings need certified monotonicity')
    monkeypatch.setattr(TransportMaterial,'__call__',refused);got=c((I(0,1),I(0,1)));assert got[1].width>0
    monkeypatch.setattr(TransportMaterial,'__call__',lambda *a,**k:[I(0),I(1000000)])
    with pytest.raises(ValueError,match='disjoint'):c((I(0,1),I(0,1)))
    def stop(*a,**k):raise Unresolved('stop')
    monkeypatch.setattr(TransportMaterial,'__call__',stop)
    with pytest.raises(Unresolved):c((I(0,1),I(0,1)))


def test_wide_region_uses_complete_aggregate_proof_instead_of_range_only(monkeypatch):
    from tools.opaque_gauss8_correction import Correction
    from tools.opaque_aggregate_peano import AggregateMaterial
    from tests.packaging.test_opaque_aggregate_transport import fixture
    from tools.opaque_shading_intervals import I,Jet
    c=Correction(fixture(),tolerance='1e-30');monkeypatch.setattr(c,'knot_count',lambda box:100)
    box=(I(0,1),I(0,1));old=c.proof.ordinary(Jet(box[0]),Jet(box[1]))[1][0,0];calls=[];real=AggregateMaterial.__call__
    def spy(self,*a,**kw):calls.append(1);return real(self,*a,**kw)
    monkeypatch.setattr(AggregateMaterial,'__call__',spy)
    got=c(box)[1];assert calls==[1] and c.used_aggregate==1 and c.used_high==0
    assert old.lo<=got.lo<=got.hi<=old.hi and got.width<old.width
