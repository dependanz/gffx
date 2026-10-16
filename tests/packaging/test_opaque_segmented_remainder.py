"""Independent exact integrals and refusal controls for segment-aware remainders."""
import math
from fractions import Fraction as Q
import pytest
from tools.opaque_shading_intervals import I,Jet


def independent_cubic_error(knot,orientation):
    # Separate 100-digit closed Gauss algebra; never use the kernel primitive.
    # A narrow numerical reference bracket avoids requiring one conservative
    # 60-digit enclosure to contain another differently rounded enclosure.
    from decimal import Decimal,localcontext
    with localcontext() as ctx:
        ctx.prec=100
        k=Decimal(str(knot));one=Decimal(1)
        a=(Decimal(5)+2*(Decimal(10)/7).sqrt()).sqrt()/6
        b=(Decimal(5)-2*(Decimal(10)/7).sqrt()).sqrt()/6
        u=(Decimal(322)-13*Decimal(70).sqrt())/1800
        v=(Decimal(322)+13*Decimal(70).sqrt())/1800
        nodes=[one/2-a,one/2-b,one/2,one/2+b,one/2+a]
        weights=[u,v,Decimal(64)/225,v,u]
        value=sum(w*max(Decimal(0),(x-k)*orientation)**3 for x,w in zip(nodes,weights))-((1-k)**4 if orientation==1 else k**4)/4
        return I(str(value-Decimal('1e-80')),str(value+Decimal('1e-80')))


@pytest.mark.parametrize('orientation',[-1,1])
@pytest.mark.parametrize('knot',['.01','.37','.5','.99'])
def test_segmented_remainder_preserves_nonzero_signed_cubic_mass(orientation,knot):
    from tools.opaque_segmented_remainder import segmented_remainder
    from tools.opaque_peano import gauss5
    k=I(knot);nodes,weights=gauss5()
    def third(s):
        if s.hi<=k.lo:return I(0 if orientation==1 else -6)
        if s.lo>=k.hi:return I(6 if orientation==1 else 0)
        return I(0,6) if orientation==1 else I(-6,0)
    full=I(0,6) if orientation==1 else I(-6,0)
    got=segmented_remainder(third,I(0,1),[k],full_range=full)
    integral=(1-k)**4/4 if orientation==1 else k**4/4
    quadrature=sum((I(max(I(0).lo,((x-k)*orientation).lo),max(I(0).hi,((x-k)*orientation).hi))**3*w for x,w in zip(nodes,weights)),I(0))
    exact=independent_cubic_error(knot,orientation)
    assert got['value'].lo<=exact.lo<=exact.hi<=got['value'].hi
    assert got['value'].width<I('1e-45').lo
    if knot!='.5':assert not (got['value'].lo<=0<=got['value'].hi)


@pytest.mark.parametrize('root',[I(0),I(1),I('.369','.371'),I(0,'.01'),I('.99',1)])
def test_uncertain_and_endpoint_roots_retain_every_piecewise_cubic(root):
    from tools.opaque_segmented_remainder import segmented_remainder
    from tools.opaque_peano import gauss5
    def third(s):
        if s.hi<=root.lo:return I(0)
        if s.lo>=root.hi:return I(6)
        return I(0,6)
    got=segmented_remainder(third,I(0,1),[root],full_range=I(0,6))['value']
    nodes,weights=gauss5()
    for k in [I(root.lo),I((Q(root.lo)+Q(root.hi))/2),I(root.hi)]:
        exact=independent_cubic_error(k.lo,1)
        assert got.lo<=exact.lo<=exact.hi<=got.hi


@pytest.mark.parametrize('constant',[-6,0,7])
def test_constant_derivative_cancels_over_complete_domain(constant):
    from tools.opaque_segmented_remainder import segmented_remainder
    result=segmented_remainder(lambda s:I(constant),I(2,4),[I('2.4','2.6'),I(3)],full_range=I(constant))
    assert result['value'].zero


def test_segmented_partition_is_bounded_and_stops_before_evaluation():
    from tools.opaque_segmented_remainder import segmented_remainder
    calls=[]
    roots=[I(I(Q(n,40)).lo,I(Q(n,40)+Q(1,100)).hi) for n in range(1,17)]
    assert segmented_remainder(lambda s:calls.append(s),I(0,1),roots,full_range=I(0,6)) is None
    assert not calls
    def stop():raise RuntimeError('stop')
    with pytest.raises(RuntimeError,match='stop'):segmented_remainder(lambda s:calls.append(s),I(0,1),[I('.4')],full_range=I(0,6),checkpoint=stop)
    assert not calls


@pytest.mark.parametrize('domain,roots',[(I(0),[]),(I(0,1),[I(-1,0)]),(I(0,1),[I(1,2)])])
def test_segmented_remainder_refuses_invalid_domains(domain,roots):
    from tools.opaque_segmented_remainder import segmented_remainder
    with pytest.raises(ValueError):segmented_remainder(lambda s:I(0),domain,roots,full_range=I(0))


@pytest.mark.parametrize('orientation',[-1,1])
def test_signed_ordinary_enclosure_keeps_every_jump_and_physical_scale(orientation):
    from tools.opaque_transverse_peano import _error
    # An externally supplied signed ordinary integral (already physical)
    # must survive separately from every original jump charge.
    row=dict(third=I(6),third_range=I(-6,6),jumps=[I('.01'),I('.02'),I('.03')],ordinary_signed=I('.001','.002')*orientation)
    got,parts=_error(row,I(2,4),I(3),False)
    old,_=_error({k:v for k,v in row.items() if k!='ordinary_signed'},I(2,4),I(3),False)
    assert parts['ordinary'].hi<=I('.0015').hi
    assert parts['global_jumps'].hi>0 and parts['signed_correction'].bound>0
    assert got.width<old.width


@pytest.mark.parametrize('sx,sy',[(-1,-1),(-1,1),(1,-1),(1,1)])
@pytest.mark.parametrize('offset',['.4988','.4992','.4998'])
def test_segmented_material_encloses_independent_log_integral(sx,sy,offset,record_property):
    import json
    from tools.opaque_peano_material import MaterialCorrection
    from tests.packaging.test_opaque_localized_peano import material_fixture
    from tests.packaging.test_opaque_transverse_peano import _log_rectangle
    e,eps,slope=material_fixture();b=I('.001')*sx;c=I('.001')*sy
    a=I(offset)+(I('.001') if sx<0 else I(0))+(I('.001') if sy<0 else I(0))
    def inputs(x,y):return x*b+y*c+a,x.lift('.75'),x.lift(1),x.lift(1)
    old=MaterialCorrection(e,inputs,curved=False,transverse=True,smooth_partition=True)
    new=MaterialCorrection(e,inputs,curved=False,transverse=True,smooth_partition=True,segmented_conditioned=True)
    before=old((I(0,1),I(0,1)))[1];after=new((I(0,1),I(0,1)))[1];exact=_log_rectangle(e,eps,slope,a,b,c)
    assert before.lo<=exact.lo<=exact.hi<=before.hi
    assert after.lo<=exact.lo<=exact.hi<=after.hi
    assert after.width<=before.width*I('1.00000000000000000000000001').hi
    record_property('segment_bound',json.dumps(dict(before=str(before.width),after=str(after.width))))


def test_segmented_material_keeps_nonmonotone_fallback():
    from tools.opaque_peano_material import MaterialCorrection
    from tests.packaging.test_opaque_localized_peano import material_fixture
    e,_,_=material_fixture()
    def inputs(x,y):return (x-I('.5'))**2*I('.01')+I('.499'),x.lift('.75'),x.lift(1),x.lift(1)
    c=MaterialCorrection(e,inputs,curved=False,transverse=True,segmented_conditioned=True)
    rows=c.axis_bounds((I(0,1),I('.5')),0)
    assert 'ordinary_signed' not in rows[1] and rows[1]['localized_jumps'][0]['max_crossings']==4


def test_segment_selection_is_oriented_and_keeps_uncertain_root_zone():
    from tools.opaque_segmented_remainder import allowed_segments
    for sign in (-1,1):
        w=dict(position=I('.4','.6'),role=0,knot=1,orientation=sign,required=True)
        assert allowed_segments([-1,0,1,2],[w],I(0,'.4'),0)==([-1,0] if sign==1 else [1,2])
        assert allowed_segments([-1,0,1,2],[w],I('.6',1),0)==([1,2] if sign==1 else [-1,0])
        assert allowed_segments([-1,0,1,2],[w],I('.4','.6'),0)==[-1,0,1,2]


def test_absent_crossing_cannot_select_a_material_branch():
    from tools.opaque_segmented_remainder import allowed_segments
    w=dict(position=I('.4','.6'),role=0,knot=1,orientation=1,required=False)
    assert allowed_segments([-1,0,1,2],[w],I(0,'.4'),0)==[-1,0,1,2]
    assert allowed_segments([-1,0,1,2],[w],I('.6',1),0)==[-1,0,1,2]


@pytest.mark.parametrize('sx,sy',[(-1,-1),(-1,1),(1,-1),(1,1)])
def test_segmented_curved_material_retains_jacobian_and_exact_integral(sx,sy):
    from tools.opaque_peano_material import MaterialCorrection
    from tests.packaging.test_opaque_localized_peano import material_fixture
    from tests.packaging.test_opaque_transverse_peano import _log_rectangle
    e,eps,slope=material_fixture();b=I('.001')*sx;c=I('.001')*sy
    a=I('.4988')+(I('.002') if sx<0 else I(0))+(I('.001') if sy<0 else I(0))
    def inputs(x,y):return (x+x*x)*b+y*c+a,x.lift('.75'),x.lift(1),1+2*x
    correction=MaterialCorrection(e,inputs,curved=True,transverse=True,smooth_partition=True,segmented_conditioned=True)
    got=correction((I(0,1),I(0,1)))[1];exact=_log_rectangle(e,eps,slope,a,b,c,I(2))
    assert got.lo<=exact.lo<=exact.hi<=got.hi


def test_signed_ordinary_refuses_disjoint_evidence():
    from tools.opaque_transverse_peano import _error
    row=dict(third=I(0),third_range=I(0),jumps=[I(0)]*3,ordinary_signed=I(1))
    with pytest.raises(ValueError,match='disjoint'):_error(row,I(0,1),I(1),False)


def test_full_original_producer_selects_qualified_segmented_checker(monkeypatch):
    import os,json,hashlib
    from pathlib import Path
    from tools import opaque_full_material_reference as module
    source=Path(os.environ['GFFX_FULL_INPUT'])
    assert hashlib.sha256(source.read_bytes()).hexdigest()=='a15bea40cb5cdead7994d828b57f5dcfac9f44fab0292a06e7d0a7b08719f73a'
    class Selected(Exception):pass
    def capture(*args,**kwargs):
        assert kwargs.get('segmented_conditioned') is True
        assert kwargs['transverse'] is True and kwargs['smooth_partition'] is True
        raise Selected
    monkeypatch.setattr(module,'MaterialCorrection',capture)
    with pytest.raises(Selected):module.full_pixel(json.loads(source.read_bytes()),selected_piece=2)
