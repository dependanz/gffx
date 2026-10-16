"""Independent controls for conditioned ordinary centering and bounded kernel reuse."""
from fractions import Fraction as Q
import math,json
import pytest
from tools.opaque_shading_intervals import I,Jet


@pytest.mark.parametrize('value',[-6,6])
@pytest.mark.parametrize('mass',[Q(1),Q(7,3)])
def test_conditioned_constant_third_cancels_but_keeps_jumps(value,mass):
    from tools.opaque_transverse_peano import _error
    row=dict(third=I(abs(value)),third_range=I(value),jumps=[I('1e-9'),I('2e-9'),I('3e-9')])
    got,parts=_error(row,I(2,4),I(mass),False)
    assert parts['ordinary'].zero
    assert parts['global_jumps'].hi>0 and got.width>0


@pytest.mark.parametrize('orientation',[-1,1])
@pytest.mark.parametrize('degree',[3,4,6])
def test_conditioned_centered_bound_encloses_exact_truncated_polynomial(orientation,degree):
    from tools.opaque_transverse_peano import _error
    from tools.opaque_peano import gauss5
    k=I('.37');nodes,weights=gauss5();q=I(0)
    for x,w in zip(nodes,weights):
        v=(x-k)*orientation;q=q+I(max(I(0).lo,v.lo),max(I(0).hi,v.hi))**degree*w
    end=I('.63') if orientation==1 else I('.37')
    exact=end**(degree+1)/(degree+1)
    upper=end**(degree-3)*math.factorial(degree)/math.factorial(degree-3)
    signed=I(0,upper.hi) if orientation==1 else I(upper.hi.copy_negate(),0)
    row=dict(third=I(signed.bound),third_range=signed,jumps=[I(0)]*3)
    error,parts=_error(row,I(0,1),I(1),False);got=q-error
    assert got.lo<=exact.lo<=exact.hi<=got.hi
    assert parts['ordinary'].hi<_error({k:v for k,v in row.items() if k!='third_range'},I(0,1),I(1),False)[1]['ordinary'].hi


def test_centered_error_refuses_inconsistent_signed_evidence():
    from tools.opaque_transverse_peano import _error
    with pytest.raises(ValueError):_error(dict(third=I(1),third_range=I(-2,2),jumps=[I(0)]*3),I(0,1),I(1),False)


def test_local_kernel_reuse_is_bounded_and_preserves_all_signed_ranges():
    from tools.opaque_peano import kernel_local_ranges,_kernel_local_ranges_cached
    _kernel_local_ranges_cached.cache_clear()
    cases=[I(0),I('.12','.13'),I('.49','.51'),I(1)]
    first=[kernel_local_ranges(p) for p in cases];info=_kernel_local_ranges_cached.cache_info()
    again=[kernel_local_ranges(I(p.lo,p.hi)) for p in cases];after=_kernel_local_ranges_cached.cache_info()
    assert after.hits-info.hits==len(cases) and after.misses==info.misses
    assert [[(x.lo,x.hi) for x in row] for row in first]==[[(x.lo,x.hi) for x in row] for row in again]
    for n in range(160):kernel_local_ranges(I(I(Q(n,200)).lo,I(Q(n+1,200)).hi))
    assert _kernel_local_ranges_cached.cache_info().currsize<=128
    before=_kernel_local_ranges_cached.cache_info()
    with pytest.raises(ValueError):kernel_local_ranges(I(-1,0))
    assert _kernel_local_ranges_cached.cache_info()==before
    _kernel_local_ranges_cached.cache_clear()


def test_cached_ranges_do_not_expose_mutable_entries():
    from tools.opaque_peano import kernel_local_ranges
    first=kernel_local_ranges(I('.12','.13'));expected=[(v.lo,v.hi) for v in first]
    first[0].lo=I(-100).lo
    assert [(v.lo,v.hi) for v in kernel_local_ranges(I('.12','.13'))]==expected


@pytest.mark.parametrize('sx,sy',[(-1,-1),(-1,1),(1,-1),(1,1)])
@pytest.mark.parametrize('offset',['.4988','.4992','.4998'])
def test_conditioned_center_retains_exact_material_and_boundaries(sx,sy,offset):
    from tools.opaque_peano_material import MaterialCorrection
    from tests.packaging.test_opaque_localized_peano import material_fixture
    from tests.packaging.test_opaque_transverse_peano import _log_rectangle
    e,eps,slope=material_fixture();b=I('.001')*sx;c=I('.001')*sy
    a=I(offset)+(I('.001') if sx<0 else I(0))+(I('.001') if sy<0 else I(0))
    def inputs(x,y):return x*b+y*c+a,x.lift('.75'),x.lift(1),x.lift(1)
    correction=MaterialCorrection(e,inputs,curved=False,transverse=True,smooth_partition=True)
    got=correction((I(0,1),I(0,1)))[1];exact=_log_rectangle(e,eps,slope,a,b,c)
    assert got.lo<=exact.lo<=exact.hi<=got.hi


@pytest.mark.parametrize('offset',['.4988','.4992'])
def test_center_and_kernel_reuse_preserve_exact_acceptance_and_charge_work(offset,monkeypatch,record_property):
    import tools.opaque_peano as kernel
    import tools.opaque_transverse_peano as transverse
    from tools.opaque_peano_material import MaterialCorrection
    from tools.opaque_square_reference import integrate_square
    from tests.packaging.test_opaque_localized_peano import material_fixture
    from tests.packaging.test_opaque_transverse_peano import _log_rectangle
    e,eps,slope=material_fixture();a=I(offset);b=c=I('.001');calls=[0];kernel_calls=[0]
    def inputs(x,y):calls[0]+=1;return x*b+y*c+a,x.lift('.75'),x.lift(1),x.lift(1)
    actual_error=transverse._error;uncached=kernel._kernel_segment_bernstein;original_cache=kernel._kernel_local_ranges_cached;original_mass_cache=kernel._kernel_local_mass_cached
    def counted(*args):kernel_calls[0]+=1;return uncached(*args)
    def previous(row,domain,mass,integrated):return actual_error({k:v for k,v in row.items() if k!='third_range'},domain,mass,integrated)
    results=[]
    for enabled in (False,True):
        calls[0]=kernel_calls[0]=0;original_cache.cache_clear();original_mass_cache.cache_clear()
        monkeypatch.setattr(kernel,'_kernel_local_mass_cached',original_mass_cache if enabled else original_mass_cache.__wrapped__)
        monkeypatch.setattr(kernel,'_kernel_segment_bernstein',counted)
        monkeypatch.setattr(kernel,'_kernel_local_ranges_cached',original_cache if enabled else original_cache.__wrapped__)
        monkeypatch.setattr(transverse,'_error',actual_error if enabled else previous)
        correction=MaterialCorrection(e,inputs,curved=False,transverse=True,smooth_partition=True)
        result=integrate_square(lambda x,y:[x.lift(0),x.lift(0)],correction_integrator=correction,tolerance='1e-12',max_cells=256,max_depth=10)
        exact=_log_rectangle(e,eps,slope,a,b,c);got=result['value'][1]
        assert got.lo<=exact.lo<=exact.hi<=got.hi and got.width<=I('2e-12').lo
        results.append(dict(enabled=enabled,cells=result['cells'],input_calls=calls[0],kernel_builds=kernel_calls[0],seconds=result['seconds'],width=str(got.width)))
    # Tighter mass can finish in one cell with no repeated kernel arguments.
    # Cache lookup must not add polynomial builds; separate tests require hits.
    assert results[1]['kernel_builds']<=results[0]['kernel_builds']
    assert results[1]['input_calls']<=results[0]['input_calls']
    record_property('conditioned_work',json.dumps(dict(offset=offset,results=results)))
