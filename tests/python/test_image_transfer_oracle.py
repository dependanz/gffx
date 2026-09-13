"""Independent high-precision transfer model; these checks do not qualify an implementation."""
from decimal import Decimal,localcontext
import math

import pytest


@pytest.mark.parametrize('dtype',['float32','float64'])
def test_decimal_transfer_one_and_represented_thresholds(dtype):
    from tools.image_transfer_oracle import reference,rounded,next_up,next_down
    for mode,threshold in (('decode',.04045),('encode',.0031308)):
        row=reference(1.,dtype,mode)
        assert row['value']==Decimal(1) and row['branch']=='nonlinear'
        threshold=rounded(threshold,dtype)
        assert reference(threshold,dtype,mode)['branch']=='linear'
        assert reference(next_down(threshold,dtype),dtype,mode)['branch']=='linear'
        assert reference(next_up(threshold,dtype),dtype,mode)['branch']=='nonlinear'


@pytest.mark.parametrize('dtype',['float32','float64'])
def test_decimal_transfer_linear_signed_zero_subnormal_and_overflow(dtype):
    from tools.image_transfer_oracle import reference,limits
    spec=limits(dtype)
    zero=reference(-0.,dtype,'decode')['value']
    assert zero.is_zero() and zero.is_signed()
    tiny=reference(spec['min_subnormal'],dtype,'decode')
    assert tiny['branch']=='linear' and tiny['value']>0
    assert reference(spec['max_finite'],dtype,'decode')['overflow']
    assert reference(-spec['max_finite'],dtype,'encode')['overflow']
    assert not reference(spec['max_finite'],dtype,'encode')['overflow']


def test_decimal_transfer_rejects_nonfinite_or_unknown_profiles():
    from tools.image_transfer_oracle import reference
    for value,dtype,mode in ((math.inf,'float64','decode'),(math.nan,'float64','decode'),
                            (1.,'float16','decode'),(1.,'float64','gamma')):
        with pytest.raises(ValueError):reference(value,dtype,mode)


def test_transfer_envelope_screen_cannot_hide_wrong_values_or_overflow():
    from tools.image_transfer_envelope import screen
    from tools.image_transfer_oracle import reference
    row=reference(.2,'float64','decode')
    screen('float64',row,0,float(row['value']),float(row['derivative']))
    for status,value,derivative in ((1,0.,0.),(0,.2,.2),(0,math.nan,1.)):
        with pytest.raises(AssertionError):screen('float64',row,status,value,derivative)
    overflow=reference(1e300,'float64','decode')
    screen('float64',overflow,1,math.inf,math.inf)
    with pytest.raises(AssertionError):screen('float64',overflow,0,1.,1.)


@pytest.mark.parametrize('dtype',['float32','float64'])
def test_decimal_overflow_boundary_uses_nearest_even_spacing(dtype):
    """An exact result slightly above max-finite can still round to a finite float."""
    from tools.image_transfer_oracle import overflow_threshold,limits
    spec=limits(dtype)
    with localcontext() as context:
        context.prec=2000
        half_spacing=Decimal.from_float(math.ldexp(1.,spec['emax']-spec['fraction_bits']-1))
        assert overflow_threshold(dtype)-Decimal.from_float(spec['max_finite'])==half_spacing


def test_transfer_envelope_requires_matching_successful_preflight():
    from tools.image_transfer_envelope import check_preflight
    identity={'library_sha256':'a'*64,'sources':{'x':'b'*64}}
    good={'status':'passed','mode':'preflight','identity':identity,'supervisor':{'exit_code':0,'failure':None}}
    assert check_preflight(good,identity) is None
    for bad in (None,{**good,'mode':'full'},{**good,'status':'stopped'},
                {**good,'identity':{}},{**good,'supervisor':{'exit_code':1,'failure':None}}):
        with pytest.raises(ValueError):check_preflight(bad,identity)


@pytest.mark.parametrize('dtype',['float32','float64'])
def test_overflow_input_neighbours_straddle_both_reachable_boundaries(dtype):
    """Large arbitrary powers are insufficient: test the last finite inputs and first overflows."""
    from tools.image_transfer_oracle import overflow_inputs,reference
    values=overflow_inputs(dtype)
    assert len(values)==34 and all(math.isfinite(x) for x in values)
    assert {reference(x,dtype,'decode')['overflow'] for x in values if x>0}=={False,True}
    assert {reference(x,dtype,'encode')['overflow'] for x in values if x<0}=={False,True}


@pytest.mark.parametrize('dtype',['float32','float64'])
@pytest.mark.parametrize('mode',['decode','negative_encode'])
def test_exact_integer_cutoffs_match_independent_decimal_and_reject_shift(dtype,mode):
    from tools.image_transfer_cutoffs import certificate,verify_certificate,represented
    from tools.image_transfer_oracle import reference
    row=certificate(dtype,mode)
    assert verify_certificate(row)
    assert row['binary_search_iterations']<=64
    assert row['cutoff_neighbours_have_relative_margin_gt_2_pow_minus_80']
    sign=-1 if mode=='negative_encode' else 1
    operation='encode' if mode=='negative_encode' else 'decode'
    assert not reference(sign*represented(row['last_finite_bits'],dtype),dtype,operation)['overflow']
    assert reference(sign*represented(row['first_overflow_bits'],dtype),dtype,operation)['overflow']
    for shift in (-1,1):
        bad={**row,'last_finite_bits':row['last_finite_bits']+shift,
             'first_overflow_bits':row['first_overflow_bits']+shift}
        with pytest.raises(ValueError):verify_certificate(bad)
    with pytest.raises(ValueError):verify_certificate({**row,'equality':'finite'})


def test_boundary_error_budget_stays_below_certified_cutoff_margin():
    from tools.image_transfer_boundary_proof import error_budget
    row=error_budget()
    assert row['iterations']==24
    assert row['normalized_value_relative_bound']<row['claimed_value_budget']<row['cutoff_margin']
    assert 'Conditional' in row['scope']


def test_exact_cutoff_inputs_reject_boolean_noninteger_and_unknown_profiles():
    from tools.image_transfer_cutoffs import represented,certificate
    for bits in (True,False,1.,-1,1<<64):
        with pytest.raises(ValueError):represented(bits,'float64')
    for dtype,mode in ((True,'decode'),('float16','decode'),('float64','unknown')):
        with pytest.raises(ValueError):certificate(dtype,mode)
