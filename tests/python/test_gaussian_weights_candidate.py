"""Interpreted Gaussian candidate checks; compiled native gates remain separate."""
from decimal import Decimal,localcontext
import math
import pytest


def test_gaussian_scaler_matches_exact_integer_rounding_transitions():
    from tools.gaussian_weights_candidate import scale
    from tools.gaussian_weights_oracle import bits_from_float,decode_bits,round_bits
    values=(0.,-0.,1.,-1.,1.5,-1.5,float.fromhex('0x1.fffffffffffffp+0'),
            float.fromhex('0x0.0000000000001p-1022'),float.fromhex('0x1p-1022'))
    for value in values:
        for exponent in (-2000,-1075,-1074,-1073,-1023,-1022,-1,0,1,52,1023,2000):
            with localcontext() as context:
                context.prec=2500
                exact=decode_bits(bits_from_float(value,'float64'),'float64')*(Decimal(2)**exponent)
                expected=round_bits(exact,'float64')
            if (expected>>52)&2047==2047:
                with pytest.raises(OverflowError):scale(value,exponent)
            else:
                assert bits_from_float(scale(value,exponent),'float64')==expected


@pytest.mark.parametrize('dtype',['float32','float64'])
def test_gaussian_candidate_all_radii_and_represented_sigma_accuracy(dtype):
    from tools.gaussian_weights_candidate import weights
    from tools.gaussian_weights_oracle import bits_from_float,decode_bits,reference
    sigmas=[bits_from_float(v,dtype) for v in (.1,.83,1.,32.)]
    sigmas.append(bits_from_float(.83,dtype)+1)
    bound=Decimal(512)*Decimal(2)**-53+Decimal(2)**-990
    derivative_bound=Decimal(8192)*Decimal(2)**-53+Decimal(2)**-990
    if dtype=='float32':bound+=Decimal(2)**-24;derivative_bound+=Decimal(2)**-21
    for radius in range(33):
        for bits in sigmas:
            row=reference(bits,dtype,radius,precision=120)
            w,d=weights(float(row['sigma']),dtype,radius)
            assert w==w[::-1] and d==d[::-1]
            for got,expected in zip(w,row['weights']):
                assert abs(decode_bits(bits_from_float(got,dtype),dtype)-expected)<=bound
            for got,expected in zip(d,row['derivatives']):
                assert abs(decode_bits(bits_from_float(got,dtype),dtype)-expected)<=derivative_bound


@pytest.mark.parametrize('dtype,sigma,radius',[('float32',.138,2),('float64',.1033,4)])
def test_gaussian_candidate_preserves_derivative_after_weight_underflow(dtype,sigma,radius):
    from tools.gaussian_weights_candidate import weights
    from tools.gaussian_weights_oracle import bits_from_float,reference,round_bits
    row=reference(bits_from_float(sigma,dtype),dtype,radius,precision=180)
    w,d=weights(float(row['sigma']),dtype,radius)
    assert bits_from_float(w[0],dtype)==round_bits(row['weights'][0],dtype)==0
    assert bits_from_float(d[0],dtype)==round_bits(row['derivatives'][0],dtype)!=0


def test_gaussian_candidate_reduction_boundary_neighbors_stay_in_envelope():
    from tools.gaussian_weights_candidate import exponential
    with pytest.raises(ValueError):exponential(.1)
    assert exponential(0.)==(1.,0)
    ln2=math.log(2.)  # Test-point selection only; Decimal supplies the reference.
    for n in (-1154,-1000,-10,-1):
        boundary=(n+.5)*ln2
        for x in (math.nextafter(boundary,-math.inf),boundary,math.nextafter(boundary,math.inf)):
            m,p=exponential(x)
            with localcontext() as context:
                context.prec=180
                expected=Decimal.from_float(x).exp()
                got=Decimal.from_float(m)*(Decimal(2)**p)
                assert abs(got/expected-1)<Decimal(100)*Decimal(2)**-53


@pytest.mark.parametrize('dtype',['float32','float64'])
def test_gaussian_candidate_vjp_uses_dtype_products_and_fixed_tree(dtype):
    from tools.gaussian_weights_candidate import vjp,weights
    from tools.gaussian_weights_oracle import bits_from_float,decode_bits,round_bits
    for radius in (0,1,2,7,32):
        cotangents=[(-1.)**i*(i+.25) for i in range(2*radius+1)]
        _,derivatives=weights(.83,dtype,radius)
        with localcontext() as context:
            context.prec=2500
            def represented(x):return decode_bits(bits_from_float(x,dtype),dtype)
            def rounded(x):return decode_bits(round_bits(x,dtype),dtype)
            values=[rounded(represented(d)*represented(g)) for d,g in zip(derivatives,cotangents)]
            while len(values)>1:
                values=[rounded(values[i]+values[i+1]) for i in range(0,len(values)-1,2)]+(values[-1:] if len(values)%2 else [])
            expected=0 if radius==0 else round_bits(values[0],dtype)
        assert bits_from_float(vjp(.83,dtype,radius,cotangents),dtype)==expected


@pytest.mark.parametrize('dtype',['float32','float64'])
def test_gaussian_candidate_vjp_validates_radius_zero_and_cotangents(dtype):
    from tools.gaussian_weights_candidate import vjp
    from tools.gaussian_weights_oracle import bits_from_float
    for g in (-3.,-0.,0.,2.):assert bits_from_float(vjp(.83,dtype,0,[g]),dtype)==0
    for bad in (math.inf,-math.inf,math.nan):
        with pytest.raises(ValueError):vjp(.83,dtype,0,[bad])
    with pytest.raises(ValueError):vjp(.83,dtype,1,[1.])
    with pytest.raises(ValueError):vjp(.01,dtype,0,[1.])


@pytest.mark.parametrize('dtype',['float32','float64'])
def test_gaussian_candidate_vjp_overflow_cannot_be_hidden_by_cancellation(dtype,monkeypatch):
    from tools import gaussian_weights_candidate as candidate
    maximum=float.fromhex('0x1.fffffep+127' if dtype=='float32' else '0x1.fffffffffffffp+1023')
    # Isolate prescribed product and tree overflow from the approximation itself.
    monkeypatch.setattr(candidate,'weights',lambda *args:([0.,1.,0.],[2.,-2.,0.]))
    with pytest.raises(OverflowError):candidate.vjp(1.,dtype,1,[maximum,maximum,1.])
    monkeypatch.setattr(candidate,'weights',lambda *args:([0.,1.,0.],[1.,1.,-1.]))
    with pytest.raises(OverflowError):candidate.vjp(1.,dtype,1,[maximum,maximum,maximum])


@pytest.mark.parametrize('dtype',['float32','float64'])
def test_gaussian_candidate_vjp_cancellation_obeys_pair_order(dtype,monkeypatch):
    from tools import gaussian_weights_candidate as candidate
    # Adjacent pairs give 0; a reassociated expression would give 1.
    large=2.**(24 if dtype=='float32' else 53)
    monkeypatch.setattr(candidate,'weights',lambda *args:([0.,1.,0.],[1.,1.,1.]))
    assert candidate.vjp(1.,dtype,1,[large,1.,-large])==0.


@pytest.mark.parametrize('dtype',['float32','float64'])
def test_gaussian_candidate_log_interior_and_endpoint_envelopes(dtype,record_property):
    import json,random
    from tools.gaussian_weights_candidate import weights,vjp
    from tools.gaussian_weights_oracle import bits_from_float,decode_bits,reference,round_bits
    rng=random.Random(913)
    inputs={bits_from_float(.1,dtype),bits_from_float(.1,dtype)+1,bits_from_float(32.,dtype)-1,bits_from_float(32.,dtype)}
    inputs.update(bits_from_float(.1*320**(i/16),dtype) for i in range(1,16))
    inputs.update(bits_from_float(.1+31.9*rng.random(),dtype) for _ in range(12))
    u=Decimal(2)**-53
    wb=512*u+Decimal(2)**-990;db=8192*u+Decimal(2)**-990
    output_u=u if dtype=='float64' else Decimal(2)**-24
    if dtype=='float32':wb+=Decimal(2)**-24;db+=Decimal(2)**-21
    maxima=dict(weight=Decimal(0),derivative=Decimal(0),vjp=Decimal(0)); cases=0; mismatches=0
    for radius in range(33):
        for bits in sorted(inputs):
            row=reference(bits,dtype,radius,precision=120)
            w,d=weights(float(row['sigma']),dtype,radius)
            cotangents=[(-1.)**i*(.125+i/16) for i in range(len(w))]
            cb=[bits_from_float(g,dtype) for g in cotangents]
            for key,actual,expected,bound in [('weight',w,row['weights'],wb),('derivative',d,row['derivatives'],db)]:
                for a,e in zip(actual,expected):
                    value=decode_bits(bits_from_float(a,dtype),dtype)
                    error=abs(value-e);maxima[key]=max(maxima[key],error)
                    assert error<=bound
                    mismatches+=int((value==0)!=(round_bits(e,dtype)&((1<<(32 if dtype=='float32' else 64))-1)==0)) if e>=0 else 0
            gs=[decode_bits(g,dtype) for g in cb]
            exact=sum((g*e for g,e in zip(gs,row['derivatives'])),Decimal(0))
            got=decode_bits(bits_from_float(vjp(float(row['sigma']),dtype,radius,cotangents),dtype),dtype)
            error=abs(got-exact);maxima['vjp']=max(maxima['vjp'],error)
            # Separate analytic error and one product plus depth-seven pair rounding.
            gamma8=8*output_u/(1-8*output_u)
            bound=sum(map(abs,gs))*db+gamma8*sum(abs(g)*(abs(e)+db) for g,e in zip(gs,row['derivatives']))+Decimal(len(w))*Decimal(2)**(-149 if dtype=='float32' else -1074)
            assert error<=bound
            cases+=1
    record_property('gaussian_candidate_sweep',json.dumps(dict(dtype=dtype,cases=cases,sigma_count=len(inputs),max_errors={k:str(v) for k,v in maxima.items()},positive_zero_classification_differences=mismatches)))
