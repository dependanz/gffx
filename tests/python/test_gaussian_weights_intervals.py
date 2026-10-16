"""Directed mathematical intervals at selected Gaussian underflow transitions."""
import json
import pytest


@pytest.mark.parametrize('dtype,radius',[('float32',2),('float64',4),('float64',8),('float64',16),('float64',32)])
@pytest.mark.parametrize('quantity',['weights','derivatives'])
def test_gaussian_directed_underflow_transition_classification(dtype,radius,quantity,record_property):
    from tools.gaussian_weights_oracle import bits_from_float,reference,round_bits
    from tools.gaussian_weights_intervals import reference_intervals
    from tools.gaussian_weights_candidate import weights
    # On this small-sigma outer tap, both positive quantities increase with sigma.
    # Locate the first represented nonzero rounded result, then certify neighbors.
    low=bits_from_float(.1,dtype);high=bits_from_float(1.,dtype)
    def nonzero(bits):return round_bits(reference(bits,dtype,radius,precision=180)[quantity][0],dtype)!=0
    assert not nonzero(low) and nonzero(high)
    while high-low>1:
        middle=(high+low)//2
        if nonzero(middle):high=middle
        else:low=middle
    rows=[]
    for bits in range(high-2,high+3):
        mathematical=reference(bits,dtype,radius,precision=180)
        interval=reference_intervals(bits,dtype,radius,precision=100)[quantity][0]
        lo,hi=interval
        assert lo<=mathematical[quantity][0]<=hi
        lower,upper=round_bits(lo,dtype),round_bits(hi,dtype)
        assert lower==upper,'directed interval does not certify the rounding decision'
        candidate=weights(float(mathematical['sigma']),dtype,radius)[0 if quantity=='weights' else 1][0]
        actual=bits_from_float(candidate,dtype)
        rows.append(dict(sigma_bits=bits,expected_bits=lower,candidate_bits=actual,
                         zero_classification_equal=(lower==0)==(actual==0)))
    record_property('gaussian_directed_transition',json.dumps(dict(dtype=dtype,radius=radius,quantity=quantity,rows=rows)))
    # An attribution control: certify the mathematical classification and record
    # candidate disagreement; do not silently call any disagreement acceptance.
