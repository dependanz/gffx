"""Explicit standalone compiled-scalar qualification; not installed blur/device gates."""
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import pytest

@pytest.fixture
def probe(record_property):
    path=os.environ.get('GFFX_GAUSSIAN_SCALAR_PROBE')
    if not path:pytest.skip('standalone scalar probe must be supplied explicitly')
    path=Path(path);library=ctypes.CDLL(str(path))
    pointer=ctypes.POINTER(ctypes.c_double)
    library.gaussian_test_scale.argtypes=[ctypes.c_double,ctypes.c_int,pointer]
    library.gaussian_test_weights.argtypes=[ctypes.c_double,ctypes.c_int,ctypes.c_int,pointer,pointer]
    library.gaussian_test_vjp.argtypes=[ctypes.c_double,ctypes.c_int,ctypes.c_int,pointer,pointer]
    for name in ('gaussian_test_scale','gaussian_test_weights','gaussian_test_vjp'):getattr(library,name).restype=ctypes.c_int
    record_property('gaussian_scalar_library',json.dumps(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())))
    return library


def test_gaussian_compiled_scaler_exact_boundaries(probe,record_property):
    from decimal import Decimal,localcontext
    from tools.gaussian_weights_oracle import bits_from_float,decode_bits,round_bits
    values=[0.,-0.,1.,-1.,1.5,-1.5,float.fromhex('0x1.fffffffffffffp+0'),float.fromhex('0x0.0000000000001p-1022'),float.fromhex('0x1p-1022')]
    # Include both signs around every significand rounding/carry bit.
    for k in range(1,53):
        for sign in (-1,1):values.append(sign*math.nextafter(2.**k,0.))
    count=0
    for value in values:
        for exponent in (-2000,-1075,-1074,-1073,-1023,-1022,-53,-1,0,1,52,1023,2000):
            with localcontext() as ctx:
                ctx.prec=2500
                expected=round_bits(decode_bits(bits_from_float(value,'float64'),'float64')*Decimal(2)**exponent,'float64')
            output=ctypes.c_double(91.);status=probe.gaussian_test_scale(value,exponent,ctypes.byref(output))
            if (expected>>52)&2047==2047:assert status!=0 and output.value==91.
            else:assert status==0 and bits_from_float(output.value,'float64')==expected
            count+=1
    for value in (-1.,1.):
        output=ctypes.c_double()
        assert probe.gaussian_test_scale(value,-2147483648,ctypes.byref(output))==0
        assert math.copysign(1.,output.value)==value and output.value==0.
        assert probe.gaussian_test_scale(value,2147483647,ctypes.byref(output))!=0
    record_property('exact_scaling_pairs',str(count))


@pytest.mark.parametrize('dtype',['float32','float64'])
def test_gaussian_compiled_weights_and_vjp_match_candidate(probe,dtype,record_property):
    import random
    from tools.gaussian_weights_candidate import weights,vjp
    from tools.gaussian_weights_oracle import bits_from_float,decode_bits
    rng=random.Random(913);precision=32 if dtype=='float32' else 64
    inputs={bits_from_float(.1,dtype),bits_from_float(.1,dtype)+1,bits_from_float(32.,dtype)-1,bits_from_float(32.,dtype),bits_from_float(.83,dtype),bits_from_float(.83,dtype)+1}
    inputs.update(bits_from_float(.1*320**(i/16),dtype) for i in range(1,16))
    inputs.update(bits_from_float(.1+31.9*rng.random(),dtype) for _ in range(12))
    cases=0
    for radius in range(33):
        for bits in sorted(inputs):
            sigma=float(decode_bits(bits,dtype));expected,derivative=weights(sigma,dtype,radius);count=2*radius+1
            w=(ctypes.c_double*count)();d=(ctypes.c_double*count)()
            assert probe.gaussian_test_weights(sigma,radius,precision,w,d)==0
            assert [bits_from_float(x,dtype) for x in w]==[bits_from_float(x,dtype) for x in expected]
            assert [bits_from_float(x,dtype) for x in d]==[bits_from_float(x,dtype) for x in derivative]
            gs=[(-1.)**i*(.125+i/16) for i in range(count)];g=(ctypes.c_double*count)(*gs);result=ctypes.c_double()
            assert probe.gaussian_test_vjp(sigma,radius,precision,g,ctypes.byref(result))==0
            assert bits_from_float(result.value,dtype)==bits_from_float(vjp(sigma,dtype,radius,gs),dtype)
            cases+=1
    record_property('compiled_candidate_cases',str(cases))


@pytest.mark.parametrize('dtype,radius,sigma',[('float32',2,.138),('float64',4,.1033)])
def test_gaussian_compiled_underflow_retains_independent_derivative(probe,dtype,radius,sigma):
    from tools.gaussian_weights_oracle import bits_from_float,reference,round_bits
    row=reference(bits_from_float(sigma,dtype),dtype,radius,precision=180)
    count=2*radius+1;w=(ctypes.c_double*count)();d=(ctypes.c_double*count)()
    assert probe.gaussian_test_weights(float(row['sigma']),radius,32 if dtype=='float32' else 64,w,d)==0
    assert bits_from_float(w[0],dtype)==round_bits(row['weights'][0],dtype)==0
    assert bits_from_float(d[0],dtype)==round_bits(row['derivatives'][0],dtype)!=0
