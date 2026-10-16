"""Bounded crossover timing with identical arithmetic endpoints."""
import os,json,time
from pathlib import Path
from contextlib import nullcontext
from fractions import Fraction as Q
from tools.opaque_shading_intervals import I,Jet
from tools.opaque_decimal_convolution import decimal_convolution

def test_decimal_convolution_crossover_is_bounded_and_exact():
    dest=Path(os.environ['GFFX_FULL_REFERENCE']);assert not dest.exists();start=time.perf_counter();rows=[];calls=0
    try:
        for order in (0,1,2,3,6,10):
            for tangent in (False,True):
                a=Jet(order=order,coeff={(i,j):I(Q((-1)**i*(i+1),17**(i+j+1))) for i in range(order+1) for j in range(1+int(tangent))});b=a+1;expected=a*b;row=dict(order=order,terms=len(a.c)*len(b.c),profiles=[]);rows.append(row)
                for mode in ('decimal','compiled','compiled','decimal'):
                    with decimal_convolution() if mode=='compiled' else nullcontext():
                        t=time.perf_counter()
                        for _ in range(20):
                            assert time.perf_counter()-start<20;calls+=1;got=a*b
                        seconds=time.perf_counter()-t
                    assert list(got.c)==list(expected.c) and all((got[k].lo,got[k].hi)==(expected[k].lo,expected[k].hi) for k in got.c)
                    row['profiles'].append(dict(mode=mode,seconds=seconds,calls=20))
    finally:dest.write_text(json.dumps(dict(rows=rows,calls=calls,seconds=time.perf_counter()-start,max_seconds=20,max_calls=960),indent=2),encoding='utf8')
