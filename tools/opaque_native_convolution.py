"""Isolated hash-pinned native high convolution; protected Decimal low coefficients."""
from contextlib import contextmanager
from decimal import Decimal as D
from pathlib import Path
import ctypes as C,hashlib,json,os
from tools.opaque_shading_intervals import I,Jet
from tools.opaque_binary_derivatives import _endpoint,_Range
_active=False
class Term(C.Structure):_fields_=[('i',C.c_int),('j',C.c_int),('lo',C.c_double),('hi',C.c_double)]
def library():
    manifest=Path(os.environ['GFFX_PROOF_NATIVE_MANIFEST']);raw=manifest.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=os.environ['GFFX_PROOF_NATIVE_MANIFEST_SHA']:raise ValueError('native proof manifest hash')
    p=json.loads(raw)
    for row in p['files']:
        if hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()!=row['sha256']:raise ValueError('native proof file hash')
    dll=C.CDLL(p['dll']);fn=dll.gffx_proof_convolve
    fn.argtypes=[C.c_int,C.POINTER(Term),C.c_int,C.POINTER(Term),C.c_int,C.POINTER(C.c_double),C.POINTER(C.c_uint32)];fn.restype=C.c_int
    return dll,fn
@contextmanager
def native_convolution():
    global _active
    if _active:raise RuntimeError('nested native derivative context')
    dll,fn=library();original,reverse=Jet.__mul__,Jet.__rmul__;stats=dict(fast=0,fallback=0,ordinary=0)
    def multiply(a,b):
        b=a.other(b)
        if a.order<3 or len(a.c)*len(b.c)<36:stats['ordinary']+=1;return original(a,b)
        try:
            aa=(Term*len(a.c))(*(Term(i,j,_endpoint(v.lo,False),_endpoint(v.hi,True)) for (i,j),v in a.c.items()))
            bb=(Term*len(b.c))(*(Term(i,j,_endpoint(v.lo,False),_endpoint(v.hi,True)) for (i,j),v in b.c.items()))
            out=(C.c_double*44)();mask=C.c_uint32();code=fn(a.order,aa,len(aa),bb,len(bb),out,C.byref(mask))
            if code in (1,3):raise _Range
            if code:raise ValueError('invalid native convolution arguments')
            lowa=Jet._from_owned(2,{k:v for k,v in a.c.items() if k[0]<=2});lowb=Jet._from_owned(2,{k:v for k,v in b.c.items() if k[0]<=2});result=original(lowa,lowb).c
            result.update({(i,j):I(D.from_float(out[2*(2*i+j)]),D.from_float(out[2*(2*i+j)+1])) for i in range(3,a.order+1) for j in range(2) if mask.value&(1<<(2*i+j))})
        except (OverflowError,_Range):stats['fallback']+=1;return original(a,b)
        stats['fast']+=1;return Jet._from_owned(a.order,result)
    _active=True;Jet.__mul__=Jet.__rmul__=multiply
    try:yield stats
    finally:Jet.__mul__,Jet.__rmul__=original,reverse;_active=False
