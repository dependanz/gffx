"""Opt-in compiled bookkeeping with identical directed Decimal operations."""
from contextlib import contextmanager
from decimal import localcontext,ROUND_FLOOR,ROUND_CEILING
from pathlib import Path
import hashlib,json,os,importlib.util
from tools.opaque_shading_intervals import I,Jet,ZERO,PRECISION
_active=False

def library():
    path=Path(os.environ['GFFX_PROOF_DECIMAL_MANIFEST']);raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=os.environ['GFFX_PROOF_DECIMAL_MANIFEST_SHA']:raise ValueError('decimal native manifest hash')
    p=json.loads(raw)
    for row in p['files']:
        if hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()!=row['sha256']:raise ValueError('decimal native file hash')
    spec=importlib.util.spec_from_file_location('_opaque_decimal_native',p['module']);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

@contextmanager
def decimal_convolution():
    global _active
    if _active:raise RuntimeError('nested decimal native context')
    module=library();original,reverse=Jet.__mul__,Jet.__rmul__;stats=dict(fast=0,ordinary=0)
    def multiply(a,b):
        b=a.other(b)
        if len(a.c)*len(b.c)<=1:stats['ordinary']+=1;return original(a,b)
        aa=[(i,j,v.lo,v.hi) for (i,j),v in a.c.items()];bb=[(i,j,v.lo,v.hi) for (i,j),v in b.c.items()]
        with localcontext() as ctx:
            ctx.prec=PRECISION;out=module.convolve(aa,bb,a.order,ctx,ZERO.lo,ROUND_FLOOR,ROUND_CEILING)
        stats['fast']+=1;return Jet._from_owned(a.order,{k:I(*v) for k,v in out.items()})
    _active=True;Jet.__mul__=Jet.__rmul__=multiply
    try:yield stats
    finally:Jet.__mul__,Jet.__rmul__=original,reverse;_active=False
