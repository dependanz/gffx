"""Bounded in-process standalone scalar/Decimal checks; not a shipped-wheel or CUDA claim.

Compile the test-only DLL with the standalone strict-FP CMake target first. Pass
its hash plus the scalar-header and C-probe hashes observed during that build.
Preflight precedes the full sampled exponent envelope under unchanged supervision.
"""
import argparse
import ctypes
from decimal import Decimal,localcontext
import hashlib
import json
import math
import os
from pathlib import Path
import time

from tools import material_loss_cost as cost
from tools.image_transfer_oracle import reference,limits,rounded,next_up,next_down,envelope_inputs,overflow_inputs

TOLERANCES={'float64':(2e-12,2e-12),'float32':(2e-5,2e-5)}
SOURCES=('native/core/image_transfer_scalar.h','tests/abi/test_image_transfer_scalar.c',
         'native/core/image_transfer_boundary.h',
         'tests/abi/image_transfer/CMakeLists.txt','tools/image_transfer_oracle.py',
         'tools/image_transfer_envelope.py')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def screen(dtype,oracle,status,value,vjp):
    """Original bounded tolerances plus stricter full-range relative/subnormal checks."""
    assert dtype in TOLERANCES,'unknown precision'
    if oracle['overflow']:
        assert status!=0,'independently classified overflow was reported as finite success'
        return dict(overflow=True,value_relative_error=0.,vjp_relative_error=0.)
    assert status==0,'representable reference unexpectedly rejected'
    absolute,relative=(Decimal(str(x)) for x in TOLERANCES[dtype])
    spec=limits(dtype);minimum=Decimal.from_float(spec['min_normal'])
    subnormal=Decimal.from_float(spec['min_subnormal']);metrics={'overflow':False}
    for name,actual,expected in (('value',value,oracle['value']),('vjp',vjp,oracle['derivative'])):
        assert math.isfinite(actual),f'{name} is nonfinite'
        assert rounded(actual,dtype)==actual,f'{name} was not rounded to the declared dtype'
        with localcontext() as context:
            context.prec=120
            error=abs(Decimal.from_float(actual)-expected);magnitude=abs(expected)
            assert error<=absolute+relative*magnitude,f'{name} exceeds the unchanged accuracy tolerance'
            if magnitude>=minimum:
                assert error<=relative*magnitude,f'{name} exceeds the additional full-range relative screen'
            else:
                assert error<=4*subnormal,f'{name} exceeds four subnormal units'
            if expected.is_zero() or actual==0:
                assert (math.copysign(1.,actual)<0)==expected.is_signed(),f'{name} zero sign differs'
            metrics[name+'_relative_error']=float(error/magnitude) if magnitude else 0.
    return metrics


def check_preflight(previous,identity):
    if previous is None:raise ValueError('full envelope requires a passed matching preflight')
    supervisor=previous.get('supervisor',{})
    if (previous.get('status')!='passed' or previous.get('mode')!='preflight' or
        previous.get('identity')!=identity or supervisor.get('exit_code')!=0 or supervisor.get('failure')):
        raise ValueError('preflight identity, mode or supervision differs')


def preflight_inputs(dtype):
    spec=limits(dtype)
    values=[-0.,0.,-spec['min_subnormal'],spec['min_subnormal'],-spec['min_normal'],spec['min_normal'],
            -spec['max_finite'],spec['max_finite'],-1.,.1,.2,.7,1.,4.,16.]
    for exponent in (-1000,-100,-10,-3,-1,0,1,10,100,1000):
        if spec['emin']<=exponent<=spec['emax']:
            values.extend(rounded(math.ldexp(m,exponent),dtype) for m in (1.,1.25,1.5))
    for threshold in (.04045,.0031308):
        x=rounded(threshold,dtype);values.extend((next_down(x,dtype),x,next_up(x,dtype)))
    if dtype=='float64':
        for x in (2.**425,2.**1018):
            values.extend((next_down(x,dtype),x,next_up(x,dtype)))
            values.extend((-next_down(x,dtype),-x,-next_up(x,dtype)))
    values.extend(overflow_inputs(dtype))
    return [rounded(value,dtype) for value in values]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',action='store_true')
    for name in ('fixtures','output','library'):
        parser.add_argument('--'+name,type=Path,required=True)
    for name in ('sha256','source-sha256','probe-sha256'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--mode',choices=('preflight','full'),required=True)
    parser.add_argument('--preflight',type=Path)
    args=parser.parse_args()
    if not args.worker:
        extra=['--library',str(args.library.resolve()),'--sha256',args.sha256,
               '--source-sha256',args.source_sha256,'--probe-sha256',args.probe_sha256,'--mode',args.mode]
        if args.preflight is not None:extra+=['--preflight',str(args.preflight)]
        return cost.supervise(args.fixtures,args.output,worker_module='tools.image_transfer_envelope',worker_args=extra)
    row=dict(status='working',worker_pid=os.getpid(),mode=args.mode,cases=0,root_cases=0,
             tolerances=TOLERANCES,full_range_extra='relative normal; four subnormal units; zero sign',
             scope='Standalone strict-FP scalar candidate and sampled Decimal envelope; not CUDA, ABI or wheel qualification')
    cost.write_report(args.output,row);deadline=time.perf_counter()+120
    try:
        identity={'library_sha256':digest(args.library),'sources':{name:digest(name) for name in SOURCES}}
        if (identity['library_sha256']!=args.sha256 or identity['sources'][SOURCES[0]]!=args.source_sha256 or
            identity['sources'][SOURCES[1]]!=args.probe_sha256):raise ValueError('compiled-library or build-source identity differs')
        row['identity']=identity
        if args.mode=='full':
            check_preflight(json.loads(args.preflight.read_text()) if args.preflight else None,identity)
            row['preflight_sha256']=digest(args.preflight)
        library=ctypes.CDLL(str(args.library.resolve()))
        transfer=library.image_test_transfer
        transfer.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_double,ctypes.c_double,
                           ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double)]
        transfer.restype=ctypes.c_int
        root=library.image_test_root;root.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_double];root.restype=ctypes.c_double
        dd=library.image_test_dd
        dd.argtypes=[ctypes.c_int,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_double,
                     ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double)]
        dd.restype=None
        row['max_relative_errors']={};row['root_max_relative_errors']={}
        for dtype,bits in (('float32',32),('float64',64)):
            values=list(envelope_inputs(dtype)) if args.mode=='full' else preflight_inputs(dtype)
            for mode,direction in (('decode',0),('encode',1)):
                key=dtype+'/'+mode;maximum=[0.,0.]
                for x in values:
                    cost.check_deadline(deadline)
                    row['last_case']=dict(dtype=dtype,mode=mode,input_hex=x.hex())
                    oracle=reference(x,dtype,mode);y=ctypes.c_double();vjp=ctypes.c_double()
                    status=transfer(bits,direction,x,1.,ctypes.byref(y),ctypes.byref(vjp))
                    try:metrics=screen(dtype,oracle,status,y.value,vjp.value)
                    except AssertionError:
                        row['failed_result']=dict(status=status,value=repr(y.value),vjp=repr(vjp.value),
                            reference_value=str(oracle['value']),reference_vjp=str(oracle['derivative']),branch=oracle['branch'])
                        raise
                    maximum=[max(maximum[0],metrics['value_relative_error']),max(maximum[1],metrics['vjp_relative_error'])]
                    row['cases']+=1
                    if row['cases']%128==0:cost.write_report(args.output,row)
                row['max_relative_errors'][key]=dict(value=maximum[0],vjp=maximum[1])
            intervals=1024 if args.mode=='full' else 16
            for q in (5,12):
                maximum=0.;bound=Decimal(16)*Decimal(2)**(-limits(dtype)['fraction_bits'])
                for i in range(intervals):
                    cost.check_deadline(deadline);x=rounded(1+i/intervals,dtype)
                    with localcontext() as context:
                        context.prec=120;expected=context.power(Decimal.from_float(x),Decimal(1)/Decimal(q))
                        actual=root(bits,q,x);error=abs(Decimal.from_float(actual)-expected)/abs(expected)
                    row['last_case']=dict(dtype=dtype,root=q,mantissa_hex=x.hex())
                    assert error<=bound,'fixed24 root exceeds the additional16-epsilon sampled convergence screen'
                    maximum=max(maximum,float(error));row['root_cases']+=1
                row['root_max_relative_errors'][dtype+'/'+str(q)]=maximum
        row['compensated_cases']=0
        # Exact rational arithmetic checks exercise cancellation and signed tails,
        # independently of the HDR reproducer and transfer power formula.
        for operation in (0,1,2):
            for a,b in ((1.,3.),(1.+2.**-27,1.-2.**-27),(-.75,1.25),(2.**-430,1.055)):
                cost.check_deadline(deadline);hi=ctypes.c_double();lo=ctypes.c_double()
                dd(operation,a,0.,b,0.,ctypes.byref(hi),ctypes.byref(lo))
                with localcontext() as context:
                    context.prec=160;da=Decimal.from_float(a);db=Decimal.from_float(b)
                    expected=(da+db,da*db,da/db)[operation]
                    actual=Decimal.from_float(hi.value)+Decimal.from_float(lo.value)
                    assert abs(actual-expected)<=abs(expected)*Decimal(2)**-98,'compensated arithmetic control failed'
                row['compensated_cases']+=1
        for r in range(5):
            hi=ctypes.c_double();lo=ctypes.c_double();dd(4,float(r),0.,0.,0.,ctypes.byref(hi),ctypes.byref(lo))
            with localcontext() as context:
                context.prec=160;actual=Decimal.from_float(hi.value)+Decimal.from_float(lo.value)
                assert abs(actual**5-Decimal(2)**r)<=Decimal(2)**(r-100),'compensated constant residual failed'
            row['compensated_cases']+=1
        intervals=1024 if args.mode=='full' else 16
        for x in [1+i/intervals for i in range(intervals)]+[math.nextafter(2.,0.)]:
            cost.check_deadline(deadline);hi=ctypes.c_double();lo=ctypes.c_double()
            dd(3,x,0.,0.,0.,ctypes.byref(hi),ctypes.byref(lo))
            with localcontext() as context:
                context.prec=160;actual=Decimal.from_float(hi.value)+Decimal.from_float(lo.value)
                expected=context.power(Decimal.from_float(x),Decimal(1)/5)
                assert abs(actual-expected)<=expected*Decimal(2)**-88,'compensated root convergence failed'
            row['compensated_cases']+=1
        if digest(args.library)!=args.sha256 or {name:digest(name) for name in SOURCES}!=identity['sources']:
            raise ValueError('source or library changed during qualification')
        row['status']='passed'
    except Exception as error:
        row.update(status='stopped',error=f'{type(error).__name__}: {error}')
        raise
    finally:cost.write_report(args.output,row)
    return 0


if __name__=='__main__':raise SystemExit(main())
