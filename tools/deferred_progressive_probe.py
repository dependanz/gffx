"""Run an unchanged resident/stream worker with approved progressive forecasting.

Only the forecast callback, report provenance and child-module routing are wrapped.
Original execution drivers, rendering, memory admission and supervision are unchanged.
"""
import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path
from tools import deferred_progressive_admission as admission


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--execution',choices=('resident','stream'),required=True)
    parser.add_argument('--anchor',nargs=2,action='append',default=[])
    args,remaining=parser.parse_known_args()
    module='tools.deferred_flame_probe' if args.execution=='resident' else 'tools.deferred_stream_probe'
    driver=importlib.import_module(module);cost=driver.cost
    original_forecast=driver.forecast;original_supervise=cost.supervise;original_write=cost.write_report
    argv=sys.argv[:];current={};metadata=dict(policy=admission.POLICY,
        wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        policy_sha256=hashlib.sha256(Path(admission.__file__).read_bytes()).hexdigest(),
        execution_module=module,explicit_anchors=[dict(path=p,sha256=s) for p,s in args.anchor])
    def write(path,row):
        current['row']=row
        if '--worker' in remaining:
            row['admission_policy']=metadata.copy()
        return original_write(path,row)
    def forecast(rows,**kwargs):
        # Original startup anchors are validated by the original forecast too.
        metadata['original_32_64_forecast_seconds']=original_forecast(rows,**kwargs)
        anchors=[]
        for flag in ('--calibration32','--calibration64'):
            if flag not in remaining:raise ValueError('original startup calibration required')
            path=Path(remaining[remaining.index(flag)+1]);sha=hashlib.sha256(path.read_bytes()).hexdigest()
            anchors.append(admission.load_anchor(path,sha))
        for path,sha in args.anchor:anchors.append(admission.load_anchor(path,sha))
        result=admission.forecast(anchors,current['row']);metadata.update(result)
        return result['seconds']
    def supervise(fixtures,output,**kwargs):
        if kwargs['worker_module']!=module:raise ValueError('unexpected execution module')
        kwargs['worker_module']='tools.deferred_progressive_probe'
        extra=['--execution',args.execution]
        for path,sha in args.anchor:extra+=['--anchor',path,sha]
        kwargs['worker_args']+=extra
        return original_supervise(fixtures,output,**kwargs)
    try:
        sys.argv=[argv[0],*remaining];driver.forecast=forecast;cost.supervise=supervise;cost.write_report=write
        return driver.main()
    finally:
        sys.argv=argv;driver.forecast=original_forecast;cost.supervise=original_supervise;cost.write_report=original_write


if __name__=='__main__':raise SystemExit(main())
