"""Bounded deferred TDD workers; explicit test selection and installed artifact."""
import argparse
import os
from pathlib import Path
import time
from tools import material_loss_cost as cost, shading_cuda_closeout as artifact


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--worker',action='store_true')
    for name in ('fixtures','output','wheel'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--sha256',required=True)
    p.add_argument('--selection',default='tests/python/test_deferred.py')
    p.add_argument('--expect-failure',action='store_true')
    p.add_argument('--deferred-source',type=Path)
    a=p.parse_args()
    if not a.worker:
        extra=['--wheel',str(a.wheel),'--sha256',a.sha256,'--selection',a.selection]
        if a.expect_failure:extra.append('--expect-failure')
        if a.deferred_source is not None:extra.extend(('--deferred-source',str(a.deferred_source)))
        return cost.supervise(a.fixtures,a.output,worker_module='tools.deferred_probe',worker_args=extra)
    row=dict(status='working',worker_pid=os.getpid(),selection=a.selection,driver_sha256=artifact.digest(__file__))
    cost.write_report(a.output,row);deadline=time.perf_counter()+120
    try:
        import torch
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        row['wheel']=artifact.identity(a.wheel,a.sha256)
        os.environ['GFFX_TEST_ARTIFACT_SHA256']=row['wheel']['sha256']
        if a.deferred_source is not None:
            import hashlib
            import importlib.util
            import sys
            source=a.deferred_source.resolve()
            if source!=Path('src/gffx/torch/_deferred.py').resolve():
                raise ValueError('overlay must be the explicitly reviewed deferred source')
            row['source_overlay']=dict(path=str(source),sha256=artifact.digest(source))
            row['qualification']='source-overlay diagnostic, not installed-wheel qualification'
            # Keep every installed artifact byte untouched. Only this worker's
            # named Python module is replaced after the base wheel is verified.
            spec=importlib.util.spec_from_file_location('gffx.torch._deferred',source)
            module=importlib.util.module_from_spec(spec)
            sys.modules[spec.name]=module;spec.loader.exec_module(module)
            combined=(row['wheel']['sha256']+row['source_overlay']['sha256']).encode('ascii')
            os.environ['GFFX_TEST_ARTIFACT_SHA256']=hashlib.sha256(combined).hexdigest()
        if not a.selection.startswith('tests/python/test_deferred'):
            raise ValueError('selection outside deferred tests')
        row['test_sha256']=artifact.digest(a.selection.split('::')[0])
        import pytest
        code=int(pytest.main(['-q','-x',a.selection,'--junitxml='+str(a.output.with_suffix('.xml'))]))
        row['pytest_exit_code']=code
        cost.check_deadline(deadline)
        if a.expect_failure:
            if code!=1:raise ValueError('expected one assertion/import red failure')
            row['status']='expected red'
        else:
            if code:raise ValueError('deferred qualification failed')
            row['status']='passed'
    except Exception as error:
        row.update(status='stopped',error=f'{type(error).__name__}: {error}')
        raise
    finally:cost.write_report(a.output,row)
    return 0


if __name__=='__main__':raise SystemExit(main())
