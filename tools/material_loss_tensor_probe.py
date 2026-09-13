"""Finite tensor-resource timing or unchanged physical regression under a watchdog."""
import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import torch
from tools import material_loss_cost as cost
from tools.material_loss_tensor import TensorLossResource


def query(resource,args,*,backward,account=False):
    saved={}
    constant={x.untyped_storage().data_ptr() for x in resource._data.values()}
    def pack(x):
        storage=x.untyped_storage()
        saved[storage.data_ptr()]=storage.nbytes()
        return x
    hooks=torch.autograd.graph.saved_tensors_hooks(pack,lambda x:x) if account else nullcontext()
    with hooks,torch.set_grad_enabled(backward):
        result=resource.evaluate(*args)
        objective=(result['loss']+result['mean_loss']).sum()
        gradients=torch.autograd.grad(objective,args) if backward else None
        if not bool(torch.isfinite(objective)) or gradients is not None and not all(bool(torch.isfinite(g).all()) for g in gradients):
            raise ValueError('nonfinite tensor measurement')
    return dict(objective=objective.item(),gradient_sums=[g.sum().item() for g in gradients] if gradients else None,
                saved_tensor_bytes=sum(saved.values()) if account else None,
                saved_nonconstant_bytes=sum(size for pointer,size in saved.items() if pointer not in constant) if account else None)


def worker(fixtures,output,mode):
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    deadline=time.perf_counter()+cost.PROFILE['cooperative_seconds']
    root=Path(__file__).parent
    profile=dict(cost.PROFILE,batch_implementation='vectorized tensor curves; no scalar query loop',
                 memory_qualification='distinct saved tensor storage plus planning estimate, not allocator peak')
    report=dict(status='working',worker_pid=os.getpid(),profile=profile,mode=mode,
        environment=dict(python=sys.version,executable=sys.executable,torch=torch.__version__,torch_file=torch.__file__,host=platform.node()),
        hashes={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in
                ('material_loss_tensor.py','material_loss_tensor_probe.py','material_loss_smooth.py','material_loss_cost.py',
                 'material_loss_reference.py','material_mean_validation.py')},resources={})
    cost.write_report(output,report)
    try:
        if report['hashes']['material_loss_smooth.py']!=cost.SMOOTH_HASH:
            raise ValueError('frozen scalar math changed')
        if mode!='timing':
            import pytest
            os.environ['GFFX_LOSS_RESOURCE_FIXTURE_DIR']=str(fixtures.resolve())
            os.environ['GFFX_TENSOR_DEADLINE']=str(deadline)
            result=pytest.main(['-q','tests/python/test_material_loss_tensor_physical.py',
                               '-k',mode,'--junitxml='+str(output.with_suffix('.xml')),
                               '--basetemp='+str(output.parent/(output.stem+'-cases'))])
            report['pytest_exit_code']=int(result)
            if result: raise ValueError('physical regression failed or stopped')
        else:
            for name,digest in cost.HASHES.items():
                cost.check_deadline(deadline)
                data=cost.load_json(fixtures/(name+'.json'),digest)
                reference=cost.construct(data)
                tensor=TensorLossResource.from_resource(reference,evidence_id='regression-'+digest)
                envelope=tensor.to_envelope()
                tensor,construction=cost.samples(lambda:TensorLossResource.from_envelope(envelope),deadline)
                row=dict(fixture_sha256=digest,construction=construction,
                         persistent_tensor_bytes=tensor.inspect()['persistent_tensor_bytes'],queries=[])
                report['resources'][name]=row
                for size in cost.PROFILE['batch_sizes']:
                    for backward in (False,True):
                        args=cost.inputs(size,gradients=backward)
                        observed,timing=cost.samples(lambda:query(tensor,args,backward=backward),deadline)
                        original=cost.run_batch(reference,args,backward=backward)
                        if abs(observed['objective']-original['objective'])>1e-11:
                            raise ValueError('scalar/tensor workload value parity')
                        if backward and any(abs(x-y)>1e-11+1e-11*abs(y) for x,y in zip(observed['gradient_sums'],original['gradient_sums'])):
                            raise ValueError('scalar/tensor workload gradient parity')
                        memory=query(tensor,args,backward=backward,account=True)
                        estimate=tensor.workspace(size,backward=backward)
                        if memory['saved_nonconstant_bytes']>estimate or estimate>64*1024**2:
                            raise ValueError('saved graph exceeds declared workspace planning bound')
                        row['queries'].append(dict(batch_size=size,mode='forward_plus_backward' if backward else 'forward_no_grad',
                            **timing,**observed,accounting=memory,workspace_estimate_bytes=estimate))
                        cost.write_report(output,report)
                del reference,tensor,data,envelope
        cost.check_deadline(deadline)
        report['status']='measured; not public admission'
    except Exception as error:
        report.update(status='stopped',error=f'{type(error).__name__}: {error}')
        cost.write_report(output,report)
        raise
    cost.write_report(output,report)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=('timing','directional','mean','furnace'),default='timing')
    parser.add_argument('--worker',action='store_true')
    args=parser.parse_args()
    if args.worker:
        worker(args.fixtures,args.output,args.mode)
        return 0
    return cost.supervise(args.fixtures,args.output,worker_module='tools.material_loss_tensor_probe',worker_args=('--mode',args.mode))


if __name__=='__main__': raise SystemExit(main())
