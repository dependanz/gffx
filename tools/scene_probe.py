"""Bounded scene qualification; preserve each report and exact wheel identity."""
import argparse
import os
from pathlib import Path
import time
from tools import material_loss_cost as cost, shading_cuda_closeout as artifact


def test_selection(mode):
    if mode=='unit':return ['tests/python/test_scene.py','tests/python/test_scene_probe.py']
    if mode=='presentation':return ['tests/python/test_scene_presentation.py','tests/python/test_textbook.py','tests/python/test_learning_pages.py','tests/python/test_base_import.py']
    names={'integration':('raster_matches_manual_and_frames','image_gradient_and_fresh_descent','parent_and_vertex_gradcheck'),
           'cuda':('inherited_packing_and_gradients',),'budget':('synthetic_flame_scale',)}
    return ['tests/python/test_scene_integration.py::test_scene_'+mode+'_'+name for name in names[mode]]


def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true')
    for name in ('fixtures','output','wheel'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--sha256',required=True);p.add_argument('--mode',choices=('unit','integration','cuda','budget','presentation'),default='unit')
    a=p.parse_args()
    if not a.worker:
        if a.mode=='cuda':cost.PROFILE['sampled_working_set_limit_bytes']=2*1024**3
        return cost.supervise(a.fixtures,a.output,worker_module='tools.scene_probe',worker_args=(
            '--wheel',str(a.wheel),'--sha256',a.sha256,'--mode',a.mode))
    row=dict(status='working',worker_pid=os.getpid(),mode=a.mode);cost.write_report(a.output,row)
    deadline=time.perf_counter()+120
    try:
        import torch
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        row['wheel']=artifact.identity(a.wheel,a.sha256)
        if a.mode=='cuda':
            if not torch.cuda.is_available():raise ValueError('CUDA required')
            torch.cuda.reset_peak_memory_stats()
        import pytest
        selection=test_selection(a.mode);paths=sorted({node.split('::')[0] for node in selection})
        row['tests']={path:artifact.digest(path) for path in paths}
        code=pytest.main(['-q','-x','-p','tools.testing_dashboard.pytest_plugin',*selection,'--junitxml='+str(a.output.with_suffix('.xml'))])
        row['pytest_exit_code']=int(code)
        if a.mode=='cuda':
            row['gpu_peak_bytes']=torch.cuda.max_memory_allocated()
            if row['gpu_peak_bytes']>64*1024**2:raise MemoryError('GPU cap')
        cost.check_deadline(deadline)
        if code:raise ValueError('scene qualification failed')
        row['status']='passed'
    except Exception as error:
        row.update(status='stopped',error=f'{type(error).__name__}: {error}');raise
    finally:cost.write_report(a.output,row)
    return 0


if __name__=='__main__':raise SystemExit(main())
