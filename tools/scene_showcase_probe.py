"""Supervised finite scene presentation; source and wheel identity recorded."""
import argparse
import os
from pathlib import Path
import time
from tools import material_loss_cost as cost, shading_cuda_closeout as artifact


def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true')
    for name in ('fixtures','output','wheel'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--sha256',required=True)
    p.add_argument('--mode',choices=('tests','render','publish'),default='tests')
    p.add_argument('--address',default='http://127.0.0.1:8791')
    p.add_argument('--run',default='scene-atelier-20260909')
    a=p.parse_args()
    if not a.worker:
        return cost.supervise(a.fixtures,a.output,worker_module='tools.scene_showcase_probe',worker_args=(
            '--wheel',str(a.wheel),'--sha256',a.sha256,'--mode',a.mode,'--address',a.address,'--run',a.run))
    row=dict(status='working',worker_pid=os.getpid(),mode=a.mode);cost.write_report(a.output,row)
    deadline=time.perf_counter()+120
    try:
        import torch
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        row['wheel']=artifact.identity(a.wheel,a.sha256)
        row['sources']={path:artifact.digest(path) for path in (
            'tools/testing_dashboard/scene_showcase.py','tools/testing_dashboard/cinematic.py',
            'tools/scene_showcase_probe.py','tests/python/test_scene_showcase.py')}
        if a.mode=='tests':
            import pytest
            code=pytest.main(['-q','-x','-p','tools.testing_dashboard.pytest_plugin',
                'tests/python/test_scene_showcase.py','tests/python/test_scene_presentation.py',
                '--junitxml='+str(a.output.with_suffix('.xml'))])
            row['pytest_exit_code']=int(code)
            if code:raise ValueError('scene showcase tests failed')
        else:
            from tools.testing_dashboard import scene_showcase as demo
            from tools.testing_dashboard.cinematic import encoded
            from tools.testing_dashboard.presentation import AssetWriter
            data=demo.payload(lambda:cost.check_deadline(deadline))
            writer=AssetWriter(a.output.parent/'images')
            row['images']=[writer.put(encoded(im)) for im in data['images']]
            row['mesh_frames']=sum(map(len,data['clips']))
            row['vertices']=len(data['hero']['vertices']);row['faces']=len(data['hero']['faces'])
            row['logical_clip_bytes']=sum(r[k].numel()*r[k].element_size()
                for clip in data['clips'] for r in clip for k in ('vertices','faces','normals','colors'))
            if a.mode=='publish':
                demo.publish(a.address,a.run,data);row['run']=a.run
        cost.check_deadline(deadline);row['status']='passed'
    except Exception as error:
        row.update(status='stopped',error=f'{type(error).__name__}: {error}');raise
    finally:cost.write_report(a.output,row)
    return 0


if __name__=='__main__':raise SystemExit(main())
