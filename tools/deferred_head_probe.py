"""Bounded DF04 constant-albedo head preflight or one explicit candidate image.

Stored sources are never executed. Shared-native reference buffers are not an
independent raster oracle; independent analytic edge tests remain separate.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from tools import material_loss_cost as cost,shading_cuda_closeout as artifact

REFERENCE_HASHES={'constant':'0efcc0eeeabd16cd7aff46d46bba6085234800d1a702187741b21f8acf5b8de3','vertex':'87bbb1da40c6db735e1dade741da71291dc18c9e8a62efcf4237b13b12cc4207','gradient':'cf333596ecb72431a494279cc0db55832ba041a9a1a43bdc0e091bc590cb7b83','checker':'fc0f92484c812996f307ea0343c55835f09ea0d0a26565c96db5ec6d5828dea3'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker',action='store_true');p.add_argument('--render',action='store_true')
    p.add_argument('--tile-rows',type=int,choices=(64,))
    for name in ('fixtures','output','wheel'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--sha256',required=True);p.add_argument('--material',choices=('classic','principled'),required=True)
    p.add_argument('--albedo',choices=('constant','vertex','gradient','checker'),default='constant')
    p.add_argument('--scale16',type=Path);p.add_argument('--scale32',type=Path)
    p.add_argument('--reference-report',type=Path,default=Path('build/principled-direct-20260908/constant-verified.json'))
    a=p.parse_args()
    if not a.worker:
        if a.output.exists():raise ValueError('preserve prior DF04 outcome')
        extra=['--wheel',str(a.wheel),'--sha256',a.sha256,'--material',a.material,'--albedo',a.albedo]
        if a.render:extra.append('--render')
        if a.tile_rows:extra.extend(('--tile-rows',str(a.tile_rows)))
        for key in ('scale16','scale32','reference_report'):
            if getattr(a,key) is not None:extra.extend(('--'+key.replace('_','-'),str(getattr(a,key))))
        return cost.supervise(a.fixtures,a.output,worker_module='tools.deferred_head_probe',worker_args=extra)
    row=dict(status='working',worker_pid=os.getpid(),mode='render' if a.render else 'preflight',
        material=a.material,profile=dict(width=128,height=128,vertices=1986,faces=3968,
        expected_covered_pixels=6536,light_samples=1,dtype='float64',device='cpu',backward=False,
        albedo=a.albedo,outputs=['colour','mask'],soft=0,supersample=1))
    cost.write_report(a.output,row);deadline=time.perf_counter()+120
    try:
        import torch
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        row['wheel']=artifact.identity(a.wheel,a.sha256)
        row['sources']={name:artifact.digest(name) for name in ('tools/deferred_head_probe.py',
            'tools/deferred_head.py','tools/shading_image_bundle.py','tools/principled_direct_images.py',
            'tools/deferred_scale.py','tools/shading_independent_evaluator.py','tools/deferred_image_partition.py')}
        from tools import corpus_registry,shading_image_bundle,principled_direct_images as reference
        from tools.deferred_head import create_scene,projected_seconds
        from tools.shading_image_reference import compare_images
        from gffx.torch import render
        from gffx.torch._deferred import _plan_workload
        if artifact.digest(reference.scalar.__file__)!=reference.SCALAR_HASH:
            raise ValueError('independent scalar formula identity changed; review required')
        parent=corpus_registry.load_entry(a.fixtures,'synthetic_face_m1',allow_development=True)
        data=json.loads(parent['files']['scene.json']);row['parent_revision']=parent['manifest']['revision']
        row['scene_sha256']=hashlib.sha256(parent['files']['scene.json']).hexdigest()
        with torch.no_grad():
            model,camera=create_scene(data,a.material,albedo=a.albedo)
            if a.tile_rows:
                from tools.deferred_image_partition import plan_partitioned,render_partitioned
                _,plans=plan_partitioned(model,camera,tile_rows=a.tile_rows)
                row['partitions']=plans;row['estimate']=plans[0]['estimate']
                if any(p['estimate']['principled_point_block_limit']!=row['estimate']['principled_point_block_limit'] for p in plans):raise ValueError('different tile material schedule')
            else:row['estimate']=_plan_workload(model,camera,1,67108864,('colour','mask'))
            cost.check_deadline(deadline)
            if a.render:
                if a.scale16 is None or a.scale32 is None:raise ValueError('render requires16/32 calibration reports')
                rows=[json.loads(path.read_text(encoding='utf-8')) for path in (a.scale16,a.scale32)]
                if any(item.get('driver_sha256')!=row['sources']['tools/deferred_scale.py'] for item in rows):
                    raise ValueError('calibration driver source changed')
                row['forecast_seconds']=projected_seconds(*rows,a.sha256,
                    point_block_limit=row['estimate']['principled_point_block_limit'])
                row['calibration_sha256']=[artifact.digest(path) for path in (a.scale16,a.scale32)]
                bundle=shading_image_bundle.load_bundle(a.fixtures,path=reference.MANIFEST,sha256=reference.MANIFEST_HASH)
                row['reference_manifest_sha256']=reference.MANIFEST_HASH
                covered=bundle['buffers']['covered']
                if int(covered.sum())!=6536:raise ValueError('fixed coverage count differs')
                if a.material=='classic':expected=bundle['images']['classic/'+a.albedo]
                else:
                    saved=reference.read_measured(a.reference_report,REFERENCE_HASHES[a.albedo],covered)
                    expected=saved['reference'];row['reference_report_sha256']=REFERENCE_HASHES[a.albedo]
                    if saved['record']['albedo']!=a.albedo:raise ValueError('reference albedo differs')
                cost.write_report(a.output,row);cost.check_deadline(deadline)
                if row['forecast_seconds']>deadline-time.perf_counter():
                    raise ValueError('prospective head time exceeds remaining cooperative allowance')
                start=time.perf_counter()
                if a.tile_rows:result,_=render_partitioned(model,camera,tile_rows=a.tile_rows,checkpoint=lambda:cost.check_deadline(deadline))
                else:result=render(model,camera,outputs=('colour','mask'))
                row['partitioned']=bool(a.tile_rows);row['render_seconds']=time.perf_counter()-start;cost.check_deadline(deadline)
                if not torch.equal(result['mask'][0],covered.to(torch.float64)):
                    raise ValueError('candidate coverage differs from fixed reference')
                row['metrics']=compare_images(result['colour'][0],expected,covered)
                if not row['metrics']['passed']:raise ValueError('DF04 declared-model image comparison failed')
                row['arrays']={}
                for name,value in (('actual',result['colour'][0]),('reference',expected)):
                    blob=shading_image_bundle.tensor_bytes(value);path=a.output.with_suffix('.'+name+'.bin')
                    with path.open('xb') as stream:stream.write(blob)
                    row['arrays'][name]=dict(path=path.name,bytes=len(blob),sha256=hashlib.sha256(blob).hexdigest())
        row['status']='passed'
        row['scope']=('Fixed declared-albedo model comparison with shared-native reference raster; no Cycles, CUDA or full-image backward claim'
                      if a.render else 'Shape/metadata/pre-allocation plan only; no renderer image or reference comparison executed')
    except Exception as error:
        row.update(status='stopped',error=f'{type(error).__name__}: {error}');raise
    finally:cost.write_report(a.output,row)
    return 0


if __name__=='__main__':raise SystemExit(main())
