"""Strict numerical replay of a complete image or an explicitly partial prefix."""
import hashlib
import json
from pathlib import Path
import torch
from tools import environment_image_runner as runner

SCHEDULES={
    ('classic','uniform'):'99877e6bcb259b07759f070d83e66032456f123343cda2270d1582998a5a7dee',
    ('classic','affine'):'72c06c472fb632e5cea062ab2d8e379efea3b76f48433d07621567ed5ffc1c15',
    ('principled','uniform'):'ae9a846b93005cc91fba7494483922e933d54db7561987a1544279a52ea42d4e',
    ('principled','affine'):'d4768c12b828c8a747845ba11262660a3c26e27d6b84063abf6d37531ec5b085'}


def snapshot(root, model, environment, expected, *, blob=None, require_complete=False):
    key=(model,environment)
    if key not in SCHEDULES: raise ValueError('unknown image profile')
    root=Path(root);directory=root/('whole-'+model+'-'+environment)
    raw=(directory/'cohort.json').read_bytes() if blob is None else blob
    if len(raw)>16*1024**2 or hashlib.sha256(raw).hexdigest()!=expected: raise ValueError('cohort hash/size')
    row=json.loads(raw)
    plan=runner.read_schedule(root/'schedules'/(model+'-'+environment+'.json'),SCHEDULES[key])
    if (row['sources']!=runner.sources() or row['schedule_sha256']!=SCHEDULES[key]
            or (row['model'],row['environment'])!=key or row['declared_counts']!=plan['counts']
            or row['aggregate_seconds']!=6000 or row['status'] not in ('working',runner.PASSED)
            or not 0<row.get('elapsed_seconds',0)<=6000): raise ValueError('cohort identity/status/limits')
    count=len(plan['groups']);complete=row['status']==runner.PASSED
    if (not 0<len(row['workers'])<=2*count or len(row['passes'])>2
            or (complete and (len(row['workers'])!=2*count or len(row['passes'])!=2))
            or (require_complete and not complete)): raise ValueError('incomplete image evidence')
    for ordinal,entry in enumerate(row['workers']):
        iteration,index=divmod(ordinal,count)
        if (entry['iteration'],entry['index'],entry['path'])!=(iteration,index,f'iteration-{iteration}-tile-{index}.json'):
            raise ValueError('worker ownership')
    wheel=Path(__file__).resolve().parents[1]/'build/light-batches-20260908/final-wheels/gffx-0.2.0.dev0-cp310-abi3-win_amd64.whl'
    producer=runner.identity(wheel,runner.old.direct.WHEEL_HASH)
    tex=runner.old.direct.texture();weights,data=runner.inputs(plan['points'])
    values=[torch.zeros((6536,3),dtype=torch.float64) for _ in range(2)]
    validated=[0,0];gradients=[];peak=0;seconds=0.;calls=0;directions=0;last_losses=[]
    for iteration in range(2):
        selected=row['workers'][iteration*count:(iteration+1)*count]
        if not selected:break
        texture_path=directory/f'texture-{iteration}.json';texture_hash=runner.old.direct.digest(texture_path)
        if json.loads(texture_path.read_text())!={'texture':tex.tolist()}:raise ValueError('fresh texture/update identity')
        tiles=[]
        for entry in selected:
            tile=runner.replay_tile(directory/entry['path'],entry['sha256'],plan,entry['index'],texture_hash,tex)
            if tile['schedule_sha256']!=SCHEDULES[key] or tile['producer']!=producer:raise ValueError('worker artifact binding')
            tiles.append(tile)
            group=plan['groups'][entry['index']];a,b=group['start'],group['end']
            values[iteration][a:b]=torch.tensor([r['values'] for r in tile['result']['offsets']],dtype=torch.float64).mean(0)
            validated[iteration]=b
            peak=max(peak,tile['supervisor']['sampled_peak_working_set_bytes'])
            seconds+=tile['supervisor']['elapsed_seconds']
            calls+=tile['result']['counts']['calls'];directions+=tile['result']['counts']['directions']
        if len(tiles)==count:
            offsets=[];average=torch.zeros_like(tex)
            for offset in range(2):
                combined=runner.combine([t['result']['offsets'][offset] for t in tiles],6536)
                checked=runner.old.inspect(tex.tolist(),weights,data,combined['values'].tolist(),combined['gradient'].tolist(),combined['loss'])
                offsets.append(dict(split=(.5,.25)[offset],values=combined['values'].tolist(),gradient=combined['gradient'].tolist(),loss=combined['loss'],**checked))
                average+=combined['gradient']/2
            if iteration<len(row['passes']) and row['passes'][iteration]!=dict(texture_sha256=texture_hash,offsets=offsets):
                raise ValueError('assembled image evidence changed')
            gradients.append(average);last_losses.append([x['loss'] for x in offsets])
            if iteration==0:
                ref=runner.old.ref.evaluate(tex.tolist(),weights,data['intercepts'],data['slopes'],data['target'])
                fd=runner.old.ref.finite_differences(tex.tolist(),weights,data['intercepts'],data['slopes'],data['target'],ref['gradient'])
                if row.get('finite_differences') is not None and (row['finite_differences']!=fd or not all(x['passed'] for x in fd)):
                    raise ValueError('finite differences changed')
                if complete and row.get('finite_differences')!=fd:raise ValueError('missing finite differences')
                tex=(tex-5*average).contiguous()
                if not bool(((tex>=0)&(tex<=1)).all()):raise ValueError('update domain')
    if complete:
        if calls!=plan['counts']['calls'] or directions!=plan['counts']['directions']:
            raise ValueError('complete work accounting')
        if any(b>=a for a,b in zip(*last_losses)):raise ValueError('complete objective did not improve')
    if len(row['passes'])>len(gradients):raise ValueError('claimed unassembled pass')
    return dict(complete=complete,workers=len(row['workers']),total_workers=2*count,
        pixels_per_image=6536,validated_pixels=validated,values=values,gradients=gradients,
        target=torch.tensor(data['target'],dtype=torch.float64),indices=[p['index'] for p in plan['points']],
        cohort_sha256=expected,calls=calls,directions=directions,peak_host_bytes=peak,
        worker_seconds=seconds,losses=last_losses,model=model,environment=environment)
