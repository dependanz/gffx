"""Finite partial-image preview from strictly replayed selected albedo tiles."""
import argparse
from pathlib import Path
import torch
from tools.albedo_tiles import replay,combine
from tools.environment_full_images import WHEEL_HASH
from .image_reference_demo import display_rgb

def panels(tiles):
    ref=torch.full((128,128,3),96,dtype=torch.uint8);actual=ref.clone();seen=set()
    for tile in tiles:
        for row in tile['rows']:
            i=row['point']['flat_index']
            if type(i) is not int or not 0<=i<16384 or i in seen:raise ValueError('duplicate/invalid display pixel')
            seen.add(i)
            values=torch.tensor([row['reference']['radiance'],
                [sum(e[k] for e in row['actual']['estimates'])/2 for k in range(3)]],dtype=torch.float64)
            if not bool(torch.isfinite(values).all()) or bool((values<0).any()):raise ValueError('invalid display colour')
            rgb=display_rgb(values/2);ref.reshape(-1,3)[i]=rgb[0];actual.reshape(-1,3)[i]=rgb[1]
    return torch.cat((ref,actual),dim=1)

def publish(result,address,run):
    from gffx.dashboard import ArrayView,connect
    from .reporter import Reporter
    reporter=Reporter(address,run)
    def send(event,**values):
        if not reporter.send(event,**values):raise RuntimeError(reporter.error)
    send('start',kind='demo',provenance=dict(artifact_sha256=WHEEL_HASH));artifacts=[]
    with connect(address,run=run,wait_acks=True) as dash:
        dash.text('albedo-tiles/scope','SELECTED IMAGE CHECKPOINTS, NOT COMPLETE IMAGES. '
            'Each panel shows independent reference LEFT and public gffx output RIGHT. '
            'Only 264 of 6,536 surface pixels per profile are measured: first/middle/short final tiles. '
            'GRAY is uncomputed, including background. Colours use half exposure then sRGB; '
            'error checks use unscaled linear RGB. Thirty-six unique tiles cover 3,168 pixel/profile '
            'comparisons. Error/allowance must be below 1. Each worker is serial and bounded; '
            'sampled whole-process memory includes Python/PyTorch, not just shader allocations. '
            'No full-image, image-gradient, arbitrary HDR-map or M1 acceptance is claimed.',step=0)
        artifacts.append(dict(kind='text',path='albedo-tiles/scope',label='What these partial images mean'))
        groups={}
        for tile in result['tiles']:
            key=(tile['model'],tile['environment'],tile['albedo_name']);groups.setdefault(key,[]).append(tile)
        for key,tiles in groups.items():
            name='-'.join(key);path='albedo-tiles/'+name;pixels=panels(tiles)
            dash.image(path,ArrayView('uint8',tuple(pixels.shape),bytes(pixels.flatten().tolist())),step=0)
            artifacts.append(dict(kind='image',path=path,label=name+' — reference left / public right; gray uncomputed'))
        for i,tile in enumerate(result['tiles']):
            ratio=max(e/t for r in tile['rows'] for errors in r['actual']['errors'] for e,t in zip(errors,r['actual']['limits']))
            dash.scalar('albedo-tiles/error/ratio',ratio,step=i)
            dash.scalar('albedo-tiles/cost/seconds',tile['supervisor']['elapsed_seconds'],step=i)
            dash.scalar('albedo-tiles/memory/MiB',tile['supervisor']['sampled_peak_working_set_bytes']/1024**2,step=i)
        for path,label in [('error','Error / allowance by declared tile order'),('cost','Whole-worker seconds'),('memory','Sampled whole-worker host memory in MiB')]:
            artifacts.append(dict(kind='plot',path='albedo-tiles/'+path,label=label))
        details=[{k:t[k] for k in ('model','environment','albedo_name','index','counts','supervisor')} for t in result['tiles']]
        dash.record('albedo-tiles/evidence',dict(batches=result['batches'],seconds=result['elapsed_seconds'],samples=result['samples'],tiles=details),step=0)
        artifacts.append(dict(kind='record',path='albedo-tiles/evidence',label='Batch hashes, exact tile identities, work and limits'))
    send('demo',page='shading/albedo-tiles-demo',title='Textured environment image checkpoints',artifacts=artifacts)
    send('finish',exit_code=0)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--sha256',action='append',required=True)
    p.add_argument('--corpus',type=Path,required=True);p.add_argument('--address',required=True);p.add_argument('--run',required=True)
    a=p.parse_args()
    if len(a.sha256)!=6:p.error('exactly six ordered batch hashes required')
    result=combine([replay(a.root/f'batch-{i}',sha,a.corpus) for i,sha in enumerate(a.sha256)])
    publish(result,a.address,a.run)

if __name__=='__main__':main()
