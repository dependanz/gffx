"""Finite progress from strictly replayed original and newly computed tiles."""
import argparse
from pathlib import Path
import torch
from tools import albedo_images as images
from tools import albedo_tiles as selected
from tools.albedo_sampling import PARENT_HASHES
from tools.environment_full_images import WHEEL_HASH
from .image_reference_demo import display_rgb

def panels(tiles):
    ref=torch.full((128,128,3),96,dtype=torch.uint8);actual=ref.clone();seen=set()
    for tile in tiles:
        for offset,row in enumerate(tile['rows']):
            i=row['point']['flat_index']
            if type(i) is not int or not 0<=i<16384 or i in seen:raise ValueError('duplicate/invalid display pixel')
            seen.add(i)
            estimates=tile['candidate']['estimates'][offset] if 'candidate' in tile else row['actual']['estimates']
            values=torch.tensor([row['reference']['radiance'],[sum(e[k] for e in estimates)/2 for k in range(3)]],dtype=torch.float64)
            if values.shape!=(2,3) or not bool(torch.isfinite(values).all()) or bool((values<0).any()):raise ValueError('invalid display colour')
            rgb=display_rgb(values/2);ref.reshape(-1,3)[i]=rgb[0];actual.reshape(-1,3)[i]=rgb[1]
    return torch.cat((ref,actual),dim=1)

def complete_panels(image):
    return torch.cat([display_rgb(image[name]/2) for name in ('reference','actual')],dim=1)

def load(root,parent,corpus,hashes):
    if not 1<=len(hashes)<=98:raise ValueError('ordered continuation batch hashes required')
    original=selected.combine([selected.replay(parent/f'batch-{i}',sha,corpus) for i,sha in enumerate(PARENT_HASHES)])
    tiles=list(original['tiles']);batches=list(original['batches']);seconds=original['elapsed_seconds'];samples=original['samples']
    for i,sha in enumerate(hashes):
        result=images.replay(root/f'batch-{i}',sha,corpus)
        if result['record']['batch']!=i:raise ValueError('continuation batch ownership')
        tiles.extend(result['tiles']);batches.append(dict(continuation_batch=i,sha256=sha))
        seconds+=result['record']['elapsed_seconds']
        samples+=sum(t['counts'][k] for t in result['tiles'] for k in ('reference_directions','candidate_directions','moment_samples'))
    if len({images.identity(t) for t in tiles})!=len(tiles):raise ValueError('duplicate tile ownership')
    return dict(tiles=tiles,batches=batches,seconds=seconds,samples=samples)

def publish(result,address,run):
    from gffx.dashboard import ArrayView,connect
    from .reporter import Reporter
    reporter=Reporter(address,run)
    def send(event,**values):
        if not reporter.send(event,**values):raise RuntimeError(reporter.error)
    send('start',kind='demo',provenance=dict(artifact_sha256=WHEEL_HASH));artifacts=[]
    tiles=result['tiles'];pixels=sum(len(t['rows']) for t in tiles)
    with connect(address,run=run,wait_acks=True) as dash:
        dash.text('albedo-images/scope',f'CHECKPOINT PROGRESS: {len(tiles)} / 624 tiles, {pixels} / 78,432 pixel/profile comparisons. '
            'Includes 36 original checkpoints; new pixels have freshly converged independent references. '
            'REFERENCE LEFT, PUBLIC GFFX RIGHT; GRAY IS UNCOMPUTED, including background. '
            'Half exposure then sRGB is display only; acceptance uses unscaled linear RGB. '
            'These are finite snapshots. Complete profiles are identified separately below; partial panels alone are not full-image acceptance. '
            'No image-gradient, arbitrary HDR or M1 acceptance is implied.',step=0)
        artifacts.append(dict(kind='text',path='albedo-images/scope',label='What is complete and what is still uncomputed'))
        for number,complete in enumerate(result.get('complete',[])):
            name='-'.join(complete['record']['profile']);image=complete_panels(complete['image'])
            path=f'albedo-images/complete-{number}'
            dash.image(path,ArrayView('uint8',tuple(image.shape),bytes(image.flatten().tolist())),step=0)
            artifacts.append(dict(kind='image',path=path,label=f'COMPLETE {name}: all6,536 surface pixels; reference left / public right, exact black background'))
            dash.record(path+'-evidence',dict(sha256=complete['sha256'],profile=complete['record']['profile'],
                metrics=complete['image']['metrics'],counts=complete['image']['counts'],arrays=complete['record']['arrays']),step=0)
            artifacts.append(dict(kind='record',path=path+'-evidence',label='Complete-profile error metrics and byte-verified image identities'))
        groups={}
        for t in tiles:groups.setdefault((t['model'],t['environment'],t['albedo_name']),[]).append(t)
        completed={tuple(c['record']['profile']) for c in result.get('complete',[])}
        for key,group in groups.items():
            if key in completed:continue
            name='-'.join(key);path='albedo-images/'+name;image=panels(group)
            dash.image(path,ArrayView('uint8',tuple(image.shape),bytes(image.flatten().tolist())),step=0)
            artifacts.append(dict(kind='image',path=path,label=f'{name}: {sum(len(t["rows"]) for t in group)} / 6,536 pixels; reference left / public right'))
        for i,t in enumerate(tiles):
            ratios=[]
            for j,row in enumerate(t['rows']):
                estimates=t['candidate']['estimates'][j] if 'candidate' in t else row['actual']['estimates']
                ratios.extend(abs(a-r)/(1e-10+1e-8*abs(r)) for e in estimates for a,r in zip(e,row['reference']['radiance']))
            dash.scalar('albedo-images/error/ratio',max(ratios),step=i)
            dash.scalar('albedo-images/cost/seconds',t['supervisor']['elapsed_seconds'],step=i)
            dash.scalar('albedo-images/memory/MiB',t['supervisor']['sampled_peak_working_set_bytes']/1024**2,step=i)
        for name,label in [('error','Error / allowance by checkpoint order'),('cost','Whole-worker seconds'),('memory','Sampled whole-worker host memory in MiB')]:
            artifacts.append(dict(kind='plot',path='albedo-images/'+name,label=label))
        dash.record('albedo-images/evidence',dict(batches=result['batches'],seconds=result['seconds'],samples=result['samples'],
            tiles=[{k:t[k] for k in ('model','environment','albedo_name','index','counts','supervisor')} for t in tiles]),step=0)
        artifacts.append(dict(kind='record',path='albedo-images/evidence',label='Exact checkpoint identities and cumulative work'))
    send('demo',page='shading/albedo-images-demo',title='New textured-image pixels',artifacts=artifacts);send('finish',exit_code=0)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','parent','corpus'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--sha256',action='append',required=True)
    for name in ('address','run'):p.add_argument('--'+name,required=True)
    a=p.parse_args();publish(load(a.root,a.parent,a.corpus,a.sha256),a.address,a.run)

if __name__=='__main__':main()
