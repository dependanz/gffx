"""Finite source-backed image snapshots; grey pixels have not been evaluated."""
import argparse
import hashlib
from pathlib import Path
import torch
from tools import environment_image_evidence as evidence
from .image_reference_demo import display_rgb


def panels(result):
    ids=result['indices'];count=len(ids);done=result['validated_pixels']
    if (not 1<=count<=6536 or len(set(ids))!=count or any(type(i) is not int or not 0<=i<16384 for i in ids)
            or len(done)!=2 or any(type(n) is not int or not 0<=n<=count for n in done)):
        raise ValueError('finite unique image ownership required')
    fields=[result['target'],*result['values']]
    if len(fields)!=3 or any(x.shape!=(count,3) or not bool(torch.isfinite(x).all()) for x in fields):
        raise ValueError('finite RGB fields required')
    result={}
    for name,field,known in zip(('target','before','after'),fields,(count,*done)):
        image=torch.tensor([24,24,28],dtype=torch.uint8).expand(128,128,3).clone()
        image.reshape(-1,3)[ids]=96
        image.reshape(-1,3)[ids[:known]]=display_rgb(field[:known])
        result[name]=image
    return result


def publish(result,address,run):
    from gffx.dashboard import ArrayView,connect
    from .reporter import Reporter
    reporter=Reporter(address,run)
    def send(event,**kw):
        if not reporter.send(event,**kw):raise RuntimeError(reporter.error)
    send('start',kind='demo',provenance=dict(artifact_sha256=evidence.runner.old.direct.WHEEL_HASH))
    try:
        artifacts=[]
        with connect(address,run=run,wait_acks=True) as dash:
            scope=('COMPLETE FIXED PROFILE' if result['complete'] else 'PARTIAL CHECKPOINT, NOT A COMPLETE IMAGE')
            dash.text('environment-image/scope',scope+'. Profile: '+result['model']+' / '+result['environment']+'. '
                'Target uses the independently evaluated lighting model. Before and after show only hash-verified public outputs. '
                'Grey surface pixels have not yet been evaluated; dark pixels are background. Both light-sampling offsets '
                'are checked individually and averaged only for display. One after-image uses a single whole-image texture '
                'update; it is not independently fitted tile by tile. sRGB conversion is display only. Fixed geometry, '
                'source-texture colour only; not CUDA image qualification, all shading, M1, or Cycles fidelity.',step=0)
            artifacts.append(dict(kind='text',path='environment-image/scope',label='What is measured and what is still missing'))
            for name,image in panels(result).items():
                image=image.repeat_interleave(3,0).repeat_interleave(3,1)
                path='environment-image/'+name
                dash.image(path,ArrayView('uint8',tuple(image.shape),bytes(image.flatten().tolist())),step=0)
                artifacts.append(dict(kind='image',path=path,label=name.capitalize()+': grey means not evaluated'))
            record={k:result[k] for k in ('model','environment','complete','workers','total_workers','pixels_per_image',
                'validated_pixels','cohort_sha256','calls','directions','peak_host_bytes','worker_seconds','losses')}
            dash.record('environment-image/evidence',record,step=0)
            artifacts.append(dict(kind='record',path='environment-image/evidence',label='Verified pixel counts, measurements and exact saved evidence'))
            for iteration,losses in enumerate(result['losses']):
                for split,loss in zip((.5,.25),losses):dash.scalar('environment-image/loss/offset-'+str(split),loss,step=iteration)
            if result['losses']:artifacts.append(dict(kind='plot',path='environment-image/loss',label='Complete image loss only; no partial-tile average'))
        send('demo',page='shading/environment-image-adjoints-demo',title='Environment image fitting: '+result['model']+' / '+result['environment'],artifacts=artifacts)
        send('finish',exit_code=0)
    except Exception as error:
        reporter.send('diagnostic',message=str(error));reporter.send('finish',exit_code=1);raise


def publish_gallery(results,address,run):
    """Display the four strict complete snapshots; never upgrade a prefix."""
    keys=[(r['model'],r['environment']) for r in results]
    if (len(keys)!=4 or set(keys)!=set(evidence.SCHEDULES)
            or any(not r['complete'] or r['workers']!=r['total_workers']
                   or r['validated_pixels']!=[r['pixels_per_image']]*2
                   or len(r['losses'])!=2 for r in results)):
        raise ValueError('four complete distinct image profiles required')
    prepared=[panels(r) for r in results]  # Validate before starting an external run.
    from gffx.dashboard import ArrayView,connect
    from .reporter import Reporter
    reporter=Reporter(address,run)
    def send(event,**kw):
        if not reporter.send(event,**kw):raise RuntimeError(reporter.error)
    send('start',kind='demo',provenance=dict(artifact_sha256=evidence.runner.old.direct.WHEEL_HASH))
    try:
        artifacts=[]
        with connect(address,run=run,wait_acks=True) as dash:
            dash.text('environment-gallery/scope',
                'FOUR COMPLETE FIXED CPU PROFILES: classic and principled, each under uniform and coloured affine world lighting. '
                'Each comparison shows the independent target, actual public output before fitting, and actual public output after '
                'one shared source-texture update. Both sampling offsets pass individually and are averaged only for display. '
                'The loss is the full covered-image mean, not a mean of tile means. Fixed geometry/UVs/material/light parameters; '
                'only source-texture colour is fitted. Images are 128x128 measurements enlarged for readability, not higher-resolution '
                'renders. This is not whole-image CUDA, arbitrary HDR lighting, optimizer convergence, Cycles fidelity or M1 acceptance.',step=0)
            artifacts.append(dict(kind='text',path='environment-gallery/scope',label='Read this first: what these four comparisons prove'))
            for result,images in zip(results,prepared):
                name=result['model']+'-'+result['environment'];prefix='environment-gallery/'+name
                for label,image in images.items():
                    image=image.repeat_interleave(3,0).repeat_interleave(3,1)
                    path=prefix+'/'+label
                    dash.image(path,ArrayView('uint8',tuple(image.shape),bytes(image.flatten().tolist())),step=0)
                    artifacts.append(dict(kind='image',path=path,label=name+' / '+label))
                record={k:result[k] for k in ('cohort_sha256','workers','total_workers','pixels_per_image',
                    'validated_pixels','calls','directions','peak_host_bytes','worker_seconds','losses')}
                dash.record(prefix+'/evidence',record,step=0)
                artifacts.append(dict(kind='record',path=prefix+'/evidence',label=name+' / exact evidence and limits'))
                for iteration,losses in enumerate(result['losses']):
                    for split,loss in zip((.5,.25),losses):dash.scalar(prefix+'/loss/offset-'+str(split),loss,step=iteration)
                artifacts.append(dict(kind='plot',path=prefix+'/loss',label=name+' / complete image error before and after'))
        send('demo',page='shading/environment-image-adjoints-demo',title='Four complete environment-image fitting comparisons',artifacts=artifacts)
        send('finish',exit_code=0)
    except Exception as error:
        reporter.send('diagnostic',message=str(error));reporter.send('finish',exit_code=1);raise


def gallery(root,address,run):
    """Replay saved evidence only. No new material evaluation or test pass claims."""
    root=Path(root);torch.set_num_threads(1)
    results=[]
    for model,environment in evidence.SCHEDULES:
        raw=(root/('whole-'+model+'-'+environment)/'cohort.json').read_bytes()
        digest=hashlib.sha256(raw).hexdigest()
        results.append(evidence.snapshot(root,model,environment,digest,blob=raw,require_complete=True))
    publish_gallery(results,address,run)
    return [{k:r[k] for k in ('model','environment','cohort_sha256','workers','losses')} for r in results]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--snapshot',type=Path,required=True)
    for key in ('model','environment','address','run'):parser.add_argument('--'+key,required=True)
    args=parser.parse_args();torch.set_num_threads(1)
    raw=(args.root/('whole-'+args.model+'-'+args.environment)/'cohort.json').read_bytes()
    digest=hashlib.sha256(raw).hexdigest()
    result=evidence.snapshot(args.root,args.model,args.environment,digest,blob=raw)
    # Preserve the exact partial input even as the live coordinator advances.
    args.snapshot.parent.mkdir(parents=True,exist_ok=True)
    with args.snapshot.open('xb') as stream:stream.write(raw)
    publish(result,args.address,args.run)
    print(dict(snapshot=str(args.snapshot),sha256=digest,workers=result['workers'],complete=result['complete']))


if __name__=='__main__':main()
