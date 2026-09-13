"""Publish a finite saved full-angle error map; no fresh physics or animation."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from gffx.dashboard import ArrayView, connect
from .reporter import Reporter


def error_map(rows):
    """Rows preserve screen order; each tile is one declared parameter triple."""
    if not 1 <= len(rows) <= 256:
        raise ValueError('error map requires 1..256 finite-screen rows')
    roughnesses = list(dict.fromkeys(row['roughness'] for row in rows))
    counts = [sum(row['roughness']==r for row in rows) for r in roughnesses]
    columns, tile = max(counts), 64
    width, height = columns*tile, ((len(rows)+columns-1)//columns)*tile
    pixels = bytearray([18, 25, 39])*(width*height)
    for index, row in enumerate(rows):
        ratio = min(max(row['relative_error']/.001, 0), 2)
        color = (round(35+100*ratio), round(190-65*ratio), round(150-55*ratio))
        ox, oy = (index%columns)*tile, (index//columns)*tile
        for y in range(oy+3, oy+tile-3):
            start = (y*width+ox+3)*3
            pixels[start:start+3*(tile-6)] = bytes(color)*(tile-6)
    return ArrayView('uint8', (height, width, 3), bytes(pixels))


def publish(address, run, source, *, label):
    source = Path(source)
    names = ('full-angle-holdouts', 'full-angle-mean', 'full-angle-furnace')
    data, blobs = {}, []
    for name in names:
        path = source/(name+'.json')
        if path.stat().st_size > 4*1024*1024:
            raise ValueError('saved diagnostic exceeds 4 MiB')
        blob = path.read_bytes()
        blobs.append(blob)
        data[name] = json.loads(blob)
    rows = data[names[0]]
    picture = error_map(rows)
    roughnesses = list(dict.fromkeys(row['roughness'] for row in rows))
    fresnels = list(dict.fromkeys(row['f0'] for row in rows))
    reporter = Reporter(address, run)
    def send(event, **values):
        if not reporter.send(event, **values):
            raise RuntimeError(reporter.error)
    digest = hashlib.sha256(b''.join(blobs)).hexdigest()
    send('start', kind='demo', provenance=dict(
        revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True)),
        artifact_sha256=digest))
    with connect(address, run=run, wait_acks=True) as dash:
        dash.text('loss-resource/read-me', label+'. Saved diagnostic measurements, not acceptance. '
            'This is an error map, not a rendered material or fresh test execution. '
            f'This saved screen contains {len(rows)} parameter triples. Roughness groups: {roughnesses}; '
            f'Fresnel values: {fresnels}. Tiles follow the saved record order, with '
            'views grouped within each roughness; some views are coupled to roughness squared. '
            'Read the exact records for the cosine of each tile. '
            'Green tiles have little relative loss error; orange/red tiles approach or exceed the 0.1 percent limit. '
            'Inspect the plots and exact records for numerical values. A gradient ratio above one fails its limit. '
            'Exact curve-average consistency does not establish physical accuracy. '
            'This producer publishes once and exits; it does not stream meshes or run in the background.',step=0)
        dash.image('loss-resource/map',picture,step=0)
        for i,row in enumerate(rows):
            dash.scalar('loss-resource/error/relative_percent',row['relative_error']*100,step=i)
            dash.scalar('loss-resource/error/limit_percent',.1,step=i)
            dash.scalar('loss-resource/gradients/error_over_limit',max(row['gradient_ratios']),step=i)
            dash.scalar('loss-resource/gradients/limit',1.,step=i)
        for i,row in enumerate(data['full-angle-mean']):
            dash.scalar('loss-resource/mean/relative_percent',row['relative_error']*100,step=i)
            dash.scalar('loss-resource/mean/limit_percent',.1,step=i)
        for name in names:
            dash.record('loss-resource/'+name,dict(label=label,rows=data[name],
                saved_evidence_sha256=digest,scope='Finite development diagnostic; not public admission.'),step=0)
    artifacts=[dict(kind=kind,path='loss-resource/'+path,label=title) for kind,path,title in (
        ('text','read-me','How to read this experiment'),('image','map','Relative loss error by parameter triple'),
        ('plot','error','Relative loss error and limit, percent'),('plot','gradients','Derivative error divided by its limit'))]
    if data['full-angle-mean']:
        artifacts.append(dict(kind='plot',path='loss-resource/mean',label='Mean physical error, percent'))
    artifacts.extend(dict(kind='record',path='loss-resource/'+name,label=name) for name in names)
    send('demo',page='shading/loss-resource-demo',title=label,artifacts=artifacts)
    send('finish',exit_code=0)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--address',required=True)
    parser.add_argument('--run',required=True)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--label',required=True)
    args=parser.parse_args()
    publish(args.address,args.run,args.source,label=args.label)
