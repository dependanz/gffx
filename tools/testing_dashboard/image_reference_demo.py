"""Finite reference-only contact sheet and optional explicit dashboard publisher.

Reads an explicit hash-verified bundle. No shader substitution or auto service.
"""
import argparse
import json
from pathlib import Path

import torch

from .presentation import png_rgb

LIMITS=('Reference-only synthetic test head: shared gffx geometry and independent scalar lighting. '
        'No public candidate, residual comparison, Cycles match or principled resource admission. '
        '128x128 measurements; the display is enlarged 2x with nearest-neighbour sampling. '
        'Linear RGB is clipped to [0,1] and converted to sRGB for display only; saved float64 evidence is unchanged.')


def display_rgb(linear):
    if not isinstance(linear,torch.Tensor) or linear.device.type!='cpu' or not torch.isfinite(linear).all():
        raise ValueError('finite CPU display input required')
    x=linear.detach().clamp(0,1)
    result=torch.where(x<=.0031308,12.92*x,1.055*x.pow(1/2.4)-.055)
    return (result*255).round().to(torch.uint8)


def presentation(bundle):
    # A deliberately bounded atlas: four colour sources across, four models down.
    from .camera_demo import FONT
    canvas=torch.full((1080,1024,3),245,dtype=torch.uint8)
    panels=[]
    descriptions=dict(albedo='Unlit surface colour: isolates interpolation and UV sampling.',
        diffuse='Independent Lambert reflection under one directional light.',
        classic='Independent diffuse plus the empirical classic highlight; not a conservation claim.',
        environment_diffuse='Analytic Lambert reflection under unit radiance from the whole hemisphere.')
    for row,model in enumerate(('albedo','diffuse','classic','environment_diffuse')):
        for column,colour in enumerate(('constant','vertex','gradient','checker')):
            name=model+'/'+colour
            values=bundle['buffers'][name] if row==0 else bundle['images'][name]
            if values.shape!=(128,128,3):raise ValueError('fixed display profile required')
            number=len(panels)+1;x=column*256;y=row*270
            pixels=display_rgb(values).repeat_interleave(2,0).repeat_interleave(2,1)
            canvas[y:y+256,x:x+256]=pixels
            # Stable panel numbers, not tiny paragraphs baked into the image.
            for i,char in enumerate(str(number)):
                for bit,on in enumerate(FONT[char]):
                    if on=='1':canvas[y+258+2*(bit//3):y+260+2*(bit//3),x+6+i*8+2*(bit%3):x+8+i*8+2*(bit%3)]=30
            panels.append(dict(number=number,name=name,explanation=descriptions[model]+' Source: '+colour+'.'))
    return dict(width=1024,height=1080,png=png_rgb(1024,1080,bytes(canvas.flatten().tolist())),panels=panels,limits=LIMITS)


def publish(address,run,bundle,record):
    from gffx.dashboard import ArrayView,connect
    from .reporter import Reporter
    reporter=Reporter(address,run)
    def send(event,**data):
        if not reporter.send(event,**data):raise RuntimeError(reporter.error)
    send('start',kind='demo',provenance=dict(artifact_sha256=record['sha256']))
    artifacts=[dict(kind='text',path='image-reference/read-me',label='What these images do and do not establish')]
    with connect(address,run=run,wait_acks=True) as dash:
        dash.text('image-reference/read-me',LIMITS,step=0)
        for model in ('albedo','diffuse','classic','environment_diffuse'):
            for colour in ('constant','vertex','gradient','checker'):
                name=model+'/'+colour
                values=bundle['buffers'][name] if model=='albedo' else bundle['images'][name]
                path='image-reference/'+name
                dash.image(path,ArrayView('uint8',(128,128,3),bytes(display_rgb(values).flatten().tolist())),step=0)
                artifacts.append(dict(kind='image',path=path,label=name.replace('/',' — ')))
        dash.record('image-reference/evidence',dict(bundle=record,producer=bundle['manifest']['producer'],details=bundle['manifest']['details']),step=0)
        artifacts.append(dict(kind='record',path='image-reference/evidence',label='Exact reference checks and provenance'))
    send('demo',page='shading/image-reference-demo',title='One head, four colour sources, three lighting answers',artifacts=artifacts)
    send('finish',exit_code=0)


def main():
    from tools.shading_image_bundle import load_bundle
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--manifest',required=True);parser.add_argument('--sha256',required=True)
    parser.add_argument('--png',type=Path);parser.add_argument('--address');parser.add_argument('--run')
    args=parser.parse_args()
    if not args.png and not args.address:parser.error('select --png or --address')
    if bool(args.address)!=bool(args.run):parser.error('--address and --run required together')
    record=dict(path=args.manifest,sha256=args.sha256)
    bundle=load_bundle(args.root,**record)
    if args.png:
        data=presentation(bundle)
        with args.png.open('xb') as handle:handle.write(data['png'])
        print(json.dumps(dict(panels=data['panels'],limits=data['limits'])))
    if args.address:publish(args.address,args.run,bundle,record)


if __name__=='__main__':main()
