"""Compare installed public classic shading with an explicit immutable reference.

Finite CPU-only evidence, never a new reference generator or qualification setter.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import torch
from gffx.torch import material, light
from tools.shading_image_reference import compare_images
from .image_reference_demo import display_rgb


@torch.no_grad()
def evaluate(bundle):
    buffers=bundle['buffers']; mask=buffers['covered']
    n,v=buffers['normal'][mask],buffers['view'][mask]
    d=light.directional(torch.tensor([[1.,1.,2.]],dtype=torch.float64),torch.ones((1,3),dtype=torch.float64))
    packet=dict(measure='delta',direction=d['direction'].expand(len(n),1,3),
                irradiance=d['irradiance'].expand(len(n),1,3),valid=torch.ones((len(n),1),dtype=torch.bool))
    result={}
    for model,spec in (('diffuse',0.),('classic',.1)):
        for colour in ('constant','vertex','gradient','checker'):
            name=model+'/'+colour
            brdf=material.classic(n,v,d['direction'],albedo=buffers['albedo/'+colour][mask],
                specular=torch.tensor(spec,dtype=torch.float64),exponent=torch.tensor(2.,dtype=torch.float64))
            actual=torch.zeros((128,128,3),dtype=torch.float64)
            actual[mask]=light.integrate(n,brdf,packet)
            reference=bundle['images'][name]
            result[name]=dict(actual=actual,reference=reference,metrics=compare_images(actual,reference,mask))
    return result


def publish(address,run,result,record):
    from gffx.dashboard import ArrayView,connect
    from .reporter import Reporter
    reporter=Reporter(address,run)
    def send(event,**values):
        if not reporter.send(event,**values):raise RuntimeError(reporter.error)
    send('start',kind='demo',provenance=dict(artifact_sha256=record['candidate_wheel_sha256']))
    artifacts=[]
    with connect(address,run=run,wait_acks=True) as dash:
        explanation=('Public classic material + public light integration versus saved independent scalar references. '
            'Shared geometry buffers isolate shading: this does not independently qualify rasterization. '
            'Each group shows reference, public result, then absolute error divided by the fixed '
            '1e-10 + 1e-8*abs(reference) tolerance (black=zero, white=at least the allowed error). '
            'Reference/result use display-only sRGB and clipping. Original linear metrics are below. '
            'No principled/Cycles/physical-budget/real-model or whole-M1 claim; the head remains a test fixture.')
        dash.text('classic/read-me',explanation,step=0)
        artifacts.append(dict(kind='text',path='classic/read-me',label='How to read the comparison'))
        for name,item in result.items():
            error=(item['actual']-item['reference']).abs()/(1e-10+1e-8*item['reference'].abs())
            for label,pixels in (('reference',display_rgb(item['reference'])),('public',display_rgb(item['actual'])),
                                 ('error-over-tolerance',(error.clamp(0,1)*255).round().byte())):
                path='classic/'+name+'/'+label
                dash.image(path,ArrayView('uint8',(128,128,3),bytes(pixels.flatten().tolist())),step=0)
                artifacts.append(dict(kind='image',path=path,label=name+' — '+label))
        dash.record('classic/evidence',record,step=0)
        artifacts.append(dict(kind='record',path='classic/evidence',label='Measured linear errors and provenance'))
    send('demo',page='shading/classic-material-demo',title='Classic material: public result versus reference',artifacts=artifacts)
    send('finish',exit_code=0)


def main():
    import gffx
    from tools.shading_image_bundle import load_bundle
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--manifest',required=True)
    parser.add_argument('--sha256',required=True);parser.add_argument('--wheel',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--address');parser.add_argument('--run')
    args=parser.parse_args()
    if bool(args.address)!=bool(args.run):parser.error('--address and --run required together')
    if args.report.exists():parser.error('report already exists; choose a new path')
    torch.set_num_threads(1)
    package=Path(gffx.__file__).resolve().parent.parent
    with zipfile.ZipFile(args.wheel) as archive:
        for name in archive.namelist():
            if name.startswith('gffx/') and not name.endswith('/'):
                if archive.read(name)!=(package/name).read_bytes():raise ValueError('installed wheel mismatch: '+name)
    result=evaluate(load_bundle(args.root,path=args.manifest,sha256=args.sha256))
    record=dict(reference=dict(path=args.manifest,sha256=args.sha256),
        candidate_wheel_sha256=hashlib.sha256(args.wheel.read_bytes()).hexdigest(),
        producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        measurements={name:item['metrics'] for name,item in result.items()})
    with args.report.open('x',encoding='utf-8') as handle:json.dump(record,handle,indent=2,allow_nan=False)
    print(json.dumps(record))
    if not all(item['metrics']['passed'] for item in result.values()):raise ValueError('public classic image tolerance failed')
    if args.address:publish(args.address,args.run,result,record)


if __name__=='__main__':main()
