"""Create finite scientific presentation boards from verified saved renderer arrays.

Standard library only. Does not render, fit, import a framework or change evidence.
"""
import argparse,ast,hashlib,json,math,struct,time
from pathlib import Path
from tools.testing_dashboard.presentation import png_rgb

WHEEL='fedde8401078f7db27f8a1ceef24f30517577cbe231a2d397a5c48c3e61d6fa9'
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    a.output.mkdir();started=time.perf_counter()
    font_source=Path('tools/testing_dashboard/image_showcase.py')
    font=next(ast.literal_eval(n.value) for n in ast.parse(font_source.read_text(encoding='utf-8')).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_FONT' for t in n.targets))
    font.update({'5':'11111 10000 10000 11110 00001 00001 11110','6':'01110 10000 10000 11110 10001 10001 01110','7':'11111 00001 00010 00100 01000 01000 01000','8':'01110 10001 10001 01110 10001 10001 01110','9':'01110 10001 10001 01111 00001 00001 01110','.':'00000 00000 00000 00000 00000 00110 00110'})
    report=dict(status='working',generator_sha256=digest(__file__),font_sha256=digest(font_source),producer_wheel_sha256=WHEEL,producers=[],images=[])
    cohort=a.root/'classic17-head-cohort1.json';assert json.loads(cohort.read_text())['status']=='passed';report['cohort_sha256']=digest(cohort)
    for material in ('classic','principled'):
        for albedo in ('constant','vertex','gradient','checker'):
            path=a.root/f'classic17-head-{material}-{albedo}-cpu1.json';r=json.loads(path.read_text());assert r['status']=='passed' and r['wheel']['sha256']==WHEEL
            arrays={}
            for key in ('actual','reference'):
                descriptor=r['arrays'][key];file=(path.parent/descriptor['path']).resolve();assert file.parent==a.root.resolve();raw=file.read_bytes();assert len(raw)==descriptor['bytes'] and digest(file)==descriptor['sha256'];arrays[key]=struct.unpack('<49152d',raw)
            width,height=1160,470;board=bytearray(bytes([13,20,32])*width*height)
            def pixel(x,y,colour):
                assert 0<=x<width and 0<=y<height
                i=(y*width+x)*3;board[i:i+3]=bytes(colour)
            def text(x,y,value,scale=2):
                value=value.upper();assert all(c==' ' or c in font for c in value),value
                for i,c in enumerate(value):
                    if c==' ':continue
                    for yy,line in enumerate(font[c].split()):
                        for xx,on in enumerate(line):
                            if on=='1':
                                for dy in range(scale):
                                    for dx in range(scale):pixel(x+(i*6+xx)*scale+dx,y+yy*scale+dy,[219,230,241])
            def srgb(v):
                v=max(0.,min(1.,v));return round(255*(12.92*v if v<=.0031308 else 1.055*v**(1/2.4)-.055))
            text(38,24,material+' / '+albedo+' albedo',4)
            text(38,68,'Fixed 128 x 128 head / two 64-row calls / exact CUDA colour and mask',2)
            for offset,label,kind in ((38,'SCALAR REFERENCE','reference'),(430,'GFFX OUTPUT','actual'),(822,'ABS ERROR X20','error')):
                text(offset,108,label,3)
                for y in range(128):
                    for x in range(128):
                        idx=(y*128+x)*3
                        if kind=='error':
                            e=min(1.,20*max(abs(arrays['actual'][idx+k]-arrays['reference'][idx+k]) for k in range(3)));rgb=[round(255*e),round(255*.25*e),round(255*.65*e)]
                        else:rgb=[srgb(v) for v in arrays[kind][idx:idx+3]]
                        for dy in range(2):
                            for dx in range(2):pixel(offset+x*2+dx,146+y*2+dy,rgb)
            text(38,420,'Maximum linear error '+format(r['metrics']['max_absolute_error'],'.5g'),2)
            text(38,446,'Material reference shares geometry. Independent silhouette gate is separate.',2)
            blob=png_rgb(width,height,bytes(board));sha=hashlib.sha256(blob).hexdigest();file=a.output/(sha+'.png');file.open('xb').write(blob)
            entry=dict(label=material.title()+' / '+albedo,caption='Actual PC Classic17 renderer output versus the frozen independent scalar material reference. Linear max-channel absolute error uses the fixed x20 display scale; images alone use sRGB and nearest enlargement. Maximum linear error '+format(r['metrics']['max_absolute_error'],'.17g')+'. Both CUDA halves have exact output bytes. Shared geometry is separate from the qualified full-mask Decimal oracle.',path=str(file),asset=dict(url='/testing/media/'+sha+'.png',sha256=sha,bytes=len(blob)))
            report['images'].append(entry);report['producers'].append(dict(path=str(path),sha256=digest(path),arrays=r['arrays']))
    recovery=a.root/'deferred-recovery1/report.json';r=json.loads(recovery.read_text());worker=a.root/'classic17-recovery1.json';q=json.loads(worker.read_text());assert r['status']==q['status']=='passed' and q['wheel']['sha256']==WHEEL
    assert digest(r['asset_path'])==r['asset']['sha256']
    report['images'].append(dict(label='Recover RGB albedo',caption=r['scope']+' MSE '+str(r['initial_mse'])+' to '+str(r['final_mse'])+'; maximum recovered parameter error '+str(r['max_parameter_error'])+'. The four-decimal display table is rounded, not an exact-parameter claim.',path=r['asset_path'],asset=r['asset']))
    report['producers'].extend([dict(path=str(recovery),sha256=digest(recovery)),dict(path=str(worker),sha256=digest(worker))]);report.update(status='passed',elapsed_seconds=time.perf_counter()-started,scope='Nine finite scientific boards from verified existing PC results; no renderer or optimizer rerun. Original failures, binary arrays and numerical criteria remain unchanged.')
    output=a.output/'report.json';output.open('x',encoding='utf-8').write(json.dumps(report,indent=2))
    manifest=dict(version=1,title='Scenes, reference images and material recovery',page='render/deferred-demo',images=report['images'],scope=report['scope'],provenance=dict(producer_report=str(output),producer_report_sha256=digest(output),wheel_sha256=WHEEL))
    file=a.output/'manifest.json';file.open('x',encoding='utf-8').write(json.dumps(manifest,indent=2));print(json.dumps(dict(manifest=str(file),sha256=digest(file),images=len(report['images']),elapsed_seconds=report['elapsed_seconds'])))
if __name__=='__main__':main()
