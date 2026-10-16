"""Saved actual streamed512-square renderer outputs; no renderer invocation."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
 digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
 sources=[]
 def read(soft):
  path=a.root/f'stream40-render-cpu-512-{soft}-0-2.json';row=json.loads(path.read_text());assert row['status']=='passed' and row['stream']['complete'];sources.append(dict(path=str(path),sha256=digest(path)))
  values={}
  for key,spec in row['arrays'].items():
   path=Path(spec['path']);assert path.stat().st_size==spec['bytes'] and digest(path)==spec['sha256'];sources.append(dict(path=str(path),sha256=spec['sha256']));values[key]=np.frombuffer(path.read_bytes(),dtype='<f8').reshape(spec['shape'])[0]
  return row,values
 hard,h=read('0.0');soft,s=read('0.2');im=Image.new('RGB',(1600,1020),'#edf1f3');d=ImageDraw.Draw(im)
 def text(x,y,v,size=22,fill='#1a3041',bold=False):d.text((x,y),v,font=ImageFont.truetype('C:/Windows/Fonts/'+('seguisb.ttf' if bold else 'segoeui.ttf'),size),fill=fill)
 text(60,32,'GFFX  /  STREAMED EXECUTION  /  DANZIEPC',17,'#087a7c',True)
 text(60,69,'Tiles preserve the complete image',43,bold=True)
 text(60,133,'512 × 512 actual samples • 5,002 vertices • 10,000 triangles • native float64 Principled',22,'#546b7b')
 for x,title,values,row in ((60,'Hard visibility',h,hard),(614,'Soft coverage',s,soft)):
  text(x,192,title,26,bold=True)
  rgb=np.clip(values['colour']+(1-values['mask'][...,None])*np.array([.007,.012,.02]),0,1)
  rgb=np.where(rgb<=.0031308,12.92*rgb,1.055*np.power(rgb,1/2.4)-.055)
  im.paste(Image.fromarray(np.rint(rgb*255).astype('uint8')),(x,238))
  text(x,776,f"Complete forward: {row['measured_seconds']:.3f} s",23,'#087a7c',True)
  text(x,813,f"{row['stream']['plan']['window_count']} sequential windows · {row['stream']['plan']['tile_rows']} rows maximum",19,'#546b7b')
 text(1180,198,'What this demonstrates',23,bold=True)
 for i,line in enumerate(('Every pixel is computed.','Original camera coordinates','continue across windows.','','One caller-owned RGB+mask','destination stays on CPU.','Each window keeps 64 MiB.','','Hard and soft use the same','display conversion. Soft edges','are a coverage approximation;','internal triangle edges remain.')):text(1180,249+i*32,line,19,'#546b7b')
 d.line((60,880,1540,880),fill='#bfcbd3',width=1)
 text(60,903,'Display: linear RGB to sRGB over a labelled dark background; no retouching or external beauty render.',20,'#546b7b')
 text(60,942,'Single evaluations, not sustained throughput. Large gradients, 1080p and final M1 visual acceptance remain open.',20,'#546b7b')
 path=a.output/'plate.png';im.save(path,optimize=True);sha=digest(path);final=a.output/(sha+'.png');path.rename(final)
 item=dict(label='Complete512-square streamed head',caption='Actual saved512-square CPU hard/soft renders at native display resolution. Complete job times include caller allocation and copies. Each sequential window retains64 MiB;whole CPU job stays under1 GiB. Dark background and sRGB conversion are display only. Soft internal-edge limitations are unchanged. This is engineering evidence,not a final beauty-render or M1 completion claim.',path=str(final),asset=dict(url='/testing/media/'+sha+'.png',sha256=sha,bytes=final.stat().st_size))
 report=dict(status='passed',generator_sha256=digest(__file__),sources=sources,images=[item]);rp=a.output/'report.json';rp.write_text(json.dumps(report,indent=2))
 manifest=dict(version=1,title='Complete streamed512-square renderer',page='render/scale-demo',scope=item['caption'],images=[item],provenance=dict(producer_report=str(rp),producer_report_sha256=digest(rp),wheel_sha256=hard['wheel']['sha256']))
 mp=a.output/'manifest.json';mp.write_text(json.dumps(manifest,indent=2));print(json.dumps(dict(manifest=str(mp),sha256=digest(mp),image=str(final))))
if __name__=='__main__':main()
