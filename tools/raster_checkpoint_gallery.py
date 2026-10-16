"""Saved raster41/cohort42 results and prior timings; no renderer invocation."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
 digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
 sources=[]
 def read(soft):
  path=a.root/f'cohort42-render-cpu-512-{soft}-0-1.json';row=json.loads(path.read_text());assert row['status']=='passed' and row['stream']['complete'];sources.append(dict(path=str(path),sha256=digest(path)))
  values={}
  for key,spec in row['arrays'].items():
   path=Path(spec['path']);assert path.stat().st_size==spec['bytes'] and digest(path)==spec['sha256'];sources.append(dict(path=str(path),sha256=spec['sha256']));values[key]=np.frombuffer(path.read_bytes(),dtype='<f8').reshape(spec['shape'])[0]
  return row,values
 hard,h=read('0.0');soft,s=read('0.2');im=Image.new('RGB',(1600,1020),'#edf1f3');d=ImageDraw.Draw(im)
 def text(x,y,v,size=22,fill='#1a3041',bold=False):d.text((x,y),v,font=ImageFont.truetype('C:/Windows/Fonts/'+('seguisb.ttf' if bold else 'segoeui.ttf'),size),fill=fill)
 text(60,32,'GFFX  /  RASTER ACCELERATION  /  DANZIEPC',17,'#087a7c',True)
 text(60,69,'Faster rendering, unchanged pixels',43,bold=True)
 text(60,133,'512 × 512 actual samples • 5,002 vertices • 10,000 triangles • native float64 Principled',22,'#546b7b')
 for x,title,values,row in ((60,'Hard visibility',h,hard),(614,'Soft coverage',s,soft)):
  text(x,192,title,26,bold=True)
  rgb=np.clip(values['colour']+(1-values['mask'][...,None])*np.array([.007,.012,.02]),0,1)
  rgb=np.where(rgb<=.0031308,12.92*rgb,1.055*np.power(rgb,1/2.4)-.055)
  im.paste(Image.fromarray(np.rint(rgb*255).astype('uint8')),(x,238))
  text(x,776,f"Complete forward: {row['measured_seconds']:.3f} s",23,'#087a7c',True)
  text(x,813,f"{row['stream']['plan']['window_count']} sequential windows · {row['stream']['plan']['tile_rows']} rows maximum",19,'#546b7b')
 text(1180,198,'CPU / 512 × 512',24,bold=True)
 for y,title,new,softness in ((255,'Hard visibility',hard,'0.0'),(454,'Soft coverage',soft,'0.2')):
  oldpath=a.root/f'stream40-render-cpu-512-{softness}-0-2.json';old=json.loads(oldpath.read_text());assert old['status']=='passed'
  sources.append(dict(path=str(oldpath),sha256=digest(oldpath)))
  assert all(new['arrays'][k]['sha256']==old['arrays'][k]['sha256'] for k in new['arrays'])
  before,after=old['measured_seconds'],new['measured_seconds']
  text(1180,y,title,22,bold=True)
  text(1180,y+38,f'Original scan  {before:.3f} s',19,'#546b7b')
  d.rounded_rectangle((1180,y+68,1510,y+82),radius=4,fill='#a9bbc7')
  text(1180,y+97,f'Limited acceleration  {after:.3f} s',19,'#087a7c')
  d.rounded_rectangle((1180,y+127,1180+330*after/before,y+141),radius=4,fill='#087a7c')
  text(1180,y+155,f'{before/after:.2f}× observed speedup',22,'#087a7c',True)
 text(1180,688,'Same 64 MiB per window.',20,'#546b7b')
 text(1180,725,'Single complete jobs;',20,'#546b7b')
 text(1180,754,'not sustained throughput.',20,'#546b7b')
 text(1180,802,'Soft internal edges remain.',20,'#546b7b')
 d.line((60,880,1540,880),fill='#bfcbd3',width=1)
 text(60,903,'Display: linear RGB to sRGB over a dark background. Actual native-resolution samples; no retouching.',20,'#546b7b')
 text(60,942,'CPU256 hard/soft forward + backward now pass. Larger gradients, 1080p and final M1 visual acceptance remain open.',20,'#546b7b')
 path=a.output/'plate.png';im.save(path,optimize=True);sha=digest(path);final=a.output/(sha+'.png');path.rename(final)
 item=dict(label='Limited raster acceleration: unchanged512-square pixels',caption='Paired old/new timing bars are single complete CPU evaluations,not sustained-throughput benchmarks. Exact saved image hashes agree. Actual saved512-square CPU hard/soft renders at native display resolution. Complete job times include caller allocation and copies. Each sequential window retains64 MiB;whole CPU job stays under1 GiB. Dark background and sRGB conversion are display only. Soft internal-edge limitations are unchanged. This is engineering evidence,not a final beauty-render or M1 completion claim.',path=str(final),asset=dict(url='/testing/media/'+sha+'.png',sha256=sha,bytes=final.stat().st_size))
 report=dict(status='passed',generator_sha256=digest(__file__),sources=sources,images=[item]);rp=a.output/'report.json';rp.write_text(json.dumps(report,indent=2))
 manifest=dict(version=1,title='Limited raster acceleration: same complete pixels',page='render/scale-demo',scope=item['caption'],images=[item],provenance=dict(producer_report=str(rp),producer_report_sha256=digest(rp),wheel_sha256=hard['wheel']['sha256']))
 mp=a.output/'manifest.json';mp.write_text(json.dumps(manifest,indent=2));print(json.dumps(dict(manifest=str(mp),sha256=digest(mp),image=str(final))))
if __name__=='__main__':main()
