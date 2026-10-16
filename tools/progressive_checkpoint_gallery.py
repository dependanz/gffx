"""Explain saved progressive-admission results; never invoke the renderer."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
 digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
 im=Image.new('RGB',(1760,1050),'#edf1f3');d=ImageDraw.Draw(im);sources=[];images=[]
 def text(x,y,v,size=22,fill='#1a3041',bold=False):d.text((x,y),v,font=ImageFont.truetype('C:/Windows/Fonts/'+('seguisb.ttf' if bold else 'segoeui.ttf'),size),fill=fill)
 text(50,27,'GFFX  /  COMPLETE FRAME EVIDENCE  /  DANZIEPC',18,'#087a7c',True)
 text(50,65,'1080p forward passes. Larger gradients remain.',43,bold=True)
 text(50,130,'Actual 1,920 × 1,080 samples · 5,002 vertices · 10,000 triangles · float64 Principled',23,'#546b7b')
 for x,soft,title in ((50,'0.0','Hard visibility'),(905,'0.2','Soft coverage')):
  path=a.root/f'admission46-resident-cuda-1920x1080-{soft}-0-1.json';r=json.loads(path.read_text());assert r['status']=='passed' and r['render_executed'];sources.append(dict(path=str(path),sha256=digest(path)))
  arrays={}
  for k,v in r['arrays'].items():
   p=Path(v['path']);assert p.stat().st_size==v['bytes'] and digest(p)==v['sha256'];sources.append(dict(path=str(p),sha256=digest(p)));arrays[k]=np.frombuffer(p.read_bytes(),dtype='<f8').reshape(v['shape'])[0]
   assert v['shape']==([1,1080,1920,3] if k=='colour' else [1,1080,1920])
  rgb=arrays['colour']+(1-arrays['mask'][...,None])*np.array([.007,.012,.02]);assert np.isfinite(rgb).all();rgb=np.clip(rgb,0,1);rgb=np.where(rgb<=.0031308,12.92*rgb,1.055*np.power(rgb,1/2.4)-.055)
  view=Image.fromarray(np.rint(rgb*255).astype('uint8'));view=view.resize((800,450),Image.Resampling.LANCZOS)
  text(x,190,title,27,bold=True);im.paste(view,(x,239));text(x,711,f"Measured {r['measured_seconds']:.2f} s   /   Admission forecast {r['forecast_seconds']:.2f} s",23,'#087a7c',True)
  images.append(r)
 text(50,779,'16 / 24 scale profiles complete',30,bold=True)
 text(905,779,'What this establishes',26,bold=True)
 text(50,826,'512² CUDA: forward 4.81 / 4.83 s; + backward 17.46 / 17.52 s.',22,'#546b7b')
 text(50,864,'Hard / soft colour and mask match saved CPU512 bytes exactly.',22,'#546b7b')
 text(905,824,'Complete finite outputs within unchanged worker limits.',22,'#546b7b')
 text(905,861,'1080p backward still refused by whole-frame memory planning.',22,'#546b7b')
 d.line((50,913,1710,913),fill='#bfcbd3',width=1)
 text(50,940,'Display only: linear RGB to sRGB over dark background; full images reduced to 800 × 450. No retouching.',21,'#546b7b')
 text(50,980,'Single-job engineering evidence. Soft internal edges remain. Independent large-image accuracy and final M1 visuals are separate.',21,'#546b7b')
 verify=a.root/'admission45-512-verification1.json';sources.append(dict(path=str(verify),sha256=digest(verify)))
 path=a.output/'plate.png';im.save(path,optimize=True);sha=digest(path);final=a.output/(sha+'.png');path.rename(final)
 caption='Saved complete1080p hard/soft CUDA forward images,displayed at800x450 with explicit sRGB conversion and background. Forecasts use the slower rate of two largest matching complete prior frames,2x margin and30 seconds. Single evaluations are not sustained throughput. CUDA512 hard/soft forward and albedo backward images match saved CPU512 bytes exactly. Large-render matrix16/24 complete;CPU512 backward,CPU1080p forward/backward andCUDA1080p backward remain. Soft internal-edge limitations and all acceptance/resource requirements remain. This engineering plate does not close the final SIGGRAPH-quality visual gate.'
 item=dict(label='Complete1080p forward: measured progress and remaining gates',caption=caption,path=str(final),asset=dict(url='/testing/media/'+sha+'.png',sha256=sha,bytes=final.stat().st_size))
 rp=a.output/'report.json';rp.write_text(json.dumps(dict(status='passed',generator_sha256=digest(__file__),sources=sources,images=[item]),indent=2))
 mp=a.output/'manifest.json';mp.write_text(json.dumps(dict(version=1,title='Complete1080p forward: progress and remaining gates',page='render/scale-demo',scope=caption,images=[item],provenance=dict(producer_report=str(rp),producer_report_sha256=digest(rp),wheel_sha256=images[0]['wheel']['sha256'])),indent=2));print(json.dumps(dict(manifest=str(mp),sha256=digest(mp),image=str(final))))
if __name__=='__main__':main()
