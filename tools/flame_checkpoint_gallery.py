"""Publication-style evidence plates from saved renderer arrays; no rendering or fitting."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
 sources=[];images=[];W,H=1600,960
 def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
 def read(name):
  path=a.root/name;sources.append(dict(path=str(path),sha256=digest(path)));return json.loads(path.read_text())
 def array(row,key):
  spec=row['arrays'][key];path=Path(spec['path']);blob=path.read_bytes();assert len(blob)==spec['bytes'] and digest(path)==spec['sha256'];sources.append(dict(path=str(path),sha256=spec['sha256']))
  return np.frombuffer(blob,dtype='<f8').reshape(spec['shape'])[0]
 def font(size,bold=False):return ImageFont.truetype('C:/Windows/Fonts/'+('seguisb.ttf' if bold else 'segoeui.ttf'),size)
 ink='#102435';muted='#536b7b';teal='#007f83';orange='#b5572c'
 def canvas():
  im=Image.new('RGB',(W,H),'#f6f8fa');return im,ImageDraw.Draw(im)
 def text(d,x,y,value,size=22,color=ink,bold=False):d.text((x,y),value,font=font(size,bold),fill=color)
 def heading(d,kicker,title,subtitle):
  text(d,64,36,kicker,17,teal,True);text(d,64,71,title,42,ink,True);text(d,64,131,subtitle,22,muted)
 def save(im,label,caption):
  path=a.output/(str(len(images))+'.png');im.save(path,optimize=True);sha=digest(path);final=a.output/(sha+'.png');path.rename(final)
  images.append(dict(label=label,caption=caption,path=str(final),asset=dict(url='/testing/media/'+sha+'.png',sha256=sha,bytes=final.stat().st_size)))
 hard=read('flame31-cuda-hard-f-128-1.json');soft=read('flame31-cuda-soft-f-128-1.json');back=read('flame31-cuda-hard-fb-128-1.json')
 assert all(r['status']=='passed' and r['profile']['vertices']==5002 and r['profile']['width']==128 for r in (hard,soft,back))
 hc,hm,sc,sm=array(hard,'colour'),array(hard,'mask'),array(soft,'colour'),array(soft,'mask')
 def display(c,m):
  rgb=np.clip(c+(1-m[...,None])*np.array([.008,.013,.021]),0,1)
  rgb=np.where(rgb<=.0031308,12.92*rgb,1.055*np.power(rgb,1/2.4)-.055)
  return Image.fromarray(np.rint(rgb*255).astype('uint8')).resize((456,456),Image.Resampling.LANCZOS)
 im,d=canvas();heading(d,'GFFX / NATIVE PRINCIPLED / DANZIEPC','A valid scale head, through the complete renderer','5,002 vertices · 10,000 nondegenerate triangles · linear float64 · one directional light')
 for x,title,tile in ((64,'Hard visibility',display(hc,hm)),(572,'Soft coverage',display(sc,sm)),(1080,'Stored soft mask',Image.fromarray(np.rint(np.clip(sm,0,1)*255).astype('uint8')).convert('RGB').resize((456,456),Image.Resampling.NEAREST))):
  text(d,x,197,title,25,ink,True);im.paste(tile,(x,241))
 text(d,64,728,'128 × 128 rendered samples, enlarged for inspection',25,ink,True)
 text(d,64,770,'The first two panels apply the same sRGB display conversion and dark background.',21,muted)
 text(d,64,801,'The mask is linear grayscale. Soft coverage is a visibility model, not physical transparency.',21,muted)
 d.line((64,852,1536,852),fill='#cbd5dc',width=1)
 text(d,64,878,f"Hard forward {hard['measured_seconds']:.3f} s    |    Forward + albedo VJP {back['measured_seconds']:.3f} s",21,teal,True)
 text(d,64,916,'Single evaluations · synthetic procedural geometry · larger profiles and final M1 showcase remain open',18,muted)
 save(im,'Complete native scale-head render','Actual saved128×128 public-renderer outputs on a5002-vertex/10000-face synthetic head. Enlarged presentation is not a128-to-512 resolution claim. Hard/soft panels use identical display conversion; mask is raw linear grayscale. Timings are single evaluations; backward fits only constant RGB albedo. This evidence plate is not the final SIGGRAPH-quality demonstration or an M1 closeout.')
 before=[read('flame29-'+device+'-soft-f-64-1.json') for device in ('cpu','cuda')];after=[read('flame31-'+device+'-soft-f-64-1.json') for device in ('cpu','cuda')]
 differences=[np.abs(array(pair[0],'mask')-array(pair[1],'mask')) for pair in (before,after)]
 assert differences[0].max()>0 and differences[1].max()==0
 limit=float(differences[0].max());im,d=canvas();heading(d,'NUMERICAL EVIDENCE / PRESERVED FAILURES','A tiny error still deserves an exact explanation','Absolute CPU–CUDA soft-mask difference · the same fixed 64 × 64 scale-head scene')
 for x,title,delta in ((64,'Before: retained failing calculation',differences[0]),(600,'After: fixed operation order',differences[1])):
  text(d,x,200,title,24,ink,True)
  u=np.clip(delta/limit,0,1)[...,None];rgb=np.array([15,34,52])*(1-u)+np.array([243,160,72])*u
  tile=Image.fromarray(np.rint(rgb).astype('uint8')).resize((456,456),Image.Resampling.NEAREST);im.paste(tile,(x,246))
  text(d,x,723,f'Maximum absolute difference: {float(delta.max()):.3e}',20,teal if x>64 else orange,True)
 text(d,1130,222,'What changed',24,ink,True)
 lines=['Identical raster buffers','were kept unchanged.','','An explicit tensor divisor','and shared logistic','operation order restore','exact mask bytes.','','All original failed reports','remain in the dashboard.']
 for i,line in enumerate(lines):text(d,1130,270+i*32,line,20,muted)
 text(d,64,795,f'One fixed heat-map scale: dark = 0, amber = {limit:.3e}. Nearest-neighbour enlargement.',21,muted)
 text(d,64,838,'Independent accuracy checks and exact device comparisons answer different questions.',22,ink,True)
 text(d,64,882,'Hard and soft full-colour/mask arrays now match at 32 × 32 and 64 × 64.',21,teal,True)
 text(d,64,922,'A repaired small image does not qualify larger memory, runtime or complete fitting profiles.',18,muted)
 save(im,'Preserved failure and exact repair','Saved old and repaired64×64 CPU/CUDA masks, with one fixed absolute-error scale and nearest-neighbour display enlargement. The old nonzero differences are preserved evidence. New colour and mask byte identity is established separately on32/64 hard and soft profiles; large-frame gradients and resource limits remain separate gates.')
 report=dict(status='passed',generator_sha256=digest(__file__),sources=sources,images=images,scope='Saved renderer evidence plates only; no numerical rerun.128-square outputs are enlarged for inspection. Final SIGGRAPH-quality visual showcase and M1 completion remain open.')
 rp=a.output/'report.json';rp.write_text(json.dumps(report,indent=2))
 manifest=dict(version=1,title='Principled scale head: coverage and execution',page='render/scale-demo',images=images,scope=report['scope'],provenance=dict(producer_report=str(rp),producer_report_sha256=digest(rp),wheel_sha256=hard['wheel']['sha256']))
 mp=a.output/'manifest.json';mp.write_text(json.dumps(manifest,indent=2));print(json.dumps(dict(manifest=str(mp),sha256=digest(mp),images=len(images))))
if __name__=='__main__':main()
