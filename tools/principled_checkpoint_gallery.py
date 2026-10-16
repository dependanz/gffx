"""Presentation-only Pillow figures from immutable numerical reports; no renderer rerun.

Pillow is an optional authoring dependency, not a gffx runtime dependency.
"""
import argparse,hashlib,json,math
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    args.output.mkdir(exist_ok=False);root=args.root;S=2;W,H=1600,900
    bg='#0c1422';panel='#142237';ink='#e9f0fa';muted='#a8bad2';cyan='#65d6d0';gold='#f4bf71';purple='#b39df2'
    fonts={};sources=[];images=[]
    def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    def read(name):
        p=root/name;sources.append(dict(path=str(p),sha256=digest(p)));return json.loads(p.read_text(encoding='utf-8'))
    def font(size,bold=False):
        key=size,bold
        if key not in fonts:fonts[key]=ImageFont.truetype('C:/Windows/Fonts/'+('seguisb.ttf' if bold else 'segoeui.ttf'),size*S)
        return fonts[key]
    def canvas():
        image=Image.new('RGB',(W*S,H*S),bg);return image,ImageDraw.Draw(image)
    def text(d,x,y,s,size=24,colour=ink,bold=False):d.text((x*S,y*S),s,font=font(size,bold),fill=colour)
    def line(d,xy,fill=muted,width=1):d.line([(x*S,y*S) for x,y in xy],fill=fill,width=width*S)
    def box(d,xy,fill=panel):d.rounded_rectangle(tuple(v*S for v in xy),radius=18*S,fill=fill)
    def save(image,label,caption):
        image=image.resize((W,H),Image.Resampling.LANCZOS);tmp=args.output/(str(len(images))+'.png');image.save(tmp,optimize=True)
        sha=digest(tmp);path=args.output/(sha+'.png');tmp.rename(path)
        images.append(dict(label=label,caption=caption,path=str(path),asset=dict(url='/testing/media/'+sha+'.png',sha256=sha,bytes=path.stat().st_size)))
    names=('vram19-principled-64-forward1.json','nativecurve22-principled-64-forward1.json','nativecurve23-principled-64-forward1.json')
    # Original vram19 is not required for presentation: only current source-bound
    # new measurements are compared, each explicitly a single evaluation.
    rows=[read(n) for n in names[1:]]
    assert all(x['status']=='passed' and x['profile']['size']==64 and x['profile']['material']=='principled' for x in rows)
    forecasts=[read('nativecurve23-principled-'+str(n)+'-forward1.json') for n in (128,256)]
    assert all(x['status']=='not_admitted' and not x.get('render_executed',False) for x in forecasts)
    image,d=canvas();text(d,64,45,'MAKING PRINCIPLED PRACTICAL',17,cyan,True);text(d,64,81,'Accurate curves. Execution still matters.',44,ink,True)
    text(d,64,143,'Native curve integration on danziepc  /  14 September 2026',23,muted)
    box(d,(48,210,975,745));box(d,(1000,210,1552,745))
    text(d,78,235,'Measured 64 × 64 frame',29,ink,True);text(d,78,282,'Hard visibility · one light · float64 · CUDA',21,muted)
    x0,x1=78,918;y0=645
    for tick in range(0,19,3):
        x=x0+(x1-x0)*tick/18;line(d,[(x,344),(x,y0)],'#29394e');text(d,x-6,y0+10,str(tick),19,muted)
    for idx,(label,row,colour) in enumerate(zip(('Native curves','Node-parallel curves'),rows,(purple,cyan))):
        y=370+idx*140;value=row['measured_seconds'];text(d,x0,y-32,label,23,ink,True)
        box(d,(x0,y+9,x0+(x1-x0)*value/18,y+53),colour);text(d,x0+(x1-x0)*value/18+13,y+14,f'{value:.3f} s',23,ink,True)
    text(d,78,697,'Seconds per complete measured frame',20,muted)
    text(d,1030,235,'Large-frame admission',27,ink,True)
    text(d,1030,287,'Forecasts, not executed renders',21,gold)
    for idx,row in enumerate(forecasts):
        y=357+idx*149;n=row['profile']['size'];text(d,1030,y,f'{n} × {n}',29,ink,True)
        text(d,1030,y+44,f"{row['forecast_seconds']:.1f} s forecast",25,gold,True);text(d,1030,y+82,'Not admitted under the time gate',19,muted)
    text(d,64,783,'ONE OBSERVATION PER PROFILE · NO SUSTAINED-THROUGHPUT CLAIM',17,cyan,True)
    text(d,64,818,'Existing 1,986-vertex calibration head. FLAME-scale, large-frame gradients and final visual showcase remain open.',20,muted)
    save(image,'Execution checkpoint','Two source-bound installed-wheel measurements at 64×64: nativecurve22 versus node-parallel nativecurve23. Each is one complete forward evaluation, not a statistical benchmark. The 128/256 figures use the unchanged conservative forecast and were not executed. This is calibration geometry, not FLAME-scale qualification.')
    physical=read('nativecurve23-physical-cuda-directional1.rows.json');assert len(physical)==80
    report=read('nativecurve23-physical-cuda-directional1.json');assert report['status']=='passed'
    cohort=read('nativecurve23-physical-cohort.json');assert cohort['status']=='passed'
    image,d=canvas();text(d,64,45,'CHECK THE PHYSICS, THEN THE PICTURE',17,cyan,True);text(d,64,81,'Independent values and fitting gradients',43,ink,True)
    text(d,64,143,'Directional-loss samples from the frozen physical screen · CUDA nativecurve23',23,muted)
    box(d,(48,210,810,734));box(d,(834,210,1552,734))
    text(d,78,235,'Directional loss',28,ink,True);text(d,78,279,'Fixed F0 = 0.07 · colour denotes roughness',20,muted)
    colors=[cyan,gold,purple,'#ec8eac','#8eb6fc'];roughness=sorted({v['roughness'] for v in physical});left,bottom,right,top=108,650,768,343
    for tick in (0,.25,.5,.75,1.):
        y=bottom-(bottom-top)*tick;line(d,[(left,y),(right,y)],'#29394e');text(d,65,y-12,f'{tick:g}',17,muted)
    for idx,r in enumerate(roughness):
        samples=sorted([v for v in physical if v['roughness']==r and v['f0']==.07],key=lambda v:v['cosine']);colour=colors[idx%len(colors)]
        points=[(left+(right-left)*v['cosine'],bottom-(bottom-top)*v['loss']) for v in samples];line(d,points,colour,2)
        for point,row in zip(points,samples):
            xx,yy=point;d.ellipse(((xx-4)*S,(yy-4)*S,(xx+4)*S,(yy+4)*S),fill=colour)
            yy=bottom-(bottom-top)*row['reference'];d.ellipse(((xx-7)*S,(yy-7)*S,(xx+7)*S,(yy+7)*S),outline=ink,width=S)
        text(d,112+idx*126,680,f'r={r:g}',16,colour,True)
    text(d,295,713,'View cosine 0 → 1',17,muted)
    text(d,865,235,'Gradient acceptance ratio',28,ink,True);text(d,865,279,'240 partials · original bound = 1',20,muted)
    left,right,top,bottom=895,1508,343,650
    for tick in (0,.25,.5,.75,1.):
        y=bottom-(bottom-top)*tick;line(d,[(left,y),(right,y)],'#29394e');text(d,852,y-12,f'{tick:g}',17,muted)
    ratios=[v for row in physical for v in row['gradient_ratios']]
    for j,v in enumerate(ratios):
        x=left+(right-left)*j/(len(ratios)-1);y=bottom-(bottom-top)*v;d.ellipse(((x-2)*S,(y-2)*S,(x+2)*S,(y+2)*S),fill=cyan)
    line(d,[(left,top),(right,top)],gold,2);text(d,1157,353,'Acceptance limit',17,gold)
    text(d,913,680,f'Maximum ratio {max(ratios):.4f}  /  all below 1',21,cyan,True)
    text(d,64,774,'FILLED POINTS: GFFX · OPEN RINGS: INDEPENDENT REFERENCE · LINES CONNECT SAMPLES',16,cyan,True)
    text(d,64,811,'80 directional + 16 mean + 192 furnace comparisons pass on this CUDA candidate. No tolerance was relaxed.',20,muted)
    save(image,'Physical qualification','Saved directional-loss values versus the independent frozen physical reference at 80 parameter combinations. The left panel selects F0=0.07; connecting lines are visual guides between measured samples. The right panel shows all 240 first-partial acceptance ratios; values below one satisfy the original criterion. Separate mean/furnace reports complete the 288-row CUDA screen. This is numerical qualification, not a final rendered visual demonstration.')
    report={'status':'passed','generator_sha256':digest(__file__),'sources':sources,'images':images,'scope':'Two presentation figures from saved verified reports; no renderer, optimizer or physical test rerun. Numerical qualification and incomplete large-frame admission remain distinct. These figures are not the final SIGGRAPH-quality renderer showcase.'}
    rp=args.output/'report.json';rp.write_text(json.dumps(report,indent=2))
    manifest={'version':1,'title':'Native Principled: accuracy and execution','page':'shading/native-principled-demo','images':images,'scope':report['scope'],'provenance':{'producer_report':str(rp),'producer_report_sha256':digest(rp),'wheel_sha256':rows[-1]['wheel']['sha256']}}
    mp=args.output/'manifest.json';mp.write_text(json.dumps(manifest,indent=2));print(json.dumps({'manifest':str(mp),'sha256':digest(mp),'images':len(images)}))

if __name__=='__main__':main()
