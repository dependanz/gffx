"""High-resolution finite camera-recovery and lighting presentations. Not a public renderer."""
import argparse
import ctypes
import math
from pathlib import Path
import time
import torch
from gffx.camera import Camera
from gffx.dashboard import connect
from .classic_demo import load_asset
from .showcase import look_at,director_frame,provenance
from .presentation import AssetWriter,png_rgb
from .reporter import Reporter
from gffx.torch import mesh,render

TARGET=[.08,-.02,.01,.30,1.15,4.2,math.log(.95)]
INITIAL=[-.24,.16,-.10,-.65,1.5,5.6,math.log(.70)]

def refine_fixture(vertices,faces):
    """One finite Loop-style display refinement, not a public gffx subdivision API.

    Non-manifold edge endpoints are left fixed; their edge points are midpoints.
    The original attributed fixture files are never changed.
    """
    v=vertices.tolist();f=faces.tolist();neighbors=[set() for _ in v];edges={}
    for a,b,c in f:
        for x,y,o in [(a,b,c),(b,c,a),(c,a,b)]:
            neighbors[x].add(y);neighbors[y].add(x);edges.setdefault(tuple(sorted((x,y))),[]).append(o)
    boundary=[[] for _ in v];irregular=set()
    for (a,b),opposite in edges.items():
        if len(opposite)==1:boundary[a].append(b);boundary[b].append(a)
        elif len(opposite)!=2:irregular.update((a,b))
    points=[]
    for i,p in enumerate(v):
        ring=neighbors[i];n=len(ring)
        if i in irregular or not n:points.append(p)
        elif len(boundary[i])==2:points.append([.75*p[k]+.125*sum(v[j][k] for j in boundary[i]) for k in range(3)])
        elif boundary[i]:points.append(p)
        else:
            beta=3/16 if n==3 else 3/(8*n)
            points.append([(1-n*beta)*p[k]+beta*sum(v[j][k] for j in ring) for k in range(3)])
    ids={}
    for (a,b),opposite in edges.items():
        ids[a,b]=len(points)
        points.append([.375*(v[a][k]+v[b][k])+.125*sum(v[o][k] for o in opposite) if len(opposite)==2 else .5*(v[a][k]+v[b][k]) for k in range(3)])
    refined=[]
    for a,b,c in f:
        ab=ids[tuple(sorted((a,b)))];bc=ids[tuple(sorted((b,c)))];ca=ids[tuple(sorted((c,a)))]
        refined.extend([[a,ab,ca],[ab,b,bc],[ca,bc,c],[ab,bc,ca]])
    return vertices.new_tensor(points),faces.new_tensor(refined)

def highlight_lobe(normal,light,view):
    reflected=2*(normal*light).sum(-1)[...,None]*normal-light
    return (reflected*view).sum(-1).clamp_min(0).pow(72)*73/(2*math.pi)

def rotation(p):
    y,x,z=p[:3];cy,sy=torch.cos(y),torch.sin(y);cx,sx=torch.cos(x),torch.sin(x);cz,sz=torch.cos(z),torch.sin(z)
    zero=y*0;one=zero+1
    ry=torch.stack([cy,zero,sy,zero,one,zero,-sy,zero,cy]).reshape(3,3)
    rx=torch.stack([one,zero,zero,zero,cx,-sx,zero,sx,cx]).reshape(3,3)
    rz=torch.stack([cz,-sz,zero,sz,cz,zero,zero,zero,one]).reshape(3,3)
    return ry@rx@rz

def shot_camera(parameters,*,height=720,width=1280):
    p=parameters;top=torch.cat([rotation(p),p[3:6,None]],dim=1)
    pose=torch.cat([top,p.new_tensor([[0,0,0,1]])]).unsqueeze(0)
    focal=p[6].exp()*width
    return Camera.from_intrinsics(focal,focal,width/2,height/2,height,width,.1,30.,pose=pose)

def shot_scene():
    # Original studio's offline fixtures; use the bunny alone as the visual protagonist.
    # Build directly from the attributed fixture to avoid slicing through any triangles.
    asset=load_asset('bunny');v=torch.tensor(asset['vertices'],dtype=torch.float32);f=torch.tensor(asset['faces'],dtype=torch.int32)
    lo=v.amin(0);hi=v.amax(0);v=(v-(lo+hi)/2)*1.8/(hi-lo).max();v[:,1]-=v[:,1].min();v[:,1]+=.06
    if (v[f[:,0].long()]*torch.linalg.cross(v[f[:,1].long()],v[f[:,2].long()])).sum()<0:f=f[:,[0,2,1]]
    v,f=refine_fixture(v,f)
    colors=torch.tensor([.62,.76,.82]).expand(len(v),3).clone()
    # Large matte stage, with a subtle procedural tile grid in the shading recipe.
    floor=v.new_tensor([[-4,0,1.8],[4,0,1.8],[4,0,-4],[-4,0,-4]])
    ff=torch.tensor([[0,1,2],[0,2,3]],dtype=torch.int32)+len(v)
    v=torch.cat([v,floor]);f=torch.cat([f,ff]);colors=torch.cat([colors,v.new_tensor([[.065,.08,.11]]).expand(4,3)])
    return {'vertices':v,'faces':f,'normals':mesh.vertex_normals(v,f),'colors':colors}

def solve_shot(*,steps=600):
    if not 100<=steps<=1200:raise ValueError('Camera solve budget exceeded')
    s=shot_scene();vertices=s['vertices'][:-4].double()
    points=vertices[torch.linspace(0,len(vertices)-1,96).long()]
    target=torch.tensor(TARGET,dtype=torch.float64)
    wanted=shot_camera(target,height=360,width=640).project(points).pixels.detach()
    parameters=torch.tensor(INITIAL,dtype=torch.float64,requires_grad=True)
    opt=torch.optim.Adam([parameters],lr=.025)
    refine=torch.optim.LBFGS([parameters],lr=.5,max_iter=2,line_search_fn='strong_wolfe',tolerance_grad=1e-12,tolerance_change=1e-14)
    saved={round((i/47)**2*steps) for i in range(48)};history=[]
    for step in range(steps+1):
        opt.zero_grad();actual=shot_camera(parameters,height=360,width=640).project(points).pixels
        loss=(actual-wanted).square().mean()
        if step in saved:history.append({'step':step,'loss':float(loss.detach()),'parameters':parameters.detach().tolist()})
        if step>100 and loss.detach()<1e-9:
            if history[-1]['step']!=step:history.append({'step':step,'loss':float(loss.detach()),'parameters':parameters.detach().tolist()})
            break
        if step<steps:
            if step<100:loss.backward();opt.step()
            else:
                def closure():
                    refine.zero_grad();pixels=shot_camera(parameters,height=360,width=640).project(points).pixels
                    value=(pixels-wanted).square().mean();value.backward();return value
                refine.step(closure)
    return {'objective':'known 3D landmark reprojection; not image loss','history':history,'target':TARGET,'landmarks':96}

@torch.no_grad()
def render_beauty(s,camera,t=.3):
    v,f=s['vertices'],s['faces'];projected=camera.project(v,in_front_only=True)
    ndc=torch.stack((projected.pixels[:,0]*2/camera.image_width-1,1-projected.pixels[:,1]*2/camera.image_height,projected.ndc_depth),-1)
    indices,bary,_,_=render.rasterize(ndc.contiguous(),f,image_height=camera.image_height,image_width=camera.image_width,cull_mode=render.CULL_NONE)
    attrs=torch.cat([v,s['normals'],s['colors'],torch.ones_like(v[:,:1])],-1)/projected.view_depth[:,None]
    sample=render.interpolate(indices,bary,attrs[f.long()].contiguous())[0,:,:,0]
    sample=sample/sample[:,:,-1:].clamp_min(1e-10);point=sample[:,:,:3];normal=torch.nn.functional.normalize(sample[:,:,3:6],dim=-1)
    albedo=sample[:,:,6:9];valid=indices[0,:,:,0]>=0
    eye=camera.pose[0,:3,3];view=torch.nn.functional.normalize(eye-point,dim=-1)
    angle=t*math.pi*1.6
    lights=[([3*math.cos(angle),3.7,2.5*math.sin(angle)],[1.,.60,.32],20.),([-2.,2.,-2.],[.12,.55,1.],26.),([0.,4.,3.],[.8,.9,1.],5.)]
    diffuse=torch.zeros_like(point);specular=torch.zeros_like(point)
    for position,color,power in lights:
        delta=v.new_tensor(position)-point;d2=delta.square().sum(-1).clamp_min(.04);direction=delta/d2.sqrt()[...,None]
        cosine=(normal*direction).sum(-1).clamp_min(0);irr=v.new_tensor(color)*power/d2[...,None]
        diffuse+=irr*cosine[...,None]
        # Reciprocal Phong recipe with a conservative hemispherical bound: integral
        # of max(R dot V,0)^n over a hemisphere is <= 2*pi/(n+1), and cos(theta)<=1.
        lobe=highlight_lobe(normal,direction,view)
        specular+=irr*(lobe*cosine)[...,None]*.12
    floor=point[:,:,1]<.02
    grid=((point[:,:,0]*2).remainder(1)<.025)|((point[:,:,2]*2).remainder(1)<.025)
    albedo=torch.where((floor&grid)[...,None],albedo*1.7,albedo)
    linear=.88*albedo*(diffuse+.045)/math.pi+specular
    rgb=(linear/(1+linear)).clamp(0,1).pow(1/2.2)
    yy=torch.linspace(0,1,camera.image_height,device=v.device)[:,None,None]
    background=v.new_tensor([.045,.065,.095])[None,None,:]*(.5+yy*.9)
    rgb=torch.where(valid[...,None],rgb,background)
    # Bounded edge smoothing in display space, never used as numeric/gradient evidence.
    x=rgb.permute(2,0,1)[None];avg=torch.nn.functional.avg_pool2d(x,3,stride=1,padding=1,count_include_pad=False)
    edge=(x-avg).abs().amax(1,keepdim=True)>.055
    rgb=torch.where(edge,x*.35+avg*.65,x)[0].permute(1,2,0)
    return (rgb.clamp(0,1)*255).round().byte()

def encoded(image):
    image=image.cpu().contiguous();data=ctypes.string_at(image.data_ptr(),image.numel())
    return png_rgb(image.shape[1],image.shape[0],data)

def publish(address,root,run,*,device='cuda',mode='fit',height=720,width=1280):
    if mode not in ('fit','lights'):raise ValueError('Unknown cinematic mode')
    if (height,width) not in ((720,1280),(180,320)):raise ValueError('Use reviewed 720p or diagnostic size')
    torch.set_num_threads(2);started=time.perf_counter();writer=AssetWriter(Path(root)/'presentation')
    s={k:v.to(device) for k,v in shot_scene().items()}
    result=solve_shot();target=shot_camera(torch.tensor(TARGET,device=device),height=height,width=width)
    fixed=writer.put(encoded(render_beauty(s,target)))
    poster=writer.put(encoded(render_beauty(s,shot_camera(torch.tensor(TARGET,device=device),height=1080 if height==720 else height,width=1920 if width==1280 else width))))
    observer=Camera.from_intrinsics(310.,310.,240.,135.,270,480,.1,30.,pose=look_at([5.,3.5,7.],[0.,1.,2.]).to(device))
    outside=render_beauty(s,observer)
    history=result['history'] if mode=='fit' else [{'step':i,'loss':None,'parameters':TARGET} for i in range(48)]
    frames=[]
    for index,item in enumerate(history):
        p=torch.tensor(item['parameters'],device=device)
        if mode=='lights':
            p=p.clone();p[0]+=.10*math.sin(index/47*math.pi*2);p[3]+=.35*math.sin(index/47*math.pi*2)
        camera=shot_camera(p,height=height,width=width)
        beauty=writer.put(encoded(render_beauty(s,camera,index/47 if mode=='lights' else .3)))
        director=writer.put(encoded(director_frame(outside,observer,camera)))
        frames.append({**item,'image':beauty,'director':director})
        print(f'{mode}: frame {index+1}/{len(history)}; compressed bundle {writer.bytes/1048576:.1f} MiB',flush=True)
    manifest={'version':1,'title':'Recover the shot' if mode=='fit' else 'Light reveals form','mode':mode,'width':width,'height':height,
              'target':fixed,'poster':poster,'frames':frames,'objective':result['objective'] if mode=='fit' else 'authored moving lights and camera choreography',
              'scope':'Existing gffx camera, normals, rasterize and interpolate. Offline fixture has one display-only Loop-style refinement. Reciprocal Phong highlight with conservative energy bound, not the pending public material. No shadows or reflections. Edge-filtered display PNGs.',
              'bytes':writer.bytes,'generation_seconds':time.perf_counter()-started}
    reporter=Reporter(address,run)
    def send(event,**kw):
        if not reporter.send(event,**kw):raise RuntimeError(reporter.error)
    send('start',kind='demo',provenance=provenance())
    science=[]
    with connect(address,run=run,wait_acks=True) as dash:
        dash.record('presentation/film',manifest,step=0)
        dash.text('presentation/explanation',manifest['objective']+'. '+manifest['scope']+' Saved optimization states are genuine; playback does not solve or write history. No correspondence-free photo reconstruction.',step=0)
        if mode=='fit':
            for frame in frames:dash.scalar('presentation/loss',frame['loss'],step=frame['step'])
            from .camera_fit_demo import derivatives
            for focal,a,n,e in derivatives():
                dash.scalar('diagnostics/derivative/autodiff',a,step=focal)
                dash.scalar('diagnostics/derivative/finite-difference',n,step=focal)
                dash.scalar('diagnostics/derivative-error',e,step=focal)
            dash.text('diagnostics/derivative-explained','These independent unprojection derivative checks sweep focal length, not optimization steps. They are not the camera-recovery loss. Compare autodiff against central finite differences.',step=0)
        else:
            from .lighting_demo import examples
            from .showcase import image_array
            data=examples();dash.image('lighting/exact-left-sh-right',image_array(data['image']),step=0)
            for index,(_,point,distant) in enumerate(data['distance']):
                dash.scalar('lighting/distance/point',point,step=index);dash.scalar('lighting/distance/distant',distant,step=index)
            for degree,exact,sh in data['angle']:
                dash.scalar('lighting/angle/exact',exact,step=int(degree));dash.scalar('lighting/angle/sh',sh,step=int(degree))
            dash.text('lighting/explained','Independent analytic fixtures, not measurements of the presentation shader. Distance=(sample+1)/2; intensity=4. Angle is degrees to the source. Exact sphere response left, signed-SH approximation right; magenta marks negative SH ringing.',step=0)
            science=[{'kind':k,'path':p,'label':label} for k,p,label in [('text','lighting/explained','Scientific light fixtures'),('image','lighting/exact-left-sh-right','Exact versus SH'),('plot','lighting/distance','Inverse-square falloff'),('plot','lighting/angle','Exact versus signed SH')]]
    artifacts=[{'kind':'record','path':'presentation/film','label':manifest['title']},{'kind':'text','path':'presentation/explanation','label':'Method and boundaries'}]
    artifacts.extend(science)
    if mode=='fit':
        artifacts.extend([{'kind':'plot','path':'presentation/loss','label':'Known landmark reprojection error'},
                          {'kind':'text','path':'diagnostics/derivative-explained','label':'Independent derivative comparison'},
                          {'kind':'plot','path':'diagnostics/derivative','label':'Autodiff versus finite differences'},
                          {'kind':'plot','path':'diagnostics/derivative-error','label':'Absolute derivative error'}])
    send('demo',page='camera/fit-demo' if mode=='fit' else 'shading/lights-demo',title=manifest['title'],artifacts=artifacts)
    send('finish',exit_code=0);print(manifest['generation_seconds'],writer.bytes,flush=True)
    return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--address',required=True);p.add_argument('--root',required=True);p.add_argument('--run',required=True);p.add_argument('--mode',choices=['fit','lights'],default='fit');p.add_argument('--device',default='cuda');p.add_argument('--preview',action='store_true');a=p.parse_args()
    publish(a.address,a.root,a.run,mode=a.mode,device=a.device,height=180 if a.preview else 720,width=320 if a.preview else 1280)
