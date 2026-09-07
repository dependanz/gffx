"""Finite studio scenes composed from gffx operations, not a public scene/shader API.

Offline fixtures, direct Lambert lighting, explicit constant environment fill. No shadows,
bounced light, physical wood, or measured interactive renderer throughput is implied.
"""
import argparse
import math
import subprocess
import hashlib
from pathlib import Path
import torch
from gffx.camera import Camera
from gffx.torch import mesh, render
from gffx.dashboard import ArrayView, connect
from .classic_demo import load_asset
from .reporter import Reporter

FRAMES=24


def provenance():
    return {'revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'dirty':bool(subprocess.check_output(['git','status','--porcelain'],text=True)),
            'artifact_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def scene():
    vertices=[];faces=[];colors=[]
    def add(v,f,c):
        offset=len(vertices);vertices.extend(v);faces.extend([[i+offset for i in row] for row in f]);colors.extend([c]*len(v))
    def quad(v,c):add(v,[(0,1,2),(0,2,3)],c)
    # Open-front Cornell-style set. Quads face the room; no Cornell reference claim.
    quad([[-2,0,2],[2,0,2],[2,0,-2],[-2,0,-2]],(.72,.72,.65))
    quad([[-2,0,-2],[2,0,-2],[2,3,-2],[-2,3,-2]],(.67,.69,.72))
    quad([[-2,0,2],[-2,0,-2],[-2,3,-2],[-2,3,2]],(.65,.10,.08))
    quad([[2,0,-2],[2,0,2],[2,3,2],[2,3,-2]],(.08,.44,.30))
    for name,x,size,color in [('bunny',-.7,1.7,(.8,.77,.64)),('teapot',.85,1.45,(.35,.64,.78))]:
        asset=load_asset(name);v=torch.tensor(asset['vertices'],dtype=torch.float32)
        lo=v.amin(0);hi=v.amax(0);v=(v-(lo+hi)/2)*size/(hi-lo).max()
        v[:,1]-=v[:,1].min();v+=torch.tensor([x,.08,0.])
        f=torch.tensor(asset['faces'],dtype=torch.int64)
        # Normalize an inward-wound fixture in this demo only; bundled data is unchanged.
        signed_volume=(v[f[:,0]]*torch.linalg.cross(v[f[:,1]],v[f[:,2]])).sum()
        if signed_volume<0:f=f[:,[0,2,1]]
        add(v.tolist(),f.tolist(),color)
    v=torch.tensor(vertices,dtype=torch.float32);f=torch.tensor(faces,dtype=torch.int32)
    normals=mesh.vertex_normals(v,f)
    return dict(vertices=v,faces=f,colors=torch.tensor(colors),normals=normals)


def look_at(eye,target):
    eye=torch.tensor(eye,dtype=torch.float32);target=torch.tensor(target,dtype=torch.float32)
    back=torch.nn.functional.normalize(eye-target,dim=0)
    right=torch.nn.functional.normalize(torch.linalg.cross(torch.tensor([0.,1.,0.]),back),dim=0)
    up=torch.linalg.cross(back,right);pose=torch.eye(4)
    pose[:3,:3]=torch.stack((right,up,back),dim=1);pose[:3,3]=eye
    return pose.unsqueeze(0)


def camera_path(mode,t,*,height=192,width=256,device='cpu'):
    if mode not in ('orbit','pan','dolly','zoom','dolly-zoom'):raise ValueError('Unknown camera path')
    if not math.isfinite(t) or not 0<=t<=1:raise ValueError('Camera time must lie in [0,1]')
    distance=4.5+2*t if mode in ('dolly','dolly-zoom') else 5.5
    focal=width*.95
    if mode=='dolly-zoom':focal*=distance/5.5
    if mode=='zoom':focal*=.7+.6*t
    eye=[0.,1.,distance];target=[0.,1.,0.]
    if mode=='orbit':
        angle=(t-.5)*.65;eye=[distance*math.sin(angle),2.,distance*math.cos(angle)]
    if mode=='pan':target=[(t-.5)*1.4,1.,0.]
    return Camera.from_intrinsics(focal,focal,width/2,height/2,height,width,.1,20.,pose=look_at(eye,target).to(device))


@torch.no_grad()
def render_frame(s,camera,t,*,preset='warm-cool'):
    if preset not in ('warm-cool','studio','approach'):raise ValueError('Unknown light preset')
    v=s['vertices'];f=s['faces'];p=camera.project(v,in_front_only=True)
    ndc=torch.stack((p.pixels[:,0]*2/camera.image_width-1,1-p.pixels[:,1]*2/camera.image_height,p.ndc_depth),-1)
    index,bary,z,_=render.rasterize(ndc.contiguous(),f,image_height=camera.image_height,
                                  image_width=camera.image_width,cull_mode=render.CULL_NONE)
    # Perspective-correct interpolated positions, normals and albedo from affine fragments.
    attributes=torch.cat((v,s['normals'],s['colors'],torch.ones_like(v[:,:1])),dim=1)
    attributes=attributes/p.view_depth[:,None]
    sample=render.interpolate(index,bary,attributes[f.long()].contiguous())[0,:,:,0]
    mask=index[0,:,:,0]>=0
    sample=sample/sample[:,:,-1:].clamp_min(1e-10)
    points=sample[:,:,:3];normals=torch.nn.functional.normalize(sample[:,:,3:6],dim=-1)
    albedo=sample[:,:,6:9];theta=2*math.pi*t
    light=v.new_tensor([1.4*math.cos(theta),2.3,1.0+1.0*math.sin(theta)])
    if preset=='approach':light=v.new_tensor([-.7,2.0,3.0-2.5*t])
    delta=light-points;distance2=delta.square().sum(-1).clamp_min(.01)
    direction=delta/distance2.sqrt()[...,None]
    cosine=(normals*direction).sum(-1).clamp_min(0)
    key=v.new_tensor([1.,.53,.23] if preset=='warm-cool' else [1.,.95,.86])
    direct=key*(16*cosine/distance2)[...,None]
    fill_dir=torch.nn.functional.normalize(v.new_tensor([-.7,.7,.5]),dim=0)
    fill=(normals*fill_dir).sum(-1).clamp_min(0)[...,None]*v.new_tensor([.2,.35,.8])
    linear=albedo*(direct+fill+.10)/math.pi
    # Explicit display transform only, never applied to scientific energy measurements.
    rgb=(linear/(1+linear)).clamp(0,1).pow(1/2.2)
    rgb[~mask]=v.new_tensor([.045,.065,.09])
    normal_image=((normals+1)/2).clamp(0,1);normal_image[~mask]=0
    depth=camera.ndc_to_view_depth(z[0,:,:,0]);depth=torch.where(mask,depth,torch.zeros_like(depth))
    depth_rgb=torch.stack((depth/10,1-depth/10,.5+depth/20),-1).clamp(0,1);depth_rgb[~mask]=0
    color=(rgb*255).round().byte()
    marker=torch.stack([light+v.new_tensor(offset) for offset in [(-.07,0,0),(.07,0,0),(0,-.07,0),(0,.07,0)]])
    color=overlay_lines(color,camera,marker,[(0,1),(2,3)],color=(255,245,165)).to(v.device)
    return dict(color=color,normals=(normal_image*255).round().byte(),
                depth_image=(depth_rgb*255).round().byte(),depth=depth,mask=mask,light=light,linear=linear)


def overlay_lines(image,camera,points,edges,color=(70,235,230)):
    """Display-only line annotations projected by the actual gffx camera."""
    image=image.cpu().clone();p=camera.project(points.to(camera.pose.device),in_front_only=True)
    pixels=p.pixels.cpu().tolist();valid=p.valid.cpu().tolist();height,width=image.shape[:2]
    for a,b in edges:
        if not valid[a] or not valid[b]:continue
        x0,y0=pixels[a];x1,y1=pixels[b]
        count=min(2000,max(int(abs(x1-x0)),int(abs(y1-y0)),1))
        for k in range(count+1):
            x=round(x0+(x1-x0)*k/count);y=round(y0+(y1-y0)*k/count)
            if 0<=x<width and 0<=y<height:image[y,x]=torch.tensor(color,dtype=torch.uint8)
    return image


def director_frame(background,observer,camera):
    device=camera.pose.device
    pixels=torch.tensor([[0.,0.],[camera.image_width,0.],[camera.image_width,camera.image_height],[0.,camera.image_height]],device=device)
    corners=camera.unproject_points(pixels,torch.ones(4,device=device)*1.1,kind='view')
    points=torch.cat((camera.pose[0,:3,3].unsqueeze(0),corners))
    return overlay_lines(background,observer,points,[(0,i) for i in range(1,5)]+[(1,2),(2,3),(3,4),(4,1)])


def publish_clips(dash,*,modes,frames=FRAMES,height=192,width=256,device='cpu'):
    """Finite byte-bounded clips shared by lighting and camera demonstration pages."""
    if not 2<=frames<=48 or not 16<=height<=240 or not 16<=width<=320:raise ValueError('Finite showcase budget exceeded')
    s={k:v.to(device) for k,v in scene().items()};artifacts=[]
    observer=Camera.from_intrinsics(width*.75,width*.75,width/2,height/2,height,width,.1,30.,pose=look_at([7.,5.,10.],[0.,1.,1.5]).to(device))
    background=render_frame(s,observer,.15)['color']
    for mode in modes:
        lighting=mode in ('lights','studio','approach')
        views=('color','normals','depth_image') if lighting else ('color','director')
        queues={view:dash.queue(f'studio/{mode}/{view}','image',recent_window=frames,snapshot_every=1) for view in views}
        for step in range(frames):
            t=step/(frames-1);camera=camera_path('orbit' if lighting else mode,t,height=height,width=width,device=device)
            result=render_frame(s,camera,t if lighting else .15,preset={'lights':'warm-cool','studio':'studio','approach':'approach'}.get(mode,'warm-cool'))
            if not lighting:result['director']=director_frame(background,observer,camera)
            for view,q in queues.items():q.push(image_array(result[view]),step=step)
            if mode=='lights':dash.scalar('studio/light-z',float(result['light'][2]),step=step)
        for view in views:artifacts.append(dict(kind='image',path=f'studio/{mode}/{view}',label=f'{mode} · {view}'))
    return artifacts


def image_array(tensor):
    tensor=tensor.detach().cpu().contiguous()
    return ArrayView('uint8',tuple(tensor.shape),bytes(tensor.reshape(-1).tolist()))


def depth_samples(camera,result,*,stride=4):
    if type(stride) is not int or stride<1:raise ValueError('stride must be positive')
    xyz,valid=camera.unproject(result['depth'][None],kind='view',valid=result['mask'][None])
    xyz=xyz[0,::stride,::stride];valid=valid[0,::stride,::stride]
    y,x=torch.meshgrid(torch.arange(0,camera.image_height,stride,device=xyz.device)+.5,
                       torch.arange(0,camera.image_width,stride,device=xyz.device)+.5,indexing='ij')
    pixels=torch.stack((x,y),-1)
    return xyz[valid],result['color'][::stride,::stride][valid].float()/255,pixels[valid]


def publish_depth(address,run,*,device='cpu',height=192,width=256):
    from .camera_demo import points_array
    s={k:v.to(device) for k,v in scene().items()};camera=camera_path('orbit',.5,device=device,height=height,width=width)
    result=render_frame(s,camera,.15);xyz,colors,_=depth_samples(camera,result)
    reporter=Reporter(address,run)
    if not reporter.send('start',kind='demo',provenance=provenance()):raise RuntimeError(reporter.error)
    with connect(address,run=run,wait_acks=True) as dash:
        dash.image('depth/color',image_array(result['color']),step=0)
        dash.image('depth/distance',image_array(result['depth_image']),step=0)
        dash.points('depth/surface',points_array(xyz),colors=points_array(colors),step=0)
        dash.text('depth/explained','LEFT: gffx rasterized color. RIGHT: colored positive view depth (fixed 0–10 world-unit display scale). Orbit the point cloud below: unprojecting every fourth visible pixel produces only the surfaces seen by this camera. Empty backs are expected, not missing geometry reconstruction. gffx Camera.unproject performs the reconstruction; the dashboard only displays the points.',step=0)
    artifacts=[dict(kind=k,path='depth/'+p,label=l) for k,p,l in [('image','color','Original camera image'),('image','distance','Positive view depth'),('mesh','surface','Orbit the visible surface · unseen backs are empty'),('text','explained','From pixels back to geometry')]]
    if not reporter.send('demo',page='camera/depth-demo',title='Depth becomes geometry',artifacts=artifacts):raise RuntimeError(reporter.error)
    if not reporter.send('finish',exit_code=0):raise RuntimeError(reporter.error)


def publish(address,run,*,frames=FRAMES,height=192,width=256,device='cpu'):
    if not 2<=frames<=48 or not 16<=height<=240 or not 16<=width<=320:raise ValueError('Finite showcase budget exceeded')
    torch.set_num_threads(2)
    reporter=Reporter(address,run)
    def report(event,**kw):
        if not reporter.send(event,**kw):raise RuntimeError(reporter.error)
    report('start',kind='demo',provenance=provenance())
    artifacts=[]
    with connect(address,run=run,wait_acks=True) as dash:
        artifacts=publish_clips(dash,modes=('lights','orbit','pan','dolly','zoom','dolly-zoom'),frames=frames,height=height,width=width,device=device)
        dash.text('studio/explained','Choose a clip and press Play. Lights move while the camera orbits; other clips isolate orbit, pan, dolly, zoom and dolly-zoom. The latter keeps the z=0 reference plane scale fixed, not every point on a deep object. Images use gffx cameras, vertex normals, rasterize and perspective-correct interpolate. Direct Lambert light and constant environment fill are demonstration tensor recipes, not public shading. Display uses x/(1+x) then gamma 1/2.2. No shadows, bounced light or Cornell reference equivalence. All frames are saved once and the producer exits. Bunny/teapot fixture terms remain in demo_assets/README.md.',step=0)
        artifacts.append(dict(kind='text',path='studio/explained',label='How the studio works'))
        artifacts.append(dict(kind='plot',path='studio/light-z',label='Light world Z · shared frame index'))
    report('demo',page='camera/studio-demo',title='Studio · lights & camera moves',artifacts=artifacts)
    report('finish',exit_code=0)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--address',required=True);parser.add_argument('--run',required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args=parser.parse_args();publish(args.address,args.run,device=args.device)
