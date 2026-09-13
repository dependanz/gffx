"""Finite public-scene demonstrations: hierarchy, shared topology and image fitting."""
import argparse
import math
import struct
import torch


def payload():
    from gffx.torch import scene as s,render
    v=torch.tensor([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],dtype=torch.float64)*.22
    f=torch.tensor([[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],[2,3,7],[2,7,6],[1,2,6],[1,6,5],[3,0,4],[3,4,7]],dtype=torch.int32)
    times=torch.arange(12,dtype=v.dtype)*2*math.pi/12
    root=torch.eye(4,dtype=v.dtype).repeat(12,1,1)
    root[:,0,0]=times.cos();root[:,0,1]=-times.sin();root[:,1,0]=times.sin();root[:,1,1]=times.cos()
    def t(x,y):
        a=torch.eye(4,dtype=v.dtype);a[0,3]=x;a[1,3]=y;return a
    colors=((.15,.55,1.),(1.,.45,.1),(.2,.85,.45))
    prims=[s.Rig('turntable',transform=root),s.Rig('arm',parent='turntable',transform=t(.65,0))]
    for name,parent,tr,c in zip(('blue','orange','green'),('turntable','arm','arm'),(t(-.6,0),t(0,0),t(0,.6)),colors):
        prims.append(s.Mesh(name,v,f,parent=parent,transform=tr,attributes={'colour':s.Attribute(torch.tensor(c,dtype=v.dtype),'constant')}))
    p=s.Scene(prims).flatten();frames=[p.vertices[i*24:(i+1)*24].detach() for i in range(12)]
    # Fixed orthographic display mapping, not a scene-owned camera or renderer.
    ndc=frames[1]*.65;ndc=ndc.clone();ndc[:,2]=ndc[:,2]*.2+.5
    fragments=render.rasterize(ndc,p.faces[:36],192,192,cull_mode=render.CULL_NONE)
    colour=render.interpolate(fragments[0],fragments[1],p.attributes['colour'].values[:36])[0,:,:,0]
    colour=torch.where((fragments[0][0,:,:,0]>=0)[...,None],colour,torch.tensor([.025,.04,.075],dtype=v.dtype))
    image=(colour.clamp(0,1)*255).round().to(torch.uint8)
    triangle=torch.tensor([[-.7,-.6,.2],[.6,-.5,.3],[0,.7,.4]],dtype=v.dtype)
    faces=torch.tensor([[0,1,2]],dtype=torch.int32)
    colours=torch.tensor([[1.,.1,.2],[.2,1.,.3],[.1,.3,1.]],dtype=v.dtype)
    def picture(x):
        matrix=torch.eye(4,dtype=v.dtype)+x*torch.tensor([[0,0,0,1.],[0,0,0,0],[0,0,0,0],[0,0,0,0]],dtype=v.dtype)
        packed=s.Scene([s.Mesh('fit',triangle,faces,transform=matrix,attributes={'colour':s.Attribute(colours,'vertex')})]).flatten()
        q=render.rasterize(packed.vertices,packed.faces,8,8,cull_mode=render.CULL_NONE)
        return render.interpolate(q[0],q[1],packed.attributes['colour'].values)[0,:,:,0]
    target=picture(torch.tensor(.014,dtype=v.dtype)).detach();x=torch.tensor(.013,dtype=v.dtype,requires_grad=True)
    losses=[];before=picture(x).detach()
    for _ in range(8):
        loss=(picture(x)-target).square().sum();losses.append(float(loss.detach()))
        x=(x-.001*torch.autograd.grad(loss,x)[0]).detach().requires_grad_()
    after=picture(x).detach()
    fit=(torch.cat((target,before,after),dim=1).clamp(0,1)*255).round().to(torch.uint8).repeat_interleave(12,0).repeat_interleave(12,1)
    return dict(vertices=frames[0],faces=p.faces[:36].detach(),frames=frames,image=image,fit_image=fit,fit_losses=losses)


def view(t):
    from gffx.dashboard import ArrayView
    t=t.detach().cpu().contiguous()
    kind='uint8' if t.dtype==torch.uint8 else 'int32' if t.dtype==torch.int32 else 'float32'
    values=t.flatten().tolist();fmt={'uint8':'B','int32':'i','float32':'f'}[kind]
    return ArrayView(kind,tuple(t.shape),struct.pack('<'+fmt*len(values),*values))


def publish(address,run):
    from gffx.dashboard import connect
    from .reporter import Reporter
    from pathlib import Path
    from tools.shading_cuda_closeout import digest
    import gffx.torch.scene as scene
    data=payload();reporter=Reporter(address,run)
    def send(event,**kwargs):
        if not reporter.send(event,**kwargs):raise RuntimeError(reporter.error)
    send('start',kind='demo',provenance={'artifact_sha256':digest(Path(scene.__file__))})
    artifacts=[]
    with connect(address,run=run,wait_acks=True) as dash:
        def add(kind,path,label,*args):
            getattr(dash,kind)(path,*args,step=0);artifacts.append(dict(kind=kind,path=path,label=label))
        add('text','scene/guide','How this scene works',
            'Three cubes share a turntable parent; orange and green share a second arm parent. '
            'The 12-frame clip is authored transform animation through public Scene.flatten, not physics or skinning. '
            'The 192x192 image is native rasterization with interpolated colours, not new lighting. '
            'The tiny fitting example uses a stable interior translation and fresh gradients; enlarged pixels are intentional. '
            'It demonstrates graph connectivity, not full-image alignment or optimizer convergence.')
        add('mesh','scene/hierarchy','Three objects in a shared hierarchy',view(data['vertices']),view(data['faces']))
        add('image','scene/rendered','Packed geometry through the existing renderer',view(data['image']))
        queue=dash.queue('scene/sequence','mesh',recent_window=12,snapshot_every=1)
        for i,v in enumerate(data['frames']):queue.push(view(v),view(data['faces']),step=i)
        artifacts.append(dict(kind='mesh',path='scene/sequence',label='12 saved frames: the parent moves both children'))
        add('image','scene/fitting','Target / before / after: small smooth pose fit',view(data['fit_image']))
        for i,loss in enumerate(data['fit_losses']):dash.scalar('scene/loss/value',loss,step=i)
        artifacts.append(dict(kind='plot',path='scene/loss',label='Eight fresh image-loss measurements'))
    send('demo',page='scene/hierarchy-demo',title='Scenes: hierarchy, frames and fitting',artifacts=artifacts)
    send('finish',exit_code=0)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--address',required=True);p.add_argument('--run',required=True);a=p.parse_args()
    torch.set_num_threads(1);publish(a.address,a.run)
