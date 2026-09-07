"""Finite synthetic intrinsic fitting and inverse-derivative illustrations, not gffit."""
import argparse
import subprocess
import torch
from gffx.camera import Camera
from gffx.dashboard import connect
from .camera_demo import dots,BLUE,ORANGE,Figure
from .reporter import Reporter


def solve(*,subject='ring'):
    dtype=torch.float64
    t=torch.linspace(0,6.1,32,dtype=dtype)
    points=torch.stack((.6*torch.cos(t),.4*torch.sin(t),-2+.2*torch.sin(t*2)),axis=-1)
    geometry=None
    if subject=='bunny':
        from .classic_demo import load_asset
        asset=load_asset('bunny');vertices=torch.tensor(asset['vertices'],dtype=dtype)
        lo=vertices.amin(0);hi=vertices.amax(0);vertices=(vertices-(lo+hi)/2)*1.3/(hi-lo).max()
        vertices[:,2]-=2.5;geometry=(vertices,asset['faces'])
        points=vertices[torch.linspace(0,len(vertices)-1,32).long()]
    elif subject!='ring':raise ValueError('Unknown calibration subject')
    target=torch.tensor([90.,110.,80.,60.],dtype=dtype)
    params=torch.tensor([60.,80.,60.,80.],dtype=dtype,requires_grad=True)
    def camera(p):return Camera.from_intrinsics(*(p[i:i+1] for i in range(4)),120,160,.1,10.)
    desired=camera(target).project(points).pixels.detach()
    initial=camera(params).project(points).pixels.detach()
    optimizer=torch.optim.Adam([params],lr=2.)
    history=[]
    for step in range(301):
        optimizer.zero_grad();pixels=camera(params).project(points).pixels
        loss=(pixels-desired).square().mean()
        if step%5==0:history.append((step,float(loss.detach()),params.detach().tolist()))
        if step==300:break
        loss.backward();optimizer.step()
    return dict(points=points,desired=desired,initial=initial,final=pixels.detach(),history=history,
                target=target.tolist(),recovered=params.detach().tolist(),geometry=geometry)


def fit_frames(data):
    """Saved landmark residuals over the recognizable target mesh, not a pose fit."""
    frames=[];background=Figure()
    if data['geometry'] is not None:
        vertices,faces=data['geometry'];target=data['target']
        pixels=Camera.from_intrinsics(*target,120,160,.1,10.).project(vertices).pixels.tolist()
        edges={tuple(sorted((f[i],f[(i+1)%3]))) for f in faces for i in range(3)}
        for a,b in edges:
            x,y=pixels[a];u,v=pixels[b];background.line(34+x,30+y,34+u,30+v,(160,180,195))
    for step,loss,params in data['history']:
        predicted=Camera.from_intrinsics(*params,120,160,.1,10.).project(data['points']).pixels.tolist()
        figure=Figure();figure.data=background.data.copy()
        for (x,y),(u,v) in zip(data['desired'].tolist(),predicted):
            figure.line(34+x,30+y,34+u,30+v,ORANGE)
            for dx in range(-1,2):
                for dy in range(-1,2):
                    figure.pixel(34+x+dx,30+y+dy,BLUE);figure.pixel(34+u+dx,30+v+dy,ORANGE)
        frames.append((step,figure.value()))
    return frames


def derivatives():
    rows=[]
    pixels=torch.tensor([[120.,40.]],dtype=torch.float64);depth=torch.tensor([2.],dtype=torch.float64)
    def value(fx):return Camera.from_intrinsics(fx,110.,80.,60.,120,160,.1,10.).unproject_points(pixels,depth,kind='view')[0,0]
    for focal in range(40,141,5):
        fx=torch.tensor([float(focal)],dtype=torch.float64,requires_grad=True)
        analytic=torch.autograd.grad(value(fx),fx)[0].item();h=1e-4
        numeric=((value(fx.detach()+h)-value(fx.detach()-h))/(2*h)).item()
        rows.append((focal,analytic,numeric,abs(analytic-numeric)))
    return rows


def publish(address,run):
    data=solve(subject='bunny');curve=derivatives();reporter=Reporter(address,run)
    provenance={'revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                'dirty':bool(subprocess.check_output(['git','status','--porcelain'],text=True))}
    if not reporter.send('start',kind='demo',provenance=provenance):raise RuntimeError(reporter.error)
    artifacts=[]
    with connect(address,run=run,name='finite synthetic camera fit',wait_acks=True) as dash:
        replay=dash.queue('camera-fit/replay','image',recent_window=61,snapshot_every=5)
        for step,image in fit_frames(data):replay.push(image,step=step)
        artifacts.append(dict(kind='image',path='camera-fit/replay',label='Recover the bunny camera · residual arrows shrink'))
        def artifact(kind,name,value):
            path='camera-fit/'+name;getattr(dash,kind)(path,value,step=0)
            artifacts.append(dict(kind=kind,path=path,label=name))
        artifact('text','fit-explained','Fit four intrinsic parameters (fx, fy, cx, cy) from 32 landmarks on the bundled bunny and target pixels. Geometry and pose are fixed. Blue is the target; orange is the estimate; arrows show the residual, not a force. The saved replay shows every fifth optimization step, synchronized with the fit curves. Before and after images bracket 300 CPU steps. This is not real-data calibration, a full-pose fit or the gffit engine. Bunny attribution remains in demo_assets/README.md.')
        for name,predicted in [('before',data['initial']),('after',data['final'])]:
            artifact('image',name,dots(torch.cat((data['desired'],predicted)).tolist(),colors=[BLUE]*32+[ORANGE]*32))
        for step,loss,values in data['history']:
            dash.scalar('camera-fit/loss',loss,step=step)
            for name,v in zip(('fx','fy','cx','cy'),values):dash.scalar('camera-fit/parameters/'+name,v,step=step)
        artifacts.extend([dict(kind='plot',path='camera-fit/loss',label='mean squared pixel error'),dict(kind='plot',path='camera-fit/parameters',label='intrinsics by optimization step')])
        artifact('record','recovered-parameters',{'target':data['target'],'recovered':data['recovered'],'order':['fx','fy','cx','cy'],'steps':300})
        artifact('text','derivative-explained','Sweep focal length from 40 to 140 pixels. Compare the world-X derivative of unprojection with respect to fx: autodiff versus a small central finite difference. The two curves should overlap; the error curve quantifies the difference. The horizontal step here means focal length, not optimization iteration. Formal gradchecks additionally cover pose, depth and all intrinsics.')
        for focal,a,n,e in curve:
            dash.scalar('camera-fit/derivative/autodiff',a,step=focal);dash.scalar('camera-fit/derivative/finite-difference',n,step=focal)
            dash.scalar('camera-fit/derivative-error',e,step=focal)
        artifacts.extend([dict(kind='plot',path='camera-fit/derivative',label='inverse derivative comparison'),dict(kind='plot',path='camera-fit/derivative-error',label='absolute derivative error')])
    if not reporter.send('demo',page='camera/fit-demo',title='Synthetic camera fitting & derivatives',artifacts=artifacts):raise RuntimeError(reporter.error)
    if not reporter.send('finish',exit_code=0):raise RuntimeError(reporter.error)
    return data


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--address',required=True);parser.add_argument('--run',required=True)
    args=parser.parse_args();publish(args.address,args.run)
