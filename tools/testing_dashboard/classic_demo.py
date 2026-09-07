"""Offline classic meshes and a finite illustrative compliant-contact ball demo.

This is stdlib developer tooling, not a public gffx physics/autodiff implementation.
"""
import argparse
import json
import math
from pathlib import Path
import subprocess

from gffx.dashboard import connect
from .mesh_animation_demo import array
from .reporter import Reporter

FRAMES=181
FPS=60
RADIUS=.35


def load_asset(name):
    if name not in ('bunny','teapot'):raise ValueError('Unknown bundled demo asset')
    return json.loads((Path(__file__).parent/'demo_assets'/(name+'.json')).read_text(encoding='utf-8'))


def inspection(name):
    """Optional developer inspection through native gffx operations, not JS reimplementation."""
    import torch
    from gffx.torch import mesh
    asset=load_asset(name);v=torch.tensor(asset['vertices'],dtype=torch.float32);f=torch.tensor(asset['faces'],dtype=torch.int32)
    normals,areas,valid=mesh.face_geometry(v,f);vertex_normals=mesh.vertex_normals(v,f)
    return dict(areas=areas.tolist(),face_normals=normals.tolist(),valid=valid.tolist(),
                normal_colors=((vertex_normals+1)/2).tolist())


def simulate(*,substeps=40):
    """1 kg mass, 9.81 m/s² gravity, unilateral spring/damper contact at y=0.

    Penalty displacement models effective ball compression, not penetration of the
    displayed floor. Semi-implicit Euler, fixed dt=1/(60*substeps), no wall clock.
    Only the centroid is integrated: the ellipsoid is a stylized volume-preserving
    display, not a continuum constitutive model or a mesh dynamics solver.
    """
    if isinstance(substeps,bool) or not isinstance(substeps,int) or not 1<=substeps<=1000:
        raise ValueError('substeps must be an integer from 1 to 1000')
    h=2.;v=0.;dt=1/(FPS*substeps);states=[]
    for frame in range(FRAMES):
        compression=max(0.,RADIUS-h)
        states.append(dict(t=frame/FPS,height=h,velocity=v,compression=compression,
                           energy=9.81*h+.5*v*v+.5*1200*compression**2,
                           potential=9.81*h,kinetic=.5*v*v,spring=.5*1200*compression**2,
                           energy_difference=19.62-(9.81*h+.5*v*v+.5*1200*compression**2)))
        if frame==FRAMES-1:break
        for _ in range(substeps):
            compression=max(0.,RADIUS-h)
            force=max(0.,1200*compression-5*v) if compression>0 else 0.
            v+=(-9.81+force)*dt;h+=v*dt
    return states


def scene(state):
    """Y-up ball and fixed wooden-coloured planks; no material/lighting engine claim."""
    h=state['height'];ry=min(RADIUS,h);rx=RADIUS*math.sqrt(RADIUS/ry)
    vertices=[(0.,h+ry,0.)];colors=[(.95,.24,.08)];faces=[]
    for ring in range(1,10):
        theta=math.pi*ring/10
        for j in range(16):
            phi=2*math.pi*j/16
            vertices.append((rx*math.sin(theta)*math.cos(phi),h+ry*math.cos(theta),rx*math.sin(theta)*math.sin(phi)))
            colors.append((.95,.24 if j%4<2 else .72,.08))
    bottom=len(vertices);vertices.append((0.,h-ry,0.));colors.append((.95,.24,.08))
    for j in range(16):
        a=1+j;b=1+(j+1)%16
        faces.append((0,b,a));faces.append((bottom,1+8*16+j,1+8*16+(j+1)%16))
    for ring in range(8):
        for j in range(16):
            a=1+ring*16+j;b=1+ring*16+(j+1)%16;c=b+16;d=a+16
            faces.extend(((a,b,c),(a,c,d)))
    for plank in range(10):
        x=-1.5+.3*plank;a=len(vertices)
        vertices.extend(((x,0.,-1.2),(x+.285,0.,-1.2),(x+.285,0.,1.2),(x,0.,1.2)))
        faces.extend(((a,a+2,a+1),(a,a+3,a+2)))
        colors.extend([(.42+.025*(plank%3),.22+.018*(plank%3),.09)]*4)
    return vertices,faces,colors


def publish(address,run,*,inspect_meshes=False):
    publish_page(address,run,inspect_meshes=inspect_meshes)
    publish_page(address,run+'-ball',ball=True)


def publish_page(address,run,*,ball=False,inspect_meshes=False):
    reporter=Reporter(address,run)
    def send(event,**fields):
        if not reporter.send(event,**fields):raise RuntimeError(reporter.error)
    send('start',kind='demo',provenance={
        'revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'dirty':bool(subprocess.check_output(['git','status','--porcelain'],text=True))})
    with connect(address,run=run,name='offline classics and finite ball contact',wait_acks=True) as dash:
        artifacts=[]
        for name,label,color in ([] if ball else [('bunny','Stanford bunny',(.76,.83,.94)),('teapot','Utah teapot',(.1,.66,.75))]):
            asset=load_asset(name);path='classics/'+name
            note=(label+': '+str(len(asset['vertices']))+' vertices and '+str(len(asset['faces']))+' triangles. '
                  'Drag to orbit; zoom and toggle wireframe to inspect connectivity. Bundled offline; no runtime download. '
                  +asset['license_note']+' Source: '+asset['source_url']+
                  ' Display fixture only, not a watertightness or reconstruction benchmark.')
            dash.text(path+'-explained',note,step=0)
            artifacts.append(dict(kind='text',path=path+'-explained',label=label+' · source and limits'))
            dash.mesh(path,array(asset['vertices']),array(asset['faces'],'uint32'),colors=array([color]*len(asset['vertices'])),step=0)
            if inspect_meshes:dash.record(path+'-inspection',inspection(name),step=0)
            artifacts.append(dict(kind='mesh',path=path,label=label))
        if not ball:
            send('demo',page='geometry/classic-meshes-demo',title='Stanford bunny & Utah teapot',artifacts=artifacts)
            send('finish',exit_code=0)
            return
        dash.text('ball/explained',
            'Press Play, then select 60 fps max for the intended 3-second timing (181 samples at 60 Hz). '
            'Lower settings play in slow motion; actual playback may be slower on your device. '
            'A 1 kg ball falls from a 2 m centre height under gravity (9.81 m/s²), meets a rigid wooden-coloured floor, '
            'compresses and rebounds. Contact uses a 1200 N/m spring and 5 N·s/m damper with 40 substeps per frame. '
            'Squash is a volume-preserving ellipsoid driven by compression, not finite-element soft-body physics. '
            'Wood is a visual appearance, not measured wood stiffness. Plots show centre height (metres) and '
            'gravity + kinetic + contact-spring energy (joules); the horizontal axis is saved frame index, seconds = index / 60. '
            'Energy decreases through contact damping. A finite saved clip, initially paused, no endless producer. '
            'This demonstrates dashboard mesh/history presentation, not gffx physics, differentiability, collision API or real-time conformance.',step=0)
        queue=dash.queue('ball/motion','mesh',recent_window=FRAMES,snapshot_every=1)
        for frame,state in enumerate(simulate()):
            vertices,faces,colors=scene(state)
            queue.push(array(vertices),array(faces,'uint32'),colors=array(colors),step=frame)
            for key in ('potential','kinetic','spring','energy_difference'):
                dash.scalar('ball/parts/'+key,state[key],step=frame)
            if frame%3==0:
                dash.scalar('ball/height',state['height'],step=frame)
                dash.scalar('ball/energy',state['energy'],step=frame)
        send('demo',page='simulation/elastic-ball-demo',title='Elastic ball meets a wooden floor',artifacts=[
            dict(kind='text',path='ball/explained',label='What the ball simulation shows'),
            dict(kind='mesh',path='ball/motion',label='Elastic ball · 3 seconds at 60 fps target'),
            dict(kind='plot',path='ball/height',label='Centre height (m) · frame / 60 = seconds'),
            dict(kind='plot',path='ball/energy',label='Mechanical energy (J) · frame / 60 = seconds')])
    send('finish',exit_code=0)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--address',required=True);parser.add_argument('--run',required=True)
    parser.add_argument('--inspect-meshes',action='store_true')
    args=parser.parse_args();publish(args.address,args.run,inspect_meshes=args.inspect_meshes)
