"""Finite, dependency-free synthetic mesh/clip producer; not the gffx animation API."""
import argparse
import math
import struct
import subprocess

from gffx.dashboard import ArrayView, connect
from .reporter import Reporter

FRAMES = 24


def geometry(planar, frame):
    phase=2*math.pi*frame/FRAMES
    vertices=[];faces=[];colors=[]
    if planar:
        for y in range(9):
            for x in range(9):
                u=(x-4)/4;v=(y-4)/4
                vertices.append((u+.16*math.sin(phase+v*math.pi)*(1-u*u),v,0.))
                colors.append((.2+.5*x/8,.35+.4*y/8,.85))
        for y in range(8):
            for x in range(8):
                a=y*9+x;faces.extend(((a,a+1,a+10),(a,a+10,a+9)))
    else:
        for i in range(16):
            u=2*math.pi*i/16
            for j in range(12):
                v=2*math.pi*j/12
                r=.28*(1+.28*math.sin(phase+3*u))
                vertices.append(((.75+r*math.cos(v))*math.cos(u),
                                 (.75+r*math.cos(v))*math.sin(u),r*math.sin(v)+.12*math.sin(phase+2*u)))
                colors.append((.25+.65*i/15,.4+.35*j/11,.85-.5*i/15))
        for i in range(16):
            for j in range(12):
                a=i*12+j;b=((i+1)%16)*12+j;c=((i+1)%16)*12+(j+1)%12;d=i*12+(j+1)%12
                faces.extend(((a,b,c),(a,c,d)))
    return vertices,faces,colors


def array(rows,dtype='float32'):
    flat=[x for row in rows for x in row]
    return ArrayView(dtype,(len(rows),len(rows[0])),struct.pack('<'+str(len(flat))+('I' if dtype=='uint32' else 'f'),*flat))


def planar_image(vertices,faces):
    """Front-on XY wireframe, fixed scale and framing for every clip frame."""
    width,height=240,180;data=bytearray([245,247,250])*(width*height)
    def line(a,b,color):
        x0,y0=a;x1,y1=b;n=max(abs(x1-x0),abs(y1-y0),1)
        for t in range(n+1):
            x=round(x0+(x1-x0)*t/n);y=round(y0+(y1-y0)*t/n)
            if 0<=x<width and 0<=y<height:data[(y*width+x)*3:(y*width+x)*3+3]=bytes(color)
    line((20,90),(220,90),(185,195,210));line((120,10),(120,170),(185,195,210))
    points=[(round(120+x*68),round(90-y*68)) for x,y,_ in vertices]
    edges={tuple(sorted((face[i],face[(i+1)%3]))) for face in faces for i in range(3)}
    rest,_,_=geometry(True,0)
    baseline=[(round(120+x*68),round(90-y*68)) for x,y,_ in rest]
    for a,b in sorted(edges):line(baseline[a],baseline[b],(190,195,205))
    for a,b in sorted(edges):line(points[a],points[b],(35,105,190))
    return ArrayView('uint8',(height,width,3),bytes(data))


def publish(address,run):
    reporter=Reporter(address,run)
    provenance={'revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                'dirty':bool(subprocess.check_output(['git','status','--porcelain'],text=True))}
    if not reporter.send('start',kind='demo',provenance=provenance):raise RuntimeError(reporter.error)
    artifacts=[]
    with connect(address,run=run,name='finite mesh and animation gallery',wait_acks=True) as dash:
        def note(name,text):
            dash.text('gallery/'+name,text,step=0);artifacts.append(dict(kind='text',path='gallery/'+name,label=name))
        def mesh(name,planar):
            vertices,faces,colors=geometry(planar,0)
            dash.mesh('gallery/'+name,array(vertices),array(faces,'uint32'),colors=array(colors),step=0)
            artifacts.append(dict(kind='mesh',path='gallery/'+name,label=name))
        note('2D-mesh-explained','A 9 by 9 planar vertex grid joined into 128 triangles. Left is the flat XY wireframe (X right, Y up); right is the same mesh in the 3D viewer with Z exactly zero. Drag to inspect, zoom, or enable wireframe. A 2D mesh is a connected surface, not just a bitmap; the bitmap is its fixed front view. Synthetic demonstration, not a conformance test.')
        vertices,faces,_=geometry(True,0)
        dash.image('gallery/mesh-2d-front',planar_image(vertices,faces),step=0)
        artifacts.append(dict(kind='image',path='gallery/mesh-2d-front',label='2D mesh · fixed XY view'))
        mesh('mesh-2d',True)
        note('3D-mesh-explained','A closed ring-shaped mesh with 192 vertices and 384 triangles. Vertex colours reveal its surface structure. Drag to orbit, pinch or scroll to zoom, and toggle wireframe to inspect connectivity. This uses the shared WebGPU display, not gffx rasterizer acceptance or a loaded external asset.')
        mesh('mesh-3d',False)
        note('2D-animation-explained','24 saved frames deform the planar grid sideways while preserving its triangles and keeping Z at zero. Grey is the rest frame; blue is the current deformation. Use Play, Pause, previous/next or the frame slider; playback runs once and stops. Speed is a requested upper limit: slow frame delivery can reduce it. Coordinates and framing stay fixed. This procedural clip demonstrates history playback, not the planned gffx animation operators.')
        q2=dash.queue('gallery/animation-2d','image',recent_window=FRAMES,snapshot_every=1)
        for frame in range(FRAMES):
            vertices,faces,_=geometry(True,frame);q2.push(planar_image(vertices,faces),step=frame)
        artifacts.append(dict(kind='image',path='gallery/animation-2d',label='2D animation · planar deformation'))
        note('3D-animation-explained','24 saved frames make the ring breathe and bend; vertex positions change but topology stays fixed. Blue-to-orange colors show Euclidean displacement from frame zero, scaled by 0.5 world units for display (not stress or strain). Compare with the static ring above. Orbit while paused or playing to inspect the motion. Both clips start paused and stop when you leave or hide the page. Each tiny, finite clip explicitly saves all 24 frames for restart replay; ordinary live-queue defaults remain unchanged. No background producer keeps running.')
        q3=dash.queue('gallery/animation-3d','mesh',recent_window=FRAMES,snapshot_every=1)
        for frame in range(FRAMES):
            vertices,faces,colors=geometry(False,frame)
            rest,_,_=geometry(False,0)
            displacement=[min(1.,math.sqrt(sum((a-b)**2 for a,b in zip(p,q)))/.5) for p,q in zip(vertices,rest)]
            colors=[(.15+.8*d,.45-.15*d,.9-.75*d) for d in displacement]
            q3.push(array(vertices),array(faces,'uint32'),colors=array(colors),step=frame)
        artifacts.append(dict(kind='mesh',path='gallery/animation-3d',label='3D animation · breathing ring'))
    if not reporter.send('demo',page='geometry/mesh-animation-demo',title='2D and 3D meshes & animations',artifacts=artifacts):raise RuntimeError(reporter.error)
    if not reporter.send('finish',exit_code=0):raise RuntimeError(reporter.error)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--address',required=True);parser.add_argument('--run',required=True)
    args=parser.parse_args();publish(args.address,args.run)
