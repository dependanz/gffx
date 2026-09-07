"""Finite camera illustrations using the public API, not acceptance evidence.

Run: python -m tools.testing_dashboard.camera_demo --address URL --run UNIQUE_ID
Only this development producer requires PyTorch; gffx.dashboard remains dependency-free.
"""
import argparse
import struct
import subprocess

import torch
from gffx.camera import Camera
from gffx.dashboard import ArrayView, connect
from .reporter import Reporter


def points_array(value):
    value=value.detach().cpu().to(torch.float32)
    flat=value.reshape(-1).tolist()
    return ArrayView('float32',tuple(value.shape),struct.pack('<%df'%len(flat),*flat))


BLUE, ORANGE, INK = (25,100,210), (230,100,30), (45,55,70)
# Tiny in-tree bitmap labels keep this finite producer independent of plotting/Pillow.
FONT = dict(zip('0123456789.XYZAB-', (
    '111101101101111','010110010010111','111001111100111',
    '111001111001111','101101111001001','111100111001111',
    '111100111101111','111001001001001','111101111101111',
    '111101111001111','000000000000010','101101010101101',
    '101101010010010','111001010100111','010101111101101',
    '110101110101110','000000111000000')))


class Figure:
    def __init__(self): self.data=bytearray([245,247,250])*(240*180)
    def pixel(self,x,y,color):
        x,y=int(x),int(y)
        if 0<=x<240 and 0<=y<180:
            index=(y*240+x)*3;self.data[index:index+3]=bytes(color)
    def line(self,x0,y0,x1,y1,color=INK):
        count=max(abs(int(x1-x0)),abs(int(y1-y0)),1)
        for i in range(count+1):self.pixel(x0+(x1-x0)*i/count,y0+(y1-y0)*i/count,color)
    def text(self,x,y,text,color=INK):
        for char in text:
            for i,bit in enumerate(FONT.get(char,'0'*15)):
                if bit=='1':
                    for dy in range(2):
                        for dx in range(2):self.pixel(x+(i%3)*2+dx,y+(i//3)*2+dy,color)
            x+=8
    def value(self): return ArrayView('uint8',(180,240,3),bytes(self.data))


def dots(pixels, *, colors=None):
    """Pixel scatter with explicit x-right/y-down axes and camera colour keys."""
    f=Figure();left,top=34,30
    for x in (0,80,160):
        f.line(left+x,top,left+x,top+120,(215,222,230));f.text(left+x-8,top+126,str(x))
    for y in (0,60,120):
        f.line(left,top+y,left+160,top+y,(215,222,230));f.text(2,top+y-5,str(y))
    f.text(226,153,'X');f.text(4,12,'Y');f.text(55,8,'A',BLUE)
    if colors:f.text(145,8,'B',ORANGE)
    colors=colors or [BLUE]*len(pixels)
    for (x,y),color in zip(pixels,colors):
        if not (0<=x<160 and 0<=y<120):continue
        for dy in range(-2,3):
            for dx in range(-2,3):f.pixel(left+x+dx,top+y+dy,color)
    return f.value()


def depth_figure(depth):
    f=Figure();lo,hi=depth.min().item(),depth.max().item()
    for y,row in enumerate(depth.tolist()):
        for x,value in enumerate(row):
            gray=int((value-lo)/(hi-lo)*255)
            for dy in range(5):
                for dx in range(5):f.pixel(16+x*5+dx,24+y*5+dy,(gray,gray,gray))
    for y in range(120):
        gray=int(y/119*255)
        for x in range(192,208):f.pixel(x,24+y,(gray,gray,gray))
    f.text(180,8,f'{lo:.2f}');f.text(180,152,f'{hi:.2f}')
    f.text(18,152,'32 X 24');return f.value()


def camera_plan():
    """Top view of the documented synthetic poses; directions are -Z, not visibility."""
    f=Figure();f.text(10,8,'X');f.line(24,14,220,14)
    f.text(4,42,'-Z');f.line(16,58,16,162)
    for x,label,color in ((86,'A',BLUE),(126,'B',ORANGE)):
        f.text(x-2,30,label,color);f.line(x,48,x,112,color)
        f.line(x,112,x-5,104,color);f.line(x,112,x+5,104,color)
        f.line(x,48,x-40,140,color);f.line(x,48,x+40,140,color)
    f.text(80,162,'0.5');return f.value()


def publish(address, run):
    reporter=Reporter(address,run)
    provenance={}
    try:
        provenance={'revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                    'dirty':bool(subprocess.check_output(['git','status','--porcelain'],text=True))}
    except (OSError,subprocess.SubprocessError):pass
    if not reporter.send('start',kind='demo',provenance=provenance):raise RuntimeError(reporter.error)
    camera=Camera.from_intrinsics(90.,90.,80.,60.,120,160,.1,10.)
    t=torch.linspace(0,6.283185,32)
    landmarks=torch.stack((.6*torch.cos(t),.4*torch.sin(t),-2+.2*torch.sin(2*t)),axis=-1)
    projection=camera.project(landmarks)
    y,x=torch.meshgrid(torch.arange(24),torch.arange(32),indexing='ij')
    depth=1.+x.to(torch.float32)/64+y.to(torch.float32)/96
    depth_camera=Camera.from_intrinsics(30.,30.,16.,12.,24,32,.1,10.)
    xyz,valid=depth_camera.unproject(depth[None],kind='view')
    pose=torch.eye(4)[None];pose[:,0,3]=.5
    second=Camera.from_intrinsics(90.,90.,80.,60.,120,160,.1,10.,pose=pose)
    rig=Camera.stack([camera,second])
    two_views=rig.project(landmarks[None].expand(2,-1,-1)).pixels
    entries=[]
    with connect(address,run=run,name='finite pinhole camera illustrations',wait_acks=True) as dash:
        def add(kind,name,value):
            path='camera-demo/'+name
            getattr(dash,kind)(path,value,step=0)
            entries.append({'kind':'mesh' if kind=='points' else kind,'path':path,'label':name})
        add('text','landmarks-explained','32 synthetic 3D points form a wavy ring. Drag the point viewer to explore the source shape. The labelled image shows Camera.project pixel locations: X increases right, Y increases down, and A identifies the blue camera. This is an illustration, not an occlusion renderer or a test result.')
        add('points','landmarks-world',points_array(landmarks))
        add('image','landmarks-projected',dots(projection.pixels.tolist()))
        add('text','depth-explained','A 24 by 32 synthetic depth image gets farther away across the image. Brighter means farther: the scale runs from 1.00 to 1.72 arbitrary world units (rounded). Camera.unproject converts pixel centres and depths into the 768 world-space points beside it. The image is enlarged for readability; no sensor data is involved.')
        add('image','depth-image',depth_figure(depth))
        add('points','depth-reconstructed',points_array(xyz[valid]))
        add('text','rig-explained','The same ring is projected by two cameras. A (blue) is the original camera; B (orange) is translated 0.5 world units right, so its projection shifts left. The top-view diagram marks both positions and their -Z viewing directions; its guide rays are schematic, not clipping or visibility evidence. This is a batched camera illustration, not a fitting engine.')
        add('image','rig-two-views',dots(two_views.reshape(-1,2).tolist(),colors=[BLUE]*32+[ORANGE]*32))
        add('points','rig-shared-world',points_array(landmarks))
        add('image','rig-camera-plan',camera_plan())
    if not reporter.send('demo',page='camera/pinhole-demo',title='Pinhole camera illustrations',artifacts=entries):raise RuntimeError(reporter.error)
    if not reporter.send('finish',exit_code=0):raise RuntimeError(reporter.error)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--address',required=True);parser.add_argument('--run',required=True)
    args=parser.parse_args();publish(args.address,args.run)
