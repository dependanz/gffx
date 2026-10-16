"""Small independent Decimal ray/triangle hard-mask oracle; no gffx/Torch import."""
from decimal import Decimal,localcontext,ROUND_CEILING,ROUND_FLOOR
import math


def hard_mask(vertices,faces,*,focal,principal,height,width,camera_z=3.,near=.1,far=10.,checkpoint=lambda:None):
    if not (type(height)==type(width)==int and 0<height<=128 and 0<width<=128 and len(vertices)<=4096 and len(faces)<=8192):raise ValueError('bounded oracle profile')
    if not all(math.isfinite(float(v)) for row in vertices for v in row):raise ValueError('finite geometry required')
    if any(len(v)!=3 for v in vertices) or any(len(f)!=3 or any(type(i)!=int or not 0<=i<len(vertices) for i in f) for f in faces):raise ValueError('triangle geometry')
    if not (math.isfinite(focal) and focal>0 and 0<near<far and all(math.isfinite(q) for q in (*principal,camera_z))):raise ValueError('pinhole inputs')
    def sub(a,b):return tuple(x-y for x,y in zip(a,b))
    def dot(a,b):return sum(x*y for x,y in zip(a,b))
    def cross(a,b):return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
    def dec(x):return Decimal.from_float(float(x))
    mask=[[False]*width for _ in range(height)]
    with localcontext() as ctx:
        ctx.prec=80;zero=Decimal(0);half=Decimal('.5');one=Decimal(1)
        v=[tuple(dec(x) for x in row) for row in vertices];origin=(zero,zero,dec(camera_z));f=dec(focal);cx,cy=map(dec,principal);lo,hi=dec(near),dec(far)
        if any(not lo<origin[2]-p[2]<hi for p in v):raise ValueError('oracle excludes near/far clipping profiles')
        projected=[(cx+f*p[0]/(origin[2]-p[2]),cy-f*p[1]/(origin[2]-p[2])) for p in v]
        rays_x=[(Decimal(x)+half-cx)/f for x in range(width)];rays_y=[-(Decimal(y)+half-cy)/f for y in range(height)]
        for index,face in enumerate(faces):
            if index%32==0:checkpoint()
            a,b,c=(v[i] for i in face);normal=cross(sub(b,a),sub(c,a))
            if dot(normal,sub(origin,a))<=0:continue
            screen=[projected[i] for i in face]
            left=max(0,int((min(q[0] for q in screen)-half).to_integral_value(rounding=ROUND_CEILING)))
            right=min(width-1,int((max(q[0] for q in screen)-half).to_integral_value(rounding=ROUND_FLOOR)))
            top=max(0,int((min(q[1] for q in screen)-half).to_integral_value(rounding=ROUND_CEILING)))
            bottom=min(height-1,int((max(q[1] for q in screen)-half).to_integral_value(rounding=ROUND_FLOOR)))
            numerator=dot(normal,sub(a,origin));edges=[(a,b),(b,c),(c,a)]
            for y in range(top,bottom+1):
                for x in range(left,right+1):
                    if mask[y][x]:continue
                    ray=(rays_x[x],rays_y[y],-one);denominator=dot(normal,ray)
                    if denominator==0:continue
                    t=numerator/denominator
                    if not lo<=t<=hi:continue
                    point=tuple(o+t*d for o,d in zip(origin,ray))
                    if all(dot(normal,cross(sub(end,start),sub(point,start)))>=0 for start,end in edges):mask[y][x]=True
        checkpoint()
    return mask
