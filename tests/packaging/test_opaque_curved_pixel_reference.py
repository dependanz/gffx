"""Independent outward full-pixel curved control,without importing the renderer."""
import json,os
from decimal import Decimal,localcontext
from pathlib import Path
from fractions import Fraction as Q
from tools.opaque_curved_reference import CurvedTriangle,perspective_numerator,integrate_curved,radial_homogeneous_fields
from tools.opaque_shading_intervals import I

def test_complete_curved_pixel_reference(record_property):
    # Exact rational interpretation of binary fixture inputs and projection.
    world=[[-.008,.008,-2],[.008,.008,-2],[-.008,-.008,-2]]
    normals=[[1,0,-.005],[1.001,0,.007],[.999,0,.001]]
    view=[[-Q(x) for x in p] for p in world]
    tri=[(Q(.5)+100*Q(p[0])/2,Q(.5)-100*Q(p[1])/2) for p in world]
    raw_normals=normals
    # Independently compose the scene's per-corner normalization and its first
    # raw-normal-z derivative before constructing the perspective numerator.
    normals=[];motion=[]
    with localcontext() as ctx:
        ctx.prec=90
        for raw in raw_normals:
            n=list(map(Decimal.from_float,map(float,raw)));length=sum(q*q for q in n).sqrt()
            normals.append([float(q/length) for q in n])
            motion.append([float((Decimal(i==2)-q*n[2]/(length*length))/length) for i,q in enumerate(n)])
    iw=[Q(1,2)]*3
    coeff=perspective_numerator(tri,inverse_w=iw,normals=normals,views=view,normal_motion=motion)
    patch=CurvedTriangle(tri,coeff)
    paths=[[[q,d] for q,d in zip(n,m)] for n,m in zip(normals,motion)]
    fields=radial_homogeneous_fields(patch,tri,inverse_w=iw,fields={'n':paths,'v':view})
    def field(t,s):
        f=fields(t,s);n=f['n'];v=f['v'];dot=lambda a,b:sum((x*y for x,y in zip(a,b)),t.lift(0))
        nn=dot(n,n).sqrt();nv=dot(n,v)/(nn*dot(v,v).sqrt());nl=n[0]/nn
        return [nl/((nv+nl)*(2*I(float.fromhex('0x1.921fb54442d18p+1'))))]
    result=integrate_curved(patch,field,radial=True)
    row=dict(result=result,world=world,raw_normals=raw_normals,normals=normals,normal_motion=motion,normalization='Independent90-digit per-corner normalization and first raw-normal-z chain rule,rounded once to binary64 reference inputs; interval bounds concern that explicit packed-input model; existing normalization acceptance remains separate.',scope='Complete pixel RGB and normal-z tangent for the finite fully opaque single triangle,black uncovered background.')
    p=Path(os.environ['GFFX_CURVED_PIXEL_REFERENCE']);assert not p.exists();p.write_text(json.dumps(row,indent=2,default=str),encoding='utf-8');record_property('reference',json.dumps(row,default=str))
