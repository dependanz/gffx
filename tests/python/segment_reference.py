"""Independent test-only finite-segment visibility; no gffx raster/shadow calls."""
import torch

@torch.no_grad()
def visibility(points,sources,triangles,block=256):
    if any(t.dtype!=torch.float64 for t in (points,sources,triangles)):
        raise TypeError('Reference requires float64 geometry')
    a=triangles[:,0];e1=triangles[:,1]-a;e2=triangles[:,2]-a
    columns=[]
    for begin in range(0,len(points),block):
        p=points[begin:begin+block];tv=p[:,None]-a
        q=torch.linalg.cross(tv,e1[None]);numerator=(e2[None]*q).sum(-1)
        values=[]
        for source in sources:
            direction=source-p
            cross=torch.linalg.cross(direction[:,None],e2[None])
            determinant=(e1[None]*cross).sum(-1)
            valid=determinant.abs()>1e-14
            inverse=1/torch.where(valid,determinant,torch.ones_like(determinant))
            u=(tv*cross).sum(-1)*inverse
            v=(direction[:,None]*q).sum(-1)*inverse
            t=numerator*inverse
            hit=valid&(u>=-1e-12)&(v>=-1e-12)&(u+v<=1+1e-12)&(t>1e-8)&(t<1-1e-8)
            values.append((~hit.any(-1)).to(points.dtype))
        columns.append(torch.stack(values,-1))
    return torch.cat(columns,0) if columns else points.new_empty((0,len(sources)))

def rectangle_sources(center,u,v,count):
    # Work in authored precision before converting to float64, matching the
    # actual sample locations without depending on gffx's light sampler.
    return torch.stack([center+((i+.5)/count-.5)*u+((j+.5)/count-.5)*v for i in range(count) for j in range(count)]).to(torch.float64)
