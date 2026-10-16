"""Research-only tensor refinement of affine flat-fan Gauss5 samples.

Not a native renderer or integration certificate. Closed-moment controls and
actual independent endpoint comparisons qualify each explicit experiment.
"""
def refine(samples,outer,inner,*,source=(5,5)):
    import torch,numpy as np
    from tools.opaque_radial_resample import N,W
    if type(outer)!=int or type(inner)!=int or outer not in (5,6,7,8) or inner not in (5,6,7,8):raise ValueError('explicit diagnostic5-8tensor rule required')
    if source not in ((5,5),(5,8)):raise ValueError('qualified source tensor rule required')
    count=source[0]*source[1]
    if samples.ndim!=2 or samples.shape[1]!=4 or not len(samples) or len(samples)%count or len(samples)//count*outer*inner>800 or samples.dtype!=torch.float64:raise ValueError('bounded float64 flat-fan sample rows required')
    if not bool(torch.isfinite(samples).all()) or bool((samples[:,3]<0).any()):raise ValueError('finite positive measure required')
    n=torch.tensor(N,dtype=samples.dtype,device=samples.device);w=torch.tensor(W,dtype=samples.dtype,device=samples.device)
    grid=samples.reshape(-1,*source,4)
    def one(grid,axis,order):
        old=source[axis-1]
        if old==5:sn,sw=n,w
        else:
            xx,ww=np.polynomial.legendre.leggauss(old);sn=torch.tensor((xx+1)/2,dtype=samples.dtype,device=samples.device);sw=torch.tensor(ww/2,dtype=samples.dtype,device=samples.device)
        shape=[1]*4;shape[axis]=old;weight=sw.reshape(shape);nodes=sn.reshape(shape)
        raw=torch.cat((grid[...,:3],grid[...,3:]/weight),-1)
        delta=(raw.select(axis,old-1)-raw.select(axis,0))/(sn[-1]-sn[0]);base=raw.select(axis,0)-sn[0]*delta
        fitted=base.unsqueeze(axis)+nodes*delta.unsqueeze(axis)
        if bool(((raw-fitted).abs()>5e-13*(1+raw.abs())).any()):raise ValueError('independently affine flat rows required')
        if order==old:return grid
        nodes,weights=np.polynomial.legendre.leggauss(order);shape[axis]=order
        nodes=torch.tensor((nodes+1)/2,dtype=samples.dtype,device=samples.device).reshape(shape);weights=torch.tensor(weights/2,dtype=samples.dtype,device=samples.device).reshape(shape)
        result=base.unsqueeze(axis)+nodes*delta.unsqueeze(axis)
        return torch.cat((result[...,:3],result[...,3:]*weights),-1)
    grid=one(grid,2,inner);grid=one(grid,1,outer);result=grid.reshape(-1,4)
    if not bool(torch.isfinite(result).all()) or bool((result[:,3]<0).any()):raise ValueError('finite refined measure required')
    return result
