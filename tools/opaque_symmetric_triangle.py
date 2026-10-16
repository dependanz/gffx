"""Research-only source-pinned42-point symmetric triangle diagnostic.

Numerical data: Xiao/Gimbutas triasymq degree14, under the preserved FreeBSD
license in the qualification source package. No downloaded code is executed.
Not a production rule or an accuracy certificate for nonpolynomial shading.
"""
def table():
    import json,hashlib
    from pathlib import Path
    p=Path('D:/devspace/gffx/qualification-20260913/m1-continuation-20260913/opaque155-20261001/xg14-table1.json')
    assert hashlib.sha256(p.read_bytes()).hexdigest()=='cc798528071984933b6dd68efc42e5fc758a9752a190644bd5ef7a553e6e7ea7'
    return [list(map(float,row)) for row in json.loads(p.read_bytes())['rows']]


def refine(samples):
    import torch,numpy as np
    from tools.opaque_radial_resample import N,W
    if samples.ndim!=2 or samples.shape[1]!=4 or not len(samples) or len(samples)%40 or len(samples)//40*42>800 or samples.dtype!=torch.float64:raise ValueError('bounded five-by-eight flat-fan rows required')
    if not bool(torch.isfinite(samples).all()) or bool((samples[:,3]<0).any()):raise ValueError('finite nonnegative rows required')
    n=samples.new_tensor(N);w=samples.new_tensor(W);xx,ww=np.polynomial.legendre.leggauss(8);m=samples.new_tensor((xx+1)/2);mw=samples.new_tensor(ww/2);grid=samples.reshape(-1,5,8,4)
    dv=(grid[:,0,-1,:3]-grid[:,0,0,:3])/((m[-1]-m[0])*(1-n[0]))
    a=grid[:,0,0,:3]-dv*(1-n[0])*m[0];b=grid[:,-1,0,:3]-dv*(1-n[-1])*m[0];du=(b-a)/(n[-1]-n[0]);origin=a-du*n[0]
    det=grid[:,0,0,3]/((1-n[0])*w[0]*mw[0]);xy=origin[:,None,None,:]+du[:,None,None,:]*n[None,:,None,None]+dv[:,None,None,:]*(1-n[None,:,None,None])*m[None,None,:,None]
    weights=det[:,None,None]*(1-n[None,:,None])*w[None,:,None]*mw[None,None,:];expected=torch.cat((xy,weights[...,None]),-1)
    if bool(((expected-grid).abs()>5e-13*(1+grid.abs())).any()):raise ValueError('affine complete flat-fan rows required')
    rule=samples.new_tensor(table());xyz=origin[:,None,:]+du[:,None,:]*rule[None,:,1,None]+dv[:,None,:]*rule[None,:,2,None];weights=det[:,None]*rule[None,:,3]
    return torch.cat((xyz,weights[...,None]),-1).reshape(-1,4)
