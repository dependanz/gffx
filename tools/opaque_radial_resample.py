"""Diagnostic affine radial-row reconstruction; not certified quadrature."""
import numpy as np
N=np.array([.0469100770306680036,.2307653449471584545,.5,.7692346550528415455,.9530899229693319964])
W=np.array([.1184634425280945438,.2393143352496832340,.2844444444444444444,.2393143352496832340,.1184634425280945438])
def resample_rows(samples,order):
    q=np.asarray(samples,dtype=np.float64)
    if type(order) is not int or order not in (5,8,12,20):raise ValueError('bounded diagnostic order required')
    if q.ndim!=2 or q.shape[1]!=4 or len(q)%5 or not len(q) or not np.isfinite(q).all() or (q[:,3]<0).any():raise ValueError('finite nonnegative five-node rows required')
    rows=q.reshape(-1,5,4).copy();rows[:,:,3]/=W
    delta=(rows[:,-1]-rows[:,0])/(N[-1]-N[0]);base=rows[:,0]-N[0]*delta
    residual=np.abs(rows-(base[:,None]+N[None,:,None]*delta[:,None]));scale=1+np.max(np.abs(rows),axis=1)[:,None]
    if (residual>5e-13*scale).any():raise ValueError('rows must be affine in the radial coordinate')
    if order==5:return q.copy()
    nodes,weights=np.polynomial.legendre.leggauss(order);nodes=(nodes+1)/2;weights=weights/2
    new=base[:,None]+nodes[None,:,None]*delta[:,None];new[:,:,3]*=weights
    if not np.isfinite(new).all() or (new[:,:,3]<0).any():raise ValueError('invalid reconstructed row')
    return new.reshape(-1,4)


def mapped_rows(samples,mapping):
    """Diagnostic change of variables, preserving five radial nodes per row."""
    if mapping not in ('square_left','square_right','smooth3','smooth5'):raise ValueError('unknown mapping')
    q=resample_rows(samples,5);rows=q.reshape(-1,5,4).copy();rows[:,:,3]/=W
    delta=(rows[:,-1]-rows[:,0])/(N[-1]-N[0]);base=rows[:,0]-N[0]*delta
    t=N
    if mapping=='square_left':u=t*t;du=2*t
    elif mapping=='square_right':u=1-(1-t)**2;du=2*(1-t)
    elif mapping=='smooth3':u=t*t*(3-2*t);du=6*t*(1-t)
    else:u=t*t*t*(10+t*(-15+6*t));du=30*t*t*(1-t)**2
    new=base[:,None]+u[None,:,None]*delta[:,None];new[:,:,3]*=W*du
    if not np.isfinite(new).all() or (new[:,:,3]<0).any():raise ValueError('invalid mapped row')
    return new.reshape(-1,4)


def horizon_rows(samples):
    """Diagnostic side inference; native production already knows its side."""
    rows=resample_rows(samples,5).reshape(-1,5,4);result=[]
    for row in rows:
        m=row[:,3]/W;delta=(m[-1]-m[0])/(N[-1]-N[0]);base=m[0]-N[0]*delta;eps=5e-13*max(abs(m))
        if abs(delta)<=eps:result.append(row.copy())
        elif delta>0:result.append(mapped_rows(row,'square_right' if abs(base)<=eps else 'square_left'))
        else:raise ValueError('unexpected decreasing radial measure')
    return np.concatenate(result)
