"""Frozen bounded-cubic coefficients; no I/O or reference-tool dependency."""
import math
import torch

def _axis(v,y):
    return v.reshape((-1,)+(1,)*(y.ndim-1))


def bounded_slopes(x,y,*,ordered_pairs=False):
    """Quadratic nodal estimates limited by a monotone Bezier-control box.

    Every signed slope is <=1.5 times both adjacent secants. Each interval's
    two interior controls therefore stay on their respective sides of its
    midpoint. A shared pair scale additionally preserves F0 endpoint ordering.
    All arithmetic through y remains in autodiff, including active limiters.
    """
    if len(x)==1:
        return y*0
    h=x[1:]-x[:-1]
    d=(y[1:]-y[:-1])/_axis(h,y)
    if len(x)==2:
        m=torch.stack((d[0],d[0]))
    else:
        hp,hn=_axis(h[:-1],y),_axis(h[1:],y)
        interior=(hn*d[:-1]+hp*d[1:])/(hp+hn)
        left=((2*h[0]+h[1])*d[0]-h[0]*d[1])/(h[0]+h[1])
        right=((2*h[-1]+h[-2])*d[-1]-h[-1]*d[-2])/(h[-1]+h[-2])
        raw=torch.cat((left[None],interior,right[None]))
        before=torch.cat((d[:1],d))
        after=torch.cat((d,d[-1:]))
        same=(before.sign()==after.sign())&(before!=0)&(after!=0)
        sign=torch.where(same,before.sign(),torch.zeros_like(before))
        bound=1.5*torch.minimum(before.abs(),after.abs())
        m=sign*torch.minimum(torch.maximum(sign*raw,torch.zeros_like(raw)),bound)
    if ordered_pairs:
        gap=y[...,0]-y[...,1]
        delta=m[...,0]-m[...,1]
        hp=_axis(torch.cat((h[:1]*0,h)),gap)
        hn=_axis(torch.cat((h,h[-1:]*0)),gap)
        risk=torch.where(delta>0,delta*hp,-delta*hn)
        safe=torch.where(risk>0,risk,torch.ones_like(risk))
        scale=torch.where(risk>0,torch.minimum(torch.ones_like(gap),3*gap/safe),torch.ones_like(gap))
        # Leave a rounding margin only on a genuinely active control constraint.
        scale=torch.where(scale<1,scale*(1-8*torch.finfo(y.dtype).eps),scale)
        m=m*scale[...,None]
    if not bool(torch.isfinite(m).all()):
        raise ValueError('cubic slopes are not representable')
    return m


def controls(x,y,m):
    h=_axis(x[1:]-x[:-1],y)
    return torch.stack((y[:-1],y[:-1]+h*m[:-1]/3,y[1:]-h*m[1:]/3,y[1:]),dim=1)


def _bezier(b,u):
    # de Casteljau: convex combinations preserve the control bounds.
    p=(1-u)*b[0]+u*b[1]
    q=(1-u)*b[1]+u*b[2]
    s=(1-u)*b[2]+u*b[3]
    return (1-u)*((1-u)*p+u*q)+u*((1-u)*q+u*s)


def bernstein_weights(widths):
    """Analytic scaled exponential integrals of four Bernstein polynomials.

    Fixed geometry only: no quadrature and no queried parameter is detached.
    Positive series at z<=8 avoids small-z cancellation; a stable exponential
    moment recurrence at z>8 avoids overflowing exp(z). Sixty-four series terms
    suffice at the declared branch bound, checked independently in tests.
    """
    rows=[]
    for width in widths.tolist():
        z=2*width
        if not math.isfinite(z) or z<=0:
            raise ValueError('positive finite angular widths required')
        if z<=8:
            row=[]
            for k in range(4):
                term=total=.25
                for n in range(1,64):
                    term*=z/n*(n+k)/(n+4)
                    total+=term
                row.append(math.exp(-z)*total)
        else:
            e=math.exp(-z)
            j=[-math.expm1(-z)/z]
            for k in range(1,4):
                j.append((k*j[-1]-e)/z)
            row=[j[3],3*(j[2]-j[3]),3*(j[1]-2*j[2]+j[3]),j[0]-3*j[1]+3*j[2]-j[3]]
        rows.append(row)
    return torch.tensor(rows, dtype=torch.float64, device=widths.device).reshape(-1,4)
