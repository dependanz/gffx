"""Bounded cubic loss resource; experimental scalar CPU float64, not public API.

Roughness uses cubic log loss; angle uses cubic loss in log cosine so its mean
has an analytic integral. Ordered Bezier controls bound both Fresnel endpoints.
The old power-segment LossResource remains available as a failure control.
"""
from dataclasses import dataclass
import math
import torch

from tools.material_budget_reference import _resolution, _validate
from tools.material_loss_resource import LossResource, _tensor, _freeze
from tools.shading_energy_probe import _gauss_unit


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
    return _tensor(rows).reshape(-1,4)


@dataclass(frozen=True,init=False)
class SmoothLossResource(LossResource):
    _logs: tuple
    _slopes: tuple
    _hlogs: tuple
    _hslopes: tuple
    _weights: tuple

    def __init__(self,roughness_nodes,cosine_nodes,losses):
        super().__init__(roughness_nodes,cosine_nodes,losses)
        rn,cn,v=map(_tensor,(self.roughness_nodes,self.cosine_nodes,self.losses))
        logs=v[:,1:].log();hlogs=v[:,0,0].log()
        values=dict(_logs=logs,_slopes=bounded_slopes(rn.log(),logs,ordered_pairs=True),
                    _hlogs=hlogs,_hslopes=bounded_slopes(rn.log(),hlogs),
                    _weights=bernstein_weights(cn[2:].log()-cn[1:-1].log()))
        for name,value in values.items():
            object.__setattr__(self,name,_freeze(value.tolist()))

    @classmethod
    def from_resource(cls,resource):
        return cls(*map(_tensor,(resource.roughness_nodes,resource.cosine_nodes,resource.losses)))

    def _at_roughness(self,r):
        _validate(roughness=r)
        rn,cn=map(_tensor,(self.roughness_nodes,self.cosine_nodes))
        if not bool((r>=rn[0])&(r<=rn[-1])):
            raise ValueError('roughness outside this resource coverage')
        i=min(int(torch.searchsorted(rn,r,right=True))-1,len(rn)-2)
        x=rn[i:i+2].log();u=(r.log()-x[0])/(x[1]-x[0])
        logs=_tensor(self._logs[i:i+2]);slopes=_tensor(self._slopes[i:i+2])
        pairs=_bezier(controls(x,logs,slopes)[0],u).exp()
        horizon=_bezier(controls(x,_tensor(self._hlogs[i:i+2]),_tensor(self._hslopes[i:i+2]))[0],u).exp()
        return cn,pairs,torch.stack((horizon,horizon*0))

    def _curve(self,r):
        cn,values,horizon=self._at_roughness(r)
        x=cn[1:].log()
        return cn,values,horizon,controls(x,values,bounded_slopes(x,values,ordered_pairs=True))

    def evaluate(self,cosine,roughness,f0):
        _validate(cosine=cosine,f0=f0)
        cn,values,horizon,b=self._curve(roughness)
        if len(cn)==2 or bool(cosine<cn[1]):
            u=cosine/cn[1]
            pair=(1-u)*horizon+u*values[0]
        else:
            i=min(int(torch.searchsorted(cn,cosine,right=True))-1,len(cn)-2)
            u=(cosine.log()-cn[i].log())/(cn[i+1].log()-cn[i].log())
            pair=_bezier(b[i-1],u)
        width=cn[2:].log()-cn[1:-1].log()
        mean=cn[1].square()*(horizon+2*values[0])/3
        mean=mean+(2*cn[2:,None,None].square()*width[:,None,None]*b*_tensor(self._weights).reshape(-1,4)[...,None]).sum(dim=(0,1))
        if not bool((torch.isfinite(pair)&(pair>=0)&(pair<=1)).all()
                    and (torch.isfinite(mean)&(mean>0)&(mean<=1)).all()
                    and pair[0]>=pair[1] and mean[0]>=mean[1]):
            raise ValueError('bounded curve arithmetic is not representable')
        loss=(1-f0)*pair[0]+f0*pair[1]
        average=(1-f0)*mean[0]+f0*mean[1]
        return dict(loss=loss,energy=1-loss,mean_loss=average,mean_energy=1-average)

    def integrate_for_test(self,roughness,f0,*,order=32):
        _validate(f0=f0);_resolution(order,maximum=64)
        cn,values,horizon,b=self._curve(roughness)
        u,w=_gauss_unit(order)
        first_curve=(1-u[:,None])*horizon+u[:,None]*values[0]
        first=(2*cn[1].square()*u[:,None]*first_curve*w[:,None]).sum(0)
        width=cn[2:].log()-cn[1:-1].log()
        cosine=cn[1:-1,None]*torch.exp(width[:,None]*u)
        basis=torch.stack(((1-u)**3,3*u*(1-u)**2,3*u*u*(1-u),u**3),dim=0)
        curve=torch.einsum('ikf,kq->iqf',b,basis)
        pair=first+(2*cosine[...,None].square()*width[:,None,None]*curve*w[None,:,None]).sum(dim=(0,1))
        return (1-f0)*pair[0]+f0*pair[1]
