"""Experimental local log-loss cells, not a full table or public material API."""
from dataclasses import dataclass
import torch
from tools import material_loss_reference as reference
from tools.material_budget_reference import _validate, _resolution


def _axis(value, name):
    if (not isinstance(value,torch.Tensor) or value.dtype!=torch.float64 or value.device.type!='cpu'
        or value.shape!=(2,) or not bool(torch.isfinite(value).all())
        or not bool((value>0).all()) or not bool((value<=1).all()) or not bool(value[1]>value[0])):
        raise ValueError(name+' needs two increasing positive CPU float64 knots at most one')
    if name=='cosines' and not bool(value[1]<1):raise ValueError('cosine cell must be strictly interior')


@dataclass(frozen=True,init=False)
class LossCell:
    # Plain immutable numbers avoid exposing mutable tensor knot storage.
    roughness: tuple
    cosines: tuple
    losses: tuple

    def __init__(self, roughness, cosines, losses):
        _axis(roughness,'roughness');_axis(cosines,'cosines')
        if (not isinstance(losses,torch.Tensor) or losses.dtype!=torch.float64 or losses.device.type!='cpu'
            or losses.shape!=(2,2,2) or not bool(torch.isfinite(losses).all())
            or not bool(((losses>0)&(losses<=1)).all()) or not bool((losses[:,:,0]>=losses[:,:,1]).all())):
            raise ValueError('loss knots must be positive ordered Fresnel endpoints in [0,1], CPU float64 shape (2,2,2)')
        object.__setattr__(self,'roughness',tuple(roughness.detach().tolist()))
        object.__setattr__(self,'cosines',tuple(cosines.detach().tolist()))
        object.__setattr__(self,'losses',tuple(tuple(tuple(pair) for pair in row) for row in losses.detach().tolist()))

    def evaluate(self, cosine, roughness, f0):
        _validate(cos_view=cosine,roughness=roughness,f0=f0)
        for name,value,axis in (('cosine',cosine,self.cosines),('roughness',roughness,self.roughness)):
            if not bool((value>=axis[0])&(value<=axis[1])):
                raise ValueError(name+' outside explicit experimental cell; no extrapolation')
        rn=torch.tensor(self.roughness,dtype=torch.float64).log()
        cn=torch.tensor(self.cosines,dtype=torch.float64).log()
        x=(roughness.log()-rn[0])/(rn[1]-rn[0])
        y=(cosine.log()-cn[0])/(cn[1]-cn[0])
        weights=torch.stack((1-x,x))[:,None]*torch.stack((1-y,y))[None,:]
        knots=torch.tensor(self.losses,dtype=torch.float64).log()
        pair=(knots*weights[:,:,None]).sum(dim=(0,1)).exp()
        loss=(1-f0)*pair[0]+f0*pair[1]
        return dict(loss=loss,energy=1-loss)


def generate(roughness, cosines, *, resolution=512):
    """Eight explicit scalar endpoint integrations; never called by evaluate."""
    _axis(roughness,'roughness');_axis(cosines,'cosines');_resolution(resolution)
    with torch.no_grad():
        values=torch.stack([torch.stack([torch.stack([
            reference.directional(c,r,torch.tensor(f,dtype=torch.float64),resolution=resolution)
            for f in (0.,1.)]) for c in cosines]) for r in roughness])
    return LossCell(roughness,cosines,values)
