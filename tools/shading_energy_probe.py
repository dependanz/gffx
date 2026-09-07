"""Independent contract diagnostic, NOT an implemented gffx material.

Integrate the PLAN's proposed (1-F)*Lambert + GGX*Smith*Fresnel material in a
unit-radiance white environment. Positive-hemisphere midpoint quadrature uses
mu=cos(theta), so d(omega)=d(mu)d(phi). No image renderer, RNG or external assets.
"""
import argparse
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import torch


@lru_cache(maxsize=128)
def furnace(cos_view, roughness, *, resolution=512, model='candidate', f0=.04):
    """Return a float64 CPU directional-hemispherical reflectance estimate.

    2*resolution azimuth bins and resolution cosine bins; chunking bounds memory.
    Normal view instead uses one azimuth sample and 4*resolution cosine bins.
    The candidate is deliberately not normalized or clamped to make it pass.
    """
    if not (0 < cos_view <= 1 and 0 < roughness <= 1):
        raise ValueError('view cosine and roughness must be in (0, 1]')
    if isinstance(resolution,bool) or not isinstance(resolution,int) or not 16<=resolution<=2048:
        raise ValueError('resolution must be an integer in [16, 2048]')
    if not 0 <= f0 <= 1:
        raise ValueError('f0 must be in [0, 1]')
    if model not in ('candidate','lambert','unit_fresnel_ggx','specular'):
        raise ValueError('unknown diagnostic model')
    # At normal view the integrand is azimuth-invariant. Spend those samples on
    # cosine resolution instead, where the narrow specular peak actually lives.
    normal_view = cos_view == 1
    if normal_view:
        resolution *= 4
    phi = (torch.zeros(1, dtype=torch.float64) if normal_view else
           (torch.arange(2*resolution,dtype=torch.float64)+.5)*(math.pi/resolution))
    vx=math.sqrt(1-cos_view*cos_view);alpha2=roughness**4;total=0.
    for start in range(0,resolution,32):
        mu=(torch.arange(start,min(start+32,resolution),dtype=torch.float64)+.5)/resolution
        nl=mu[:,None];x=(1-nl*nl).sqrt()*phi.cos()[None,:]
        vl=vx*x+cos_view*nl
        length=(2+2*vl).sqrt();nh=(nl+cos_view)/length;vh=(1+vl)/length
        d=alpha2/(math.pi*(nh*nh*(alpha2-1)+1)**2)
        visibility=.5/(nl*math.sqrt(cos_view*cos_view*(1-alpha2)+alpha2)+
                       cos_view*(nl*nl*(1-alpha2)+alpha2).sqrt())
        fresnel=f0+(1-f0)*(1-vh)**5
        if model=='candidate':brdf=(1-fresnel)/math.pi+d*visibility*fresnel
        elif model=='unit_fresnel_ggx':brdf=d*visibility
        elif model=='specular':brdf=d*visibility*fresnel
        else:brdf=torch.ones_like(x)/math.pi
        total+=float((brdf*nl).sum())
    return total*2*math.pi/(resolution*len(phi))


def coupled_diffuse(albedo, e_view, e_light, e_average):
    """Reciprocal Kelemen--Szirmay-Kalos budget coupling, per RGB channel.

    The E arguments MUST be integrals of the same specular BRDF, not local Fresnel.
    Independent diagnostic only: no public broadcasting/device contract implied.
    See https://cg.iit.bme.hu/~szirmay/scook.pdf section 2.2.
    """
    for value in (albedo, e_view, e_light, e_average):
        if not bool(torch.isfinite(value).all() and ((value >= 0) & (value <= 1)).all()):
            raise ValueError('albedo and energy budgets must be finite in [0, 1]')
    zero = e_average == 1
    if bool((zero & ((e_view != 1) | (e_light != 1))).any()):
        raise ValueError('zero average residual requires zero directional residuals')
    denominator = torch.where(zero, torch.ones_like(e_average), 1-e_average)
    value = albedo*(1-e_view)*(1-e_light)/(math.pi*denominator)
    return torch.where(zero, torch.zeros_like(value), value)


@lru_cache(maxsize=4)
def _gauss_unit(resolution):
    # Golub-Welsch for Legendre quadrature, shifted from [-1,1] to [0,1].
    indices = torch.arange(1, resolution, dtype=torch.float64)
    off_diagonal = indices/(4*indices.square()-1).sqrt()
    matrix = torch.diag(off_diagonal, 1)+torch.diag(off_diagonal, -1)
    nodes, vectors = torch.linalg.eigh(matrix)
    return (nodes+1)/2, vectors[0].square()


def directional_albedo(cos_view, roughness, *, f0=.04, resolution=256, radial_rule='uniform'):
    """Independent float64 VNDF quadrature for single-scatter GGX reflectance.

    Heitz 2018 sections 3--4 and Appendix B, deterministic unit-square quadrature
    rather than RNG: https://jcgt.org/published/0007/04/01/paper.pdf.
    Gauss-Legendre radius-squared nodes resolve near-mirror distribution tails;
    azimuth uses midpoint bins. Weight normalization is validated independently.
    Weight is F*G2/G1, with zero contribution for reflected rays below the plane.
    This detached integration is NOT a differentiable runtime implementation.
    """
    if not (0 <= cos_view <= 1 and 0 < roughness <= 1 and 0 <= f0 <= 1):
        raise ValueError('invalid diagnostic cosine, roughness or f0')
    if isinstance(resolution, bool) or not isinstance(resolution, int) or not 16 <= resolution <= 512:
        raise ValueError('VNDF resolution must be an integer in [16, 512]')
    if radial_rule not in ('uniform','tail-dense'):
        raise ValueError("radial_rule must be 'uniform' or 'tail-dense'")
    alpha = roughness**2
    vx = math.sqrt(1-cos_view*cos_view)
    length = math.hypot(alpha*vx, cos_view)
    hx, hz = alpha*vx/length, cos_view/length
    u, weights = _gauss_unit(resolution)
    if radial_rule=='tail-dense':
        weights=weights*2*(1-u)
        u=u*(2-u)
    phi = (torch.arange(2*resolution, dtype=torch.float64)+.5)*(math.pi/resolution)
    disk_x = u[:, None].sqrt()*phi.cos()[None, :]
    disk_y = u[:, None].sqrt()*phi.sin()[None, :]
    mix = (1+hz)/2
    disk_y = (1-mix)*(1-disk_x.square()).sqrt()+mix*disk_y
    # Numerical roundoff guard on the geometric square root, NOT an energy clamp.
    projected_z = (1-disk_x.square()-disk_y.square()).clamp_min(0).sqrt()
    nx = alpha*(-hz*disk_y+hx*projected_z)
    ny = alpha*disk_x
    nz = hx*disk_y+hz*projected_z
    scale = (nx.square()+ny.square()+nz.square()).sqrt()
    nx, nz = nx/scale, nz/scale
    vh = vx*nx+cos_view*nz
    nl = 2*vh*nz-cos_view
    valid = nl > 0
    # Evaluate only positive hemisphere values; excluded samples contribute zero.
    positive_nl = torch.where(valid, nl, torch.ones_like(nl))
    root_v = math.sqrt(alpha*alpha+(1-alpha*alpha)*cos_view*cos_view)
    root_l = (alpha*alpha+(1-alpha*alpha)*positive_nl.square()).sqrt()
    ratio = positive_nl*(cos_view+root_v)/(positive_nl*root_v+cos_view*root_l)
    fresnel = f0+(1-f0)*(1-vh)**5
    return float((torch.where(valid, fresnel*ratio, 0.).mean(dim=1)*weights).sum())


@dataclass(frozen=True)
class EnergyBudget:
    roughness: float
    f0: float
    nodes: torch.Tensor
    energy: torch.Tensor
    average: torch.Tensor

    def at(self, cosine):
        index = torch.searchsorted(self.nodes, cosine).clamp(1, len(self.nodes)-1)
        low, high = self.nodes[index-1], self.nodes[index]
        fraction = (cosine-low)/(high-low)
        return self.energy[index-1]*(1-fraction)+self.energy[index]*fraction


def make_budget(roughness, *, f0=.04, resolution=256, knots=129):
    """Finite diagnostic curve, not an approved shipping LUT or universal bound.

    Quadratically spaced cosines resolve grazing angles. Integrate the piecewise
    linear curve times cosine exactly for its mean; furnace uses a different grid.
    """
    if isinstance(knots, bool) or not isinstance(knots, int) or not 3 <= knots <= 257:
        raise ValueError('knots must be an integer in [3, 257]')
    nodes = torch.linspace(0, 1, knots, dtype=torch.float64).square()
    energy = torch.tensor([directional_albedo(float(c), roughness, f0=f0,
                                            resolution=resolution) for c in nodes], dtype=torch.float64)
    if not bool(torch.isfinite(energy).all() and ((energy >= 0) & (energy <= 1)).all()):
        raise ValueError('invalid integrated specular budget; do not clamp it')
    lo, hi = nodes[:-1], nodes[1:]
    slope = (energy[1:]-energy[:-1])/(hi-lo)
    intercept = energy[:-1]-slope*lo
    average = 2*(slope*(hi**3-lo**3)/3 + intercept*(hi**2-lo**2)/2).sum()
    return EnergyBudget(roughness, f0, nodes, energy, average)


def coupled_furnace(cos_view, budget, *, albedo=1., resolution=2048):
    """Integrate revised material with an independent outgoing-direction grid.

    Specular uses direct hemisphere quadrature, NOT the budget's VNDF estimate.
    Diffuse is azimuth-invariant; its 1-D integral uses the full furnace grid.
    """
    specular = furnace(cos_view, budget.roughness, f0=budget.f0,
                       resolution=resolution, model='specular')
    mu = (torch.arange(resolution, dtype=torch.float64)+.5)/resolution
    diffuse = coupled_diffuse(torch.tensor(albedo, dtype=torch.float64),
                              budget.at(torch.tensor(cos_view, dtype=torch.float64)),
                              budget.at(mu), budget.average)
    return specular+float((diffuse*mu).mean())*2*math.pi


def report():
    rows=[]
    for view in (1.,.5,.1):
        for roughness in (.2,.5,1.):
            energies=[furnace(view,roughness,resolution=n) for n in (256,512,1024)]
            rows.append(dict(cos_view=view,roughness=roughness,energies=energies,
                             refinement_delta=abs(energies[-1]-energies[-2]),
                             excess_above_one=energies[-1]-1))
    return dict(scope='independent proposed-formula diagnostic, not gffx operation evidence',
                dtype='float64',device='cpu',albedo=1.,f0=.04,
                integrand='((1-F)/pi + D*V*F) * NdotL',
                resolution=[256,512,1024],azimuth_multiplier=2,rows=rows)


def revision_report():
    rows = []
    for roughness in (.2, .5, 1.):
        for f0 in (0., .04, .5, 1.):
            budget = make_budget(roughness, f0=f0, resolution=128, knots=65)
            refined = make_budget(roughness, f0=f0, resolution=256, knots=129)
            for view in (.1, .5, 1.):
                for albedo in (0., .5, 1.):
                    coarse = coupled_furnace(view, budget, albedo=albedo, resolution=1024)
                    fine = coupled_furnace(view, refined, albedo=albedo, resolution=2048)
                    rows.append(dict(roughness=roughness, f0=f0, view=view, albedo=albedo,
                                     coarse=coarse, fine=fine, delta=abs(fine-coarse)))
    return dict(scope='independent reciprocal coupled formula; not gffx runtime evidence',
                dtype='float64', device='cpu', rows=rows,
                quadrature=dict(vndf_radial='Gauss-Legendre', vndf_resolution=[128,256],
                                cosine_knots=[65,129], light_resolution=[1024,2048],
                                normal_view='one azimuth sample, 4096/8192 cosine bins'),
                max_refinement_delta=max(row['delta'] for row in rows),
                max_energy=max(row['fine'] for row in rows),
                max_white_error=max(abs(row['fine']-1) for row in rows if row['albedo']==1))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--revised', action='store_true')
    args=parser.parse_args();torch.set_num_threads(1)
    result=revision_report() if args.revised else report();args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
