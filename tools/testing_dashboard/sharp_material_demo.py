"""Finite polished-surface preview and high-precision budget comparisons."""
import argparse
import ctypes
import hashlib
import math
from pathlib import Path
import subprocess

import torch
from gffx.dashboard import ArrayView,connect
from tools import material_budget_sharp as sharp
from tools import material_budget_reference as direct
from tools import shading_energy_probe as oracle
from .reporter import Reporter


@torch.no_grad()
def surface_plate(roughnesses):
    """Own analytic spheres + 16x32 deterministic VNDF environment quadrature.

    This display is intentionally separate from the high-precision reference.
    Soft Gaussian windows reduce sampling aliasing, not material roughness.
    """
    y,x=torch.meshgrid((torch.arange(360,dtype=torch.float64)+.5-180)/142,
        (torch.arange(320,dtype=torch.float64)+.5-160)/142,indexing='ij')
    mask=x.square()+y.square()<1
    n=torch.stack((x[mask],-y[mask],(1-x[mask].square()-y[mask].square()).sqrt()),-1)
    c=n[:,2];s=(1-c.square()).sqrt();view=n.new_tensor([0.,0.,1.])
    tangent=(view-c[:,None]*n)/s[:,None]
    bitangent=torch.linalg.cross(n,tangent)
    f0=n.new_tensor([.8,.43,.14])
    window1=n.new_tensor([-.45,.65,.8]);window1/=window1.norm()
    window2=n.new_tensor([.7,.1,.7]);window2/=window2.norm()
    u,weights=oracle._gauss_unit(16)
    images=[]
    for roughness in roughnesses:
        alpha=roughness**2;a2=alpha**2
        length=torch.hypot(alpha*s,c);hx=alpha*s/length;hz=c/length
        rv=(a2+(1-a2)*c.square()).sqrt()
        linear=torch.zeros_like(n)
        for radial,weight in zip(u,weights):
            for index in range(32):
                phi=(index+.5)*2*math.pi/32
                dx=radial.sqrt()*math.cos(phi)
                dy=(1-hz)/2*(1-dx.square()).sqrt()+(1+hz)/2*radial.sqrt()*math.sin(phi)
                pz=(1-dx.square()-dy.square()).clamp_min(0).sqrt()
                h=torch.stack((alpha*(-hz*dy+hx*pz),alpha*dx.expand_as(c),hx*dy+hz*pz),-1)
                h/=h.norm(dim=-1,keepdim=True)
                vh=s*h[:,0]+c*h[:,2]
                local=2*vh[:,None]*h-torch.stack((s,torch.zeros_like(s),c),-1)
                ray=local[:,0,None]*tangent+local[:,1,None]*bitangent+local[:,2,None]*n
                nl=local[:,2];positive=torch.where(nl>0,nl,torch.ones_like(nl))
                rl=(a2+(1-a2)*positive.square()).sqrt()
                ratio=positive*(c+rv)/(positive*rv+c*rl)
                fresnel=f0+(1-f0)*(1-vh[:,None]).pow(5)
                light=.04+.04*(ray[:,1,None]+1)+22*torch.exp(((ray@window1)-1)/.016)[:,None]
                light=light+8*torch.exp(((ray@window2)-1)/.04)[:,None]*n.new_tensor([.4,.7,1.])
                linear+=torch.where((nl>0)[:,None],light*fresnel*ratio[:,None],0.)*weight/32
        image=torch.full((360,320,3),0.,dtype=torch.float64)
        image[:]=n.new_tensor([.035,.05,.075])
        image[mask]=(linear/(1+linear)).pow(1/2.2)
        images.append((image*255).round().to(torch.uint8))
    return torch.cat(images,1)


def examples():
    roughnesses=[.01,.05,.1,.199];rows=[]
    for roughness in roughnesses:
        for cosine in (.025,.11,.43,.79):
            c=torch.tensor(cosine,dtype=torch.float64,requires_grad=True)
            r=c.new_tensor(roughness).requires_grad_();f=c.new_tensor(.04)
            value=sharp.directional(c,r,f,resolution=256)
            dc,dr=torch.autograd.grad(value,(c,r))
            with torch.no_grad():legacy=direct.directional(c,r,f,resolution=256)
            rows.append(dict(roughness=roughness,cosine=cosine,energy=value.item(),
                view_gradient=dc.item(),roughness_gradient=dr.item(),direct_grid=legacy.item()))
    return dict(image=surface_plate(roughnesses),roughnesses=roughnesses,rows=rows,
        scope='Analytic spheres with 512 deterministic VNDF environment samples per pixel; not furnace acceptance, public material, mesh renderer or converged image. Scalar budgets use separate high-precision quadrature.')


def publish(address,run,preview=None):
    old=torch.get_num_threads();torch.set_num_threads(1)
    try:data=examples()
    finally:torch.set_num_threads(old)
    pixels=data.pop('image').contiguous()
    raw=ctypes.string_at(pixels.data_ptr(),pixels.numel())
    if preview:
        from .presentation import png_rgb
        Path(preview).write_bytes(png_rgb(1280,360,raw))
    reporter=Reporter(address,run)
    source=b''.join(Path(p).read_bytes() for p in (__file__,sharp.__file__,direct.__file__,oracle.__file__))
    provenance=dict(revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True)),artifact_sha256=hashlib.sha256(source).hexdigest())
    def send(event,**values):
        if not reporter.send(event,**values):raise RuntimeError(reporter.error)
    send('start',kind='demo',provenance=provenance)
    with connect(address,run=run,wait_acks=True) as dash:
        dash.image('sharp/surfaces',ArrayView('uint8',tuple(pixels.shape),raw),step=0)
        dash.text('sharp/read-me','Left to right: roughness 0.01, 0.05, 0.1, 0.199. Gold-colored single-scattering GGX spheres reflect two smooth bright windows. '
            'All share the environment, F0=(0.8,0.43,0.14), camera and x/(1+x), gamma 1/2.2 display transform. No roughness floor or matte fill. '
            'This 512-sample-per-pixel preview is approximate: it is not the principled material or an energy test. '
            'Plots instead use F0=0.04 and four view cosines 0.025,0.11,0.43,0.79 in that order. Each curve fixes roughness. '
            'The direct-grid comparison can miss a narrow highlight; it is a diagnostic, not truth. Inspect exact coordinates in the record. '
            'The new scalar reference remains offline tooling, not the fast table or a public API.',step=0)
        for r in data['roughnesses']:
            for index,row in enumerate(x for x in data['rows'] if x['roughness']==r):
                for key in ('energy','direct_grid','roughness_gradient'):
                    dash.scalar('sharp/'+key+'/r-'+str(r),row[key],step=index)
        dash.record('sharp/samples',data,step=0)
    send('demo',page='shading/sharp-demo',title='Polished surfaces: resolve the reflection',artifacts=[
        dict(kind=k,path='sharp/'+p,label=l) for k,p,l in (
            ('image','surfaces','Polished gold: four roughnesses, one environment'),
            ('text','read-me','Preview and measurement boundaries'),
            ('plot','energy','Integrated reflection: transformed reference'),
            ('plot','direct_grid','Regular-grid comparison: may be unresolved'),
            ('plot','roughness_gradient','How changing roughness changes reflected energy'),
            ('record','samples','Exact samples, view gradients and limitations'))])
    send('finish',exit_code=0)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--address',required=True);parser.add_argument('--run',required=True)
    parser.add_argument('--preview');args=parser.parse_args()
    publish(args.address,args.run,args.preview)
