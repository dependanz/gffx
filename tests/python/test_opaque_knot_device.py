"""Device material/geometry checks against the saved independent knot pixel."""
import json,hashlib,os,math
from decimal import Decimal as D,localcontext
from pathlib import Path
import torch

def test_moving_geometry_knot_pixel_matches_independent_transport(record_property):
    from gffx.torch.material import PrincipledBudget,principled
    from gffx.torch.render import opaque_quadrature
    from tools.material_loss_cost import check_deadline
    import time
    start=time.perf_counter();path=Path(os.environ['GFFX_KNOT_PIXEL']);assert hashlib.sha256(path.read_bytes()).hexdigest()==os.environ['GFFX_KNOT_PIXEL_SHA'];saved=json.loads(path.read_bytes());ref=saved['reference'];assert saved['producer']['report']['status']=='passed'
    device=os.environ.get('GFFX_KNOT_DEVICE','cpu');t=lambda x:torch.tensor(x,dtype=torch.float64,device=device)
    budget=PrincipledBudget.bundled('ggx-smith-loss-'+device+'-f64-v1',device=device,dtype=torch.float64)
    if device=='cuda':torch.cuda.reset_peak_memory_stats();baseline=torch.cuda.memory_allocated()
    theta=t(0).requires_grad_();base=t([[-2,-2,.2],[6,-2,.2],[-2,6,.2]]);v=base+theta*t([1,0,0]);f=torch.tensor([[0,1,2]],dtype=torch.int32,device=device);cuts=v.new_empty((1,3,0));owners,q=opaque_quadrature(v,f,cuts,1,1)
    nodes=q[0][owners[0]>=0];corners=ref['z0']+ref['dz']*(base[:,0]-.5);values=[];gradients=[]
    for i in range(0,len(nodes),4):
        check_deadline(start+120);piece=nodes[i:i+4];z=(piece[:,:3]*corners).sum(-1);view=torch.stack([torch.ones_like(z),torch.zeros_like(z),z],-1)
        n=t([[0,0,1]]).expand(len(z),3).contiguous();light=t([[.2,.1,1.]])
        rgb=principled(n,view,light,albedo=t([.2,.4,.6]),roughness=t(1.),f0=t([.04]*3),budget=budget)[:,0,:]/math.sqrt(1.05)
        value=(rgb*piece[:,3,None]).sum(0);values.append(value.detach().cpu().tolist());gradients.append([torch.autograd.grad(value[c],theta,retain_graph=True)[0].item() for c in range(3)])
    got={k:[math.fsum(p[c] for p in rows) for c in range(3)] for k,rows in [('value',values),('derivative',gradients)]};comparisons={}
    for key in got:
        comparisons[key]=[]
        for value,bounds in zip(got[key],ref[key]):
            with localcontext() as ctx:
                ctx.prec=90;lo,hi=map(D,bounds.strip('[]').split(','))
                if key=='derivative':lo,hi=-hi,-lo  # Rigid rightward geometry motion transports the field left.
                assert hi-lo<=D('2e-12');v=D.from_float(value);distance=max(lo-v,v-hi,D(0));assert distance<=D('1e-10')+D('1e-9')*abs((lo+hi)/2)
                comparisons[key].append(dict(value=value,distance=str(distance)))
    result=dict(device=device,samples=len(nodes),native=got,comparison=comparisons,seconds=time.perf_counter()-start,scope='One fixed-visibility pixel; rigid projected triangle translation and attached view field. Native quadrature plus installed material in4-point batches. Not full scene or arbitrary geometry qualification.')
    if device=='cuda':torch.cuda.synchronize();result['cuda_incremental_bytes']=torch.cuda.max_memory_allocated()-baseline;assert torch.cuda.max_memory_allocated()<67108864
    record_property('geometry_knot',json.dumps(result))
