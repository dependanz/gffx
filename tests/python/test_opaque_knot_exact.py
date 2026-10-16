"""Separate caller reduction order from native moving-knot device parity."""
import torch,math,json,os,hashlib
from pathlib import Path

def test_knot_device_exactness_requires_explicit_caller_reductions(record_property):
    from gffx.torch.material import PrincipledBudget,principled
    from gffx.torch.render import opaque_quadrature
    from gffx.torch._reference_channels import sum3,expand,multiply,_sum_last
    path=Path(os.environ['GFFX_KNOT_PIXEL']);assert hashlib.sha256(path.read_bytes()).hexdigest()==os.environ['GFFX_KNOT_PIXEL_SHA'];ref=json.loads(path.read_bytes())['reference']
    cases={}
    for fixed in (False,True):
        cases[str(fixed)]=[]
        for device in ('cpu','cuda'):
            t=lambda x:torch.tensor(x,dtype=torch.float64,device=device)
            budget=PrincipledBudget.bundled('ggx-smith-loss-'+device+'-f64-v1',device=device,dtype=torch.float64)
            theta=t(0).requires_grad_();base=t([[-2,-2,.2],[6,-2,.2],[-2,6,.2]]);direction=t([1,0,0]);v=base+(expand(theta,base.shape) if fixed else theta)*direction
            f=torch.tensor([[0,1,2]],dtype=torch.int32,device=device);owners,q=opaque_quadrature(v,f,v.new_empty((1,3,0)),1,1);nodes=q[0][owners[0]>=0];corners=ref['z0']+ref['dz']*(base[:,0]-.5);values=[];gradients=[];vertex=[]
            for i in range(0,len(nodes),4):
                piece=nodes[i:i+4];product=piece[:,:3]*corners;z=sum3(product) if fixed else product.sum(-1)
                view=torch.stack([torch.ones_like(z),torch.zeros_like(z),z],-1);n=t([[0,0,1]]).expand(len(z),3).contiguous()
                rgb=principled(n,view,t([[.2,.1,1.]]),albedo=t([.2,.4,.6]),roughness=t(1.),f0=t([.04]*3),budget=budget)[:,0,:]
                rgb=rgb/(torch.full_like(rgb,math.sqrt(1.05)) if fixed else math.sqrt(1.05))
                value=_sum_last(multiply(rgb,piece[:,3,None]).T) if fixed else (rgb*piece[:,3,None]).sum(0)
                values.append(value.detach().cpu().tolist());gradients.append([torch.autograd.grad(value[c],theta,retain_graph=True)[0].item() for c in range(3)])
                vertex.append([torch.autograd.grad(value[c],v,retain_graph=True)[0].cpu().tolist() for c in range(3)])
            cases[str(fixed)].append(dict(value=values,derivative=gradients,vertex_vjp=vertex))
    equal={k:{field:torch.tensor(rows[0][field],dtype=torch.float64).numpy().tobytes()==torch.tensor(rows[1][field],dtype=torch.float64).numpy().tobytes() for field in rows[0]} for k,rows in cases.items()}
    record_property('knot_exact',json.dumps(dict(equal=equal,cases=cases,scope='Caller expression controls use the same installed native and material kernels; preserved original harness remains numerical evidence.')))
    assert all(equal['True'].values())
    for row in cases['True']:
        for key in ('value','derivative'):
            for channel,bound in enumerate(ref[key]):
                lo,hi=map(float,bound.strip('[]').split(','))
                if key=='derivative':lo,hi=-hi,-lo
                value=math.fsum(part[channel] for part in row[key]);assert max(lo-value,value-hi,0)<=1e-10+1e-9*abs((lo+hi)/2)
