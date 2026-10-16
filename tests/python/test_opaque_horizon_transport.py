"""View/light horizon transport with an independent closed-form integral."""
import json,math,os,time
from fractions import Fraction as Q
from decimal import Decimal as D,localcontext
import torch,pytest
from tools.opaque_shading_intervals import I

@pytest.mark.parametrize('kind',['view','light'])
def test_partitioned_horizon_pixel_matches_closed_integral_and_boundary_flux(kind,record_property):
    from tools.opaque_horizon_reference import horizon_pixel
    from gffx.torch.material import PrincipledBudget,principled
    from gffx.torch.render import opaque_quadrature
    device=os.environ.get('GFFX_KNOT_DEVICE','cpu');start=time.perf_counter();t=lambda x:torch.tensor(x,dtype=torch.float64,device=device)
    reference=horizon_pixel(kind,Q(1,2));assert all(q.width<=D('2e-12') for q in reference['value']+reference['derivative'])
    budget=PrincipledBudget.bundled('ggx-smith-loss-'+device+'-f64-v1',device=device,dtype=torch.float64)
    if device=='cuda':torch.cuda.reset_peak_memory_stats();baseline=torch.cuda.memory_allocated()
    results={}
    for split in (False,True):
        theta=t(0.).requires_grad_();edge=.5+theta;vertices=t([[-2,-2,.2],[6,-2,.2],[-2,6,.2]]);faces=torch.tensor([[0,1,2]],dtype=torch.int32,device=device)
        cuts=(vertices[faces.long(),0]-edge)[...,None].contiguous() if split else vertices.new_empty((1,3,0))
        owners,q=opaque_quadrature(vertices,faces,cuts,1,1);nodes=q[0][owners[0]>=0];values=[];gradients=[]
        for i in range(0,len(nodes),4):
            from tools.material_loss_cost import check_deadline
            check_deadline(start+120);chunk=nodes[i:i+4];x=(chunk[:,:3]*vertices[:,0]).sum(-1);s=x-edge;variable=torch.stack([torch.ones_like(s),torch.zeros_like(s),s],-1);normal=t([[0,0,1]]).expand(len(s),3).contiguous();fixed=normal
            view=variable if kind=='view' else fixed;directions=fixed if kind=='view' else variable
            # Matched diagonal pairs avoid creating cross-point lights.
            diagonal=[]
            for j in range(len(s)):
                rgb=principled(normal[j:j+1].contiguous(),view[j:j+1].contiguous(),directions[j:j+1].contiguous(),albedo=t([0,0,0]),roughness=t(1.),f0=t([1,1,1]),budget=budget)[0,0]
                cosine=(normal[j]*directions[j]/directions[j].norm()).sum().clamp_min(0);diagonal.append(rgb*cosine)
            rgb=torch.stack(diagonal);value=(rgb*chunk[:,3,None]).sum(0);values.append(value.detach().cpu().tolist());gradients.append([torch.autograd.grad(value[c],theta,retain_graph=True)[0].item() for c in range(3)])
        results['partitioned' if split else 'unpartitioned']=dict(value=[math.fsum(q[c] for q in values) for c in range(3)],derivative=[math.fsum(q[c] for q in gradients) for c in range(3)],samples=len(nodes))
    for key in ('value','derivative'):
        for got,bound in zip(results['partitioned'][key],reference[key]):
            with localcontext() as ctx:
                ctx.prec=90;v=D.from_float(got);assert max(bound.lo-v,v-bound.hi,D(0))<=D('1e-10')+D('1e-9')*abs((bound.lo+bound.hi)/2)
    if kind=='view':
        assert results['unpartitioned']['derivative'][0]>0 and results['partitioned']['derivative'][0]<0
    result=dict(kind=kind,device=device,reference=reference,native=results,seconds=time.perf_counter()-start,scope='Full Principled roughness1,F0=1,albedo0. Explicit linear horizon cut through native quadrature. Unpartitioned candidate remains a negative control; general production horizon partitioning is not qualified.')
    if device=='cuda':torch.cuda.synchronize();result['cuda_incremental_bytes']=torch.cuda.max_memory_allocated()-baseline;assert torch.cuda.max_memory_allocated()<67108864
    record_property('horizon_transport',json.dumps(result,default=str))
