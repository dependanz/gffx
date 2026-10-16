"""One actual resource knot: value, moving-view tangent and independent boundary."""
import json,math,time
from decimal import Decimal as D,localcontext
from fractions import Fraction as Q
import torch
from tools.opaque_shading_intervals import I,Jet,integrate_triangle,Unresolved
from tools.opaque_principled_reference import PreparedPrincipled
from tools.opaque_loss_extension import bounded_cosines
from tools.opaque_knot_transport import MovingLossExtension,material_transport,finish_transport
from tools.principled_material_oracle import material_pair

def test_moving_view_across_resource_knot_has_bounded_pixel_derivative(record_property):
    from gffx.torch.material import PrincipledBudget,principled
    from tools.material_loss_cost import check_deadline
    start=time.perf_counter();budget=PrincipledBudget.bundled('ggx-smith-loss-cpu-f64-v1',device='cpu',dtype=torch.float64)
    data={k:v.tolist() for k,v in budget._data.items()};curve=PreparedPrincipled(data,1.);extension=MovingLossExtension(curve)
    k=min(range(2,len(curve.cn)-2),key=lambda i:abs(float(curve.cn[i].lo)-.8));cn=float(curve.cn[k].lo);z0=cn/math.sqrt(1-cn*cn);dz=.0001
    def inputs(x,y):
        z=x.lift(z0)+dz*(x-Q(1,2))+x.lift(0,tangent=dz)
        return [x.lift(0),x.lift(0),x.lift(1)],[x.lift(1),x.lift(0),z],[x.lift(.2),x.lift(.1),x.lift(1)],[x.lift(v) for v in (.2,.4,.6)],[x.lift(.04)]*3
    def angles(x,y):
        n,v,l,_,_=inputs(x,y);return [v[2]/sum((q*q for q in v),x.lift(0)).sqrt(),l[2]/sum((q*q for q in l),x.lift(0)).sqrt()]
    transport=material_transport(curve,angles,(I(0,1),I(0,1)),[.2,.4,.6],[.04]*3)
    assert any(q['crossings'] for q in transport['roles'])
    def smooth(x,y):return extension.radiance(*inputs(x,y))
    def correction(x,y):return extension.correction_radiance(*inputs(x,y),cosine_bounds=bounded_cosines(angles,x[0,0],y[0,0]))
    def prepare(x,y):extension.indices=[None,None];smooth(Jet(x),Jet(y))
    def check():
        check_deadline(start+120)
        if time.perf_counter()-start>90:raise Unresolved('whole knot reference deadline')
    total=dict(value=[I(0)]*3,derivative=[I(0)]*3);pieces=[];cells=0
    for triangle in (((0,0),(1,0),(0,1)),((1,1),(0,1),(1,0))):
        row=integrate_triangle(smooth,triangle,tolerance='4e-13',max_cells=256-cells,gauss_points=5,correction=correction,prepare=prepare,checkpoint=check);cells+=row['cells'];pieces.append(row)
        for key in total:total[key]=[a+b for a,b in zip(total[key],row[key])]
    result=finish_transport(total,transport['bound']);result.update(cells=cells,pieces=pieces,transport=transport,knot=k,cosine=cn,z0=z0,dz=dz)
    with localcontext() as c:
        c.prec=90
        def edge(x):
            z=D.from_float(z0)+D.from_float(dz)*(D(x)-D('.5'));p=list(map(D.from_float,[0.,0.,1.,1.,0.,0.,.2,.1,1.,.2,.4,.6,1.,.04,.04,.04]));p[5]=z
            cosine=1/(p[6]**2+p[7]**2+1).sqrt();return [v*cosine for v in material_pair(data,p)]
        target=[b-a for a,b in zip(edge(0),edge(1))]
        for actual,wanted in zip(result['derivative'],target):assert actual.lo<=wanted<=actual.hi
    # Independent rectangular Gauss16 with installed material and first VJP.
    import numpy as np
    nodes,weights=np.polynomial.legendre.leggauss(16);values=[];gradients=[]
    for i in range(0,16,4):
        theta=torch.tensor(0.,dtype=torch.float64,requires_grad=True);x=torch.tensor((nodes[i:i+4]+1)/2,dtype=torch.float64);z=z0+dz*(x-.5+theta)
        v=torch.stack((torch.ones_like(z),torch.zeros_like(z),z),-1);n=torch.tensor([[0.,0.,1.]],dtype=torch.float64).expand(4,3).contiguous();light=torch.tensor([[.2,.1,1.]],dtype=torch.float64)
        rgb=principled(n,v,light,albedo=torch.tensor([.2,.4,.6],dtype=torch.float64),roughness=torch.tensor(1.,dtype=torch.float64),f0=torch.tensor([.04]*3,dtype=torch.float64),budget=budget)[:,0,:]/math.sqrt(1.05)
        val=(rgb*torch.tensor(weights[i:i+4]/2,dtype=torch.float64)[:,None]).sum(0);values.append(val.detach().tolist());gradients.append([torch.autograd.grad(val[c],theta,retain_graph=c<2)[0].item() for c in range(3)])
    native={key:[math.fsum(v[c] for v in rows) for c in range(3)] for key,rows in [('value',values),('derivative',gradients)]}
    for key in native:
        for got,enclosure in zip(native[key],result[key]):
            with localcontext() as c:
                c.prec=90;q=D.from_float(got);assert max(enclosure.lo-q,q-enclosure.hi,D(0))<=D('1e-10')+D('1e-9')*abs((enclosure.lo+enclosure.hi)/2)
    record_property('knot_material',json.dumps(dict(reference=result,boundary=target,native=native,seconds=time.perf_counter()-start),default=str))
