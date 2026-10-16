"""Closed area and transport checks for the diagnostic remapping, independent of renderer."""
import torch,pytest
from tools.opaque_composite_probe import remap
from tools.opaque_curved_transport_map import N,W
@pytest.mark.parametrize('panels,order,power,positive',[(4,4,2,None),(3,9,2,[0,1]),(1,27,2,[0,1]),(2,13,2,[0,1]),(5,5,2,[0,1]),(4,6,2,[0,1]),(3,9,3,[0,1])])
def test_composite_probe_preserves_exact_area_and_horizon_flux(panels,order,power,positive):
    t=lambda x:torch.tensor(x,dtype=torch.float64);theta=t(.3).requires_grad_();an=t(N);aw=t(W);n=t([.0694318442029737124,.330009478207571868,.669990521792428132,.930568155797026288]);w=t([.173927422568726929,.326072577431273071,.326072577431273071,.173927422568726929]);nodes=torch.cat([(j+n)/4 for j in range(4)]);weights=w.repeat(4)/4;out=[]
    for block in range(2):
        for side in range(2):
            u=nodes**2 if side else 1-(1-nodes)**2;du=2*nodes if side else 2*(1-nodes);span=1-theta if side else theta;rad=u*span+(theta if side else 0)
            x=rad[None,:].expand(12,-1);y=x*an[:,None];beta=torch.stack((1-x-y,x,y),-1);measure=x*span*du*weights*aw[:,None];out.append(torch.cat((beta,measure[:,:,None]),-1).reshape(-1,4))
    grade=([0,.001,.03,1] if panels==3 else None);z=remap(torch.cat(out),panels,order,power,positive,grade);assert abs(z[:,3].sum().item()-1)<1e-13
    g,=torch.autograd.grad(z[:,3].sum(),theta,retain_graph=True);assert abs(g.item())<1e-12
    # Recovered first side is r in[0,theta] regardless of which side is favoured.
    count=16 if positive is None else (panels*order if positive[0]==0 else 32-panels*order)
    area=z[:12*count,3].sum();g,=torch.autograd.grad(area,theta)
    assert abs(area.item()-.045)<1e-13 and abs(g.item()-.3)<1e-12
