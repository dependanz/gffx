"""Analytical area, first moments and moving-boundary controls for diagnostic maps."""
import pytest
@pytest.mark.parametrize('power',[2,4,8])
def test_mapped_curved_rows_preserve_exact_area_moments_and_motion(power):
    import torch
    from tools.opaque_curved_transport_map import remap,rule
    t=lambda x:torch.tensor(x,dtype=torch.float64);nodes,weights=rule(t);a=t(.375).requires_grad_();rows=[]
    for side in (0,1):
        for s,ws in zip(nodes,weights):
            u=nodes**2 if side else 1-(1-nodes)**2;du=2*nodes if side else 2*(1-nodes);rad=a+(1-a)*u if side else a*u;span=1-a if side else a
            rows.append(torch.stack((1-rad,rad*(1-s),rad*s,ws*weights*du*span*rad),-1))
    original=torch.cat(rows);mapped=remap(original,power)
    if power==2:assert torch.equal(mapped,original)
    area=mapped[:,3].sum();assert abs(area.item()-.5)<1e-13;assert abs(torch.autograd.grad(area,a,retain_graph=True)[0].item())<1e-12
    for k in range(3):
        value=(mapped[:,k]*mapped[:,3]).sum();assert abs(value.item()-1/6)<1e-13;assert abs(torch.autograd.grad(value,a,retain_graph=True)[0].item())<1e-12
    left=mapped[:144,3].sum();assert abs(left.item()-.5*.375**2)<1e-13;assert abs(torch.autograd.grad(left,a)[0].item()-.375)<1e-12

def test_map_rejects_unqualified_shape_and_power():
    import torch
    from tools.opaque_curved_transport_map import remap
    with pytest.raises(ValueError):remap(torch.ones((25,4)),4)
    with pytest.raises(ValueError):remap(torch.ones((288,4)),3)
