"""Closed Reynolds transport identities for clipped affine triangles."""
from fractions import Fraction as Q
import math,pytest,torch


def moment(i,j):
    # Integral of x**i*y**j over (.25,.25),(.5,.25),(.25,.5).
    return sum(Q(math.comb(i,a)*math.comb(j,b)*math.factorial(a)*math.factorial(b),math.factorial(a+b+2))*Q(1,4)**(i+j+2) for a in range(i+1) for b in range(j+1))


@pytest.mark.parametrize('profile',[(5,6,4),(5,7,3),(6,6,3)])
@pytest.mark.parametrize('case',['fixed_clip','moving_clip','affine_expand'])
def test_flat_transport_matches_closed_material_and_boundary_derivatives(case,profile):
    from tools.opaque_flat_transport import nodes
    t=torch.tensor(0.,dtype=torch.float64,requires_grad=True);tri=[[Q(0),Q(0)],[Q(2),Q(0)],[Q(0),Q(2)]];part=[[Q(1,4),Q(1,4)],[Q(1,2),Q(1,4)],[Q(1,4),Q(1,2)]]
    vel=[[Q(1),Q(0)]]*3;pv=[[Q(0),Q(0)]]*3
    if case=='moving_clip':pv=vel
    if case=='affine_expand':vel=[[Q(0),Q(0)],[Q(1),Q(0)],[Q(0),Q(0)]];pv=[[x/2,Q(0)] for x,y in part]
    q=nodes(t,tri,vel,[dict(triangle=part,motion=pv)],profile=profile)
    n=profile[0]*profile[1];assert q.shape==(n+3*profile[2],4) and torch.count_nonzero(q[n:,3])==0
    for i,j in ([(0,0),(1,0),(0,1),(2,2),(3,1)]+([(3,2),(4,3)] if profile[2]==4 else [])):
        value=(q[:,1]**i*q[:,2]**j*q[:,3]).sum();g=torch.autograd.grad(value,t,retain_graph=True)[0];expect=moment(i,j)/2**(i+j)
        slope=-i*moment(i-1,j)/2**(i+j) if case=='fixed_clip' and i else (expect/2 if case=='affine_expand' else Q(0))
        assert abs(value.item()-float(expect))<2e-15 and abs(g.item()-float(slope))<3e-14,(case,i,j,g,slope)


@pytest.mark.parametrize('profile',[(5,6,4),(5,7,3),(6,6,3)])
def test_flat_transport_refuses_degenerate_over_capacity_and_nonfinite(profile):
    from tools.opaque_flat_transport import nodes
    t=torch.tensor(0.,dtype=torch.float64,requires_grad=True);tri=[[0,0],[1,0],[0,1]];vel=[[0,0]]*3;part=dict(triangle=tri,motion=vel)
    for tr,parts in (([[0,0]]*3,[part]),(tri,[part]*(800//(profile[0]*profile[1]+3*profile[2])+1))):
        with pytest.raises(ValueError):nodes(t,tr,vel,parts,profile=profile)

    for value in (float('nan'),float('inf'),.1):
        bad=torch.tensor(value,dtype=torch.float64,requires_grad=True)
        with pytest.raises(ValueError):nodes(bad,tri,vel,[part],profile=profile)
