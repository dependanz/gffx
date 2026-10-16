"""Test-only radial resampling of frozen candidate1 curved rows; never an overlay."""
import torch

def remap(samples,panels,order,power,positive_sides=None,grade=None):
    import numpy as np
    if samples.dtype!=torch.float64 or samples.shape!=(768,4) or (panels*order!=16 if positive_sides is None else not 20<=panels*order<=27) or power not in (1,2,3,4):raise ValueError('explicit two-curve diagnostic only')
    qn=samples.new_tensor([.0694318442029737124,.330009478207571868,.669990521792428132,.930568155797026288]);qw=samples.new_tensor([.173927422568726929,.326072577431273071,.326072577431273071,.173927422568726929])
    oldnodes=torch.cat([(j+qn)/4 for j in range(4)]);oldweights=qw.repeat(4)/4
    n,w=np.polynomial.legendre.leggauss(order);newnodes=torch.cat([(j+samples.new_tensor((n+1)/2))/panels for j in range(panels)]);newweights=samples.new_tensor(w/2).repeat(panels)/panels
    answer=[]
    for bi,block in enumerate(samples.reshape(2,2,12,16,4)):
        for side,rows in enumerate(block):
            if positive_sides is not None:
                pp,oo=(panels,order) if side==positive_sides[bi] else (1,32-panels*order)
                n,w=np.polynomial.legendre.leggauss(oo);newnodes=torch.cat([(j+samples.new_tensor((n+1)/2))/pp for j in range(pp)]);newweights=samples.new_tensor(w/2).repeat(pp)/pp
            if grade is not None and (positive_sides is None or side==positive_sides[bi]):
                assert len(grade)==panels+1 and grade[0]==0 and grade[-1]==1 and all(b>a for a,b in zip(grade,grade[1:]))
                edges=grade if side else [1-a for a in reversed(grade)]
                n,w=np.polynomial.legendre.leggauss(order)
                newnodes=torch.cat([a+(b-a)*samples.new_tensor((n+1)/2) for a,b in zip(edges,edges[1:])]);newweights=torch.cat([(b-a)*samples.new_tensor(w/2) for a,b in zip(edges,edges[1:])])
            u=oldnodes**2 if side else 1-(1-oldnodes)**2;du=2*oldnodes if side else 2*(1-oldnodes)
            data=torch.cat((rows[:,:,:3],(rows[:,:,3]/(oldweights*du))[:,:,None]),-1)
            slope=(data[:,-1]-data[:,0])/(u[-1]-u[0]);origin=data[:,0]-u[0]*slope
            assert torch.all((data-(origin[:,None]+u[None,:,None]*slope[:,None])).detach().abs()<5e-11)
            uu=newnodes**power if side else 1-(1-newnodes)**power;dd=power*(newnodes if side else 1-newnodes)**(power-1)
            z=origin[:,None]+uu[None,:,None]*slope[:,None];answer.append(torch.cat((z[:,:,:3],(z[:,:,3]*newweights*dd)[:,:,None]),-1).reshape(-1,4))
    return torch.cat(answer)
