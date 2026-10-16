"""Diagnostic remapping of known Gauss12x12 curved rows; not production code.

Only complete pairs of native curved sides are accepted. Ordinary rows must not
enter this helper. Reconstruct affine radial coordinates AND the full measure,
retain their autograd graph, then change variables with its complete Jacobian.
"""
N=[.0092196828766403782,.047941371814762601,.11504866290284765,.20634102285669126,.31608425050090994,.43738329574426554,.5626167042557344,.68391574949909006,.79365897714330869,.88495133709715235,.95205862818523745,.99078031712335957]
W=[.023587668193256011,.053469662997659442,.080039164271673055,.10158371336153282,.11674626826917732,.12457352290670134,.12457352290670134,.11674626826917732,.10158371336153282,.080039164271673055,.053469662997659442,.023587668193256011]
def rule(tensor):return tensor(N),tensor(W)
def remap(samples,power):
    import torch
    if type(power)!=int or power not in (2,4,8) or samples.dtype!=torch.float64 or samples.ndim!=2 or samples.shape[1]!=4 or len(samples) not in (288,576):raise ValueError('qualified complete Gauss12 curved pairs and power required')
    if not bool(torch.isfinite(samples).all()) or bool((samples[:,3]<0).any()):raise ValueError('finite nonnegative rows required')
    if power==2:return samples.clone()
    nodes,weights=rule(samples.new_tensor);blocks=samples.reshape(-1,2,12,12,4);answer=[]
    for block in blocks:
        for side,rows in enumerate(block):
            u=nodes**2 if side else 1-(1-nodes)**2;du=2*nodes if side else 2*(1-nodes)
            measure=rows[:,:,3]/(weights*du);old=torch.cat((rows[:,:,:3],measure[:,:,None]),-1)
            slope=(old[:,-1]-old[:,0])/(u[-1]-u[0]);origin=old[:,0]-u[0]*slope
            residual=(old-(origin[:,None]+u[None,:,None]*slope[:,None])).detach().abs()
            if bool((residual>5e-11*(1+old.detach().abs().amax(dim=1,keepdim=True))).any()):raise ValueError('curved radial rows must be affine after removing Jacobian')
            new_u=nodes**power if side else 1-(1-nodes)**power;new_du=power*(nodes if side else 1-nodes)**(power-1)
            result=origin[:,None]+new_u[None,:,None]*slope[:,None]
            result=torch.cat((result[:,:,:3],(result[:,:,3]*weights*new_du)[:,:,None]),-1)
            if not bool(torch.isfinite(result).all()) or bool((result[:,:,3]<0).any()):raise ValueError('invalid transformed measure')
            answer.append(result.reshape(-1,4))
    return torch.cat(answer)
