"""Research-only material-following volume plus relative-boundary transport.

For affine source-triangle velocity w and fragment boundary velocity v,
I'=integral(material F' + F div w)+boundary integral(F (v-w).n).
A30-point volume and12 edge points charge42 slots per flat fragment. All
geometry here is exact rational diagnostic input, not production admission.
"""
from fractions import Fraction as Q


def nodes(theta,triangle,velocity,parts,*,profile=(5,6,4)):
    if profile not in ((5,6,4),(6,6,3),(5,7,3)):raise ValueError("unqualified transport rule")
    outer,inner,edge_order=profile
    import torch,numpy as np
    if theta.dtype!=torch.float64 or theta.numel()!=1 or not theta.requires_grad or not 1<=len(parts)<=800//(outer*inner+3*edge_order) or theta.detach().item()!=0:raise ValueError('bounded float64 zero-anchor first-motion diagnostic required')
    def rows(x):
        if len(x)!=3 or any(len(r)<2 for r in x):raise ValueError('three two-dimensional corners required')
        return [[Q(v) for v in r[:2]] for r in x]
    tri=rows(triangle);vel=rows(velocity);sub=lambda a,b:[x-y for x,y in zip(a,b)];cross=lambda a,b:a[0]*b[1]-a[1]*b[0]
    u,v=sub(tri[1],tri[0]),sub(tri[2],tri[0]);du,dv=sub(vel[1],vel[0]),sub(vel[2],vel[0]);det=cross(u,v)
    if not det:raise ValueError('nondegenerate source triangle required')
    divergence=(cross(du,v)+cross(u,dv))/det
    def beta(x):
        q=sub(x,tri[0]);b=cross(q,v)/det;c=cross(u,q)/det;return [1-b-c,b,c]
    def point(a,b,z):return [x+(y-x)*z for x,y in zip(a,b)]
    rules={}
    for order in set(profile):
        n,w=np.polynomial.legendre.leggauss(order);rules[order]=[(Q(float((a+1)/2)),Q(float(b/2))) for a,b in zip(n,w)]
    base=[];slopes=[]
    for part in parts:
        a,b,c=rows(part['triangle']);va,vb,vc=rows(part['motion']);area=cross(sub(b,a),sub(c,a))
        if area<=0:raise ValueError('positive fragment orientation required')
        for x,wx in rules[outer]:
            for y,wy in rules[inner]:
                pos=[aa+x*(bb-aa)+(1-x)*y*(cc-aa) for aa,bb,cc in zip(a,b,c)];weight=area*(1-x)*wx*wy
                base.append([*beta(pos),weight]);slopes.append(weight*divergence)
        for e0,e1,d0,d1 in ((a,b,va,vb),(b,c,vb,vc),(c,a,vc,va)):
            edge=sub(e1,e0)
            for z,w in rules[edge_order]:
                pos=point(e0,e1,z);bb=beta(pos);actual=point(d0,d1,z);flow=[sum(bb[k]*vel[k][j] for k in range(3)) for j in range(2)];flux=cross(sub(actual,flow),edge)*w
                base.append([*bb,Q(0)]);slopes.append(flux)
    q=theta.new_tensor([[float(x) for x in r] for r in base]);slope=theta.new_tensor([float(x) for x in slopes]);return torch.cat((q[:,:3],(q[:,3]+theta*slope)[:,None]),-1)
