"""Outward integrated curvature-jump mass under two-axis monotonicity.

At q(x,y)=k, dy/dt=|qx/qy| hx for normalized x=t. The K3
contribution has coefficient qx**2 P |qx/qy| hx**4. When every x
crosses the knot between the two outer edges, t spans [0,1] and
integral K3=0 (Gauss5 integrates the cubic truncated-power moment).
Only coefficient variation remains. Partial sweeps retain full absolute mass.
All value/slope jumps and their second-derivative product terms remain uniform.
"""
from fractions import Fraction as Q
from bisect import bisect_left,bisect_right
from tools.opaque_shading_intervals import I,Jet
from tools.opaque_peano import kernel_bounds

def integrated_jumps(proof,box,axis):
    c=proof.c;index=proof.index;c.checkpoint();other_axis=1-axis
    args=[Jet(v,order=3) for v in box];args[axis]=Jet.variable(box[axis],order=3)
    values=c.inputs(*args);args2=[Jet(v,order=1) for v in box];args2[other_axis]=Jet.variable(box[other_axis],order=1);outer=c.inputs(*args2)
    h=I(box[axis].hi)-I(box[axis].lo);w=I(box[other_axis].hi)-I(box[other_axis].lo);kernels,l1=kernel_bounds();total=I(0)
    for role,q in enumerate(values[:2]):
        c.checkpoint()
        if q[1,0].zero:continue
        sums=index.jump_sums(q[0,0]);z,s,t=[c.mix([sums[0][j],sums[1][j]]) for j in range(3)]
        if all(v.zero for v in [z,s,t]):continue
        if q[1,0].lo<0<q[1,0].hi:raise ValueError('aggregate angular crossings need certified monotonicity')
        actual=c.mix(index.actual_jet(values[1-role]));p=values[1]*actual*values[3]/(q.lift(c.extension.pi)*q.lift(c.mean))
        a,b,d=[I(p[j,0].bound) for j in range(3)];qx=q[1,0];u=I(qx.bound);uu=2*I(q[2,0].bound)
        lower=z*a*h*kernels[0]+(s*u*a+z*b)*h**2*kernels[1]+(s*(uu*a+2*u*b)+2*z*d)*h**3*kernels[2]
        uniform=t*u*u*a*h**3*kernels[2]*w;qy=outer[role][1,0]
        if qy.lo<=0<=qy.hi or qx.lo<=0<=qx.hi:
            total=total+lower*w+uniform;continue
        ratio=qx/qy
        if ratio.hi<0:ratio=-ratio
        coefficient=qx*qx*p[0,0]*ratio*h**4
        edges=[]
        for edge in (box[other_axis].lo,box[other_axis].hi):
            at=[Jet(v) for v in box];at[other_axis]=Jet(I(edge));edges.append(c.inputs(*at)[role][0,0])
        if qy.hi<0:edges.reverse()
        first=bisect_left(index.upper[:-1],q[0,0].lo);last=bisect_right(index.lower[:-1],q[0,0].hi)
        full_first=max(first,bisect_right(index.lower[:-1],edges[0].hi));full_last=min(last,bisect_left(index.upper[:-1],edges[1].lo))
        def sum_curvature(a,b):
            if a>=b:return I(0)
            return c.mix([I(0,max(I(0).hi,(index.jump_prefix[b][ch][2]-index.jump_prefix[a][ch][2]).hi)) for ch in range(2)])
        if full_first<full_last:
            full=sum_curvature(full_first,full_last);partial=sum_curvature(first,full_first)+sum_curvature(full_last,last)
        else:full=I(0);partial=t
        center=I(I((Q(coefficient.lo)+Q(coefficient.hi))/2).lo);variation=I((coefficient-center).bound)
        transported=(full*variation+partial*I(coefficient.bound))*l1[2]
        total=total+lower*w+I(0,min(uniform.hi,transported.hi))
    return I(0,total.hi)
