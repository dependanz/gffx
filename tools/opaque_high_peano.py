"""All Gauss5 Peano kernels through ten, from exact polynomial identities.

K_m(t)=sum_{x_i>t} w_i (x_i-t)^(m-1)/(m-1)!-(1-t)^m/m!.
Outward power-to-Bernstein conversion includes both sides of every node.
No sampled extrema or smoothness across material knots is assumed.
"""
import math
from functools import lru_cache
from fractions import Fraction as Q
from tools.opaque_shading_intervals import I
from tools.opaque_peano import gauss5

@lru_cache(maxsize=60)
def _power(order,data):
    nodes=[(I(a,b),I(c,d)) for a,b,c,d in data];row=[]
    for k in range(order):
        moment=sum((w*x**(order-1-k) for x,w in nodes),I(0))
        row.append((-1)**k*(moment/math.factorial(k)/math.factorial(order-1-k)-I(Q(1,math.factorial(k)*math.factorial(order-k)))))
    row.append(I(Q(-(-1)**order,math.factorial(order))))
    return tuple((v.lo,v.hi) for v in row)

def _bernstein(lo,hi,coeff):
    order=len(coeff)-1;h=hi-lo
    # Each old power started with I(1), then multiplied in ascending order.
    # These local prefixes perform exactly the same rounded operations once.
    # No interval reassociation, persistent cache, or changed polynomial terms.
    lp=[I(1)];hp=[I(1)]
    for _ in range(order):lp.append(lp[-1]*lo);hp.append(hp[-1]*h)
    p=[sum((coeff[k]*math.comb(k,j)*lp[k-j] for k in range(j,order+1)),I(0))*hp[j] for j in range(order+1)]
    return [sum((p[j]*Q(math.comb(i,j),math.comb(order,j)) for j in range(i+1)),I(0)) for i in range(order+1)]

def kernel_range(position,order):
    if type(order)!=int or not 1<=order<=10:raise ValueError('kernel order1..10 required')
    position=I(position)
    if position.lo<0 or position.hi>1:raise ValueError('normalized kernel domain required')
    nodes,weights=gauss5();edges=[I(0),*nodes,I(1)];result=[]
    for segment,(a,b) in enumerate(zip(edges,edges[1:])):
        lo=max(position.lo,a.lo);hi=min(position.hi,b.hi)
        if lo>hi:continue
        key=tuple((x.lo,x.hi,w.lo,w.hi) for x,w in zip(nodes[segment:],weights[segment:]));coeff=[I(*v) for v in _power(order,key)]
        result.extend(_bernstein(I(lo),I(hi),coeff))
    return I(min(v.lo for v in result),max(v.hi for v in result))


@lru_cache(maxsize=10)
def _mass(order,data):
    nodes=[I(a,b) for a,b,_,_ in data];weights=[I(c,d) for _,_,c,d in data];edges=[I(0),*nodes,I(1)];total=I(0)
    for segment,(a,b) in enumerate(zip(edges,edges[1:])):
        key=tuple((x.lo,x.hi,w.lo,w.hi) for x,w in zip(nodes[segment:],weights[segment:]));coeff=[I(*v) for v in _power(order,key)];rows=[_bernstein(a,b,coeff)]
        for _ in range(4):
            children=[]
            for row in rows:
                levels=[row]
                while len(levels[-1])>1:levels.append([(x+y)/2 for x,y in zip(levels[-1],levels[-1][1:])])
                children.extend([[v[0] for v in levels],[v[-1] for v in reversed(levels)]])
            rows=children
        total=total+sum((I(max(v.bound for v in row)) for row in rows),I(0))*(b-a)/16
    return total.hi

def high_error(row,domain,mass,*,checkpoint=lambda:None,integrated=False):
    order=row.get('order',10)
    if type(order)!=int or not 2<=order<=10:raise ValueError('qualified derivative order2..10 required')
    domain,mass=map(I,(domain,mass));h=I(domain.hi)-I(domain.lo);d=I(row['derivative'] if 'derivative' in row else row['tenth'])
    if h.lo<=0 or mass.lo<0 or d.lo<0:raise ValueError('positive domain and nonnegative proof bounds required')
    nodes,weights=gauss5();data=tuple((x.lo,x.hi,w.lo,w.hi) for x,w in zip(nodes,weights));ordinary=d*h**(order+1)*I(_mass(order,data));signed=I(0);witnesses=row['jumps']
    if not isinstance(witnesses,list) or len(witnesses)>1022:raise ValueError('finite two-role512node resource required')
    for witness in witnesses:
        checkpoint();p=I(witness['position']);values=list(map(I,witness['values']))
        if p.lo<domain.lo or p.hi>domain.hi or len(values)!=order or type(witness['required'])!=bool:raise ValueError('complete signed witness through selected order required')
        t=(p-I(domain.lo))/h;t=I(max(I(0).lo,t.lo),min(I(1).hi,t.hi));term=I(0)
        for j,v in enumerate(values):
            if v.zero:continue
            term=term+v*h**(j+1)*kernel_range(t,j+1)
        if not witness['required']:term=I(min(I(0).lo,term.lo),max(I(0).hi,term.hi))
        term=term*mass
        if integrated and 'transverse_integral' in witness:
            supplied=I(witness['transverse_integral']);lo=max(term.lo,supplied.lo);hi=min(term.hi,supplied.hi)
            if lo>hi:raise ValueError('disjoint integrated jump evidence')
            term=I(lo,hi)
        signed=signed+term
    return signed+I(ordinary.hi.copy_negate(),ordinary.hi)*mass

def high_integral(field,box,axis_bounds,*,checkpoint=lambda:None,integrated_axis=1,error_components=None,integrated_bounds=None):
    from tools.opaque_shading_intervals import Jet
    if type(integrated_axis)!=int or integrated_axis not in [0,1] or len(box)!=2:raise ValueError('two-dimensional fixed tensor identity required')
    box=tuple(map(I,box));h=[I(v.hi)-I(v.lo) for v in box]
    if any(v.lo<=0 for v in h):raise ValueError('positive box required')
    nodes,weights=gauss5();accum=None
    for x,w in zip(nodes,weights):
        for y,v in zip(nodes,weights):
            checkpoint();values=field(Jet(I(box[0].lo)+h[0]*x),Jet(I(box[1].lo)+h[1]*y))
            if not isinstance(values,(list,tuple)) or not 1<=len(values)<=4 or any(not isinstance(z,Jet) or z.order or not z[0,1].zero for z in values):raise ValueError('fixed-geometry point jets required')
            if accum is None:accum=[I(0)]*len(values)
            if len(values)!=len(accum):raise ValueError('fixed channel count required')
            accum=[a+z[0,0]*w*v*h[0]*h[1] for a,z in zip(accum,values)]
    errors=[[I(0)]*len(accum) for _ in [0,1]];inner=1-integrated_axis
    for y,w in zip(nodes,weights):
        line=list(box);line[integrated_axis]=I(box[integrated_axis].lo)+h[integrated_axis]*y;checkpoint();rows=axis_bounds(tuple(line),inner)
        if len(rows)!=len(accum):raise ValueError('fixed channel count required')
        for k,row in enumerate(rows):errors[inner][k]=errors[inner][k]+high_error(row,box[inner],h[integrated_axis]*w,checkpoint=checkpoint)
    checkpoint();rows=(integrated_bounds or axis_bounds)(box,integrated_axis)
    if len(rows)!=len(accum):raise ValueError('fixed channel count required')
    for k,row in enumerate(rows):errors[integrated_axis][k]=high_error(row,box[integrated_axis],h[inner],checkpoint=checkpoint,integrated=True)
    if error_components:error_components(errors)
    return [v-errors[0][k]-errors[1][k] for k,v in enumerate(accum)]
