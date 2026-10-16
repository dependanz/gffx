"""Conditional implicit-curve jets and exact weighted Gauss5 kernel moments.

Caller must establish a root branch inside the supplied rectangle. This helper
bounds derivatives on that branch; it does not establish existence by sampling.
The implicit denominator is certified over the entire rectangle. Weighted Taylor
remainders retain a bound over the complete possible integration support.
"""
import math
from fractions import Fraction as Q
from tools.opaque_shading_intervals import I,Jet
from tools.opaque_peano import gauss5
from tools.opaque_high_peano import kernel_range

def curve_jet(equation,x,y,*,order=3,checkpoint=lambda:None):
    if type(order)!=int or not 1<=order<=4:raise ValueError('qualified implicit order1..4 required')
    x,y=map(I,(x,y));checkpoint();den=equation(Jet(x),Jet(y,tangent=1))[0,1]
    if den.lo<=0<=den.hi:raise ValueError('implicit denominator must exclude zero over the full rectangle')
    coeff={(0,0):y}
    for n in range(1,order+1):
        checkpoint();value=equation(Jet.variable(x,order=n),Jet(order=n,coeff=coeff))
        coeff[n,0]=-value[n,0]/den
    return Jet(order=order,coeff=coeff)

def moment(a,b,center,order,power):
    a,b,center=map(I,(a,b,center))
    if type(order)!=int or not 1<=order<=10 or type(power)!=int or not 0<=power<=3 or min(a.lo,b.lo)<0 or max(a.hi,b.hi)>1 or a.lo>b.hi:raise ValueError('bounded normalized weighted moment required')
    def truncated(x,m):
        left=x-a;right=x-b;left=I(max(I(0).lo,left.lo),max(I(0).hi,left.hi));right=I(max(I(0).lo,right.lo),max(I(0).hi,right.hi))
        return sum((math.comb(power,k)*(x-center)**(power-k)*(-1)**k*(left**(m+k)-right**(m+k))/(m+k) for k in range(power+1)),I(0))
    nodes,weights=gauss5()
    return sum((w*truncated(x,order)/math.factorial(order-1) for x,w in zip(nodes,weights)),I(0))-truncated(I(1),order+1)/math.factorial(order)

def weighted_taylor(coeff,remainder,center,a,b,order):
    coeff=list(map(I,coeff));remainder,center,a,b=map(I,(remainder,center,a,b))
    if not 1<=len(coeff)<=4 or remainder.lo<0 or center.width or not a.hi<=center.lo<=b.lo:raise ValueError('certified common center and nonnegative Taylor remainder required')
    value=sum((c*moment(a,b,center,order,k) for k,c in enumerate(coeff)),I(0))
    support=I(a.lo,b.hi);radius=I((support-center).bound)
    error=remainder*radius**len(coeff)*I(kernel_range(support,order).bound)*(I(b.hi)-I(a.lo))
    return value+I(error.hi.copy_negate(),error.hi)

def clipped_root(evaluate,domain,knot,*,checkpoint=lambda:None):
    from tools.opaque_transverse_peano import clamped_crossing
    domain,knot=map(I,(domain,knot))
    if domain.width<=0:raise ValueError('positive root domain required')
    at=domain
    for _ in range(3):
        checkpoint()
        if not at.width:return at
        derivative=evaluate(Jet.variable(at,order=1))[1,0]
        if derivative.lo<=0<=derivative.hi:raise ValueError('strict monotonicity required')
        left=evaluate(Jet(I(at.lo)))[0,0];right=evaluate(Jet(I(at.hi)))[0,0]
        if (left.lo>=knot.hi if derivative.lo>0 else left.hi<=knot.lo):return I(at.lo)
        if (right.hi<=knot.lo if derivative.lo>0 else right.lo>=knot.hi):return I(at.hi)
        middle=evaluate(Jet(I((Q(at.lo)+Q(at.hi))/2)))[0,0]
        at=clamped_crossing(at,left,right,middle,derivative,knot)
    return at

def supports(a,b,count):
    a,b=map(I,(a,b))
    if type(count)!=int or count not in (2,4) or a.lo<0 or b.hi>1 or a.lo>b.hi:raise ValueError('two or four fixed normalized supports required')
    rows=[]
    for j in range(count):
        lo,hi=I(Q(j,count)).lo,I(Q(j+1,count)).hi
        def clip(v):return I(max(lo,min(hi,v.lo)),max(lo,min(hi,v.hi)))
        rows.append((clip(a),clip(b)))
    return rows
