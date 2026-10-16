"""Outward exact-form primitives of a finite piecewise angular log cubic.

For m=0..max_order-1 (at most5) cache M_m(q)=integral_0^q t^m f(t)dt at every knot.
On a log branch, integrate exp((m+1)z) times its cubic by four exact
integration-by-parts terms. The horizon branch is linear and includes zero.
P_n(q;a)=sum_j binom(n-1,j)q^(n-1-j)(-1)^j[M_j(q)-M_j(a)]/(n-1)!.
No continuity of f is assumed; every branch's separate integral is retained.
This is an experimental arithmetic building block, not a pixel certificate.
"""
import math
from fractions import Fraction as Q
from bisect import bisect_left,bisect_right
from tools.opaque_shading_intervals import I
from tools.opaque_residual_range import ResidualRangeIndex,polynomial

class Primitives:
    def __init__(self,curve,*,max_order=3,checkpoint=lambda:None):
        if type(max_order)!=int or not 1<=max_order<=6:raise ValueError("primitive maximum order1..6 required")
        self.order=max_order
        self.checkpoint=checkpoint;self.index=curve if isinstance(curve,ResidualRangeIndex) else ResidualRangeIndex(curve,checkpoint=checkpoint)
        z=self.index
        if z.horizon is None:raise ValueError('explicit linear horizon branch required')
        self.slope=[(z.first[ch]-z.horizon[ch])/z.cn[0] for ch in range(2)]
        self.coeff=[]
        for i,row in enumerate(z.powers):
            checkpoint();moments=[]
            for m in range(self.order):
                lam=I(m+1);channels=[]
                for co in row:
                    # r'+lambda*r=p in the logarithmic coordinate.
                    r=[I(0)]*4
                    for k in reversed(range(4)):r[k]=(co[k]-(k+1)*r[k+1] if k<3 else co[k])/lam
                    channels.append(tuple(r))
                moments.append(tuple(channels))
            self.coeff.append(tuple(moments))
        self.prefix=[]
        for m in range(self.order):
            rows=[self.branch_moment(-1,z.cn[0],m)]
            for i in range(len(z.cn)-1):
                checkpoint();left=self.branch_moment(i,z.cn[i],m);right=self.branch_moment(i,z.cn[i+1],m)
                rows.append([a+b-c for a,b,c in zip(rows[-1],right,left)])
            self.prefix.append(rows)
    def branch_moment(self,i,q,m):
        z=self.index
        if i<0:return [z.horizon[ch]*q**(m+1)/(m+1)+self.slope[ch]*q**(m+2)/(m+2) for ch in range(2)]
        v=q.ln()-z.x[i];return [q**(m+1)*polynomial(co,v) for co in self.coeff[i][m]]
    def moments(self,q):
        q=I(q)
        if q.lo<0 or q.hi>1:raise ValueError('angular domain0..1 required')
        z=self.index;out=[[] for _ in range(self.order)]
        if q.lo<=z.cn[0].hi:
            part=I(q.lo,min(q.hi,z.cn[0].hi))
            for m in range(self.order):out[m].append(self.branch_moment(-1,part,m))
        first=max(0,bisect_left(z.upper[1:],q.lo));last=min(len(z.cn)-2,bisect_right(z.lower[:-1],q.hi)-1)
        for i in range(first,last+1):
            self.checkpoint();part=I(max(q.lo,z.cn[i].lo),min(q.hi,z.cn[i+1].hi))
            for m in range(self.order):
                at=self.branch_moment(i,part,m);start=self.branch_moment(i,z.cn[i],m)
                out[m].append([a+b-c for a,b,c in zip(self.prefix[m][i],at,start)])
        if any(not row for row in out):raise ValueError('uncovered angular domain')
        return [[I(min(row[ch].lo for row in pieces),max(row[ch].hi for row in pieces)) for ch in range(2)] for pieces in out]
    def integral(self,q,anchor,order,*,_moments=None):
        if type(order)!=int or not 1<=order<=self.order:raise ValueError('primitive order exceeds prepared range')
        q,anchor=I(q),I(anchor)
        if min(q.lo,anchor.lo)<0 or max(q.hi,anchor.hi)>1:raise ValueError('angular domain0..1 required')
        self.checkpoint()
        if q.lo==q.hi==anchor.lo==anchor.hi:return [I(0),I(0)]
        lookup=_moments or self.moments;end=lookup(q);start=lookup(anchor)
        return [sum((math.comb(order-1,j)*(-1)**j*q**(order-1-j)*(end[j][ch]-start[j][ch]) for j in range(order)),I(0))/math.factorial(order-1) for ch in range(2)]

    def residual(self,q,anchor,order,base,*,_moments=None):
        q,anchor=I(q),I(anchor);z=self.index
        if type(order)!=int or not 1<=order<=self.order or type(base)!=int or not -1<=base<len(z.cn)-1:raise ValueError('qualified order and base branch required')
        if min(q.lo,anchor.lo)<0 or max(q.hi,anchor.hi)>1:raise ValueError('angular domain0..1 required')
        self.checkpoint();lo=min(q.lo,anchor.lo);hi=max(q.hi,anchor.hi)
        left=I(0).lo if base==-1 else z.cn[base].hi;right=z.cn[0].lo if base==-1 else z.cn[base+1].lo
        if left<=lo and hi<=right:return [I(0),I(0)]
        actual=self.integral(q,anchor,order,_moments=_moments);end=[self.branch_moment(base,q,j) for j in range(order)];start=[self.branch_moment(base,anchor,j) for j in range(order)]
        extension=[sum((math.comb(order-1,j)*(-1)**j*q**(order-1-j)*(end[j][ch]-start[j][ch]) for j in range(order)),I(0))/math.factorial(order-1) for ch in range(2)]
        natural=[a-b for a,b in zip(actual,extension)]
        # Parameterize t=a+s(q-a). The nonnegative beta weight has integral1/n,
        # so P_n is (q-a)^n/n! times an average of the COMPLETE residual range.
        # This remains valid for reversed/uncertain endpoints and discontinuities.
        delta=z.delta(I(lo,hi),base);factor=(q-anchor)**order/math.factorial(order);out=[]
        for a,d in zip(natural,delta):
            b=d*factor;l=max(a.lo,b.lo);h=min(a.hi,b.hi)
            if l>h:raise ValueError('disjoint primitive residual bounds')
            out.append(I(l,h))
        if q.width:
            # P_n has n-1 continuous derivatives even when f jumps. Taylor's
            # integral remainder uses the complete residual value range; no
            # higher derivative or continuity at a table knot is assumed.
            mid=I(I((Q(q.lo)+Q(q.hi))/2).lo);d=q-mid
            centered=[self.residual(mid,anchor,order-j,base,_moments=_moments) for j in range(order)]
            local=z.delta(q,base)
            for ch in range(2):
                bound=sum((centered[j][ch]*d**j/math.factorial(j) for j in range(order)),I(0))+local[ch]*d**order/math.factorial(order)
                l=max(out[ch].lo,bound.lo);h=min(out[ch].hi,bound.hi)
                if l>h:raise ValueError('disjoint centered primitive evidence')
                out[ch]=I(l,h)
        return out

    def residual_family(self,q,anchor,order,base):
        """Reuse at most three immutable moment vectors within this call only."""
        if type(order)!=int or not 1<=order<=self.order:raise ValueError('qualified prepared primitive order required')
        memo={}
        def lookup(at):
            self.checkpoint();key=(at.lo,at.hi)
            if key not in memo:
                if len(memo)>=3:raise ValueError('primitive moment cache bound exceeded')
                memo[key]=self.moments(at)
            return memo[key]
        return [self.residual(q,anchor,n,base,_moments=lookup) for n in range(1,order+1)]
