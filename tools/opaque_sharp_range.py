"""Experimental branchwise enclosures intersected with the original range proof.

Every finite angular segment is retained, including uncertain boundaries and
horizon. No new table capacity or persistent pair cache. Ordinary derivatives
are within individual branches; callers must still account for all jumps.
"""
from bisect import bisect_left,bisect_right
from tools.opaque_shading_intervals import I,Jet
from tools.opaque_residual_range import ResidualRangeIndex,shift

class SharpRangeIndex(ResidualRangeIndex):
    def individual(self,q,base,maximum):
        if q.order>maximum or any(j for i,j in q.c):raise ValueError('qualified fixed spatial jet order required')
        v=q[0,0];n=len(self.cn)-1
        if type(base)!=int or not -1<=base<n or v.lo<0 or v.hi>1:raise ValueError('front angular range and base required')
        self.checkpoint();rows=[]
        def restrict(lo,hi):return Jet(order=q.order,coeff={**q.c,(0,0):I(lo,hi)})
        if v.lo<=self.cn[0].hi:
            at=restrict(v.lo,min(v.hi,self.cn[0].hi));rows.append([q.lift(0) if base==-1 else self.base_jet(at,-1,ch)-self.base_jet(at,base,ch) for ch in range(2)])
        first=max(0,bisect_left(self.upper,v.lo)-1);last=min(n-1,bisect_right(self.lower,v.hi)-1)
        for i in range(first,last+1):
            self.checkpoint();lo=max(v.lo,self.cn[i].lo);hi=min(v.hi,self.cn[i+1].hi)
            if lo>hi:continue
            at=restrict(lo,hi);channels=[]
            for ch in range(2):
                if i==base:channels.append(q.lift(0));continue
                coeff=self.powers[i][ch]
                if base>=0:coeff=tuple(a-b for a,b in zip(coeff,shift(self.powers[base][ch],self.x[i]-self.x[base])))
                z=at.ln()-q.lift(self.x[i]);value=q.lift(coeff[-1])
                for a in reversed(coeff[:-1]):value=value*z+q.lift(a)
                if base==-1:value=value-self.base_jet(at,-1,ch)
                channels.append(value)
            rows.append(channels)
        if not rows:raise ValueError('uncovered angular domain')
        return [Jet(order=q.order,coeff={(k,0):I(min(row[ch][k,0].lo for row in rows),max(row[ch][k,0].hi for row in rows)) for k in range(q.order+1)}) for ch in range(2)]
    @staticmethod
    def intersection(a,b):
        lo=max(a.lo,b.lo);hi=min(a.hi,b.hi)
        if lo>hi:raise ValueError('disjoint branchwise range evidence')
        return I(lo,hi)
    def delta(self,q,base):
        old=super().delta(q,base);new=self.individual(Jet(I(q)),base,0)
        return [self.intersection(a,b[0,0]) for a,b in zip(old,new)]
    def _delta_jet(self,q,base,maximum):
        if q.order>maximum or any(j for i,j in q.c):raise ValueError("qualified fixed spatial jet order required")
        if not q.order:return [q.lift(v) for v in self.delta(q[0,0],base)]
        old=super()._delta_jet(q,base,maximum);new=self.individual(q,base,maximum)
        return [Jet(order=q.order,coeff={(k,0):self.intersection(a[k,0],b[k,0]) for k in range(q.order+1)}) for a,b in zip(old,new)]
