"""Order-zero residual envelopes over frozen piecewise log cubics.

Each sparse block encloses polynomial coefficient differences from its first
branch. Two overlapping power-of-two blocks cover any consecutive branch range.
All jumps remain in the hull; no smoothness or crossing-count claim is made.
This alternative never relaxes the Peano checker's eight-segment guard.
"""
import math
from bisect import bisect_left,bisect_right
from tools.opaque_shading_intervals import I,Jet

def shift(coeff,d):
    return tuple(sum((math.comb(j,k)*coeff[j]*d**(j-k) for j in range(k,4)),I(0)) for k in range(4))

def polynomial(coeff,z):
    value=coeff[-1]
    for a in reversed(coeff[:-1]):value=a+z*value
    return value

class ResidualRangeIndex:
    def __init__(self,curve,*,checkpoint=lambda:None):
        self.checkpoint=checkpoint;checkpoint();self.cn=tuple(I(v) for v in curve.cn);self.x=tuple(I(v) for v in curve.x)
        if not 2<=len(self.cn)<=512 or len(self.x)!=len(self.cn) or self.cn[0].lo<=0 or not self.cn[-1].lo==self.cn[-1].hi==1 or any(a.hi>=b.lo for a,b in zip(self.cn,self.cn[1:])) or any(a.hi>=b.lo for a,b in zip(self.x,self.x[1:])):raise ValueError('bounded ordered log resource required')
        n=len(self.cn)-1;self.powers=tuple(tuple(tuple(I(v) for v in curve.power(i,c,self.x[i])) for c in range(2)) for i in range(n))
        if any(len(p)!=4 for row in self.powers for p in row):raise ValueError('two cubic coefficient vectors required')
        self.horizon=tuple(I(v) for v in curve.horizon) if hasattr(curve,'horizon') else None
        self.first=tuple(I(v) for v in curve.values[0]) if self.horizon is not None else None
        self.lower=tuple(v.lo for v in self.cn);self.upper=tuple(v.hi for v in self.cn)
        zero=tuple(I(0) for _ in range(4));self.blocks=[tuple((zero,zero) for _ in range(n))];size=2
        while size<=n:
            half=size//2;previous=self.blocks[-1];rows=[]
            for i in range(n-size+1):
                checkpoint();j=i+half;channels=[];d=self.x[i]-self.x[j]
                for c in range(2):
                    right=shift(previous[j][c],d);reference=shift(self.powers[j][c],d)
                    right=tuple(v+a-b for v,a,b in zip(right,reference,self.powers[i][c]));left=previous[i][c]
                    channels.append(tuple(I(min(a.lo,b.lo),max(a.hi,b.hi)) for a,b in zip(left,right)))
                rows.append(tuple(channels))
            self.blocks.append(tuple(rows));size*=2
        self.last_query_blocks=0
    def _base(self,q,base,c):
        if base==-1:
            if self.horizon is None:raise ValueError('horizon resource unavailable')
            return self.horizon[c]+q*(self.first[c]-self.horizon[c])/self.cn[0]
        return polynomial(self.powers[base][c],q.ln()-self.x[base])
    def delta(self,q,base):
        q=I(q);n=len(self.cn)-1
        if type(base)!=int or not -1<=base<n or q.lo<0 or q.hi>1:raise ValueError('fixed front range and valid base required')
        self.checkpoint();rows=[];self.last_query_blocks=0
        if q.lo<=self.cn[0].hi:
            if self.horizon is None:raise ValueError('horizon resource unavailable')
            v=I(q.lo,min(q.hi,self.cn[0].hi))
            rows.append([I(0) if base==-1 else self._base(v,-1,c)-self._base(v,base,c) for c in range(2)]);self.last_query_blocks+=1
        if q.hi>=self.cn[0].lo:
            first=max(0,bisect_left(self.upper,q.lo)-1);last=min(n-1,bisect_right(self.lower,q.hi)-1)
            if first<=last:
                level=(last-first+1).bit_length()-1;size=1<<level
                for i in sorted({first,last-size+1}):
                    self.checkpoint();v=I(max(q.lo,self.cn[i].lo),min(q.hi,self.cn[i+size].hi));z=v.ln()-self.x[i];channels=[]
                    for c in range(2):
                        coeff=self.blocks[level][i][c]
                        if base>=0:
                            if base!=i:
                                ref=shift(self.powers[base][c],self.x[i]-self.x[base]);coeff=tuple(a+b-d for a,b,d in zip(coeff,self.powers[i][c],ref))
                            value=polynomial(coeff,z)
                        else:value=polynomial(tuple(a+b for a,b in zip(coeff,self.powers[i][c])),z)-self._base(v,-1,c)
                        channels.append(value)
                    rows.append(channels);self.last_query_blocks+=1
        if not rows:raise ValueError('empty angular range')
        return [I(min(row[c].lo for row in rows),max(row[c].hi for row in rows)) for c in range(2)]

    def base_jet(self,q,base,c):
        if base==-1:
            if self.horizon is None:raise ValueError('horizon resource unavailable')
            return q.lift(self.horizon[c])+q/q.lift(self.cn[0])*q.lift(self.first[c]-self.horizon[c])
        z=q.ln()-q.lift(self.x[base]);value=q.lift(self.powers[base][c][-1])
        for a in reversed(self.powers[base][c][:-1]):value=value*z+q.lift(a)
        return value
    def delta_jet(self,q,base):
        return self._delta_jet(q,base,3)
    def high_delta_jet(self,q,base):
        return self._delta_jet(q,base,10)
    def weighted_delta_jet(self,q,base):
        """Enclose q times the residual before hulling branch ranges.

        Restrict only the value to each certified branch block. All spatial
        coefficients remain, including tiny nonzero derivatives. This is an
        ordinary derivative enclosure; knot jumps are accounted separately.
        """
        return self._delta_jet(q,base,3,weighted=True)
    def _delta_jet(self,q,base,maximum,weighted=False):
        if q.order>maximum or any(j for i,j in q.c):raise ValueError('qualified fixed spatial jet order required')
        if q.order==0 and not weighted:return [q.lift(v) for v in self.delta(q[0,0],base)]
        v=q[0,0];n=len(self.cn)-1
        if type(base)!=int or not -1<=base<n or v.lo<0 or v.hi>1:raise ValueError('fixed front range and valid base required')
        self.checkpoint();rows=[]
        def restrict(lo,hi):
            coeff=dict(q.c);coeff[0,0]=I(lo,hi);return Jet(order=q.order,coeff=coeff)
        if v.lo<=self.cn[0].hi:
            at=restrict(v.lo,min(v.hi,self.cn[0].hi))
            row=[q.lift(0) if base==-1 else self.base_jet(at,-1,c)-self.base_jet(at,base,c) for c in range(2)]
            rows.append([at*v for v in row] if weighted else row)
        if v.hi>=self.cn[0].lo:
            first=max(0,bisect_left(self.upper,v.lo)-1);last=min(n-1,bisect_right(self.lower,v.hi)-1)
            if first<=last:
                level=(last-first+1).bit_length()-1;size=1<<level
                for i in sorted({first,last-size+1}):
                    self.checkpoint();at=restrict(max(v.lo,self.cn[i].lo),min(v.hi,self.cn[i+size].hi));z=at.ln()-q.lift(self.x[i]);channels=[]
                    for c in range(2):
                        coeff=self.blocks[level][i][c]
                        if base>=0:
                            if base!=i:
                                ref=shift(self.powers[base][c],self.x[i]-self.x[base]);coeff=tuple(a+b-d for a,b,d in zip(coeff,self.powers[i][c],ref))
                        else:coeff=tuple(a+b for a,b in zip(coeff,self.powers[i][c]))
                        value=q.lift(coeff[-1])
                        for a in reversed(coeff[:-1]):value=value*z+q.lift(a)
                        if base==-1:value=value-self.base_jet(at,-1,c)
                        if weighted:value=at*value
                        channels.append(value)
                    rows.append(channels)
        if not rows:raise ValueError('empty angular range')
        return [Jet(order=q.order,coeff={(k,0):I(min(row[c][k,0].lo for row in rows),max(row[c][k,0].hi for row in rows)) for k in range(q.order+1)}) for c in range(2)]
    def actual_jet(self,q):
        base=-1 if q[0,0].lo<=self.cn[0].hi and self.horizon is not None else min(max(bisect_right(self.lower,q[0,0].lo)-1,0),len(self.cn)-2)
        delta=self.delta_jet(q,base)
        return [self.base_jet(q,base,c)+delta[c] for c in range(2)]
    def jump_sums(self,value):
        value=I(value)
        if value.lo<0 or value.hi>1:raise ValueError('front angular jump range required')
        first=bisect_left(self.upper[:-1],value.lo);last=bisect_right(self.lower[:-1],value.hi)
        if first==last:return [[I(0)]*3 for _ in range(2)]
        if not hasattr(self,'jump_prefix'):
            prefix=[[[I(0)]*3 for _ in range(2)]];signed=[]
            for k in range(len(self.cn)-1):
                self.checkpoint();q=Jet.variable(self.cn[k],order=2);row=[];jump=[]
                for c in range(2):
                    d=q.lift(0) if k==0 and self.horizon is None else self.base_jet(q,k,c)-self.base_jet(q,k-1,c)
                    jump.append(tuple(d[j,0]*math.factorial(j) for j in range(3)))
                    row.append([prefix[-1][c][j]+I(d[j,0].bound)*math.factorial(j) for j in range(3)])
                prefix.append(row);signed.append(tuple(jump))
            self.jump_prefix=prefix;self.jump_values=tuple(signed)
        first=bisect_left(self.upper[:-1],value.lo);last=bisect_right(self.lower[:-1],value.hi)
        self.checkpoint()
        return [[I(0,max(I(0).hi,(self.jump_prefix[last][c][j]-self.jump_prefix[first][c][j]).hi)) for j in range(3)] for c in range(2)]
