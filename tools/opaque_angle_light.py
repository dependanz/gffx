"""Exact-polynomial light differential in viewing-angle coordinates.

F=A/sqrt(N V), L=C/sqrt(6 N), with N=n.n, V=v.v, A=n.v, C=n.(1,1,2).
P_d=2 N C_d-C N_d; Q_d=2 N V A_d-A(N_d V+N V_d).
On a certified F_t>0 branch, eliminate dt from dF and dL:
  dL = P_t V^(3/2)/(sqrt(6) Q_t) dF
       +(P_s Q_t-P_t Q_s)/(2 sqrt(6) N^(3/2) Q_t) ds
       +(P_theta Q_t-P_t Q_theta)/(2 sqrt(6) N^(3/2) Q_t) dtheta.
Products/cancellation are exact rational polynomials before outward evaluation.
Only first spatial/parameter coefficients are bounded; no mixed term is omitted
from the original Jet. Physical radius and transverse domains remain [0,1].
"""
import sys
import tools.opaque_shading_intervals as scalar
from fractions import Fraction as Q
from tools.opaque_shading_intervals import I,Jet

def _clean(p):
    p={k:Q(v) for k,v in p.items() if v}
    if len(p)>256 or any(max(k)>16 or abs(v.numerator).bit_length()>8192 or v.denominator.bit_length()>8192 for k,v in p.items()):raise ValueError('differential polynomial budget')
    return p

def _add(a,b,scale=1):
    p=dict(a)
    for k,v in b.items():p[k]=p.get(k,0)+scale*v
    return _clean(p)

def _mul(a,b):
    p={}
    for x,u in a.items():
        for y,v in b.items():
            k=tuple(a+b for a,b in zip(x,y))
            if k[2]>1:continue # exact first-parameter dual algebra
            p[k]=p.get(k,0)+u*v
    return _clean(p)

def _diff(p,axis):
    out={}
    for k,v in p.items():
        if k[axis]:
            key=list(k);key[axis]-=1;out[tuple(key)]=k[axis]*v
    return _clean(out)

def _evaluate(p,t,s):
    # All callers evaluate at theta=0 after symbolic differentiation.
    p={k[:2]:v for k,v in p.items() if not k[2]}
    if not p:return I(0)
    answer=I(0)
    for i in range(max(k[0] for k in p),-1,-1):
        row=I(0);degree=max((k[1] for k in p if k[0]==i),default=0)
        for j in range(degree,-1,-1):row=row*s+I(p.get((i,j),0))
        answer=answer*t+row
    return answer

def _scalar_horner(coefficients,x):
    """Identical separately rounded Horner products/additions in one context."""
    from decimal import localcontext,ROUND_FLOOR,ROUND_CEILING
    from tools.opaque_shading_intervals import _product_endpoints
    answer=I(0)
    with localcontext() as context:
        context.prec=scalar.PRECISION
        for coefficient in coefficients:
            if answer.zero or x.zero:
                answer=I(coefficient);continue
            lower,upper=_product_endpoints(answer,x)
            context.rounding=ROUND_FLOOR;lo=min(a*b for a,b in lower)
            context.rounding=ROUND_CEILING;hi=max(a*b for a,b in upper)
            if lo==0 and hi==0:
                answer=I(coefficient);continue
            if not coefficient.zero:
                context.rounding=ROUND_FLOOR;lo=lo+coefficient.lo
                context.rounding=ROUND_CEILING;hi=hi+coefficient.hi
            answer=I(lo,hi)
    return answer

class PreparedIntervalPolynomial:
    """Same outward Horner operations with bounded converted coefficients.

    Own one conversion for the current arithmetic context. Changing context
    discards that conversion; no roots, sampled values or certificates cache.
    """
    def __init__(self,polynomial):
        self.polynomial=_clean(polynomial);self.context=None;self.rows=();self.rebuilds=0;self.entries={};self.row_evaluations=0;self.row_hits=0;self.jet_rows={};self.jet_row_evaluations=0;self.jet_row_hits=0
    def prepare(self):
        """Convert coefficients for this context without evaluating a value."""
        flint=sys.modules.get('flint');context=(scalar.PRECISION,I.__init__,None if flint is None else (flint.ctx.prec,flint.ctx.cap))
        if context!=self.context:
            p={k[:2]:v for k,v in self.polynomial.items() if not k[2]}
            self.rows=tuple(tuple(I(p.get((i,j),0)) for j in range(max((k[1] for k in p if k[0]==i),default=0),-1,-1)) for i in range(max((k[0] for k in p),default=-1),-1,-1))
            self.context=context;self.rebuilds+=1;self.entries.clear();self.jet_rows.clear()
        return self.rows
    def __call__(self,t,s):
        self.prepare()
        key=(s.lo,s.hi)
        if key in self.entries:self.row_hits+=1;values=self.entries[key]
        else:
            self.row_evaluations+=1;values=[]
            for row in self.rows:
                value=_scalar_horner(row,s)
                values.append((value.lo,value.hi))
            values=tuple(values)
            if len(self.entries)>=64:self.entries.pop(next(iter(self.entries)))
            self.entries[key]=values
        return _scalar_horner((I(lo,hi) for lo,hi in values),t)

class AngleLightDifferential:
    def __init__(self,coefficients):
        def field(rows):
            if len(rows)!=3 or any(len(row)!=3 for row in rows):raise ValueError('three affine normal/view components required')
            return [_clean({(i,j,k):v[k] for (i,j),v in zip(((0,0),(1,0),(1,1)),row) for k in (0,1)}) for row in rows]
        n,v=field(coefficients['n']),field(coefficients['v'])
        def dot(a,b):
            out={}
            for x,y in zip(a,b):out=_add(out,_mul(x,y))
            return out
        N,V,A=dot(n,n),dot(v,v),dot(n,v);C={}
        for q,c in zip(n,(1,1,2)):C=_add(C,q,c)
        pp=[];qq=[];NV=_mul(N,V)
        for axis in range(3):
            pp.append(_add(_mul(N,_diff(C,axis)),_mul(C,_diff(N,axis)),Q(-1,2)))
            qq.append(_add(_mul(NV,_diff(A,axis)),_mul(A,_diff(NV,axis)),Q(-1,2)))
        # P/2 and Q/2 have the same ratios; determinant denominator loses factor2.
        Hs=_add(_mul(pp[1],qq[0]),_mul(pp[0],qq[1]),-1)
        Htheta=_add(_mul(pp[2],qq[0]),_mul(pp[0],qq[2]),-1)
        self.polynomials=(N,V,pp[0],qq[0],Hs,Htheta)
        self.prepared=tuple(PreparedIntervalPolynomial(p) for p in self.polynomials)
        self.last_scalar=None;self.last_coefficients=None;self.scalar_coefficients={};self.scalar_hits=0;self.scalar_misses=0
    def bound(self,t,s,target,key):
        if any(not isinstance(q,Jet) for q in (t,s,target)) or len({t.order,s.order,target.order})!=1 or key not in ((1,0),(0,1)) or key==(1,0) and t.order<1:raise ValueError('matching first-derivative jets required')
        if any(q[0,0].lo<0 or q[0,0].hi>1 for q in (t,s)):raise ValueError('physical unit domain required')
        flint=sys.modules.get('flint');context=(scalar.PRECISION,I.__init__,None if flint is None else (flint.ctx.prec,flint.ctx.cap))
        scalar_key=(context,id(self.prepared),t[0,0].lo,t[0,0].hi,s[0,0].lo,s[0,0].hi)
        if scalar_key in self.scalar_coefficients:
            self.scalar_hits+=1;self.last_coefficients=self.scalar_coefficients[scalar_key];self.last_scalar=scalar_key
        else:
            self.scalar_misses+=1
            N,V,Pt,Qt,Hs,Htheta=[p(t[0,0],s[0,0]) for p in self.prepared]
            if N.lo<=0 or V.lo<=0 or Qt.lo<=0:raise ValueError('positive normal/view norms and radial angle slope required')
            a=Pt*(V**3).sqrt()/(I(6).sqrt()*Qt)
            b=Hs/((6*N**3).sqrt()*Qt)
            c=Htheta/((6*N**3).sqrt()*Qt)
            self.last_coefficients=tuple((v.lo,v.hi) for v in (a,b,c));self.last_scalar=scalar_key
            if len(self.scalar_coefficients)>=64:self.scalar_coefficients.pop(next(iter(self.scalar_coefficients)))
            self.scalar_coefficients[scalar_key]=self.last_coefficients
        # At most64 immutable tuples; coordinate and parameter coefficients are
        # applied freshly with the original separately rounded operation order.
        a,b,c=(I(*v) for v in self.last_coefficients)
        value=a*target[key]+b*s[key]
        if key==(0,1):value=value+c
        return value


    def coefficient_bounds(self,t,s,target):
        """Differentiate the exact one-form in the full spatial series.

        All coefficient polynomials are evaluated at theta=0. For parameter
        motion use L_theta=a F_theta+b s_theta+c. For spatial order n use
        n L_n=(a F_prime+b s_prime)_(n-1). The intrinsic c term is retained
        at every spatial order; no mixed coefficient is inferred from zero.
        """
        if any(not isinstance(q,Jet) for q in (t,s,target)) or len({q.order for q in (t,s,target)})!=1:raise ValueError('matching differential jets required')
        if any(q[0,0].lo<0 or q[0,0].hi>1 for q in (t,s)):raise ValueError('physical unit domain required')
        order=t.order
        radial=Jet(order=order,coeff={(i,0):t[i,0] for i in range(order+1)})
        transverse=Jet(order=order,coeff={(i,0):s[i,0] for i in range(order+1)})
        def evaluate(p):
            p.prepare() # Exact coefficient conversion; no discarded scalar value.
            result=radial.lift(0)
            for row in p.rows:
                value=transverse.lift(0)
                for c in row:value=value*transverse+c
                result=result*radial+value
            return result
        N,V,Pt,Qt,Hs,Htheta=[evaluate(p) for p in self.prepared]
        if N[0,0].lo<=0 or V[0,0].lo<=0 or Qt[0,0].lo<=0:raise ValueError('positive differential norm and radial slope required')
        a=Pt*(V**3).sqrt()/(Qt*I(6).sqrt());den=(6*N**3).sqrt()*Qt
        b=Hs/den;c=Htheta/den
        theta=lambda q:Jet(order=order,coeff={(i,0):q[i,1] for i in range(order+1)})
        motion=a*theta(target)+b*theta(s)+c
        result={(i,1):motion[i,0] for i in range(order+1)}
        if order:
            derivative=lambda q:Jet(order=order,coeff={(i,0):q[i+1,0]*(i+1) for i in range(order)})
            spatial=a*derivative(target)+b*derivative(s)
            result.update({(i,0):spatial[i-1,0]/i for i in range(1,order+1)})
        return result
