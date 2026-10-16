"""Explicit quartic residual for the independently certified positive angle root.

For homogeneous normal/view vectors, F=A/sqrt(B), B=|n|^2 |v|^2>0.
After the existing F-based existence/branch proof and Newton bracket, the same
positive root satisfies R=A^2-k^2 B=0. At that root R_u=2 A sqrt(B) F_u>0.
The caller certifies a positive R_u enclosure before each implicit recurrence.
This changes interval arithmetic; it does not claim identical old endpoints.
"""
import sys
from tools.opaque_shading_intervals import I,Jet
import tools.opaque_shading_intervals as scalar

class QuarticAngleResidual:
    _squared=True
    def __init__(self,prepared):
        from tools.opaque_prepared_angle import PreparedAngle
        if not isinstance(prepared,PreparedAngle):raise ValueError('qualified prepared outside angle required')
        self.prepared=prepared;self.entries={};self.hits=0;self.misses=0
    def coefficients(self,s):
        flint=sys.modules.get('flint');context=None if flint is None else (flint.ctx.prec,flint.ctx.cap)
        key=(Jet.__mul__,Jet.reciprocal,Jet.sqrt,Jet.ln,scalar.PRECISION,context,s.order,tuple(sorted((k,v.lo,v.hi) for k,v in s.c.items())))
        if key in self.entries:self.hits+=1;return self.entries[key]
        self.misses+=1;root,disc,width,aa,fields=self.prepared._prepare(s)
        norms=[]
        for name in ('n','v'):
            linear=[(a+root*b,width*b) for a,b in fields[name]]
            norms.append([sum((a*a for a,b in linear),s.lift(0)),sum((2*a*b for a,b in linear),s.lift(0)),sum((b*b for a,b in linear),s.lift(0))])
        product=[sum((norms[0][j]*norms[1][i-j] for j in range(3) if 0<=i-j<3),s.lift(0)) for i in range(5)]
        a1=self.prepared.patch.original_factor*width*disc;a2=self.prepared.patch.original_factor*aa*width*width
        numerator=[s.lift(0),s.lift(0),a1*a1,2*a1*a2,a2*a2] if self._squared else [s.lift(0),a1,a2]
        result=(numerator,product)
        if len(self.entries)>=64:self.entries.pop(next(iter(self.entries)))
        self.entries[key]=result;return result
    def __call__(self,u,s,k):
        if any(not isinstance(x,Jet) for x in (u,s,k)) or u.order!=s.order or u.order!=k.order:raise ValueError('matching residual jets required')
        if not 0<=u[0,0].lo<=u[0,0].hi<=1 or not 0<=s[0,0].lo<=s[0,0].hi<=1 or k[0,0].lo<=0:raise ValueError('unit parameters and strictly positive target required')
        a,b=self.coefficients(s)
        def evaluate(co):
            v=co[-1]
            for c in reversed(co[:-1]):v=v*u+c
            return v
        return evaluate(a)-k*k*evaluate(b) if self._squared else evaluate(a)-k*evaluate(b).sqrt()
    def implicit_coefficients(self,root,s,k,slope):
        """Solve the triangular quartic series once per coefficient.

        The caller supplies the certified positive root and its positive R_u.
        For each mixed key, all proper predecessor coefficients are already
        enclosed. Set the unknown coefficient to zero, evaluate that residual
        coefficient, then divide by R_u. Update only that key of powers u^m.
        No residual term or small coefficient is discarded.
        """
        if not isinstance(root,I) or not isinstance(slope,I) or slope.lo<=0 or root.lo<0 or root.hi>1:raise ValueError('positive certified root slope and unit root required')
        if not isinstance(s,Jet) or not isinstance(k,Jet) or s.order!=k.order or k[0,0].lo<=0:raise ValueError('matching positive-target jets required')
        aa,bb=self.coefficients(s);kk=k*k;co=[a-kk*b for a,b in zip(aa,bb)]
        zero=I(0);powers=[{(0,0):I(1)},{(0,0):root}]
        for m in range(2,5):powers.append({(0,0):powers[-1][0,0]*root})
        u=powers[1]
        def coefficient(left,right,key):
            i,j=key;value=zero
            for (a,b),v in left.items():
                if a<=i and b<=j:
                    other=right.get((i-a,j-b),zero)
                    if not v.zero and not other.zero:value=value+v*other
            return value
        for key in sorted(((i,j) for i in range(s.order+1) for j in (0,1) if i+j),key=lambda v:(sum(v),v)):
            for m in range(2,5):powers[m][key]=coefficient(u,powers[m-1],key)
            residual=sum((coefficient(c.c,v,key) for c,v in zip(co,powers)),zero)
            u[key]=-residual/slope
            for m in range(2,5):powers[m][key]=coefficient(u,powers[m-1],key)
        return dict(u)

    def root_slope(self,u,s,k,normalized_slope):
        # Used only at an existing F=k root, never as a Newton derivative on
        # arbitrary u. R_u=2*k*B*F_u there, avoiding subtractive cancellation.
        if any(not isinstance(x,I) for x in (u,s,k,normalized_slope)) or k.lo<=0 or normalized_slope.lo<=0:raise ValueError('positive certified root and normalized slope required')
        b=self.coefficients(Jet(s))[1];v=b[-1][0,0]
        for c in reversed(b[:-1]):v=v*u+c[0,0]
        if v.lo<=0:raise ValueError('positive root norm product required')
        return 2*k*v*normalized_slope if self._squared else v.sqrt()*normalized_slope
    def report(self):return dict(hits=self.hits,misses=self.misses,entries=len(self.entries),entry_cap=64)


class UnsquaredAngleResidual(QuarticAngleResidual):
    """R=A-k sqrt(B) avoids the squared residual's vanishing small-k slope.

    The positive normalized root is unchanged. At that root R_u=sqrt(B) F_u.
    The triangular recurrence retains a positive sqrt(B) series, solving its
    coefficients alongside powers of u; this is interval algebra, not iterative
    convergence or a small-term approximation.
    """
    _squared=False
    def implicit_coefficients(self,root,s,k,slope):
        if not isinstance(root,I) or not isinstance(slope,I) or slope.lo<=0 or root.lo<0 or root.hi>1:raise ValueError('positive certified root slope and unit root required')
        if not isinstance(s,Jet) or not isinstance(k,Jet) or s.order!=k.order or k[0,0].lo<=0:raise ValueError('matching positive-target jets required')
        aa,bb=self.coefficients(s);zero=I(0);powers=[{(0,0):I(1)},{(0,0):root}]
        for m in range(2,5):powers.append({(0,0):powers[-1][0,0]*root})
        u=powers[1]
        def coefficient(left,right,key):
            i,j=key;value=zero
            for (a,b),v in left.items():
                if a<=i and b<=j:
                    other=right.get((i-a,j-b),zero)
                    if not v.zero and not other.zero:value=value+v*other
            return value
        def polynomial(row,key):return sum((coefficient(c.c,v,key) for c,v in zip(row,powers)),zero)
        w={(0,0):polynomial(bb,(0,0)).sqrt()}
        if w[0,0].lo<=0:raise ValueError('positive norm root required')
        def update(key):
            for m in range(2,5):powers[m][key]=coefficient(u,powers[m-1],key)
            # W_key is unknown here: exclude both constant/target products.
            previous={ij:v for ij,v in w.items() if ij!=(0,0) and ij!=key}
            w[key]=(polynomial(bb,key)-coefficient(previous,previous,key))/(2*w[0,0])
        for key in sorted(((i,j) for i in range(s.order+1) for j in (0,1) if i+j),key=lambda v:(sum(v),v)):
            update(key)
            u[key]=-(polynomial(aa,key)-coefficient(k.c,w,key))/slope
            update(key)
        return dict(u)
    def report(self):return dict(super().report(),residual_form='A-k*sqrt(B)',series='incremental unsquared')
