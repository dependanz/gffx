"""Positive physical-radius inverse of the independently certified angle field.

The caller proves existence on the mapped physical patch, supplies a lower
radius enclosure, and retains the original ordered-cover proof. Interval Newton
uses a newly certified positive physical derivative over the entire bracket.
No horizon-root derivatives enter the implicit physical-radius recurrence.
"""
import sys
from fractions import Fraction as Q
from tools.opaque_shading_intervals import I,Jet
import tools.opaque_shading_intervals as scalar
from tools.opaque_algebraic_angle_inverse import UnsquaredAngleResidual
from tools.opaque_angle_light import _clean,_add,_mul,_diff,_evaluate,PreparedIntervalPolynomial

class PhysicalAngleResidual(UnsquaredAngleResidual):
    def __init__(self,coefficients,*,compiled_series=False,contraction_stop=False):
        if type(compiled_series)!=bool:raise ValueError("explicit compiled-series switch required")
        if type(contraction_stop)!=bool:raise ValueError('explicit contraction stopping switch required')
        self.contraction_stop=contraction_stop;self.contraction_stops=0;self.newton_steps=0
        self.compiled_series=compiled_series;self.compiled_calls=0;self.shared_slope_calls=0
        def field(rows):
            if len(rows)!=3 or any(len(row)!=3 for row in rows):raise ValueError('three affine radial field components required')
            return [_clean({(i,j,k):v[k] for (i,j),v in zip(((0,0),(1,0),(1,1)),row) for k in (0,1)}) for row in rows]
        def dot(a,b):
            result={}
            for x,y in zip(a,b):result=_add(result,_mul(x,y))
            return result
        n,v=field(coefficients['n']),field(coefficients['v']);A=dot(n,v);B=_mul(dot(n,n),dot(v,v))
        self.polynomials=(A,B);self.slope_polynomial=_add(_mul(_diff(A,0),B),_mul(A,_diff(B,0)),Q(-1,2))
        self.entries={};self.hits=0;self.misses=0;self.evaluations=0
        self.scalar_polynomials=tuple(PreparedIntervalPolynomial(p) for p in (A,B,self.slope_polynomial))
    def coefficients(self,s):
        flint=sys.modules.get('flint');context=None if flint is None else (flint.ctx.prec,flint.ctx.cap)
        key=(Jet.__mul__,Jet.reciprocal,Jet.sqrt,scalar.PRECISION,context,s.order,tuple(sorted((ij,v.lo,v.hi) for ij,v in s.c.items())))
        if key in self.entries:self.hits+=1;return self.entries[key]
        self.misses+=1;result=[]
        for p,degree in zip(self.polynomials,(2,4)):
            row=[]
            for i in range(degree+1):
                q=s.lift(0);last=max((j for a,j,k in p if a==i),default=0)
                for j in range(last,-1,-1):q=q*s+s.lift(p.get((i,j,0),0),tangent=p.get((i,j,1),0))
                row.append(q)
            result.append(row)
        if len(self.entries)>=64:self.entries.pop(next(iter(self.entries)))
        self.entries[key]=tuple(result);return self.entries[key]
    @staticmethod
    def evaluate(row,t):
        q=row[-1]
        for a in reversed(row[:-1]):q=q*t+a
        return q
    def radial(self,t,s):
        a,b=self.coefficients(s);A,B=self.evaluate(a,t),self.evaluate(b,t)
        da=self.evaluate([c*i for i,c in enumerate(a) if i],t);db=self.evaluate([c*i for i,c in enumerate(b) if i],t)
        return (da-A*db/(2*B))/B.sqrt()
    def radial_polynomial(self,t,s):
        # Differentiate A/sqrt(B) in exact rational polynomial algebra before
        # enclosing it. This retains every spatial and intrinsic-motion term
        # while removing dependency between the two cancelling numerator terms.
        from tools.opaque_polynomial_inputs import PolynomialCosines
        _,b=self.coefficients(s);B=self.evaluate(b,t)
        if not hasattr(self,'_slope_dual'):self._slope_dual=tuple(PreparedIntervalPolynomial(p) for p in (self.slope_polynomial,_diff(self.slope_polynomial,2)))
        a,theta=self._slope_dual
        numerator=PolynomialCosines._evaluate(a,t,s)+t.lift(0,tangent=1)*PolynomialCosines._evaluate(theta,t,s)
        return numerator/(B**3).sqrt()
    def _scalar(self,t,s):
        self.evaluations+=1;A,B=[p(t,s) for p in self.scalar_polynomials[:2]]
        if B.lo<=0:raise ValueError('positive physical norm product required')
        return A/B.sqrt()
    def _slope(self,t,s):
        self.evaluations+=1;B=self.scalar_polynomials[1](t,s)
        if B.lo<=0:raise ValueError('positive physical norm product required')
        q=self.scalar_polynomials[2](t,s)/(B**3).sqrt()
        if q.lo<=0:raise ValueError('certified positive physical angle derivative required')
        return q
    def inverse(self,target,s,*,lower,checkpoint=lambda:None,with_radial=False):
        if type(with_radial)!=bool or with_radial and not self.compiled_series:raise ValueError('shared radial proof requires explicit compiled series')
        if not isinstance(target,Jet) or not isinstance(s,Jet) or target.order!=s.order or target[0,0].lo<=0 or not isinstance(lower,I) or not 0<=lower.lo<=lower.hi<=1 or not 0<=s[0,0].lo<=s[0,0].hi<=1:raise ValueError('matching positive target and certified unit physical bracket required')
        root=I(lower.lo,1);slope=self._slope(root,s[0,0])
        for _ in range(8):
            checkpoint();previous_width=root.width;self.newton_steps+=1;mid=(Q(root.lo)+Q(root.hi))/2
            proposal=I(mid)-(self._scalar(I(mid),s[0,0])-target[0,0])/slope
            narrowed=I(max(root.lo,proposal.lo),min(root.hi,proposal.hi))
            if narrowed.lo==root.lo and narrowed.hi==root.hi:break
            root=narrowed;slope=self._slope(root,s[0,0])
            # Every current bracket is already a root certificate. Stop extra
            # contraction only; keep its full width in all mixed recurrences.
            # These thresholds select work, never admit an integral or root.
            if self.contraction_stop and (root.width<=I('1e-40').lo or I(previous_width-root.width).hi<=(I(root.width)*I('1e-6')).lo):
                self.contraction_stops+=1;break
        norm=self.scalar_polynomials[1](root,s[0,0]).sqrt()
        checkpoint()
        if with_radial:
            from tools.opaque_compiled_inverse import solve
            self.compiled_calls+=1;self.shared_slope_calls+=1
            values=solve(self,root,s,target,norm*slope,with_radial=True)
            result=tuple(Jet(order=target.order,coeff=row) for row in values)
        else:result=Jet(order=target.order,coeff=self.implicit_coefficients(root,s,target,norm*slope))
        checkpoint();return result
    def implicit_coefficients(self,root,s,k,slope):
        if self.compiled_series:
            from tools.opaque_compiled_inverse import solve
            self.compiled_calls+=1
            return solve(self,root,s,k,slope)
        return super().implicit_coefficients(root,s,k,slope)
    def report(self):return dict(super().report(),compiled_series=self.compiled_series,compiled_calls=self.compiled_calls,shared_slope_calls=self.shared_slope_calls,contraction_stop=self.contraction_stop,contraction_stops=self.contraction_stops,newton_steps=self.newton_steps,coordinate='physical radius',evaluations=self.evaluations)


def physical_angle_chart(field,cover,index,t,y,*,lower):
    """Reuse the certified ordered cover in physical radial coordinates.

    The cover proves existence for every target, including closed exits. Its
    zero-angle horizon charts retain their original mapping. This map changes only
    coordinates, retaining moving targets, transverse events and Jacobians.
    """
    if type(index)!=int or not 0<=index<len(cover.charts):raise ValueError('valid chart index required')
    if not isinstance(t,Jet) or not isinstance(y,Jet) or t.order!=y.order or any(not 0<=z[0,0].lo<=z[0,0].hi<=1 for z in (t,y)):raise ValueError('matching unit chart jets required')
    c=cover.charts[index]
    if c['low'] is None:return None
    cover.checkpoint();low=cover.levels[c['low']]
    def event(i,default):return t.lift(default) if i is None else t.lift(cover.events[i][0,0],tangent=cover.events[i][0,1])
    a=event(c['start'],0);b=event(c['end'],1);span=b-a
    if c['start'] is not None and c['end'] is not None:
        bound=(cover.levels[c['start']]-cover.levels[c['end']])/(-cover.edge_slope)
        span.c[0,0]=I(max(span[0,0].lo,bound.lo),min(span[0,0].hi,bound.hi))
    if span[0,0].lo<=0:raise ValueError('positive transverse span required')
    s=a+span*y;s.c[0,0]=I(max(s[0,0].lo,a[0,0].lo),min(s[0,0].hi,b[0,0].hi))
    if c['high'] is not None:
        high=cover.levels[c['high']];delta=t.lift(high-low)
    else:
        edge=getattr(cover._field,'edge',None)
        delta=(cover.field(t.lift(1),s) if edge is None else edge(s))-low
        delta.c[0,0]=I(max(I(0).lo,delta[0,0].lo),delta[0,0].hi)
        high=cover.outer[0] if c['band']==len(cover.levels) else cover.levels[c['band']]
    target=t*delta+low
    shared=field.compiled_series
    key=('physical_angle',id(field),shared,target.order,tuple(sorted((ij,v.lo,v.hi) for ij,v in target.c.items())),tuple(sorted((ij,v.lo,v.hi) for ij,v in s.c.items())))
    if key in cover.cache:values=cover.cache[key]
    else:
        values=field.inverse(target,s,lower=lower(s),checkpoint=cover.checkpoint,**({'with_radial':True} if shared else {}))
        if len(cover.cache)>=64:cover.cache.pop(next(iter(cover.cache)))
        cover.cache[key]=values
    cover.checkpoint()
    if shared:radius,slope=values
    else:radius=values;slope=field.radial(radius,s)
    cover.checkpoint()
    if slope[0,0].lo<=0:raise ValueError('positive physical radial derivative required')
    jacobian=delta*span/slope;jacobian.c[0,0]=I(max(I(0).lo,jacobian[0,0].lo),jacobian[0,0].hi)
    return dict(radius=radius,s=s,angle=target,jacobian=jacobian,band=c['band'],angular_range=I(low.lo,high.hi),coordinate_kind='physical_angle')
