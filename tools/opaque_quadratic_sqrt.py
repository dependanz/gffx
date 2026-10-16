"""Outward Taylor coefficients of an exact moving quadratic square root.

For q(x)>0, f^(n)(x)/n! = P_n(x) / q(x)^(n-1/2), n>=1.
P_1=q'/2 and P_(n+1)=(q P_n'-(n-1/2)q' P_n)/(n+1).
Exact rational polynomial reduction preserves identities before interval
substitution. Coefficient first motion and coordinate first motion are distinct
and both retained. This is a development checker, not a renderer primitive.
"""
from fractions import Fraction as Q
from tools.opaque_moving_reference import P
from tools.opaque_shading_intervals import I,Jet


def trim(p):return P([p[0],p[1]])
def add(a,b):return [trim((a[i] if i<len(a) else P(0))+(b[i] if i<len(b) else P(0))) for i in range(max(len(a),len(b)))]
def scale(a,c):return [trim(p*c) for p in a]
def mul(a,b):
    out=[P(0) for _ in range(len(a)+len(b)-1)]
    for i,p in enumerate(a):
        for j,q in enumerate(b):
            # Multiply in the first-motion quotient directly. The old product
            # built theta**2 only to discard it; each consumed product and sum
            # still goes through P's unchanged rational-complexity guard.
            value=(P(p[0])*P(q[0]))[0]
            tangent=(P(p[0])*P(q[1])+P(p[1])*P(q[0]))[0]
            out[i+j]=trim(out[i+j]+P([value,tangent]))
    while len(out)>1 and out[-1].zero:out.pop()
    return out

def diff(a):return [p*i for i,p in enumerate(a) if i] or [P(0)]
def at(a,x,j=0):
    v=I(a[-1][j])
    for p in reversed(a[:-1]):v=v*x+I(p[j])
    return v


class QuadraticSqrt:
    """Immutable exact derivative numerators; one prepared instance per patch."""
    def __init__(self,coefficients,*,normalize=False):
        if not isinstance(coefficients,(list,tuple)) or not 1<=len(coefficients)<=3:raise ValueError('exact quadratic coefficients required')
        if type(normalize) is not bool:raise ValueError('explicit normalization boolean required')
        self.original=tuple(trim(P(p)) for p in coefficients)
        self.scale=self.original[0][0] if normalize else Q(1)
        if self.scale<=0:raise ValueError('positive constant normalization required')
        # This constant is frozen at the expansion point. All first-motion
        # coefficients are divided by it too; none of its motion is discarded.
        self.q=tuple(P([p[0]/self.scale,p[1]/self.scale]) for p in self.original);qp=diff(self.q)
        rows=[scale(qp,Q(1,2))]
        # One extra spatial order is needed for x^order*theta when the
        # coordinate's constant has first motion. No term is dropped by size.
        for n in range(1,11):rows.append(scale(add(mul(self.q,diff(rows[-1])),scale(mul(qp,rows[-1]),-Q(2*n-1,2))),Q(1,n+1)))
        self.numerators=tuple(tuple(r) for r in rows)
    def range(self,x):
        return self._range(self.original,x)
    @staticmethod
    def _range(coefficients,x):
        # Exact endpoints and the quadratic vertex prove positivity over the
        # entire interval, including an interior minimum; no point sampling.
        q=[p[0] for p in coefficients]+[Q(0)]*(3-len(coefficients));lo,hi=Q(x.lo),Q(x.hi)
        evaluate=lambda z:q[0]+z*(q[1]+z*q[2])
        values=[evaluate(lo),evaluate(hi)]
        if q[2]:
            vertex=-q[1]/(2*q[2])
            if lo<vertex<hi:values.append(evaluate(vertex))
        return I(I(min(values)).lo,I(max(values)).hi)
    def __call__(self,x):
        if not isinstance(x,Jet):raise ValueError('Taylor jet required')
        q=self._range(self.q,x[0,0])
        if q.lo<=0:raise ValueError('strict positive quadratic required')
        root=q.sqrt();qt=at(self.q,x[0,0],1)
        result=x.lift(root,tangent=qt/(2*root))
        delta=Jet(order=x.order,coeff={k:v for k,v in x.c.items() if k!=(0,0)})
        power=x.lift(1)
        for n in range(1,x.order+2):
            power=power*delta
            if not power.c:break
            p=self.numerators[n-1];den=root*q**(n-1);value=at(p,x[0,0])/den
            tangent=at(p,x[0,0],1)/den-value*Q(2*n-1,2)*qt/q
            result=result+power*x.lift(value,tangent=tangent)
        return result if self.scale==1 else result*I(self.scale).sqrt()
