"""Exact rational normal/view polynomials followed by outward full Jets.

Cancel algebraic terms before interval evaluation, including an intrinsic
rotational norm tangent. This never zeros a coordinate-motion contribution:
Horner evaluation uses the original spatial and parameter jets in full.
Only the existing first-parameter dual algebra is used; every spatial order
and mixed first parameter coefficient remains present.
"""
from fractions import Fraction as Q
from tools.opaque_shading_intervals import I,Jet
from tools.opaque_angle_light import _clean,_add,_mul,_diff,PreparedIntervalPolynomial


class PolynomialCosines:
    def __init__(self,coefficients,*,cancelled_motion=False):
        if type(cancelled_motion)!=bool:raise ValueError("explicit cancelled motion flag required")
        self.cancelled_motion=cancelled_motion
        def field(rows):
            if not isinstance(rows,(tuple,list)) or len(rows)!=3 or any(len(row)!=3 for row in rows):raise ValueError('three affine normal/view components required')
            try:return [_clean({(i,j,k):v[k] for (i,j),v in zip(((0,0),(1,0),(1,1)),row) for k in (0,1)}) for row in rows]
            except (IndexError,TypeError) as e:raise ValueError('exact first-parameter coefficient pairs required') from e
        n,v=field(coefficients['n']),field(coefficients['v'])
        def dot(a,b):
            out={}
            for x,y in zip(a,b):out=_add(out,_mul(x,y))
            return out
        def light(a):
            out={}
            for x,w in zip(a,(1,1,2)):out=_add(out,x,w)
            return out
        N,V,A,C,L=dot(n,n),dot(v,v),dot(n,v),light(n),light(v)
        # q=A/sqrt(B): differentiate in exact rational polynomial algebra
        # before enclosing, retaining the full intrinsic parameter numerator.
        self.motion_numerators=tuple(PreparedIntervalPolynomial(_add(_mul(B,_diff(A,2)),_mul(A,_diff(B,2)),Q(-1,2))) for A,B in ((A,_mul(N,V)),(C,_clean({k:6*v for k,v in N.items()})))) if cancelled_motion else None
        self.normal_norm=N
        self.prepared=tuple((PreparedIntervalPolynomial(p),PreparedIntervalPolynomial(_diff(p,2))) for p in (N,V,A,C,L))

    @staticmethod
    def _evaluate(p,u,s):
        p.prepare() # Context-owned conversion; the scalar value is not consumed.
        # Rows depend on s, not u. Keep their original operation/coefficient
        # order and private jets; every u-Horner evaluation remains fresh.
        key=(p.context,Jet.__mul__,Jet.__add__,s.order,tuple((k,v.lo,v.hi) for k,v in s.c.items()))
        if key in p.jet_rows:
            p.jet_row_hits+=1;rows=p.jet_rows[key]
        else:
            p.jet_row_evaluations+=1;rows=[]
            for row in p.rows:
                q=s.lift(row[0])
                for coefficient in row[1:]:q=q*s+coefficient
                rows.append(q)
            rows=tuple(rows)
            if len(p.jet_rows)>=64:p.jet_rows.pop(next(iter(p.jet_rows)))
            p.jet_rows[key]=rows
        value=u.lift(0)
        for i,q in enumerate(rows):value=(value if i==0 else value*u)+q
        return value

    def __call__(self,u,s,*,material_only=False):
        if not isinstance(u,Jet) or not isinstance(s,Jet) or u.order!=s.order or type(material_only)!=bool:raise ValueError('matching full jets and explicit material query required')
        if self.cancelled_motion and material_only:
            # Base jets carry ALL coordinate motion. The separate intrinsic
            # derivative is a spatial jet; multiply by theta by adding each
            # spatial coefficient only to the corresponding first-motion slot.
            N,V,A,C=[self._evaluate(a,u,s) for a,b in self.prepared[:4]]
            if N[0,0].lo<=0 or V[0,0].lo<=0:raise ValueError('positive normal/view norms required')
            Bs=(N*V,6*N);roots=[B.sqrt() for B in Bs];values=[A/roots[0],C/roots[1]]
            spatial=lambda q:Jet(order=q.order,coeff={(k,0):q[k,0] for k in range(q.order+1)})
            uu,ss=spatial(u),spatial(s)
            for q,B,root,numerator in zip(values,Bs,roots,self.motion_numerators):
                intrinsic=self._evaluate(numerator,uu,ss)/(spatial(B)*spatial(root))
                for k in range(q.order+1):
                    value=q[k,1]+intrinsic[k,0]
                    if value.zero:q.c.pop((k,1),None)
                    else:q.c[k,1]=value
            return *values,u.lift(0)
        theta=u.lift(0,tangent=1)
        values=[self._evaluate(a,u,s)+theta*self._evaluate(b,u,s) for a,b in (self.prepared[:4] if material_only else self.prepared)]
        N,V,A,C=values[:4];L=None if material_only else values[4]
        if N[0,0].lo<=0 or V[0,0].lo<=0:raise ValueError('positive normal/view norms required')
        nv=A/(N*V).sqrt();nl=C/(6*N).sqrt()
        vh=u.lift(0) if material_only else ((1+L/(6*V).sqrt())/2).sqrt()
        return nv,nl,vh

    def light_highlight(self,u,s,*,material_only=False):
        """Full light/half-vector jets without computing an unused view cosine.

        Material-only queries consume only the exact normal norm and light dot
        polynomials. The enclosing angular cover still owns its view-norm proof.
        Intrinsic and coordinate motion remain in every polynomial coefficient.
        """
        if not isinstance(u,Jet) or not isinstance(s,Jet) or u.order!=s.order or type(material_only)!=bool:raise ValueError('matching full jets and explicit material query required')
        theta=u.lift(0,tangent=1)
        def value(index):
            a,b=self.prepared[index]
            return self._evaluate(a,u,s)+theta*self._evaluate(b,u,s)
        N,C=value(0),value(3)
        if N[0,0].lo<=0:raise ValueError('positive normal norm required')
        nl=C/(6*N).sqrt()
        if material_only:return nl,u.lift(0)
        V,L=value(1),value(4)
        if V[0,0].lo<=0:raise ValueError('positive view norm required')
        return nl,((1+L/(6*V).sqrt())/2).sqrt()
