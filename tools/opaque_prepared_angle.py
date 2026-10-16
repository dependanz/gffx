"""Exact-operation reuse for the original outside-horizon angle field.

Prepare only quantities independent of the varying radial input. Preserve all
original product/addition orders, normalized fields and first motion. The64
entry cache is per prepared piece and arithmetic context, never global.
"""
import sys
from tools.opaque_shading_intervals import Jet
import tools.opaque_shading_intervals as scalar


class PreparedAngle:
    def __init__(self,p):
        if not p['curved'] or p['patch'].inside:raise ValueError('outside curved profile required')
        self.patch=p['patch'];self.coefficients=p['fields'].coefficients
        self._entries={};self.hits=0;self.misses=0
    def _prepare(self,s,value_only=False):
        flint=sys.modules.get('flint');context=None if flint is None else (flint.ctx.prec,flint.ctx.cap)
        key=(value_only,Jet.__mul__,Jet.reciprocal,Jet.sqrt,Jet.ln,scalar.PRECISION,context,s.order,tuple(sorted((k,v.lo,v.hi) for k,v in s.c.items())))
        if key in self._entries:self.hits+=1;return self._entries[key]
        self.misses+=1
        if value_only:
            root,disc,width,aa,fields=self._prepare(s)
            spatial=lambda v:Jet(order=v.order,coeff={k:z for k,z in v.c.items() if not k[1]})
            row=(spatial(root),spatial(disc),spatial(width),spatial(aa),{name:[(spatial(a),spatial(b)) for a,b in values] for name,values in fields.items()})
            if len(self._entries)>=64:self._entries.pop(next(iter(self._entries)))
            self._entries[key]=row;return row
        root,disc=self.patch._root_discriminant(s);width=1-root
        aa=self.patch.lift(s,self.patch.radial[0][-1])
        for c in reversed(self.patch.radial[0][:-1]):aa=aa*s+self.patch.lift(s,c)
        fields={name:[(s.lift(a[0],tangent=a[1]),s.lift(b[0],tangent=b[1])+s*s.lift(c[0],tangent=c[1])) for a,b,c in self.coefficients[name]] for name in ('n','v')}
        row=(root,disc,width,aa,fields)
        if len(self._entries)>=64:self._entries.pop(next(iter(self._entries)))
        self._entries[key]=row;return row
    def _inputs(self,u,s,*,value_only=False):
        if not isinstance(u,Jet) or not isinstance(s,Jet) or u.order!=s.order:raise ValueError('matching jets required')
        if any(x[0,0].lo<0 or x[0,0].hi>1 for x in (u,s)):raise ValueError('unit parameters required')
        root,disc,width,aa,fields=self._prepare(s,value_only);t=root+u*width
        n=[a+t*b for a,b in fields['n']];v=[a+t*b for a,b in fields['v']]
        dot=lambda a,b:sum((x*y for x,y in zip(a,b)),u.lift(0))
        nn,vv=dot(n,n),dot(v,v);den=(nn*vv).sqrt()
        delta=width*u;numerator=self.patch.original_factor*delta*(disc+aa*delta)
        return numerator,den,nn,vv,n,v,fields,width,disc,aa,delta,dot
    def __call__(self,u,s):
        numerator,den,*_=self._inputs(u,s);return numerator/den
    def edge(self,s):
        """Exact physical outer edge, with no moving-horizon cancellation.

        At u=1 the physical radius is identically one. Form each quadratic
        dot product in exact rational value/tangent coefficients before
        outward Horner evaluation. All spatial and first-motion terms remain.
        """
        if not isinstance(s,Jet) or not 0<=s[0,0].lo<=s[0,0].hi<=1:raise ValueError('unit edge jet required')
        polynomials=getattr(self,'_edge_polynomials',None)
        if polynomials is None:
            def plus(a,b):return (a[0]+b[0],a[1]+b[1])
            def times(a,b):return (a[0]*b[0],a[0]*b[1]+a[1]*b[0])
            fields={name:[(plus(a,b),c) for a,b,c in self.coefficients[name]] for name in ('n','v')}
            def dot(left,right):
                result=[(0,0)]*3
                for a,b in zip(left,right):
                    for i,x in enumerate(a):
                        for j,y in enumerate(b):result[i+j]=plus(result[i+j],times(x,y))
                return tuple(result)
            n,v=fields['n'],fields['v'];polynomials=(dot(n,v),dot(n,n),dot(v,v));self._edge_polynomials=polynomials
        def evaluate(row):
            result=s.lift(row[-1][0],tangent=row[-1][1])
            for a in reversed(row[:-1]):result=result*s+s.lift(a[0],tangent=a[1])
            return result
        nv,nn,vv=map(evaluate,polynomials)
        return nv/(nn*vv).sqrt()

    def value_only(self,u,s):
        # Newton brackets consume only spatial values/slopes at theta=0.
        # Full mixed coefficients continue through __call__ and radial.
        if not isinstance(u,Jet) or not isinstance(s,Jet) or max(u.order,s.order)>1 or any(j for x in (u,s) for i,j in x.c):raise ValueError('bracket jets without parameter motion required')
        numerator,den,*_=self._inputs(u,s,value_only=True);return numerator/den
    def radial(self,u,s):
        numerator,den,nn,vv,n,v,fields,width,disc,aa,delta,dot=self._inputs(u,s)
        dn=[width*b for a,b in fields['n']];dv=[width*b for a,b in fields['v']]
        derivative_numerator=self.patch.original_factor*width*(disc+2*aa*delta)
        return derivative_numerator/den-(numerator/den)*(dot(n,dn)/nn+dot(v,dv)/vv)
    def report(self):return dict(hits=self.hits,misses=self.misses,entries=len(self._entries),entry_cap=64)


class MaterialInputCache:
    """At most64 exact field tuples shared across this invocation's charts.

    Scope includes chart identity and the complete current angular bounds.
    Store immutable endpoints; no mutable chart state or failed result is reused.
    """
    def __init__(self):self.entries={};self.hits=0;self.misses=0
    def _key(self,scope,evaluate,t,y):
        if not callable(evaluate) or not isinstance(t,Jet) or not isinstance(y,Jet) or t.order!=y.order:raise ValueError('matching input jets and evaluator required')
        flint=sys.modules.get('flint');context=None if flint is None else (flint.ctx.prec,flint.ctx.cap)
        encode=lambda x:(x.order,tuple(sorted((k,v.lo,v.hi) for k,v in x.c.items())))
        return (scope,evaluate,Jet.__mul__,Jet.reciprocal,Jet.sqrt,Jet.ln,scalar.PRECISION,context,encode(t),encode(y))
    def _rows(self,values,order):
        if not isinstance(values,(list,tuple)) or len(values)!=4 or any(not isinstance(v,Jet) or v.order!=order for v in values):raise ValueError('four matching field jets required')
        return tuple(tuple((k,v.lo,v.hi) for k,v in q.c.items()) for q in values)
    def _store(self,key,rows):
        if key not in self.entries and len(self.entries)>=64:self.entries.pop(next(iter(self.entries)))
        self.entries[key]=rows
    def seed(self,scope,evaluate,t,y,values):
        key=self._key(scope,evaluate,t,y);rows=self._rows(values,t.order)
        if key in self.entries and self.entries[key]!=rows:raise ValueError('conflicting raw material fields')
        self._store(key,rows)
    def __call__(self,scope,evaluate,t,y):
        key=self._key(scope,evaluate,t,y)
        if key in self.entries:self.hits+=1;rows=self.entries[key]
        else:
            self.misses+=1;rows=self._rows(evaluate(t,y),t.order);self._store(key,rows)
        from tools.opaque_shading_intervals import I
        return tuple(Jet(order=t.order,coeff={k:I(lo,hi) for k,lo,hi in row}) for row in rows)
    def report(self):return dict(hits=self.hits,misses=self.misses,entries=len(self.entries),entry_cap=64)
