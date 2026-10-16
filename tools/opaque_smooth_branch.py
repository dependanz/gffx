"""Equivalent centered cubic branches for fixed-angle correction jets only.

The Bernstein-to-power identity is exact over the reals. Every coefficient
operation rounds outward, retaining uncertain controls. This does not change
segments, clipping, jumps, or the independently evaluated smooth base integral.
"""
from tools.opaque_shading_intervals import I


class SmoothBranches:
    max_order=3
    def __init__(self,extension):
        self.extension=extension;self._cache={}

    def at(self,q,index):
        if type(index)!=int or index < -1:raise ValueError('smooth branch index')
        if q.order>self.max_order:raise ValueError('smooth branch exceeds qualified order')
        if any(j for i,j in q.c):raise ValueError('fixed angle only; moving knot transport required separately')
        e=self.extension
        if index==-1:
            if not hasattr(e,'horizon'):raise ValueError('horizon branch unavailable')
            return e.at(q,index)
        if index>=len(e.controls):raise ValueError('smooth branch index')
        x0,x1=e.x[index:index+2];h=x1-x0
        if h.lo<=0:raise ValueError('positive log branch width')
        controls=e.controls[index]
        if len(controls)!=2 or any(len(row)!=4 for row in controls):raise ValueError('two cubic channels required')
        key=(index,x0.lo,x0.hi,x1.lo,x1.hi,tuple(tuple((b.lo,b.hi) for b in row) for row in controls))
        if key not in self._cache:
            coefficients=[]
            for a,b,c,d in controls:
                row=((a+3*b+3*c+d)/8,3*(-a-b+c+d)/4,3*(a-b-c+d)/2,-a+3*b-3*c+d)
                coefficients.append(tuple((v.lo,v.hi) for v in row))
            # Only the current view/light branches. Immutable endpoints avoid
            # sharing mutable Interval or Jet instances with callers.
            if len(self._cache)==2:self._cache.pop(next(iter(self._cache)))
            self._cache[key]=tuple(coefficients)
        z=(q.ln()-q.lift(x0))/q.lift(h)-q.lift(I('.5'))
        result=[]
        for row in self._cache[key]:
            value=q.lift(I(*row[-1]))
            for endpoints in reversed(row[:-1]):value=q.lift(I(*endpoints))+z*value
            result.append(value)
        return result


class SmoothBaseBranches(SmoothBranches):
    """Same exact centered cubic, qualified for fixed-angle base jets through10.

    Shares the existing two-branch cache with correction consumers. The base
    Gauss5 remainder still uses outward order10 derivatives, unchanged radii,
    horizon handling and all separately accounted knot corrections.
    """
    max_order=10
