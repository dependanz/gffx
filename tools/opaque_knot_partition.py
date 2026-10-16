"""Bounded knot-targeted partition guidance; never a root or accuracy certificate."""
from fractions import Fraction as Q
from tools.opaque_shading_intervals import I,Unresolved


def knot_cut(knots,field,box,*,axis,checkpoint=lambda:None):
    if type(axis)!=int or axis not in (0,1) or len(box)!=4:raise ValueError('partition axis/box')
    a,b,c,d=map(Q,box)
    if not 0<=a<b<=1 or not 0<=c<d<=1:raise ValueError('unit partition box')
    lo,hi=(a,b) if axis==0 else (c,d);other=(c+d)/2 if axis==0 else (a+b)/2;calls=0
    def at(z):
        nonlocal calls
        checkpoint();calls+=1
        v=field(z,other) if axis==0 else field(other,z)
        if not isinstance(v,I):raise ValueError('interval guidance value required')
        checkpoint();return v
    left,right=at(lo),at(hi)
    increasing=left.hi<right.lo;decreasing=right.hi<left.lo
    if not increasing and not decreasing:return None
    lower,upper=(left.hi,right.lo) if increasing else (right.hi,left.lo)
    eligible=[v for v in knots if lower<v.lo and v.hi<upper]
    if not eligible:return None
    target=eligible[len(eligible)//2]
    # Exact dyadic cuts bracket a knot on one centre line. This deliberately
    # does not assert monotonicity, exact root equality, or a constant-knot edge.
    for _ in range(12):
        mid=(lo+hi)/2;value=at(mid)
        if value.lo<=target.hi and value.hi>=target.lo:break
        below=value.hi<target.lo
        if below==increasing:lo=mid
        else:hi=mid
    return dict(axis=axis,point=mid,knot=target,value=value,evaluations=calls,root_certified=False)


def knot_root(field,lo,hi,s,knot,*,checkpoint=lambda:None):
    """Enclose a unique root and first implicit derivatives of a smooth Jet field.

    The caller supplies a continuously differentiable field on the full box.
    Strict whole-interval derivative sign and endpoint signs certify the root;
    samples alone never do. This is not a high-order integration chart.
    """
    from tools.opaque_shading_intervals import Jet
    lo,hi,s=Q(lo),Q(hi),Q(s);calls=0
    if not 0<=lo<hi<=1 or not 0<=s<=1 or not isinstance(knot,I) or knot.lo!=knot.hi:raise ValueError('unit root domain and exact knot required')
    def at(u,v):
        nonlocal calls
        checkpoint();calls+=1;q=field(u,v)
        if not isinstance(q,Jet):raise ValueError('smooth Jet field required')
        checkpoint();return q
    interval=lambda a,b:I(I(a).lo,I(b).hi)
    slope=at(Jet.variable(interval(lo,hi),order=1),Jet(s,order=1))[1,0]
    sign=1 if slope.lo>0 else -1 if slope.hi<0 else 0
    if not sign:raise ValueError('knot root not certified monotone')
    left=sign*(at(Jet(lo),Jet(s))[0,0]-knot);right=sign*(at(Jet(hi),Jet(s))[0,0]-knot)
    if left.hi>=0 or right.lo<=0:raise ValueError('knot not strictly bracketed')
    for _ in range(12):
        mid=(lo+hi)/2;value=sign*(at(Jet(mid),Jet(s))[0,0]-knot)
        if value.hi<0:lo=mid
        elif value.lo>0:hi=mid
        else:break
    root=interval(lo,hi);du=at(Jet.variable(root,order=1),Jet(s,order=1));ds=at(Jet(root,order=1),Jet.variable(s,order=1))
    return dict(root=root,parameter_derivative=-du[0,1]/du[1,0],transverse_derivative=-ds[1,0]/du[1,0],radial_derivative=du[1,0],evaluations=calls,unique=True,scope='One fixed transverse coordinate and smooth local parameter direction; not a complete curved chart or integral.')


def knot_region(field,lo,hi,s,knot,*,checkpoint=lambda:None,tight_endpoints=False):
    """Uniform interior root and implicit Taylor jet over a transverse interval.

    Caller guarantees C^(order+1) smoothness of field throughout the box, including
    the local parameter direction. Strict F_u and uniform endpoint signs prove
    existence, uniqueness and no boundary entry/exit at the evaluation parameter.
    Interval Newton only tightens that enclosure. It never replaces those proofs.
    Every Taylor coefficient is isolated using the same certified nonzero F_u;
    the root interval is retained as the constant term, never a sampled root.
    """
    from tools.opaque_shading_intervals import Jet
    if type(tight_endpoints)!=bool:raise ValueError('explicit endpoint refinement flag required')
    lo,hi=Q(lo),Q(hi);calls=0;endpoint_calls=0
    if not isinstance(s,Jet) or not 0<=lo<hi<=1 or not 0<=s[0,0].lo<=s[0,0].hi<=1 or not isinstance(knot,I) or knot.lo!=knot.hi:
        raise ValueError('unit region and exact knot required')
    def at(u,v):
        nonlocal calls
        checkpoint();calls+=1;q=field(u,v)
        if not isinstance(q,Jet) or q.order!=u.order:raise ValueError('matching smooth Jet field required')
        checkpoint();return q
    interval=lambda a,b:I(I(a).lo,I(b).hi)
    root=interval(lo,hi);sv=s[0,0]
    slope=at(Jet.variable(root,order=1),Jet(sv,order=1))[1,0]
    sign=1 if slope.lo>0 else -1 if slope.hi<0 else 0
    if not sign:raise ValueError('knot region not certified monotone')
    left=sign*(at(Jet(lo),Jet(sv))[0,0]-knot)
    right=sign*(at(Jet(hi),Jet(sv))[0,0]-knot)
    if tight_endpoints and (left.hi>=0 or right.lo<=0):
        before=calls
        def endpoint(u):
            natural=at(Jet(u),Jet(sv))[0,0]
            ds=at(Jet(u,order=1),Jet.variable(sv,order=1))[1,0]
            if ds.lo>=0 or ds.hi<=0:
                ends=[at(Jet(u),Jet(v))[0,0] for v in (sv.lo,sv.hi)]
                natural=I(max(natural.lo,min(v.lo for v in ends)),min(natural.hi,max(v.hi for v in ends)))
            return sign*(natural-knot)
        left,right=endpoint(lo),endpoint(hi);endpoint_calls=calls-before
    if left.hi>=0 or right.lo<=0:raise ValueError('knot not uniformly bracketed; boundary entry/exit unresolved')
    for _ in range(8):
        mid=(Q(root.lo)+Q(root.hi))/2
        proposal=I(mid)-(at(Jet(mid),Jet(sv))[0,0]-knot)/slope
        lower=max(root.lo,proposal.lo);upper=min(root.hi,proposal.hi)
        if lower>upper:raise ValueError('inconsistent knot enclosure')
        tightened=I(lower,upper)
        if tightened.lo==root.lo and tightened.hi==root.hi:break
        root=tightened
        slope=at(Jet.variable(root,order=1),Jet(sv,order=1))[1,0]
        if not (slope.lo>0 or slope.hi<0):raise ValueError('knot region lost monotonicity')
    coefficients={(0,0):root}
    # Total-degree order makes all lower coefficients known before (i,j).
    # With the current coefficient zero, F[i,j] is its remainder. The unknown
    # appears linearly as F_u*r[i,j], even for mixed parameter coefficients.
    for i,j in sorted(((i,j) for i in range(s.order+1) for j in range(2) if i+j),key=lambda k:(sum(k),k)):
        local=Jet(order=i,coeff={k:v for k,v in coefficients.items() if k[0]<=i})
        transverse=Jet(order=i,coeff={k:v for k,v in s.c.items() if k[0]<=i})
        coefficients[i,j]=-at(local,transverse)[i,j]/slope
    return dict(root=Jet(order=s.order,coeff=coefficients),radial_derivative=slope,
                evaluations=calls,endpoint_evaluations=endpoint_calls,unique=True,interior=True,
                scope='Uniform interior root over the supplied transverse interval; caller owns smoothness and physical map.')


def knot_strip(t,lower,upper):
    """Map t in [0,1] between two certified boundaries; return u and du/dt.

    The transverse coordinate is unchanged. Multiplying by the physical map's
    Jacobian retains its full parameter derivative. Shared moving edges cancel
    for continuous fields; unequal traces retain the boundary flux automatically.
    Caller supplies ordered boundaries independent of t (but possibly of s/theta).
    """
    width=upper-lower
    if width[0,0].lo<=0:raise ValueError('positive strip width must be certified')
    return lower+t*width,width


def knot_band(field,lo,hi,t,s,lower_knot,upper_knot,*,checkpoint=lambda:None):
    """Certified moving band between two exact levels of one smooth angle.

    Uniform radial monotonicity proves ordering even if separately enclosed root
    ranges overlap. The mean value theorem encloses the width as delta/F_u; only
    its constant coefficient is intersected, never its derivatives. Positive
    physical measure and all mixed terms remain explicit. No sampled branch
    selection or automatic parameter-flux correction is performed here.
    """
    from tools.opaque_shading_intervals import Jet
    if not all(isinstance(k,I) and k.lo==k.hi for k in (lower_knot,upper_knot)) or lower_knot.hi>=upper_knot.lo:
        raise ValueError('ordered exact knots required')
    if not isinstance(t,Jet) or not isinstance(s,Jet) or t.order!=s.order or not 0<=t[0,0].lo<=t[0,0].hi<=1:
        raise ValueError('matching unit chart jets required')
    first=knot_region(field,lo,hi,s,lower_knot,checkpoint=checkpoint)
    second=knot_region(field,lo,hi,s,upper_knot,checkpoint=checkpoint)
    checkpoint();slope=field(Jet.variable(I(I(Q(lo)).lo,I(Q(hi)).hi),order=1),Jet(s[0,0],order=1))[1,0];checkpoint()
    sign=1 if slope.lo>0 else -1 if slope.hi<0 else 0
    if not sign:raise ValueError('band not certified monotone')
    left,right=(first['root'],second['root']) if sign>0 else (second['root'],first['root'])
    width=right-left;bound=(upper_knot-lower_knot)/(sign*slope)
    width.c[0,0]=I(max(width[0,0].lo,bound.lo),min(width[0,0].hi,bound.hi))
    if width[0,0].lo<=0:raise ValueError('positive band width required')
    u=left+t*width
    # Both roots are interior and t is in [0,1]; use that proved image bound.
    u.c[0,0]=I(max(u[0,0].lo,I(Q(lo)).lo),min(u[0,0].hi,I(Q(hi)).hi))
    return dict(coordinate=u,jacobian=width,angle_range=I(lower_knot.lo,upper_knot.hi),
                orientation=sign,evaluations=first['evaluations']+second['evaluations']+1,
                scope='One interior band with fixed exact angular endpoints; transverse and physical Jacobians remain caller obligations.')


def knot_edge_events(field,box,knot,*,checkpoint=lambda:None):
    """Certify transverse crossings of a level through the two radial edges.

    Strict whole-box monotonicity in both coordinates excludes folds/tangencies.
    Endpoint uncertainty refuses. Events enclose locations and first parameter
    motion, but do not yet construct or integrate a clipped moving-domain chart.
    """
    from tools.opaque_shading_intervals import Jet
    a,b,c,d=map(Q,box);calls=0
    if not 0<=a<b<=1 or not 0<=c<d<=1 or not isinstance(knot,I) or knot.lo!=knot.hi:raise ValueError('unit edge box and exact knot')
    def at(u,s):
        nonlocal calls
        checkpoint();calls+=1;q=field(u,s);checkpoint();return q
    ix=I(I(a).lo,I(b).hi);iy=I(I(c).lo,I(d).hi)
    du=at(Jet.variable(ix,order=1),Jet(iy,order=1))[1,0]
    ds=at(Jet(ix,order=1),Jet.variable(iy,order=1))[1,0]
    if not (du.lo>0 or du.hi<0) or not (ds.lo>0 or ds.hi<0):raise ValueError('edge event not certified monotone')
    events=[];edges=[]
    for name,u in [('lower',a),('upper',b)]:
        values=[at(Jet(u),Jet(s))[0,0]-knot for s in (c,d)]
        signs=[1 if q.lo>0 else -1 if q.hi<0 else 0 for q in values]
        if 0 in signs:raise ValueError('uncertain or touching edge endpoint')
        edges.append(dict(edge=name,signs=signs))
        if signs[0]!=signs[1]:
            root=knot_region(lambda s,u:at(u,s),c,d,Jet(u),knot,checkpoint=checkpoint)
            events.append(dict(edge=name,location=root['root'][0,0],parameter_derivative=root['root'][0,1],unique=True))
    return dict(events=events,edges=edges,evaluations=calls,radial_sign=1 if du.lo>0 else -1,transverse_sign=1 if ds.lo>0 else -1,scope='Certified edge events only; no clipped chart, complete domain or integral acceptance.')


class KnotEntryChart:
    """Three smooth charts for one lower-edge entry (F_u>0, F_s<0).

    The parent is the disjoint union, up to shared edges, of the pre-entry slab
    and two post-entry strips. The entry and radial root retain their parameter
    derivatives. Only the below strip collapses at the entry; its zero boundary
    measure is permitted, never a reason to discard a positive interior sliver.
    Caller supplies a smooth field also on the explicit extended radial bracket
    and multiplies each returned Jacobian by the complete physical Jacobian.
    """
    def __init__(self,field,box,knot,*,checkpoint=lambda:None):
        from tools.opaque_shading_intervals import Jet
        self.box=tuple(map(Q,box));self.knot=knot;self.checkpoint=checkpoint
        self.evaluations=0;self._field=field;self.cache={}
        self.certificate=knot_edge_events(self.field,self.box,knot,checkpoint=checkpoint)
        proof=self.certificate
        if proof['radial_sign']!=1 or proof['transverse_sign']!=-1 or len(proof['events'])!=1 or proof['events'][0]['edge']!='lower' or proof['edges'][1]['signs']!=[1,1]:
            raise ValueError('single lower entry with positive upper edge required')
        a,b,c,d=self.box;self.extended_lower=max(a/2,a-(b-a))
        if self.extended_lower==a:raise ValueError('entry needs an interior extended radial bracket')
        event=proof['events'][0];self.entry=event['location'];self.entry_tangent=event['parameter_derivative']
        if not I(c).hi<self.entry.lo<=self.entry.hi<I(d).lo:raise ValueError('strict interior entry required')
        # Certify a smooth root extension across the whole closed post-entry
        # strip, including its degenerate boundary. The root theorem plus the
        # event ordering proves r(s)>=a there, even when intervals lose the
        # dependence between the entry and s at y=0.
        knot_region(self.field,self.extended_lower,b,Jet(I(self.entry.lo,I(d).hi)),knot,checkpoint=checkpoint,tight_endpoints=True)
    def field(self,u,s):
        self.checkpoint();self.evaluations+=1;result=self._field(u,s);self.checkpoint();return result
    def map(self,t,y):
        from tools.opaque_shading_intervals import Jet
        if not isinstance(t,Jet) or not isinstance(y,Jet) or t.order!=y.order or any(not 0<=z[0,0].lo<=z[0,0].hi<=1 for z in (t,y)):
            raise ValueError('matching unit chart jets required')
        self.checkpoint();a,b,c,d=self.box;e=t.lift(self.entry,tangent=self.entry_tangent)
        before=e-c;after=d-e
        pre=c+before*y;post=e+after*y
        pre.c[0,0]=I(max(pre[0,0].lo,I(c).lo),min(pre[0,0].hi,self.entry.hi))
        post.c[0,0]=I(max(post[0,0].lo,self.entry.lo),min(post[0,0].hi,I(d).hi))
        key=(post.order,tuple(sorted((ij,str(v.lo),str(v.hi)) for ij,v in post.c.items())))
        if key not in self.cache:
            if len(self.cache)>=64:self.cache.pop(next(iter(self.cache)))
            root=knot_region(self.field,self.extended_lower,b,post,self.knot,checkpoint=self.checkpoint,tight_endpoints=True)['root']
            root.c[0,0]=I(max(root[0,0].lo,I(a).lo),min(root[0,0].hi,I(b).hi))
            self.cache[key]=root
        root=self.cache[key];low=root-a;high=b-root
        for width in (low,high):width.c[0,0]=I(max(I(0).lo,width[0,0].lo),width[0,0].hi)
        return [dict(u=a+(b-a)*t,s=pre,jacobian=(b-a)*before,side='above'),
                dict(u=a+t*low,s=post,jacobian=low*after,side='below'),
                dict(u=root+t*high,s=post,jacobian=high*after,side='above')]

class MonotoneAngularCover:
    """Complete unit-square partition by ordered levels, including edge exits.

    Caller supplies a smooth field through the required derivative order. Its
    radial derivative must be strictly positive everywhere; the u=0 horizon
    has zero value and first parameter motion. The u=1 edge decreases strictly
    with s. Thus every positive level is either an interior root for all s, or
    has one certified exit. Adjacent levels partition every radial interval.
    Each band uses at most two charts, with no sampled masks or omitted wedges.
    This is a coordinate construction, not a shading accuracy certificate.
    """
    def __init__(self,field,levels,*,checkpoint=lambda:None,max_cells=256,max_depth=10,value_field=None,inverse_residual=None):
        from tools.opaque_shading_intervals import Jet
        if type(max_cells)!=int or not 1<=max_cells<=256 or type(max_depth)!=int or not 0<=max_depth<=10:
            raise ValueError('existing coverage limits required')
        if value_field is not None and not callable(value_field):raise ValueError('qualified value field must be callable')
        if inverse_residual is not None and not callable(inverse_residual):raise ValueError('qualified inverse residual must be callable')
        self.inverse_residual=inverse_residual
        self._field=field;self._value_field=value_field;self.checkpoint=checkpoint;self.evaluations=0;self.cache={}
        self.levels=tuple(levels)
        if not self.levels or any(not isinstance(k,I) or k.lo!=k.hi or k.lo<=0 for k in self.levels) or any(a.hi>=b.lo for a,b in zip(self.levels,self.levels[1:])):
            raise ValueError('positive ordered exact levels required')
        unit=I(0,1)
        self.radial=self.field(Jet.variable(unit,order=1),Jet(unit,order=1))[1,0]
        if self.radial.lo<=0:raise ValueError('strict positive whole radial derivative required')
        horizon=self.field(Jet(0),Jet(unit))
        if not horizon[0,0].zero or not horizon[0,1].zero:raise ValueError('fixed zero horizon and first motion required')
        self.edge_slope=self.field(Jet(1,order=1),Jet.variable(unit,order=1))[1,0]
        if self.edge_slope.hi>=0:raise ValueError('strict decreasing outer edge required')
        self.outer=[self.field(Jet(1),Jet(s))[0,0] for s in (0,1)]
        self.events={}
        for i,k in enumerate(self.levels):
            if k.hi<self.outer[1].lo:continue
            if not self.outer[1].hi<k.lo<=k.hi<self.outer[0].lo:
                raise ValueError('uncertain or outside outer edge level')
            event=knot_region(lambda s,u:self.field(u,s),Q(0),Q(1),Jet(1),k,checkpoint=checkpoint)['root']
            self.events[i]=event
        self.charts=[];n=len(self.levels)
        for band in range(n+1):
            low=None if band==0 else band-1
            high=None if band==n else band
            end=low if low in self.events else None
            if high is not None:
                # Before the upper level exits, both level roots are present.
                split=high if high in self.events else None
                self.charts.append(dict(band=band,low=low,high=high,start=None,end=split))
                if split is not None:
                    self.charts.append(dict(band=band,low=low,high=None,start=split,end=end))
            else:self.charts.append(dict(band=band,low=low,high=None,start=None,end=end))
        # Explicit balanced band tree, followed by a binary exit split where
        # needed. Count all internal nodes and leaves, as for rectangular covers.
        def assign(lo,hi,depth):
            if hi-lo==1:
                rows=[r for r in self.charts if r['band']==lo]
                for r in rows:r['depth']=depth+(len(rows)-1)
            else:
                mid=(lo+hi)//2;assign(lo,mid,depth+1);assign(mid,hi,depth+1)
        assign(0,n+1,0)
        self.cells=2*len(self.charts)-1;self.depth=max(r['depth'] for r in self.charts)
        if self.cells>max_cells:raise Unresolved('angular cover cell budget',cells=self.cells,max_cells=max_cells)
        if self.depth>max_depth:raise Unresolved('angular cover depth budget',depth=self.depth,max_depth=max_depth)
    def field(self,u,s):
        self.checkpoint();self.evaluations+=1;value=self._field(u,s);self.checkpoint();return value
    def field_value(self,u,s):
        """Caller-qualified spatial-only path for Newton values and slopes."""
        if self._value_field is None:return self.field(u,s)
        self.checkpoint();self.evaluations+=1;value=self._value_field(u,s);self.checkpoint();return value
    def residual(self,u,s,k):
        self.checkpoint();self.evaluations+=1;result=self.inverse_residual(u,s,k);self.checkpoint()
        from tools.opaque_shading_intervals import Jet
        if not isinstance(result,Jet) or result.order!=u.order:raise ValueError('matching residual output required')
        return result
    def residual_slope(self,root,s,k,normalized_slope):
        from tools.opaque_shading_intervals import Jet
        root_identity=getattr(self.inverse_residual,'root_slope',None)
        if root_identity is None:slope=self.residual(Jet.variable(root,order=1),Jet(s[0,0],order=1),Jet(k,order=1))[1,0]
        else:
            self.checkpoint();slope=root_identity(root,s[0,0],k,normalized_slope);self.checkpoint()
            if not isinstance(slope,I):raise ValueError('interval residual root slope required')
        if slope.lo<=0:raise ValueError('positive residual radial derivative required')
        return slope
    def _root(self,index,s):
        """Implicit Jet on a domain whose closed root existence was proved.

        Only map() calls this after constructing the relevant pre-exit domain.
        Monotonicity and event order prove the root lies in [0,1], including
        its boundary endpoint. Newton intersections refine that enclosure;
        they do not establish existence or turn a failed bracket into a root.
        """
        from tools.opaque_shading_intervals import Jet
        key=(index,s.order,tuple(sorted((ij,str(v.lo),str(v.hi)) for ij,v in s.c.items())))
        if key in self.cache:return self.cache[key]
        k=self.levels[index];root=I(0,1);slope=self.radial
        for _ in range(8):
            mid=(Q(root.lo)+Q(root.hi))/2
            proposal=I(mid)-(self.field_value(Jet(mid),Jet(s[0,0]))[0,0]-k)/slope
            refined=I(max(root.lo,proposal.lo),min(root.hi,proposal.hi))
            if refined.lo==root.lo and refined.hi==root.hi:break
            root=refined
            slope=self.field_value(Jet.variable(root,order=1),Jet(s[0,0],order=1))[1,0]
            # The globally proved derivative enclosure also applies here.
            slope=I(max(slope.lo,self.radial.lo),min(slope.hi,self.radial.hi))
        if self.inverse_residual is not None:slope=self.residual_slope(root,s,k,slope)
        coefficients={(0,0):root}
        for i,j in sorted(((i,j) for i in range(s.order+1) for j in range(2) if i+j),key=lambda z:(sum(z),z)):
            u=Jet(order=i,coeff={z:v for z,v in coefficients.items() if z[0]<=i})
            v=Jet(order=i,coeff={z:v for z,v in s.c.items() if z[0]<=i})
            coefficients[i,j]=-(self.field(u,v)[i,j] if self.inverse_residual is None else self.residual(u,v,Jet(k,order=i))[i,j])/slope
        result=Jet(order=s.order,coeff=coefficients)
        if len(self.cache)>=64:self.cache.pop(next(iter(self.cache)))
        self.cache[key]=result;return result
    def map(self,index,t,y):
        from tools.opaque_shading_intervals import Jet
        if type(index)!=int or not 0<=index<len(self.charts):raise ValueError('valid chart index required')
        if not isinstance(t,Jet) or not isinstance(y,Jet) or t.order!=y.order or any(not 0<=v[0,0].lo<=v[0,0].hi<=1 for v in (t,y)):
            raise ValueError('matching unit chart jets required')
        self.checkpoint();chart=self.charts[index]
        def event(i,default):
            return t.lift(default) if i is None else t.lift(self.events[i][0,0],tangent=self.events[i][0,1])
        a=event(chart['start'],0);b=event(chart['end'],1);span=b-a
        if chart['start'] is not None and chart['end'] is not None:
            bound=(self.levels[chart['start']]-self.levels[chart['end']])/(-self.edge_slope)
            span.c[0,0]=I(max(span[0,0].lo,bound.lo),min(span[0,0].hi,bound.hi))
        if span[0,0].lo<=0:raise ValueError('positive transverse span required')
        s=a+span*y;s.c[0,0]=I(max(s[0,0].lo,a[0,0].lo),min(s[0,0].hi,b[0,0].hi))
        low=t.lift(0) if chart['low'] is None else self._root(chart['low'],s)
        high=t.lift(1) if chart['high'] is None else self._root(chart['high'],s)
        width=high-low
        if chart['high'] is not None:
            delta=self.levels[chart['high']]-(I(0) if chart['low'] is None else self.levels[chart['low']])
            bound=delta/self.radial
            width.c[0,0]=I(max(width[0,0].lo,bound.lo),min(width[0,0].hi,bound.hi))
        else:width.c[0,0]=I(max(I(0).lo,width[0,0].lo),width[0,0].hi)
        u=low+t*width;u.c[0,0]=I(max(I(0).lo,u[0,0].lo),min(I(1).hi,u[0,0].hi))
        jacobian=width*span
        # Arb's outward ball export may cross zero at a collapsing boundary.
        # Nonnegativity follows from ordered roots and positive transverse span;
        # intersect only the value coefficient, retaining all mixed derivatives.
        jacobian.c[0,0]=I(max(I(0).lo,jacobian[0,0].lo),jacobian[0,0].hi)
        band=chart['band'];lower=I(0) if band==0 else self.levels[band-1]
        upper=self.outer[0] if band==len(self.levels) else self.levels[band]
        return dict(u=u,s=s,jacobian=jacobian,band=band,angular_range=I(lower.lo,upper.hi))

    def map_angle(self,index,t,y,radial_derivative):
        """Fixed-level interior band in angle coordinates, retaining transport.

        The existing cover proves existence and uniqueness for every target
        angle in this closed band. An implicit Taylor recurrence encloses the
        inverse and all mixed coefficients. Caller must supply the analytic
        partial F_u through the same order; positivity is checked again here.
        Edge-exit charts are deliberately refused, not silently approximated.
        """
        from tools.opaque_shading_intervals import Jet
        if type(index)!=int or not 0<=index<len(self.charts):raise ValueError('valid chart index required')
        if not isinstance(t,Jet) or not isinstance(y,Jet) or t.order!=y.order or any(not 0<=z[0,0].lo<=z[0,0].hi<=1 for z in (t,y)):
            raise ValueError('matching unit chart jets required')
        chart=self.charts[index]
        if chart['high'] is None or chart['start'] is not None or chart['end'] is not None:
            raise ValueError('fixed-level interior angle chart required')
        low=I(0) if chart['low'] is None else self.levels[chart['low']]
        high=self.levels[chart['high']];delta=high-low;target=t*delta+low
        u=self._angle_inverse(target,y)
        self.checkpoint();slope=radial_derivative(u,y);self.checkpoint()
        if slope[0,0].lo<=0:raise ValueError('positive analytic radial derivative required')
        return dict(u=u,s=y,angle=target,jacobian=t.lift(delta)/slope,band=chart['band'],angular_range=I(low.lo,high.hi))

    def _angle_inverse(self,target,y):
        from tools.opaque_shading_intervals import Jet
        key=('angle',target.order,tuple(sorted((ij,str(v.lo),str(v.hi)) for ij,v in target.c.items())),tuple(sorted((ij,str(v.lo),str(v.hi)) for ij,v in y.c.items())))
        if key in self.cache:u=self.cache[key]
        else:
            root=I(0,1);slope=self.radial
            for _ in range(8):
                mid=(Q(root.lo)+Q(root.hi))/2
                proposal=I(mid)-(self.field_value(Jet(mid),Jet(y[0,0]))[0,0]-target[0,0])/slope
                narrowed=I(max(root.lo,proposal.lo),min(root.hi,proposal.hi))
                if narrowed.lo==root.lo and narrowed.hi==root.hi:break
                root=narrowed
                slope=self.field_value(Jet.variable(root,order=1),Jet(y[0,0],order=1))[1,0]
                slope=I(max(slope.lo,self.radial.lo),min(slope.hi,self.radial.hi))
            if self.inverse_residual is not None:slope=self.residual_slope(root,y,target[0,0],slope)
            solver=getattr(self.inverse_residual,'implicit_coefficients',None)
            if solver is not None:
                self.checkpoint();coeff=solver(root,y,target,slope);self.checkpoint()
            else:
                coeff={(0,0):root}
                for i,j in sorted(((i,j) for i in range(target.order+1) for j in range(2) if i+j),key=lambda z:(sum(z),z)):
                    u=Jet(order=i,coeff={z:v for z,v in coeff.items() if z[0]<=i})
                    v=Jet(order=i,coeff={z:w for z,w in y.c.items() if z[0]<=i})
                    coeff[i,j]=(target[i,j]-self.field(u,v)[i,j])/slope if self.inverse_residual is None else -self.residual(u,v,Jet(order=i,coeff={z:w for z,w in target.c.items() if z[0]<=i}))[i,j]/slope
            u=Jet(order=target.order,coeff=coeff)
            if len(self.cache)>=64:self.cache.pop(next(iter(self.cache)))
            self.cache[key]=u
        return u

    def map_exit_angle(self,index,t,y,radial_derivative):
        """Positive exiting band: F=low+t*(F(1,s)-low), s=a+(b-a)y.

        Existing ordered exit events prove the angle width nonnegative and the
        root unique. Only the boundary can collapse. Preserve the full moving
        target, inverse and physical Jacobian; horizon charts are excluded.
        """
        from tools.opaque_shading_intervals import Jet
        if type(index)!=int or not 0<=index<len(self.charts):raise ValueError('valid chart index required')
        if not isinstance(t,Jet) or not isinstance(y,Jet) or t.order!=y.order or any(not 0<=z[0,0].lo<=z[0,0].hi<=1 for z in (t,y)):raise ValueError('matching unit chart jets required')
        chart=self.charts[index]
        if chart['high'] is not None or chart['low'] is None:raise ValueError('positive exiting angle chart required')
        def event(i,default):return t.lift(default) if i is None else t.lift(self.events[i][0,0],tangent=self.events[i][0,1])
        a=event(chart['start'],0);b=event(chart['end'],1);span=b-a
        if chart['start'] is not None and chart['end'] is not None:
            bound=(self.levels[chart['start']]-self.levels[chart['end']])/(-self.edge_slope)
            span.c[0,0]=I(max(span[0,0].lo,bound.lo),min(span[0,0].hi,bound.hi))
        if span[0,0].lo<=0:raise ValueError('positive transverse span required')
        s=a+span*y;s.c[0,0]=I(max(s[0,0].lo,a[0,0].lo),min(s[0,0].hi,b[0,0].hi))
        low=self.levels[chart['low']];edge_method=getattr(self._field,'edge',None)
        edge=self.field(t.lift(1),s) if edge_method is None else edge_method(s);delta=edge-low
        delta.c[0,0]=I(max(I(0).lo,delta[0,0].lo),delta[0,0].hi)
        target=t*delta+low;u=self._angle_inverse(target,s)
        self.checkpoint();slope=radial_derivative(u,s);self.checkpoint()
        if slope[0,0].lo<=0:raise ValueError('positive analytic radial derivative required')
        jacobian=delta*span/slope
        jacobian.c[0,0]=I(max(I(0).lo,jacobian[0,0].lo),jacobian[0,0].hi)
        upper=self.outer[0] if chart['band']==len(self.levels) else self.levels[chart['band']]
        return dict(u=u,s=s,angle=target,jacobian=jacobian,band=chart['band'],angular_range=I(low.lo,upper.hi),coordinate_kind='angle')

    def map_angle_cover(self,index,t,y,radial_derivative):
        """Use the angle inverse only on fully interior positive bands.

        The existing radial map remains responsible for horizon and edge-exit
        charts. Both mappings cover the same certified regions with complete
        Jacobians. This avoids extending the angle inverse into broad collapsing
        regions whose high-order integration bounds have not qualified.
        """
        if type(index)!=int or not 0<=index<len(self.charts):raise ValueError('valid chart index required')
        chart=self.charts[index]
        if chart['low'] is None or chart['high'] is None or chart['start'] is not None or chart['end'] is not None:
            row=self.map(index,t,y);row['coordinate_kind']='radial';return row
        row=self.map_angle(index,t,y,radial_derivative);row['coordinate_kind']='angle';return row
