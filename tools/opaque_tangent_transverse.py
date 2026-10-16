"""Certified transverse integration of spatial jumps in both tangent channels.

Each ordinary parameter derivative is a scalar piecewise spatial integrand.
For q(x,y)=k with both partials strict, dy=abs(q_x/q_y) dx along
the crossing. Use clamped endpoint roots for absent/partial/complete sweeps,
retain coefficient variation and every jump order, then intersect with the
original bound. No distributional parameter flux is removed or supplied here.
"""
from fractions import Fraction as Q
from tools.opaque_shading_intervals import I,Jet
from tools.opaque_transverse_peano import clamped_crossing
from tools.opaque_high_transport import weighted_mass as weighted


def contract_crossing(inputs,box,axis,role,knot,position,*,partial_bound=None,checkpoint=lambda:None):
    """Up to four interval-Newton contractions retain every initial root.

    The transverse interval stays whole. At each iteration the mean-value
    derivative covers the entire current root bracket, so intersection cannot
    exclude a root. Unknown slopes retain the bracket; this never proves that
    an interior crossing exists and must not contract a clipped endpoint.
    """
    from tools.opaque_peano import crossing_location
    if type(axis)!=int or axis not in (0,1) or type(role)!=int or role not in (0,1) or len(box)!=2:raise ValueError('two-dimensional declared crossing required')
    if partial_bound is not None and not callable(partial_bound):raise ValueError('independently qualified full-bracket derivative required')
    position=I(position);knot=I(knot)
    if not box[axis].lo<=position.lo<=position.hi<=box[axis].hi:raise ValueError('root bracket inside physical domain required')
    # Work selection only: keep the entire nonzero certified bracket.
    # Downstream jump bounds and the complete-piece sum retain its uncertainty.
    target=(I(box[axis].width)*I('1e-8')).lo
    for _ in range(4):
        checkpoint()
        if position.width<=target:return position
        narrow=list(box);narrow[axis]=position
        # The explicit callback is an independently qualified full-bracket
        # derivative enclosure. Rebuilding a field jet is unnecessary for the
        # mean-value proof; the midpoint field value is still evaluated below.
        if partial_bound is not None:
            slope=I(partial_bound(role,tuple(narrow),axis))
        else:
            args=[Jet(v,order=1) for v in narrow];args[axis]=Jet.variable(position,order=1)
            slope=inputs(*args)[role][1,0]
        if slope.lo<=0<=slope.hi:return position
        args=[Jet(v) for v in narrow];args[axis]=Jet(I((Q(position.lo)+Q(position.hi))/2))
        center=inputs(*args)[role][0,0];new=crossing_location(position,center,slope,knot)
        if new is None:return None
        if new.lo==position.lo and new.hi==position.hi:return position
        position=new
    checkpoint();return position


def refine(rows,inputs,box,axis,knots,*,checkpoint=lambda:None,partial_bound=None,partial_ratio=None,root_parts=1,jump_evaluator=None,refine_small=False):
    if type(refine_small)!=bool:raise ValueError('explicit small-bound refinement selection required')
    if type(root_parts)!=int or root_parts not in (1,2,4) or root_parts>1 and not callable(jump_evaluator):raise ValueError('qualified two/four-part root evaluator required')
    if axis not in (0,1) or len(box)!=2:raise ValueError('two-dimensional transverse proof required')
    if partial_bound is not None and not callable(partial_bound):raise ValueError('callable exact partial proof required')
    if partial_ratio is not None and not callable(partial_ratio):raise ValueError('callable exact slope ratio required')
    other=1-axis;measure=I(box[other].hi)-I(box[other].lo)
    if measure.lo<=0:return rows
    h=I(box[axis].hi)-I(box[axis].lo)
    if h.lo<=0:raise ValueError('positive integration domain required')
    if not any(row["jumps"] for row in rows):return rows
    derivatives=[]
    for active in (axis,other):
        checkpoint();args=[Jet(v,order=1) for v in box];args[active]=Jet.variable(box[active],order=1);derivatives.append([q[1,0] for q in inputs(*args)[:2]])
    candidates={}
    for role in {w['role'] for r in rows for w in r['jumps']}:
        slopes=[v[role] for v in derivatives]
        if partial_bound is not None:
            for j,active in enumerate((axis,other)):
                checkpoint();tight=partial_bound(role,box,active);old=slopes[j];slopes[j]=I(max(old.lo,tight.lo),min(old.hi,tight.hi))
        inner,outer=slopes
        if any(v.lo<=0<=v.hi for v in (inner,outer)):continue
        edges=[]
        for edge in (box[other].lo,box[other].hi):
            values=[]
            for x in (box[axis].lo,box[axis].hi,I((Q(box[axis].lo)+Q(box[axis].hi))/2)):
                checkpoint();args=[Jet(v) for v in box];args[other]=Jet(I(edge));args[axis]=Jet(I(x));values.append(inputs(*args)[role][0,0])
            edges.append(values)
        ratio=inner/outer
        if partial_ratio is not None:
            exact=partial_ratio(role,box,axis);ratio=I(max(ratio.lo,exact.lo),min(ratio.hi,exact.hi))
        if ratio.hi<0:ratio=-ratio
        for k in {w['knot'] for r in rows for w in r['jumps'] if w['role']==role}:
            roots=[]
            for endpoint,edge in zip((box[other].lo,box[other].hi),edges):
                knot=knots[k];root=clamped_crossing(box[axis],*edge,inner,knot)
                # Only an independently established interior crossing can use
                # Newton contraction. Absent/uncertain/clipped cases retain
                # their complete original endpoint possibilities.
                interior=(edge[0].hi<knot.lo and edge[1].lo>knot.hi) if inner.lo>0 else (edge[1].hi<knot.lo and edge[0].lo>knot.hi)
                if interior and root.width:
                    line=list(box);line[other]=I(endpoint)
                    narrowed=contract_crossing(inputs,tuple(line),axis,role,knot,root,partial_bound=partial_bound,checkpoint=checkpoint)
                    if narrowed is None:raise ValueError('certified interior crossing lost')
                    root=narrowed
                t=(root-I(box[axis].lo))/h;roots.append(I(max(I(0).lo,t.lo),min(I(1).hi,t.hi)))
            if (-outer/inner).hi<0:roots.reverse()
            candidates[role,k]=(roots,ratio)
    # The unpartitioned change-of-variable result is already a complete
    # enclosure. Refinement is optional work, never a small-term omission.
    baseline={}
    for channel,row in enumerate(rows):
        for witness in row['jumps']:
            key=(witness['role'],witness['knot']);proof=candidates.get(key)
            if proof is None:continue
            checkpoint();roots,ratio=proof
            baseline[channel,*key]=sum((weighted(v*ratio*h**(j+2),*roots,j+1) for j,v in enumerate(witness['values']) if not v.zero),I(0))
    partitioned={}
    if root_parts>1:
        import math
        for (role,k),(roots,whole_ratio) in candidates.items():
            checkpoint()
            existing=[v for (channel,r,index),v in baseline.items() if (r,index)==(role,k)]
            if not refine_small and existing and max(v.width for v in existing)<=I('1e-14').lo:continue
            lo,hi=Q(roots[0].lo),Q(roots[1].hi)
            if lo>=hi:continue
            total=[I(0),I(0)];order=rows[0]['order']
            for part in range(root_parts):
                checkpoint();left=I(lo+(hi-lo)*Q(part,root_parts));right=I(lo+(hi-lo)*Q(part+1,root_parts))
                # Outward clipping preserves uncertain, absent and partial sweeps.
                clamp=lambda z:I(max(left.lo,min(right.lo,z.lo)),max(left.hi,min(right.hi,z.hi)))
                a,b=map(clamp,roots)
                if a.lo==a.hi==b.lo==b.hi:continue
                support=list(box);inner_domain=I(box[axis].lo)+h*I(a.lo,b.hi)
                support[axis]=I(max(box[axis].lo,inner_domain.lo),min(box[axis].hi,inner_domain.hi))
                outer_domain=contract_crossing(inputs,tuple(support),other,role,knots[k],box[other],partial_bound=partial_bound,checkpoint=checkpoint)
                if outer_domain is None:continue
                support[other]=outer_domain;support=tuple(support)
                if partial_ratio is not None:ratio=partial_ratio(role,support,axis)
                else:
                    partials=[]
                    for active in (axis,other):
                        args=[Jet(v,order=1) for v in support];args[active]=Jet.variable(support[active],order=1)
                        partials.append(inputs(*args)[role][1,0])
                    ratio=partials[0]/partials[1]
                if ratio.hi<0:ratio=-ratio
                ratio=I(max(ratio.lo,whole_ratio.lo),min(ratio.hi,whole_ratio.hi))
                if partial_ratio is None:slope=partials[0]
                else:
                    args=[Jet(v,order=1) for v in support];args[axis]=Jet.variable(support[axis],order=1)
                    slope=inputs(*args)[role][1,0]
                if partial_bound is not None:
                    tight=partial_bound(role,support,axis);slope=I(max(slope.lo,tight.lo),min(slope.hi,tight.hi))
                if slope.lo<=0<=slope.hi:raise ValueError('strict root-partition orientation required')
                orientation=1 if slope.lo>0 else -1
                jump=jump_evaluator(support,axis,role,k,order)
                for channel in (0,1):
                    total[channel]=total[channel]+sum((weighted(jump[j,channel]*math.factorial(j)*orientation*ratio*h**(j+2),a,b,j+1) for j in range(order) if not jump[j,channel].zero),I(0))
            partitioned[role,k]=total
    for channel,row in enumerate(rows):
        for witness in row['jumps']:
            proof=candidates.get((witness['role'],witness['knot']))
            if proof is None:continue
            checkpoint();roots,ratio=proof
            witness['transverse_integral']=baseline[channel,witness['role'],witness['knot']]
            if (witness['role'],witness['knot']) in partitioned:
                old=witness['transverse_integral'];new=partitioned[witness['role'],witness['knot']][channel]
                witness['transverse_integral']=I(max(old.lo,new.lo),min(old.hi,new.hi))
            witness['transverse_measure']=measure
    return rows


class CosinePartials:
    """Outward spatial cosine slopes after exact rational cancellation.

    q=A/sqrt(B), q_i=(B A_i-A B_i/2)/B**(3/2).
    Keep the full first-parameter polynomial while forming the identity, then
    evaluate the requested spatial or first-parameter partial at theta=0.
    """
    def __init__(self,coefficients):
        from tools.opaque_angle_light import _clean,_add,_mul,_diff,PreparedIntervalPolynomial
        def field(rows):
            if len(rows)!=3 or any(len(row)!=3 for row in rows):raise ValueError('three affine normal/view components required')
            return [_clean({(i,j,k):v[k] for (i,j),v in zip(((0,0),(1,0),(1,1)),row) for k in (0,1)}) for row in rows]
        def dot(a,b):
            result={}
            for x,y in zip(a,b):result=_add(result,_mul(x,y))
            return result
        n,v=field(coefficients['n']),field(coefficients['v']);N=dot(n,n);C={}
        for q,c in zip(n,(1,1,2)):C=_add(C,q,c)
        self.rows=[]
        for A,B in [(dot(n,v),_mul(N,dot(v,v))),(C,_clean({k:6*v for k,v in N.items()}))]:
            numerators=[_add(_mul(B,_diff(A,axis)),_mul(A,_diff(B,axis)),Q(-1,2)) for axis in (0,1,2)]
            self.rows.append(tuple(PreparedIntervalPolynomial(p) for p in [B,*numerators]))
    def bound(self,role,box,axis):
        if type(role)!=int or role not in (0,1) or type(axis)!=int or axis not in (0,1,2) or len(box)!=2:raise ValueError('declared cosine and spatial/parameter axis required')
        if any(not isinstance(v,I) or v.lo<0 or v.hi>1 for v in box):raise ValueError('physical unit domain required')
        B=self.rows[role][0](*box)
        if B.lo<=0:raise ValueError('positive cosine norm required')
        return self.rows[role][axis+1](*box)/(B*B.sqrt())

    def ratio(self,role,box,axis):
        """q_axis/q_other cancels the identical positive B**(3/2).

        Exact rational numerator polynomials retain the whole spatial box.
        The caller separately certifies both partial signs and monotonicity.
        """
        if type(role)!=int or role not in (0,1) or type(axis)!=int or axis not in (0,1) or len(box)!=2:raise ValueError('declared spatial slope ratio required')
        if any(not isinstance(v,I) or v.lo<0 or v.hi>1 for v in box):raise ValueError('physical unit domain required')
        return self.rows[role][axis+1](*box)/self.rows[role][2-axis](*box)

    def motion_bound(self,role,u,s):
        if not isinstance(u,Jet) or not isinstance(s,Jet) or u.order!=s.order:raise ValueError('matching coordinate jets required')
        box=(u[0,0],s[0,0]);value=self.bound(role,box,2)
        for axis,q in enumerate((u,s)):
            if not q[0,1].zero:value=value+self.bound(role,box,axis)*q[0,1]
        return value
