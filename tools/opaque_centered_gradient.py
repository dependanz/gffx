"""Mean-value inclusions for smooth geometric inputs, never material branches.

For g=partial_i f, C2 geometry gives
  g(B) in g(mid(B)) + sum_j partial_j g(B)*(B_j-mid(B)_j).
Intersect with the natural jet coefficient. Taylor coefficient(2,0) is half
of the same-axis second derivative; coefficient(1,1) is the mixed derivative.
Plain unit primary coordinates or one constant unit auxiliary direction are
recognized. For composed primaries, only the constant auxiliary coefficient is
refined; higher primary and mixed coefficients are never reinterpreted.
"""
from fractions import Fraction as Q
from tools.opaque_shading_intervals import I,Jet
from tools.opaque_boundary_material import BoundaryMaterial

class CenteredInputs:
    def __init__(self,inputs,*,checkpoint=lambda:None,refine_values=False):
        if type(refine_values)!=bool:raise ValueError("Boolean geometric value refinement required")
        self.refine_values=refine_values
        self.inputs=inputs;self.checkpoint=checkpoint;self.refinements=0;self.recovered_signs=0
    def __call__(self,x,y):
        self.checkpoint();values=self.inputs(x,y);self.checkpoint()
        if not isinstance(x,Jet) or not isinstance(y,Jet) or x.order!=y.order :return values
        terms=[{k:v for k,v in q.c.items() if k!=(0,0) and not v.zero} for q in (x,y)]
        active=[i for i,t in enumerate(terms) if set(t)=={(1,0)} and t[1,0].lo==t[1,0].hi==1 and not terms[1-i]]
        coefficient=(1,0)
        if len(active)!=1:
            tangents=[{k:v for k,v in q.c.items() if k[1] and not v.zero} for q in (x,y)]
            active=[i for i,t in enumerate(tangents) if set(t)=={(0,1)} and t[0,1].lo==t[0,1].hi==1 and not tangents[1-i]]
            coefficient=(0,1)
        if len(active)!=1:return values
        selected=[i for i,q in enumerate(values[:2]) if isinstance(q,Jet) and q[coefficient].lo<0<q[coefficient].hi]
        if not selected:return values
        axis=active[0];other=1-axis;box=[x[0,0],y[0,0]];mid=[I((Q(v.lo)+Q(v.hi))/2) for v in box]
        whole=[Jet(v,order=2) for v in box];whole[axis]=Jet.variable(box[axis],order=2);whole[other]=Jet.variable(box[other],axis=1,order=2)
        center=[Jet(v,order=2) for v in mid];center[axis]=Jet.variable(mid[axis],order=2)
        self.checkpoint();derivatives=self.inputs(*whole);self.checkpoint();at=self.inputs(*center);self.checkpoint();result=list(values);deltas=[v-m for v,m in zip(box,mid)]
        for i in selected:
            q=values[i];a=derivatives[i];z=at[i];enclosure=z[1,0]+2*a[2,0]*deltas[axis]+a[1,1]*deltas[other]
            lo=max(q[coefficient].lo,enclosure.lo);hi=min(q[coefficient].hi,enclosure.hi)
            if lo>hi:raise ValueError('disjoint geometric derivative inclusions')
            copy=Jet(order=q.order,coeff=q.c);copy.c[coefficient]=I(lo,hi)
            if self.refine_values:
                # C1 mean-value inclusion on the physical coordinate rectangle.
                # Reuse already computed geometry; never differentiate material
                # branches or reinterpret higher composed jet coefficients.
                value=z[0,0]+a[1,0]*deltas[axis]+a[0,1]*deltas[other]
                vlo=max(q[0,0].lo,value.lo);vhi=min(q[0,0].hi,value.hi)
                if vlo>vhi:raise ValueError('disjoint geometric value inclusions')
                copy.c[0,0]=I(vlo,vhi)
            result[i]=copy
            self.refinements+=1;self.recovered_signs+=int(lo>0 or hi<0)
        return tuple(result) if isinstance(values,tuple) else result

class CenteredBoundaryMaterial(BoundaryMaterial):
    refine_values=False
    def __call__(self,box,*,integrated_axis=0):
        c=self.c;original=c.inputs;role=getattr(c,'role_inputs',None);wrapped=CenteredInputs(original,checkpoint=c.checkpoint,refine_values=self.refine_values);c.inputs=wrapped;roles=[]
        if callable(role):
            roles=[CenteredInputs(lambda x,y,k=k:[role(k,x,y)],checkpoint=c.checkpoint,refine_values=self.refine_values) for k in (0,1)]
            c.role_inputs=lambda k,x,y:roles[k](x,y)[0]
        try:return super().__call__(box,integrated_axis=integrated_axis)
        finally:
            c.inputs=original
            if callable(role):c.role_inputs=role
            self.refinements=getattr(self,'refinements',0)+sum(v.refinements for v in [wrapped,*roles])
            self.recovered_signs=getattr(self,'recovered_signs',0)+sum(v.recovered_signs for v in [wrapped,*roles])

class ValueCenteredBoundary(CenteredBoundaryMaterial):
    """Opt-in geometric value intersection as well as derivative refinement."""
    refine_values=True
