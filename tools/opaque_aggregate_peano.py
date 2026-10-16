"""Complete residual third derivatives and absolute sums of every knot jump.

Directional monotonicity over the complete box proves at most one crossing of
each knot per line. Prefix sums bound all angular jumps; chain/product rules
retain value, slope, curvature and simultaneous crossings conservatively.
"""
from tools.opaque_shading_intervals import I,Jet
from tools.opaque_residual_range import ResidualRangeIndex
from tools.opaque_peano import residual_integral

class AggregateMaterial:
    def __init__(self,correction):
        self.c=correction;self.checkpoint=correction.checkpoint
        self.index=correction.range_index or ResidualRangeIndex(correction.extension,checkpoint=self.checkpoint)
        correction.range_index=self.index
    def ordinary_inputs(self,u,values):
        nv,nl,_,jac=values;e=self.c.extension;mix=self.c.mix
        if getattr(self.c,'weighted_light_residual',False):
            dv=self.index.delta_jet(nv,e.indices[0]);wdl=self.index.weighted_delta_jet(nl,e.indices[1])
            av=self.c.smooth.at(nv,e.indices[0]);bl=self.c.smooth.at(nl,e.indices[1])
            # nl*(dv*bl + av*dl + dv*dl), with nl attached to its own
            # branch restriction before taking the outward derivative hull.
            return [u.lift(0),(nl*mix(dv)*mix(bl)+(mix(av)+mix(dv))*mix(wdl))*jac/(u.lift(e.pi)*u.lift(self.c.mean))]
        dv=self.index.delta_jet(nv,e.indices[0]);dl=self.index.delta_jet(nl,e.indices[1])
        av=self.c.smooth.at(nv,e.indices[0]);bl=self.c.smooth.at(nl,e.indices[1])
        return [u.lift(0),nl*(mix(dv)*mix(bl)+mix(av)*mix(dl)+mix(dv)*mix(dl))*jac/(u.lift(e.pi)*u.lift(self.c.mean))]
    def ordinary(self,u,s):return self.ordinary_inputs(u,self.c.inputs(u,s))
    def axis_bounds(self,box,axis):
        self.checkpoint();args=[Jet(v,order=3) for v in box];args[axis]=Jet.variable(box[axis],order=3)
        values=self.c.inputs(*args);nv,nl,_,jac=values;ordinary=self.ordinary_inputs(args[0],values)[1];jumps=[I(0)]*3
        for role,q in enumerate((nv,nl)):
            if q[1,0].zero:continue
            sums=self.index.jump_sums(q[0,0]);sums=[self.c.mix([sums[0][j],sums[1][j]]) for j in range(3)]
            if all(v.zero for v in sums):continue
            if q[1,0].lo<0<q[1,0].hi:raise ValueError('aggregate angular crossings need certified monotonicity')
            other=(nl,nv)[role];actual=self.c.mix(self.index.actual_jet(other))
            p=nl*actual*jac/(q.lift(self.c.extension.pi)*q.lift(self.c.mean))
            a,b,c=[I(p[j,0].bound) for j in range(3)];d=I(q[1,0].bound);dd=2*I(q[2,0].bound);z,s,t=sums
            row=[z*a,s*d*a+z*b,t*d*d*a+s*(dd*a+2*d*b)+2*z*c]
            jumps=[x+y for x,y in zip(jumps,row)]
        return [dict(third=I(0),jumps=[I(0)]*3),dict(third=6*I(ordinary[3,0].bound),third_range=6*ordinary[3,0],jumps=jumps)]
    def integrated_bounds(self,box,axis):
        from tools.opaque_aggregate_transport import integrated_jumps
        rows=self.axis_bounds(box,axis);rows[1]['global_integral']=integrated_jumps(self,box,axis)
        return rows
    def __call__(self,box,*,integrated_axis=None):
        def directions(rows):self.c.preferred_split_axis=0 if max(v.hi for v in rows[0])>=max(v.hi for v in rows[1]) else 1
        def breakdown(rows):self.c.error_breakdown=rows
        if integrated_axis is not None:
            from tools.opaque_transverse_peano import conditioned_integral
            return conditioned_integral(self.ordinary,box,self.axis_bounds,self.integrated_bounds,checkpoint=self.checkpoint,error_components=directions,error_breakdown=breakdown,integrated_axis=integrated_axis)
        return residual_integral(self.ordinary,box,self.axis_bounds,checkpoint=self.checkpoint,error_components=directions,error_breakdown=breakdown)
