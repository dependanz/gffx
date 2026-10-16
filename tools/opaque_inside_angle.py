"""Explicit inside-horizon angle orientation and constant-edge coverage.

The positive island contains its radial origin. Use u=1-z, where z is the
original inside map: u=0 is the moving horizon; u=1 is the fixed radial origin.
The physical measure remains positive and its complete motion is retained.
This profile does not alter or admit the outside-horizon constructors.
"""
from fractions import Fraction as Q
from tools.opaque_shading_intervals import I,Jet,Unresolved
from tools.opaque_prepared_angle import PreparedAngle
from tools.opaque_knot_partition import MonotoneAngularCover

class InsideAngle(PreparedAngle):
    def __init__(self,p):
        if not p['curved'] or not p['patch'].inside:raise ValueError('inside curved profile required')
        self.patch=p['patch'];self.coefficients=p['fields'].coefficients
        self._entries={};self.hits=0;self.misses=0
    def _prepare(self,s,value_only=False):
        root,disc,_,aa,fields=super()._prepare(s,value_only)
        return root,disc,-root,aa,fields
    def _inputs(self,u,s,*,value_only=False):
        if not isinstance(u,Jet) or not isinstance(s,Jet) or u.order!=s.order:raise ValueError('matching jets required')
        if any(x[0,0].lo<0 or x[0,0].hi>1 for x in (u,s)):raise ValueError('unit parameters required')
        root,disc,_,aa,fields=self._prepare(s,value_only);width=-root;t=root+u*width
        # Exact t=root*(1-u) gives the certified physical range. Tighten only
        # the value; intrinsic and mixed motion coefficients remain complete.
        exact=root*(1-u);t.c[0,0]=I(max(t[0,0].lo,exact[0,0].lo),min(t[0,0].hi,exact[0,0].hi))
        n=[a+t*b for a,b in fields['n']];v=[a+t*b for a,b in fields['v']]
        dot=lambda a,b:sum((x*y for x,y in zip(a,b)),u.lift(0))
        nn,vv=dot(n,n),dot(v,v);den=(nn*vv).sqrt()
        delta=width*u;numerator=self.patch.original_factor*delta*(disc+aa*delta)
        return numerator,den,nn,vv,n,v,fields,width,disc,aa,delta,dot
    def __call__(self,u,s):
        if isinstance(u,Jet) and u[0,0].lo==u[0,0].hi==1 and all(k==(0,0) for k in u.c):return self.edge(s)
        return super().__call__(u,s)
    def edge(self,s):
        if not isinstance(s,Jet) or not 0<=s[0,0].lo<=s[0,0].hi<=1:raise ValueError('unit edge jet required')
        fields={name:[s.lift(a[0],tangent=a[1]) for a,b,c in self.coefficients[name]] for name in ('n','v')}
        dot=lambda a,b:sum((x*y for x,y in zip(a,b)),s.lift(0))
        n,v=fields['n'],fields['v'];return dot(n,v)/(dot(n,n)*dot(v,v)).sqrt()
    def radial(self,u,s):
        numerator,den,nn,vv,n,v,fields,width,disc,aa,delta,dot=self._inputs(u,s)
        dn=[width*b for a,b in fields['n']];dv=[width*b for a,b in fields['v']]
        derivative_numerator=self.patch.original_factor*width*(disc+2*aa*delta)
        return derivative_numerator/den-(numerator/den)*(dot(n,dn)/nn+dot(v,dv)/vv)

class ConstantEdgeAngularCover(MonotoneAngularCover):
    """Every level lies below a certified constant-in-s edge: no exit wedges.

    Intrinsic parameter motion of that edge is allowed and retained by the
    existing inverse/exit-map recurrence. The edge must be constant throughout
    the entire transverse interval, including its first parameter derivative.
    """
    def __init__(self,field,levels,*,checkpoint=lambda:None,max_cells=256,max_depth=10,value_field=None,inverse_residual=None):
        if type(max_cells)!=int or not 1<=max_cells<=256 or type(max_depth)!=int or not 0<=max_depth<=10:raise ValueError('existing coverage limits required')
        if value_field is not None and not callable(value_field) or inverse_residual is not None and not callable(inverse_residual):raise ValueError('qualified field callbacks required')
        self.inverse_residual=inverse_residual;self._field=field;self._value_field=value_field;self.checkpoint=checkpoint;self.evaluations=0;self.cache={};self.levels=tuple(levels)
        if not self.levels or any(not isinstance(k,I) or k.lo!=k.hi or k.lo<=0 for k in self.levels) or any(a.hi>=b.lo for a,b in zip(self.levels,self.levels[1:])):raise ValueError('positive ordered exact levels required')
        unit=I(0,1);self.radial=self.field(Jet.variable(unit,order=1),Jet(unit,order=1))[1,0]
        if self.radial.lo<=0:raise ValueError('strict positive whole radial derivative required')
        horizon=self.field(Jet(0),Jet(unit))
        if not horizon[0,0].zero or not horizon[0,1].zero:raise ValueError('fixed zero horizon and first motion required')
        edge=self.field(Jet(1,order=1),Jet.variable(unit,order=1));self.edge_slope=edge[1,0]
        if not edge[1,0].zero or not edge[1,1].zero:raise ValueError('constant transverse edge and motion required')
        self.outer=[self.field(Jet(1),Jet(s))[0,0] for s in (0,1)]
        if any(k.hi>=edge[0,0].lo for k in self.levels):raise ValueError('strictly interior edge levels required')
        self.events={};n=len(self.levels);self.charts=[dict(band=i,low=None if i==0 else i-1,high=None if i==n else i,start=None,end=None) for i in range(n+1)]
        def assign(lo,hi,depth):
            if hi-lo==1:self.charts[lo]['depth']=depth
            else:
                mid=(lo+hi)//2;assign(lo,mid,depth+1);assign(mid,hi,depth+1)
        assign(0,n+1,0);self.cells=2*len(self.charts)-1;self.depth=max(r['depth'] for r in self.charts)
        if self.cells>max_cells:raise Unresolved('angular cover cell budget',cells=self.cells,max_cells=max_cells)
        if self.depth>max_depth:raise Unresolved('angular cover depth budget',depth=self.depth,max_depth=max_depth)


def inside_material_cover(data,*,piece,motion,anchor=0,checkpoint=lambda:None,polynomial_light=False,world_profile=None):
    """Fresh inside geometry and constant-edge angular bands; no acceptance."""
    from tools.opaque_original_motion_reference import prepare,angular_material_chart,mapped_inputs,_angular_selection
    from tools.opaque_horizon_cache import HorizonCache
    from tools.opaque_prepared_angle import MaterialInputCache
    from tools.opaque_principled_reference import PreparedPrincipled
    from tools.opaque_loss_extension import bounded_cosines
    if type(polynomial_light)!=bool:raise ValueError('explicit polynomial light profile required')
    _angular_selection(piece,motion,anchor)
    checkpoint();p=prepare(data,piece=piece,motion=motion,anchor=anchor,tight_discriminant=True,checkpoint=checkpoint,**({'world_profile':world_profile} if world_profile is not None else {}))
    if not p['curved'] or not p['patch'].inside:raise ValueError('inside curved profile required')
    p['horizon_cache']=HorizonCache(p['patch'].sqrt_discriminant);p['patch'].sqrt_discriminant=p['horizon_cache']
    p['prepared_angle']=InsideAngle(p);p['material_inputs_cache']=MaterialInputCache()
    from tools.opaque_algebraic_angle_inverse import UnsquaredAngleResidual
    p['inverse_residual']=UnsquaredAngleResidual(p['prepared_angle'])
    p['physical_angle']=InsidePhysicalAngleResidual(p['fields'].coefficients,compiled_series=True,contraction_stop=True)
    p['polynomial_light']=polynomial_light
    if polynomial_light:
        from tools.opaque_polynomial_inputs import PolynomialCosines
        p['polynomial_light_model']=PolynomialCosines(p['fields'].coefficients)
    p.update(inside_angle=True,curved_transverse=True,material_inputs=True,mixed_curve=True,conditioned_correction=True,correction_order=4,gradient_guidance=True,full_light_differential=False,exit_angle=True,mapped_transport=True)
    curve=PreparedPrincipled(data['budget'],1.);angle=p['prepared_angle'];edge=angle.edge(Jet(0))[0,0]
    indices=[i for i in range(6,len(curve.cn),6) if curve.cn[i].hi<edge.lo]
    cover=ConstantEdgeAngularCover(angle,[curve.cn[i] for i in indices],checkpoint=checkpoint,value_field=angle.value_only,inverse_residual=p['inverse_residual'])
    bounds=bounded_cosines(lambda u,s:list(mapped_inputs(p,1-u,s,flatten=False)[:2]),I(0,1),I(0,1),monotone=True,partial=True)
    charts=[]
    for i,c in enumerate(cover.charts):
        checkpoint();row=angular_material_chart(p,curve,cover,i,bounds,checkpoint=checkpoint)
        if c['low'] is not None:row['tensor_rule']='gauss3x8-local4'
        charts.append(row)
    checkpoint();return p,curve,cover,charts


from tools.opaque_square_reference import integrate_square

def inside_piece_reference(data,*,piece,motion,anchor=0,progress=lambda row:None,initial_leaves=None,tolerance='5e-13',polynomial_light=False,world_profile=None):
    """One complete inside material integral with a construction-inclusive clock."""
    import time
    from tools.opaque_original_motion_reference import _angular_selection
    from tools.opaque_reference_profile import stronger_target
    tolerance=stronger_target(tolerance)
    if type(polynomial_light)!=bool:raise ValueError('explicit polynomial light profile required')
    _angular_selection(piece,motion,anchor);start=time.monotonic();p=None;cover=None;last=None
    def check():
        if time.monotonic()-start>=90:raise Unresolved('inside original reference deadline')
    def update(row):
        nonlocal last
        last=row;progress(row)
    try:
        p,curve,cover,charts=inside_material_cover(data,piece=piece,motion=motion,anchor=anchor,checkpoint=check,**({'world_profile':world_profile} if world_profile is not None else {}),**({"polynomial_light":True} if polynomial_light else {}))
        basis=integrate_square(None,chart_cover=charts,tolerance=tolerance,max_cells=256,max_depth=10,checkpoint=check,progress=update,**({'initial_leaves':initial_leaves} if initial_leaves is not None else {}))
        check();values=[basis['value'][0]+I(a)*basis['value'][1] for a in data['material']['albedo']];derivatives=[basis['derivative'][0]+I(a)*basis['derivative'][1] for a in data['material']['albedo']]
        if any(v.width>I('2e-12').lo for v in values+derivatives):raise Unresolved('complete piece RGB radius')
        return dict(polynomial_light=polynomial_light,tolerance=tolerance,status='complete_original_piece_integrated',inside_angle=True,piece=piece,face=p['face'],motion=motion,anchor=anchor,basis=basis,value=values,derivative=derivatives,cover_cells=cover.cells,cover_depth=cover.depth,evaluations=cover.evaluations,initial_schedule=initial_leaves,prepared_angle=p['prepared_angle'].report(),physical_inverse=None if 'physical_angle' not in p else p['physical_angle'].report(),inverse_residual=None if 'inverse_residual' not in p else p['inverse_residual'].report(),horizon_cache=p['horizon_cache'].report(),seconds=time.monotonic()-start,scope='Fresh inside original piece and explicitly selected motion only; no installed or whole-pixel admission.')
    except ValueError as error:
        raise Unresolved('inside original piece unresolved',polynomial_light=polynomial_light,inside_angle=True,piece=piece,motion=motion,anchor=anchor,cause=getattr(error,'details',str(error)),cover_cells=None if cover is None else cover.cells,evaluations=None if cover is None else cover.evaluations,seconds=time.monotonic()-start,last=last) from error


from tools.opaque_physical_angle import PhysicalAngleResidual

class InsidePhysicalAngleResidual(PhysicalAngleResidual):
    """Same normalized field in rho=1-t, with positive physical angle slope.

    Substitute every exact rational coefficient before outward evaluation.
    The existing root proof supplies rho in [1-horizon,1]. Norms, intrinsic
    motion and all mixed coefficients remain; no moving-horizon recurrence is
    needed by the material inverse itself. The cover still owns that horizon.
    """
    def __init__(self,coefficients,*,compiled_series=False,contraction_stop=False):
        import math
        from tools.opaque_angle_light import _clean,PreparedIntervalPolynomial
        super().__init__(coefficients,compiled_series=compiled_series,contraction_stop=contraction_stop)
        def reverse(polynomial):
            out={}
            for (i,j,k),v in polynomial.items():
                for r in range(i+1):out[r,j,k]=out.get((r,j,k),0)+v*math.comb(i,r)*(-1)**r
            return _clean(out)
        self.polynomials=tuple(reverse(p) for p in self.polynomials)
        self.slope_polynomial=_clean({k:-v for k,v in reverse(self.slope_polynomial).items()})
        self.scalar_polynomials=tuple(PreparedIntervalPolynomial(p) for p in (*self.polynomials,self.slope_polynomial))
        self.entries.clear()
    def report(self):return dict(super().report(),coordinate='reversed inside physical radius')
