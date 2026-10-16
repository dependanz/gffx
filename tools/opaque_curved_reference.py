"""Independent finite conic-horizon integration on an exact visible triangle.

A strict Bernstein certificate proves one transverse radial root from the
isolated-sign vertex to every point of the opposite edge. An algebraic root
maps each curved region to a square. Its Jacobian and parameter derivative
include boundary transport. Outward Gauss5/order10 bounds cover both integrals.
No straight-chord approximation,native renderer or sampled convergence test.
This is a development reference for certified patches,not a general conic CAD.
"""
from fractions import Fraction as Q
from decimal import Decimal as D
import time
from tools.opaque_shading_intervals import I,Jet,Unresolved,integrate_triangle
from tools.opaque_moving_reference import P,BranchCertificate

class CurvedTriangle:
    def __init__(self,triangle,quadratic,*,motion=None,positive=True,checkpoint=lambda:None):
        if len(triangle)!=3 or any(len(p)!=2 for p in triangle) or len(quadratic)!=6 or type(positive)!=bool:raise ValueError('curved triangle profile')
        triangle=[[Q(x) for x in p] for p in triangle]
        motion=[[Q(0),Q(0)] for _ in range(3)] if motion is None else [[Q(x) for x in p] for p in motion]
        if len(motion)!=3 or any(len(p)!=2 for p in motion) or any(abs(x)>16 for p in triangle+motion for x in p):raise ValueError('curved coordinate/tangent profile')
        tri=[[P([x,d]) for x,d in zip(p,m)] for p,m in zip(triangle,motion)]
        q=list(map(P,quadratic));cert=BranchCertificate(checkpoint)
        def evaluate(p):
            x,y=p;return q[0]*x*x+q[1]*x*y+q[2]*y*y+q[3]*x+q[4]*y+q[5]
        values=[evaluate(p) for p in tri];signs=[cert.classify(v) for v in values]
        if 0 in signs or len(set(signs))!=2:raise Unresolved('curved patch needs one strict isolated-sign vertex')
        first=next(i for i,s in enumerate(signs) if signs.count(s)==1);indices=[first]+[i for i in range(3) if i!=first]
        self.inside=(signs[first]>0)==positive
        # Normalize the root problem to q(A)<0,q(opposite edge)>0. This does
        # not change which physical side the caller asked us to integrate.
        factor=-signs[first];tri=[tri[i] for i in indices];values=[factor*values[i] for i in indices]
        ab,ac,bc=[factor*2*evaluate([(tri[i][k]+tri[j][k])*Q(1,2) for k in range(2)])-(values[i]+values[j])*Q(1,2) for i,j in ((0,1),(0,2),(1,2))]
        a,b,c=values
        guards={'opposite_b':b,'opposite_bc':bc,'opposite_c':c,
                'radial_at_zero_ab':2*(ab-a),'radial_at_zero_ac':2*(ac-a),
                'radial_at_one_b':2*(b-ab),'radial_at_one_bc':2*bc-ab-ac,'radial_at_one_c':2*(c-ac)}
        for name,value in guards.items():
            if cert.classify(value)<=0:raise Unresolved('curved transversality/edge sign not certified',predicate=name)
        u=[tri[1][k]-tri[0][k] for k in range(2)];v=[tri[2][k]-tri[0][k] for k in range(2)]
        det=u[0]*v[1]-u[1]*v[0];sgn=cert.classify(det)
        if not sgn:raise Unresolved('degenerate curved integration triangle')
        self.original_factor=factor;self.triangle=tri;self.controls=[a,b,c,ab,ac,bc];self.det=sgn*det
        # Reduce exact polynomial identities before interval evaluation. Keeping
        # repeated(1-s) Bernstein factors in an interval expression would lose
        # cancellation even for a linear horizon and exhaust the fixed budget.
        aa=[b-2*ab+a,2*(bc-b-ac+ab),b-2*bc+c];bb=[2*(ab-a),2*(ac-ab)]
        disc=[P(0) for _ in range(3)]
        for i,z in enumerate(bb):
            for j,w in enumerate(bb):disc[i+j]=disc[i+j]+z*w
        self.radial=(aa,bb,a);self.discriminant=[z-4*w*a for z,w in zip(disc,aa)]
        self.certificate=dict(**cert.report(),vertex_order=indices,positive=positive,region='radial_inside' if self.inside else 'radial_outside',proof='strict opposite-edge quadratic Bernstein signs and positive radial derivative on the entire square',guards={k:[str(x) for x in v.c] for k,v in guards.items()})
    @staticmethod
    def lift(x,p):return x.lift(p[0],tangent=p[1])
    def root(self,s):return self._root_discriminant(s)[0]
    def sqrt_discriminant(self,s):
        """Default interval algebra; qualified alternatives must be explicit."""
        q=self.lift(s,self.discriminant[-1])
        for v in reversed(self.discriminant[:-1]):q=q*s+self.lift(s,v)
        return q.sqrt()
    def _root_discriminant(self,s):
        def horner(coefficients):
            value=self.lift(s,coefficients[-1])
            for p in reversed(coefficients[:-1]):value=value*s+self.lift(s,p)
            return value
        aa,bb,a=self.radial;bb=horner(bb);a=self.lift(s,a)
        # Rationalized formula also handles aa=0 and its nonzero tangent.
        discriminant_root=self.sqrt_discriminant(s);r=-2*a/(bb+discriminant_root)
        # Existence/uniqueness in(0,1) is proved above independently of this
        # interval expression. Tighten only the value,never derivative terms.
        r.c[0,0]=I(max(D(0),r[0,0].lo),min(D(1),r[0,0].hi))
        return r,discriminant_root
    def radial_map(self,u,s):
        r=self.root(s);width=r if self.inside else 1-r;t=u*width if self.inside else r+u*width
        return t,self.lift(u,self.det)*t*width
    def horizon_numerator(self,distance,s):
        """Exact original quadratic at a distance from the certified radial root.

        distance is 1-u for the inside radial map, u for the outside map.
        q(r+delta)=factor*delta*(sqrt(discriminant)+a*delta), since q(r)=0.
        This cancels the root symbolically, including all jet derivatives.
        """
        if distance[0,0].lo<0 or distance[0,0].hi>1:raise ValueError('horizon distance in[0,1] required')
        def at(coeff):
            out=self.lift(s,coeff[-1])
            for v in reversed(coeff[:-1]):out=out*s+self.lift(s,v)
            return out
        r=self.root(s);delta=-r*distance if self.inside else (1-r)*distance
        return self.original_factor*delta*(self.sqrt_discriminant(s)+at(self.radial[0])*delta)

    def map(self,u,s):
        t,jac=self.radial_map(u,s)
        a,b,c=[[self.lift(u,z) for z in p] for p in self.triangle]
        x,y=[a[k]+t*((1-s)*(b[k]-a[k])+s*(c[k]-a[k])) for k in range(2)]
        return x,y,jac

def flattened_radial_map(patch,u,s,*,power=8,with_numerator=False):
    """Exact endpoint flattening with its complete change-of-variable Jacobian.

    Qualified powers4/8 map[0,1] monotonically onto the SAME conic patch.
    Returning distance separately preserves exact horizon cancellation.
    """
    if type(power)!=int or power not in (4,8):raise ValueError('qualified horizon power4or8 required')
    if type(with_numerator)!=bool:raise ValueError('explicit numerator reuse selection required')
    v=1-u if patch.inside else u;distance=v**power
    z=1-distance if patch.inside else distance
    if with_numerator:
        # Same s and immutable coefficients within this call: compute the root
        # and discriminant once. Preserve every original operation and order.
        root,disc=patch._root_discriminant(s);width=root if patch.inside else 1-root
        t=z*width if patch.inside else root+z*width
        jac=patch.lift(u,patch.det)*t*width
        delta=-root*distance if patch.inside else (1-root)*distance
        aa=patch.lift(s,patch.radial[0][-1])
        for value in reversed(patch.radial[0][:-1]):aa=aa*s+patch.lift(s,value)
        numerator=patch.original_factor*delta*(disc+aa*delta)
        return t,jac*(power*v**(power-1)),distance,numerator
    t,jac=patch.radial_map(z,s)
    return t,jac*(power*v**(power-1)),distance


def integrate_curved(patch,field,*,tolerance='1e-12',max_cells=256,checkpoint=lambda:None,radial=False):
    if type(radial)!=bool:raise TypeError('radial field flag')
    if type(max_cells)!=int or not 1<=max_cells<=256:raise ValueError('curved pixel cell budget1..256')
    tol=I(tolerance)
    if tol.lo<=0 or tol.hi>D('1e-6'):raise ValueError('reference tolerance domain')
    start=time.monotonic();rows=[];cells=0;value=None;derivative=None
    def check():
        checkpoint()
        if time.monotonic()-start>90:raise Unresolved('curved reference90s deadline',cells=cells)
    def mapped(u,s):
        if radial:
            t,jac=patch.radial_map(u,s);values=field(t,s)
        else:
            x,y,jac=patch.map(u,s);values=field(x,y)
        return [v*jac for v in values]
    # These are two parameter triangles,not straight substitutes for the conic.
    for triangle in (((0,0),(1,0),(1,1)),((0,0),(1,1),(0,1))):
        check()
        if cells>=max_cells:raise Unresolved('curved total cell budget',cells=cells)
        row=integrate_triangle(mapped,triangle,tolerance=str((tol/2).lo),max_cells=max_cells-cells,max_depth=10,gauss_points=5,checkpoint=check)
        cells+=row['cells'];rows.append(row)
        if value is None:value=[I(0)]*len(row['value']);derivative=[I(0)]*len(row['derivative'])
        for i in range(len(value)):value[i]=value[i]+row['value'][i];derivative[i]=derivative[i]+row['derivative'][i]
    check()
    if any(v.width>2*tol.lo for v in value+derivative):raise Unresolved('curved total error budget')
    return dict(value=value,derivative=derivative,cells=cells,parameter_triangles=2,certificate=patch.certificate,parts=rows,seconds=time.monotonic()-start,tolerance=tolerance,scope='Certified single-root conic patch; ordinary field derivative plus moving-boundary and triangle-Jacobian transport.')

def perspective_numerator(triangle,*,inverse_w,normals,views,normal_motion=None,view_motion=None):
    """Exact homogeneous n.view quadratic from fixed projected binary inputs.

    Normal/view trajectories are linear. This deliberately does not pretend that
    world vertex/camera motion leaves the projected face and clip w unchanged.
    """
    if len(triangle)!=3 or any(len(v)<2 for v in triangle) or len(inverse_w)!=3 or min(inverse_w)<=0:raise ValueError('perspective triangle profile')
    if any(len(rows)!=3 or any(len(v)!=3 for v in rows) for rows in (normals,views)):raise ValueError('normal/view corner shape')
    a,b,c=[[Q(z) for z in p[:2]] for p in triangle];det=(b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    if not det:raise ValueError('degenerate projected face')
    b1=[(c[1]-a[1])/det,-(c[0]-a[0])/det,(-a[0]*(c[1]-a[1])+a[1]*(c[0]-a[0]))/det]
    b2=[-(b[1]-a[1])/det,(b[0]-a[0])/det,(a[0]*(b[1]-a[1])-a[1]*(b[0]-a[0]))/det]
    b0=[-b1[0]-b2[0],-b1[1]-b2[1],1-b1[2]-b2[2]];raw=[[z*Q(w) for z in row] for row,w in zip((b0,b1,b2),inverse_w)]
    def linear(rows,motion):
        motion=[[0]*3]*3 if motion is None else motion
        if len(motion)!=3 or any(len(v)!=3 for v in motion):raise ValueError('corner tangent shape')
        return [[sum((P([Q(rows[i][k]),Q(motion[i][k])])*raw[i][j] for i in range(3)),P(0)) for j in range(3)] for k in range(3)]
    n=linear(normals,normal_motion);v=linear(views,view_motion);q=[P(0) for _ in range(6)]
    for a,b in zip(n,v):
        terms=[a[0]*b[0],a[0]*b[1]+a[1]*b[0],a[1]*b[1],a[0]*b[2]+a[2]*b[0],a[1]*b[2]+a[2]*b[1],a[2]*b[2]]
        q=[x+y for x,y in zip(q,terms)]
    return q


def radial_homogeneous_fields(patch,triangle,*,inverse_w,fields):
    """Compose fixed-face perspective numerators exactly into radial coordinates.

    Positive denominator is certified on the complete moving visible triangle.
    Field samples may be polynomial trajectories. Normalize homogeneous normal
    and view vectors directly: their common positive denominator cancels exactly.
    This avoids wide thin-triangle barycentric interval cancellation.
    """
    if len(triangle)!=3 or any(len(p)<2 for p in triangle) or len(inverse_w)!=3 or min(inverse_w)<=0:raise ValueError('homogeneous radial profile')
    a,b,c=[[Q(z) for z in p[:2]] for p in triangle];det=(b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    if not det:raise ValueError('degenerate projected face')
    def beta(x,y):
        b1=((x-a[0])*(c[1]-a[1])-(y-a[1])*(c[0]-a[0]))*Q(1,det)
        b2=((b[0]-a[0])*(y-a[1])-(b[1]-a[1])*(x-a[0]))*Q(1,det)
        return [1-b1-b2,b1,b2]
    raw=[[z*Q(w) for z,w in zip(beta(*p),inverse_w)] for p in patch.triangle]
    cert=BranchCertificate();den=[sum(row,P(0)) for row in raw]
    for p in den:
        if cert.classify(p)<=0:raise Unresolved('radial perspective denominator')
    def coefficients(values):return [values[0],values[1]-values[0],values[2]-values[1]]
    prepared={'denominator':[coefficients(den)]}
    for name,rows in fields.items():
        if name=='denominator' or len(rows)!=3 or not 1<=len(rows[0])<=4 or any(len(r)!=len(rows[0]) for r in rows):raise ValueError('bounded radial field shape')
        prepared[name]=[coefficients([sum((row[i]*P(rows[i][k]) for i in range(3)),P(0)) for row in raw]) for k in range(len(rows[0]))]
    def evaluate(t,s):
        lift=lambda p:t.lift(p[0],tangent=p[1])
        return {name:[lift(a)+t*(lift(b)+s*lift(c)) for a,b,c in components] for name,components in prepared.items()}
    evaluate.certificate=cert.report();evaluate.coefficients=prepared;return evaluate
