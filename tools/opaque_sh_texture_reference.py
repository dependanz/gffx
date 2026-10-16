"""Independent Gauss5 x Gauss5 SH/bilinear triangle error enclosure.

Development-only, Decimal60 directed intervals; no renderer import. Inputs are
exact supplied binary64 piece-corner fields, NOT a certificate of clipping,
face ownership, texture knot admission, or their construction/derivatives.
SH uses effective monomial coefficients [1,x,y,z,xy,yz,3z*z-1,xz,x*x-y*y].
First differentials include all supplied corner fields and moving area.
"""
from fractions import Fraction as Q
from tools.opaque_shading_intervals import I,Jet
import math

NODES=(.0469100770306680036,.2307653449471584545,.5,.7692346550528415455,.9530899229693319964)
WEIGHTS=(.1184634425280945438,.2393143352496832340,.2844444444444444444,.2393143352496832340,.1184634425280945438)

def rule():
    s=I(Q(10,7)).sqrt();a=((I(5)+2*s)/9).sqrt();b=((I(5)-2*s)/9).sqrt()
    w1=(I(322)-13*I(70).sqrt())/1800;w2=(I(322)+13*I(70).sqrt())/1800
    return [(1-a)/2,(1-b)/2,I(Q(1,2)),(1+b)/2,(1+a)/2],[w1,w2,I(Q(64,225)),w2,w1]

def remainder(fn):
    """Integrate a smooth supplied square field with its first differential.

    For n=5, E <= sup|f^(10)/10!| * (5!)^4/(11*(10!)^2).
    Tensor identity E_u I_v + Q_u E_v and positive unit mass give sum of
    the two directional bounds. Jet coefficient (10,1) is the corresponding
    tenth spatial derivative of the FIRST parameter differential /10!.
    Nodes and weights are algebraic intervals; summation is outward-rounded.
    """
    factor=I(Q(math.factorial(5)**4,11*math.factorial(10)**2))
    total=None
    for axis in (0,1):
        u=Jet.variable(I(0,1),order=10) if axis==0 else Jet(I(0,1),order=10)
        v=Jet.variable(I(0,1),order=10) if axis==1 else Jet(I(0,1),order=10)
        rows=fn(u,v)
        if total is None:total=[[I(0) for _ in rows] for _ in range(2)]
        for d in (0,1):
            for k,f in enumerate(rows):total[d][k]=total[d][k]+factor*I(f[10,d].bound)
    return total

def certify(fn):
    total=remainder(fn)
    nodes,weights=rule();sums=[[I(0) for _ in total[0]] for _ in range(2)]
    for u,wu in zip(nodes,weights):
        for v,wv in zip(nodes,weights):
            for k,f in enumerate(fn(Jet(u),Jet(v))):
                for d in (0,1):sums[d][k]=sums[d][k]+wu*wv*f[0,d]
    out={name:[x+I(-e.hi,e.hi) for x,e in zip(sums[d],total[d])] for d,name in enumerate(('value','tangent'))}
    out['remainder']=total;out['quadrature']=sums
    return out


def certify_partition(fn,subdivisions=1):
    """Independent integration over one square or four fixed sub-squares.

    The affine map and its Jacobian are differentiated by Jet, so each
    transformed tenth-derivative remainder and first differential retain the
    original proof. Positive outward sums enclose the complete original square.
    This changes only offline reference work, never the production Gauss rule.
    """
    if type(subdivisions) is not int or subdivisions not in (1,2):
        raise ValueError('reference grid must be 1 or 2 per axis')
    if subdivisions==1:return certify(fn)
    result=None
    for i in range(2):
        for j in range(2):
            cell=certify(lambda u,v:[x/4 for x in fn((u+i)/2,(v+j)/2)])
            if result is None:result=cell
            else:
                for key in ('value','tangent'):
                    result[key]=[a+b for a,b in zip(result[key],cell[key])]
                for key in ('remainder','quadrature'):
                    result[key]=[[a+b for a,b in zip(ra,rb)] for ra,rb in zip(result[key],cell[key])]
    return result


def fixture():
    return dict(normal=[[0.,0.,1.]]*3,uv=[[0.,0.],[1.,0.],[0.,1.]],
                tex=[[.2,.3,.4],[.3,.4,.5],[.4,.5,.6],[.7,.8,.9]],
                sh=[[1.,1.,1.]]+[[0.,0.,0.]]*8,det=1.)

def validate(a):
    for key,shape in [('normal',(3,3)),('uv',(3,2)),('tex',(4,3)),('sh',(9,3))]:
        for name in (key,'d_'+key):
            if name not in a:continue
            rows=a[name]
            if len(rows)!=shape[0] or any(len(r)!=shape[1] for r in rows):raise ValueError('piece shape')
            if any(not (isinstance(x,I) and name.startswith('d_')) and not math.isfinite(x) for r in rows for x in r):raise ValueError('finite piece fields')
    if not math.isfinite(a['det']) or a['det']<=0 or not (isinstance(a.get('d_det',0),I) or math.isfinite(a.get('d_det',0))):raise ValueError('positive finite determinant')
    if any(x<0 or x>1 for r in a['uv'] for x in r):raise ValueError('triangle must lie inside one bilinear texture cell')

def field(a,u,v):
    validate(a)
    def j(x,d=0):return Jet(x,order=u.order,tangent=d)
    def data(key):
        ds=a.get('d_'+key,[[0]*len(row) for row in a[key]])
        return [[j(x,dx) for x,dx in zip(row,drow)] for row,drow in zip(a[key],ds)]
    def interpolate(rows):return [rows[0][k]+u*(rows[1][k]-rows[0][k])+(1-u)*v*(rows[2][k]-rows[0][k]) for k in range(len(rows[0]))]
    n=interpolate(data('normal'));length=sum(x*x for x in n).sqrt()
    nx,ny,nz=[x/length for x in n]
    s,t=interpolate(data('uv'));tex=data('tex');sh=data('sh')
    albedo=[tex[0][k]+s*(tex[1][k]-tex[0][k])+t*(tex[2][k]-tex[0][k])+s*t*(tex[3][k]-tex[1][k]-tex[2][k]+tex[0][k]) for k in range(3)]
    basis=[j(1),nx,ny,nz,nx*ny,ny*nz,3*nz*nz-1,nx*nz,nx*nx-ny*ny]
    jac=j(a['det'],a.get('d_det',0))*(1-u)
    return [jac*albedo[k]*sum(b*row[k] for b,row in zip(basis,sh)) for k in range(3)]

def numeric(a):
    """Experimental binary64 candidate, explicit analytic first differential.

    Separate NumPy implementation; intentionally does not use Jet/intervals.
    Uses the exact 5-node constants already present in the native quadrature.
    """
    import numpy as np
    validate(a);total=np.zeros(3);dt=np.zeros(3)
    def rows(key):
        r=np.asarray(a[key]);return r,np.asarray(a.get('d_'+key,np.zeros_like(r)))
    normals,dnormals=rows('normal');uv,duv=rows('uv');tex,dtex=rows('tex');sh,dsh=rows('sh')
    for u,wu in zip(NODES,WEIGHTS):
        for v,wv in zip(NODES,WEIGHTS):
            beta=np.array([(1-u)*(1-v),u,(1-u)*v]);n=beta@normals;dn=beta@dnormals
            length=np.linalg.norm(n);dl=np.dot(n,dn)/length;dn=dn/length-n*(dl/(length*length));n=n/length
            (s,t),(ds,dtuv)=beta@uv,beta@duv
            w=np.array([(1-s)*(1-t),s*(1-t),(1-s)*t,s*t]);dw=np.array([-ds*(1-t)-(1-s)*dtuv,ds*(1-t)-s*dtuv,-ds*t+(1-s)*dtuv,ds*t+s*dtuv])
            colour=w@tex;dc=dw@tex+w@dtex;x,y,z=n;dx,dy,dz=dn
            b=np.array([1,x,y,z,x*y,y*z,3*z*z-1,x*z,x*x-y*y]);db=np.array([0,dx,dy,dz,dx*y+x*dy,dy*z+y*dz,6*z*dz,dx*z+x*dz,2*x*dx-2*y*dy])
            light=b@sh;dlight=db@sh+b@dsh
            jac=a['det']*(1-u);djac=a.get('d_det',0)*(1-u)
            total+=wu*wv*jac*colour*light
            dt+=wu*wv*(djac*colour*light+jac*(dc*light+colour*dlight))
    return total,dt
