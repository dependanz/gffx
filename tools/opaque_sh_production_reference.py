"""Offline enclosure of SH25 integration on witnessed binary64 triangles.
No claim about ideal FLAME geometry: projected input vertices/normals are the
observed renderer inputs. Native region corners are the integration domains.
All arithmetic below is independently outward-rounded; no production import.
"""
from tools.opaque_shading_intervals import I,Jet
from tools.opaque_sh_texture_reference import certify_partition
import math

def normal_square(n,rows,u,v):
    """Tighten only the value coefficient of a Duffy-interpolated normal.

    On [0,1]^2 its three barycentric weights are nonnegative and sum to one.
    Its squared norm is a convex combination of the nine corner dot products
    (weights wi*wj). Intersect that outward enclosure with the natural Jet
    enclosure. All spatial and parameter coefficients remain untouched.
    Outside the square the convex argument is unavailable: keep the old box.
    """
    q=sum(x*x for x in n)
    if u[0,0].lo<0 or u[0,0].hi>1 or v[0,0].lo<0 or v[0,0].hi>1:return q
    dots=[sum(I(x)*I(y) for x,y in zip(a,b)) for a in rows for b in rows]
    lo=max(q[0,0].lo,min(x.lo for x in dots));hi=min(q[0,0].hi,max(x.hi for x in dots))
    if lo>hi:raise ValueError('inconsistent normal-square enclosures')
    q.c[0,0]=I(lo,hi)
    return q

def polynomial(a,u,v,*,density=False):
 def j(x,d=0):return Jet(x,order=u.order,tangent=d)
 def data(key):
  ds=a.get('d_'+key,[[0]*len(row) for row in a[key]])
  return [[j(x,d) for x,d in zip(row,dr)] for row,dr in zip(a[key],ds)]
 def interp(key):
  z=data(key);return [z[0][k]+u*(z[1][k]-z[0][k])+(1-u)*v*(z[2][k]-z[0][k]) for k in range(len(z[0]))]
 n=interp('normal');length=normal_square(n,a['normal'],u,v).sqrt();x,y,z=[t/length for t in n];s,t=interp('uv');tex=data('tex');sh=data('sh')
 b=[j(1),x,y,z,x*y,y*z,3*z*z-1,x*z,x*x-y*y]
 albedo=[tex[0][k]+s*(tex[1][k]-tex[0][k])+t*(tex[2][k]-tex[0][k])+s*t*(tex[3][k]-tex[1][k]-tex[2][k]+tex[0][k]) for k in range(3)]
 return [j(a['det'],a.get('d_det',0))*(1 if density else 1-u)*albedo[k]*sum(bb*row[k] for bb,row in zip(b,sh)) for k in range(3)]

def geometry(vertices,corners):
 v=[[I(float(x)) for x in row[:2]] for row in vertices];p=[[I(float(x)) for x in row] for row in corners]
 sub=lambda a,b:[x-y for x,y in zip(a,b)];cross=lambda a,b:a[0]*b[1]-a[1]*b[0]
 a,b=sub(v[1],v[0]),sub(v[2],v[0]);d=cross(a,b);bary=[]
 for q in p:
  e=sub(q,v[0]);y=cross(e,b)/d;z=cross(a,e)/d;bary.append([1-y-z,y,z])
 det=cross(sub(p[1],p[0]),sub(p[2],p[0]))
 if det.hi<0:det=-det
 if det.lo<=0:raise ValueError('uncertain positive region area')
 return bary,det

def build(vertices,corners,normals,uv,texture,sh):
 bary,det=geometry(vertices,corners)
 normal=[[sum(b[k]*I(float(normals[k,j])) for k in range(3)) for j in range(3)] for b in bary]
 h,w,_=texture.shape
 tc=[[sum(b[k]*I(float(uv[k,j])) for k in range(3))*size-I('.5') for j,size in enumerate((w,h))] for b in bary]
 cell=[math.floor(float(sum(row[j] for row in tc).lo/3)) for j in (0,1)]
 local=[[row[j]-cell[j] for j in (0,1)] for row in tc]
 # Retain and bound every excursion; a positive-width knot crossing is never dropped.
 excursion=[I(max(I(0).lo,max(-row[j].lo for row in local),max(row[j].hi-1 for row in local))) for j in (0,1)]
 if any(e.hi>I('1e-8').lo for e in excursion):raise ValueError('production region crosses an unsplit texture knot')
 # Current certificates cover cells with four real texels. WRAP_BORDER needs
 # explicit zero-border coefficients and edge Lipschitz bounds, not clamping.
 if not (0<=cell[0]<w-1 and 0<=cell[1]<h-1):raise ValueError('texture-border field not qualified')
 tex=[[I(float(z)) for z in texture[min(max(cell[1]+dy,0),h-1),min(max(cell[0]+dx,0),w-1)]] for dx,dy in ((0,0),(1,0),(0,1),(1,1))]
 return dict(normal=normal,uv=local,tex=tex,sh=sh,det=det),excursion,cell

def enclose(a,excursion,texture_lipschitz,*,subdivisions=1):
 c=certify_partition(lambda u,v:polynomial(a,u,v),subdivisions)
 # Globally continuous bilinear border extension and this cell polynomial agree
 # on the cell. A segment to the clamped point gives two Lipschitz errors.
 # |normalized SH monomials| <= 2 (z2 term) or 1. Integrate with area det/2.
 light=[sum(I(2 if j==6 else 1)*I(row[k].bound) for j,row in enumerate(a['sh'])) for k in range(3)]
 extension=[]
 for k in range(3):
  tx=max((a['tex'][1][k]-a['tex'][0][k]).bound,(a['tex'][3][k]-a['tex'][2][k]).bound)
  ty=max((a['tex'][2][k]-a['tex'][0][k]).bound,(a['tex'][3][k]-a['tex'][1][k]).bound)
  cross=I((a['tex'][3][k]-a['tex'][1][k]-a['tex'][2][k]+a['tex'][0][k]).bound)
  e=a['det']/2*light[k]*(excursion[0]*(texture_lipschitz[0][k]+I(tx)+cross*excursion[1])+excursion[1]*(texture_lipschitz[1][k]+I(ty)+cross*excursion[0]))
  extension.append(e);c['value'][k]=c['value'][k]+I(-e.hi,e.hi)
 c['knot_extension']=extension
 return c


def add_direction(a,vertices,corners,dvertices,dcorners,normals,dnormals,uv,dtexture,dsh,cell):
    def jets(v,d):return [[Jet(I(float(x)),order=0,tangent=I(float(dx))) for x,dx in zip(row,dr)] for row,dr in zip(v,d)]
    v=jets(vertices,dvertices);p=jets(corners,dcorners)
    sub=lambda x,y:[aa-bb for aa,bb in zip(x,y)];cross=lambda x,y:x[0]*y[1]-x[1]*y[0]
    x,y=sub(v[1],v[0]),sub(v[2],v[0]);det=cross(x,y);bary=[]
    for pt in p:
        e=sub(pt,v[0]);b=cross(e,y)/det;c=cross(x,e)/det;bary.append([1-b-c,b,c])
    n=jets(normals,dnormals)
    a['d_normal']=[[sum(b[k]*n[k][j] for k in range(3))[0,1] for j in range(3)] for b in bary]
    h,w,_=dtexture.shape
    a['d_uv']=[[sum(b[k]*I(float(uv[k,j])) for k in range(3))[0,1]*size for j,size in enumerate((w,h))] for b in bary]
    jac=cross(sub(p[1],p[0]),sub(p[2],p[0]));a['d_det']=jac[0,1] if jac[0,0].lo>0 else -jac[0,1]
    a['d_tex']=[[I(float(z)) for z in dtexture[min(max(cell[1]+dy,0),h-1),min(max(cell[0]+dx,0),w-1)]] for dx,dy in ((0,0),(1,0),(0,1),(1,1))]
    a['d_sh']=dsh
    return a

def outside_fraction(a):
    # Exact rational polygon clipping gives an upper bound on the portion lying
    # outside a cell. Lower/upper corner enclosures enlarge each violation set.
    from fractions import Fraction as Q
    total=Q(0)
    for axis in (0,1):
        for upper in (False,True):
            vals=[Q(x[axis].hi)-1 if upper else -Q(x[axis].lo) for x in a['uv']]
            poly=[(Q(0),Q(0)),(Q(1),Q(0)),(Q(0),Q(1))]
            f=lambda p:vals[0]+p[0]*(vals[1]-vals[0])+p[1]*(vals[2]-vals[0])
            out=[];p=poly[-1];fp=f(p)
            for q in poly:
                fq=f(q)
                if (fp>0)!=(fq>0):
                    t=fp/(fp-fq);out.append(tuple(p[j]+t*(q[j]-p[j]) for j in (0,1)))
                if fq>0:out.append(q)
                p,fp=q,fq
            if len(out)>2:total+=abs(sum(out[i][0]*out[(i+1)%len(out)][1]-out[i][1]*out[(i+1)%len(out)][0] for i in range(len(out))))
    return I(min(Q(1),total))

def tangent_extension(a,texture_max,dtexture_max):
    # Error is continuous and zero at each cell boundary. The moving-domain
    # derivative has no jump delta there. Bound both first-differential densities
    # over the conservatively clipped outside portion, including moving area.
    fraction=outside_fraction(a)
    if fraction.zero:return [I(0)]*3
    generic=dict(a)
    generic['uv']=[[I(0,1),I(0,1)] for _ in range(3)]
    generic['tex']=[[I(-texture_max[k].hi,texture_max[k].hi) for k in range(3)] for _ in range(4)]
    generic['d_tex']=[[I(-dtexture_max[k].hi,dtexture_max[k].hi) for k in range(3)] for _ in range(4)]
    u=Jet(I(0,1),order=0);v=Jet(I(0,1),order=0)
    actual=polynomial(generic,u,v,density=True);extension=polynomial(a,u,v,density=True)
    return [fraction/2*(I(x[0,1].bound)+I(y[0,1].bound)) for x,y in zip(actual,extension)]
