"""Independent exact moving-face perspective numerators.

The common positive projected determinant is retained rather than differentiated
through division. Normalizing the homogeneous normal/view cancels that common
scale. This module supplies algebra, not a complete integral certificate.
"""
from fractions import Fraction as Q
from tools.opaque_moving_reference import P,BranchCertificate


class ProjectedFields:
    def __init__(self,*,triangle,triangle_motion,inverse_w,inverse_w_motion,
                 normals,views,normal_motion,view_motion,checkpoint=lambda:None):
        def rows(value,width,label):
            if len(value)!=3 or any(len(row)!=width for row in value):
                raise ValueError(label+' requires three fixed-width rows')
            return [[Q(x) for x in row] for row in value]
        tri=rows(triangle,2,'triangle');dt=rows(triangle_motion,2,'triangle motion')
        nn=rows(normals,3,'normals');vv=rows(views,3,'views')
        dn=rows(normal_motion,3,'normal motion');dv=rows(view_motion,3,'view motion')
        if len(inverse_w)!=3 or len(inverse_w_motion)!=3:raise ValueError('three perspective weights required')
        iw=list(map(Q,inverse_w));diw=list(map(Q,inverse_w_motion))
        if min(iw)<=0:raise ValueError('positive perspective weights required')
        cert=BranchCertificate(checkpoint)
        self.triangle=[[P([a,b]) for a,b in zip(row,motion)] for row,motion in zip(tri,dt)]
        a,b,c=self.triangle;det=(b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
        sign=cert.classify(det)
        if not sign:raise ValueError('degenerate projected triangle')
        self.det=sign*det;weights=[P([a,b]) for a,b in zip(iw,diw)]
        for w in weights:
            if cert.classify(w)<=0:raise ValueError('positive perspective weight interval required')
        self.raw=[]
        for i,(j,k) in enumerate(((1,2),(2,0),(0,1))):
            a,b=self.triangle[j],self.triangle[k]
            line=[a[1]-b[1],b[0]-a[0],a[0]*b[1]-b[0]*a[1]]
            self.raw.append([sign*v*weights[i] for v in line])
        self.fields={}
        for name,values,motion in [('n',nn,dn),('v',vv,dv)]:
            self.fields[name]=[[sum((self.raw[i][j]*P([values[i][k],motion[i][k]]) for i in range(3)),P(0)) for j in range(3)] for k in range(3)]
        self.certificate=cert.report();self._checkpoint=checkpoint

    def at(self,x,y):
        return [a*x+b*y+c for a,b,c in self.raw]

    def numerator(self):
        q=[P(0) for _ in range(6)]
        for a,b in zip(self.fields['n'],self.fields['v']):
            terms=[a[0]*b[0],a[0]*b[1]+a[1]*b[0],a[1]*b[1],a[0]*b[2]+a[2]*b[0],a[1]*b[2]+a[2]*b[1],a[2]*b[2]]
            q=[x+y for x,y in zip(q,terms)]
        return q

    def radial(self,patch):
        if len(patch.triangle)!=3 or any(len(p)!=2 for p in patch.triangle):raise ValueError('moving triangular patch required')
        cert=BranchCertificate(self._checkpoint);den=[sum(self.at(*point),P(0)) for point in patch.triangle]
        for value in den:
            if cert.classify(value)<=0:raise ValueError('positive complete patch perspective denominator required')
        def coefficients(values):return [values[0],values[1]-values[0],values[2]-values[1]]
        prepared={'denominator':[coefficients(den)]}
        for name,components in self.fields.items():
            prepared[name]=[coefficients([a*x+b*y+c for x,y in patch.triangle]) for a,b,c in components]
        def evaluate(t,s):
            lift=lambda p:t.lift(p[0],tangent=p[1])
            return {name:[lift(a)+t*(lift(b)+s*lift(c)) for a,b,c in components] for name,components in prepared.items()}
        evaluate.coefficients=prepared;evaluate.certificate=cert.report()
        return evaluate


def aligned_world_motion(data,profile,*,kind,input_sha256):
    """Exact first motion for the pinned aligned camera and three affine paths.

    X translation and XZ shear leave camera depth and clip w constant, so their
    screen trajectories are exactly linear. Area-weighted vertex sums under shear transform by the cofactor. Each
    vertex then needs its own normalization derivative before interpolation.
    The represented base normal is held fixed; no signed term is omitted.
    """
    if kind not in ('world_x','camera_x','world_shear_xz'):raise ValueError('declared aligned world motion required')
    if profile.get('status')!='verified_source_profile' or profile.get('input_sha256')!=input_sha256 or profile.get('scene_sha256')!=data.get('scene_sha256'):
        raise ValueError('verified matching original source profile required')
    if not isinstance(input_sha256,str) or len(input_sha256)!=64 or any(c not in '0123456789abcdef' for c in input_sha256):raise ValueError('input SHA256 required')
    camera=profile['camera'];pose=camera['pose']
    if pose!=[[1,0,0,0],[0,1,0,0],[0,0,1,3],[0,0,0,1]]:raise ValueError('explicit aligned original camera required')
    fx,fy=Q(camera['fx']),Q(camera['fy'])
    if min(fx,fy)<=0:raise ValueError('positive focal lengths required')
    ids=data['selected']['original_vertex_ids']
    if len(ids)!=len(set(ids)) or not 1<=len(data['faces'])<=32:raise ValueError('bounded unique original geometry required')
    omega=Q(.01);depths={};faces=[];velocities={}
    for source in data['faces']:
        if len(source['vertices'])!=3 or len(source['clip_w'])!=3 or len(source['normal'])!=3:raise ValueError('face corner shape')
        row=dict(triangle_motion=[],inverse_w_motion=[Q(0)]*3,normal_motion=[],view_motion=[])
        for index,w,n in zip(source['vertices'],source['clip_w'],source['normal']):
            if index not in ids or len(n)!=3:raise ValueError('declared corner required')
            world=list(map(Q,profile['vertices'][str(index)]));w=Q(w)
            if len(world)!=3 or w<=0 or (index in depths and depths[index]!=w):raise ValueError('consistent positive represented depth required')
            depths[index]=w
            dx=omega*(world[2] if kind=='world_shear_xz' else (0 if kind=='camera_x' else 1));dc=omega if kind=='camera_x' else Q(0)
            screen=[fx*(dx-dc)/w,Q(0),Q(0)]
            if index in velocities and velocities[index]!=screen:raise ValueError('inconsistent shared vertex motion')
            velocities[index]=screen;row['triangle_motion'].append(screen[:2]);row['view_motion'].append([dc-dx,Q(0),Q(0)])
            normal=list(map(Q,n));dn=[Q(0),Q(0),-omega*normal[0]] if kind=='world_shear_xz' else [Q(0)]*3
            norm2=sum(x*x for x in normal)
            if norm2<=0:raise ValueError('nonzero represented vertex normal required')
            radial=sum(x*d for x,d in zip(normal,dn))/norm2
            row['normal_motion'].append([d-x*radial for x,d in zip(normal,dn)])
        faces.append(row)
    if set(ids)!=set(velocities):raise ValueError('complete selected geometry required')
    return dict(kind=kind,input_sha256=input_sha256,screen_motion=[velocities[i] for i in ids],faces=faces,scope='Exact aligned affine path; no arbitrary camera rotation/depth or gradient acceptance.')
