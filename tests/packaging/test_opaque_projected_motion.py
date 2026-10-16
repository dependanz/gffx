"""Closed moving-face algebra independent of the renderer and shading oracle."""
from fractions import Fraction as Q
from types import SimpleNamespace
import pytest
from tools.opaque_moving_reference import P
from tools.opaque_shading_intervals import I,Jet


def _inputs(reverse=False,scale=Q(1)):
    tri=[[Q(0),Q(0)],[Q(2),Q(0)],[Q(0),Q(3)]]
    velocity=[[scale/7,-scale/11],[scale/5,scale/13],[-scale/17,scale/19]]
    iw=[Q(1),Q(2),Q(3)];diw=[scale/23,-scale/29,scale/31]
    n=[[Q(1),Q(0),Q(2)],[Q(2),Q(1),Q(1)],[Q(1),Q(3),Q(2)]]
    v=[[Q(2),Q(1),Q(3)],[Q(1),Q(2),Q(3)],[Q(3),Q(1),Q(2)]]
    dn=[[scale/37,-scale/41,scale/43] for _ in tri];dv=[[-scale/47,scale/53,scale/59] for _ in tri]
    if reverse:
        for rows in (tri,velocity,iw,diw,n,v,dn,dv):rows[1],rows[2]=rows[2],rows[1]
    return dict(triangle=tri,triangle_motion=velocity,inverse_w=iw,inverse_w_motion=diw,normals=n,views=v,normal_motion=dn,view_motion=dv)


@pytest.mark.parametrize('reverse',[False,True])
@pytest.mark.parametrize('scale',[Q(1),-Q(1,10**30)])
def test_moving_projected_basis_has_exact_vertex_identities(reverse,scale):
    from tools.opaque_projected_motion import ProjectedFields
    x=_inputs(reverse,scale);p=ProjectedFields(**x)
    for i,point in enumerate(p.triangle):
        raw=p.at(*point)
        for j,value in enumerate(raw):
            expected=p.det*P([x['inverse_w'][j],x['inverse_w_motion'][j]]) if i==j else P(0)
            assert (value-expected).zero
    assert p.det[0]>0 and p.certificate['radius']!='0'


@pytest.mark.parametrize('reverse',[False,True])
@pytest.mark.parametrize('scale',[Q(1),-Q(1,10**30)])
@pytest.mark.parametrize('order',[1,4,10])
def test_moving_projected_fields_keep_closed_perspective_and_horizon_motion(reverse,scale,order):
    from tools.opaque_projected_motion import ProjectedFields
    x=_inputs(reverse,scale);p=ProjectedFields(**x);patch=SimpleNamespace(triangle=p.triangle);field=p.radial(patch)
    u=Jet.variable(I(Q(2,5)),order=order)+Jet(0,order=order,tangent=scale/61);s=Jet(I(Q(1,3)),order=order,tangent=-scale/67);got=field(u,s)
    lift=lambda a,b=0:u.lift(a,tangent=b)
    beta=[1-u,u*(1-s),u*s];weights=[a*lift(w,d) for a,w,d in zip(beta,x['inverse_w'],x['inverse_w_motion'])];den=sum(weights,lift(0))
    want={}
    for key,name,motion in [('n','normals','normal_motion'),('v','views','view_motion')]:
        want[key]=[sum((weights[i]*lift(x[name][i][k],x[motion][i][k]) for i in range(3)),lift(0))/den for k in range(3)]
        for a,b in zip(got[key],want[key]):
            a=a/got['denominator'][0]
            for index in set(a.c)|set(b.c):
                aa=a.c.get(index,I(0));bb=b.c.get(index,I(0));assert max(aa.lo,bb.lo)<=min(aa.hi,bb.hi),(key,index)
    point=[lift(p.triangle[0][k][0],p.triangle[0][k][1])+u*(lift((p.triangle[1][k]-p.triangle[0][k])[0],(p.triangle[1][k]-p.triangle[0][k])[1])+s*lift((p.triangle[2][k]-p.triangle[1][k])[0],(p.triangle[2][k]-p.triangle[1][k])[1])) for k in range(2)]
    xx,yy=point;co=p.numerator();numerator=sum((lift(q[0],q[1])*z for q,z in zip(co,(xx*xx,xx*yy,yy*yy,xx,yy,1))),lift(0));dot=sum((a*b for a,b in zip(got['n'],got['v'])),lift(0))
    for index in set(numerator.c)|set(dot.c):
        a=numerator.c.get(index,I(0));b=dot.c.get(index,I(0));assert max(a.lo,b.lo)<=min(a.hi,b.hi),index
    # Tiny nonzero directional coefficients survive all interpolation and geometry terms.
    assert any(v.c.get((0,1),I(0)).lo!=0 or v.c.get((0,1),I(0)).hi!=0 for v in want['n'])


@pytest.mark.parametrize('bad',['zero_area','bad_weights','shape','zero_denominator'])
def test_moving_projected_fields_refuse_invalid_geometry(bad):
    from tools.opaque_projected_motion import ProjectedFields
    x=_inputs()
    if bad=='zero_area':x['triangle'][2]=x['triangle'][1]
    if bad=='bad_weights':x['inverse_w'][0]=0
    if bad=='shape':x['triangle_motion']=[]
    if bad=='zero_denominator':x['inverse_w']=[-1,-2,-3]
    with pytest.raises((ValueError,ArithmeticError)):ProjectedFields(**x)


@pytest.mark.parametrize('kind',['world_x','camera_x','world_shear_xz'])
def test_aligned_world_motion_has_closed_projection_and_view_chain(kind):
    from tools.opaque_projected_motion import aligned_world_motion
    vertices={str(i):v for i,v in enumerate([[1,0,1],[0,1,0],[-1,0,-2]])};w=[2,3,5]
    data=dict(scene_sha256='a'*64,selected=dict(original_vertex_ids=[0,1,2]),faces=[dict(vertices=[0,1,2],clip_w=w,normal=[[1,2,3]]*3)])
    profile=dict(status='verified_source_profile',input_sha256='b'*64,scene_sha256='a'*64,camera=dict(fx=2,fy=3,cx=0,cy=0,pose=[[1,0,0,0],[0,1,0,0],[0,0,1,3],[0,0,0,1]]),vertices=vertices)
    r=aligned_world_motion(data,profile,kind=kind,input_sha256='b'*64);omega=Q(.01)
    for i,(vertex,depth) in enumerate(zip(vertices.values(),w)):
        dx=omega*(Q(vertex[2]) if kind=='world_shear_xz' else (0 if kind=='camera_x' else 1));dc=omega if kind=='camera_x' else 0
        assert r['screen_motion'][i]==[Q(2)*(dx-dc)/depth,0,0]
        assert r['faces'][0]['triangle_motion'][i]==r['screen_motion'][i][:2]
        assert r['faces'][0]['view_motion'][i]==[dc-dx,0,0]
        assert r['faces'][0]['normal_motion'][i]==([3*omega/14,6*omega/14,-5*omega/14] if kind=='world_shear_xz' else [0,0,0])
        assert r['faces'][0]['inverse_w_motion'][i]==0
    if kind=='camera_x':
        opposite=aligned_world_motion(data,profile,kind='world_x',input_sha256='b'*64)
        assert all(a==-b for row,other in zip(r['screen_motion'],opposite['screen_motion']) for a,b in zip(row,other))
    with pytest.raises(ValueError):aligned_world_motion(data,profile,kind=kind,input_sha256='c'*64)
    with pytest.raises(ValueError):aligned_world_motion(data,profile,kind='unknown',input_sha256='b'*64)


def test_affine_shear_recomputed_area_normals_follow_exact_cofactor():
    # Independently cross the transformed edges, including every polynomial term.
    from tools.opaque_moving_reference import cross
    omega=Q(.01);a=list(map(P,[2,3,5]));b=list(map(P,[-3,7,11]));normal=cross(a,b)
    shear=lambda v:[v[0]+P([0,omega])*v[2],v[1],v[2]]
    actual=cross(shear(a),shear(b));expected=[normal[0],normal[1],normal[2]-P([0,omega])*normal[0]]
    assert all((x-y).zero for x,y in zip(actual,expected))
    # Any finite area-weighted sum obeys the same linear cofactor before normalization.
    c=list(map(P,[13,-2,1]));d=list(map(P,[3,4,-1]));second=cross(c,d)
    total=[x+y for x,y in zip(normal,second)];changed=[x+y for x,y in zip(cross(shear(a),shear(b)),cross(shear(c),shear(d)))]
    assert all((x-y).zero for x,y in zip(changed,[total[0],total[1],total[2]-P([0,omega])*total[0]]))


@pytest.mark.parametrize('piece',[0,8,11])
@pytest.mark.parametrize('motion',['world_x','world_shear_xz'])
def test_original_world_prepare_keeps_base_fields_and_complete_motion(piece,motion):
    import json,os,hashlib
    from pathlib import Path
    from tools.opaque_original_motion_reference import prepare,mapped_inputs
    path=Path(os.environ['GFFX_FULL_INPUT']);assert hashlib.sha256(path.read_bytes()).hexdigest()=='a15bea40cb5cdead7994d828b57f5dcfac9f44fab0292a06e7d0a7b08719f73a';data=json.loads(path.read_bytes())
    path=Path(os.environ['GFFX_WORLD_PROFILE']);assert hashlib.sha256(path.read_bytes()).hexdigest()==os.environ['GFFX_WORLD_PROFILE_SHA'];profile=json.loads(path.read_bytes())
    p=prepare(data,piece=piece,motion=motion,anchor=0,world_profile=profile)
    old=prepare(data,piece=piece,motion='normal_tilt',anchor=0)
    assert p['motion']==motion and p['face']==old['face'] and p['curved']==old['curved'] and p['world_motion']['kind']==motion
    assert p['visibility'] is not None
    u=Jet.variable('.3',order=1);v=Jet('.4',order=1)
    got=mapped_inputs(p,u,v);want=mapped_inputs(old,u,v)
    for a,b in zip(got,want):assert max(a[0,0].lo,b[0,0].lo)<=min(a[0,0].hi,b[0,0].hi)
    assert any(not a[0,1].zero for a in got)
    with pytest.raises(ValueError):prepare(data,piece=piece,motion=motion,anchor=0)
    with pytest.raises(ValueError):prepare(data,piece=piece,motion='normal_tilt',anchor=0,world_profile=profile)


@pytest.mark.parametrize('family',['front','outside','inside'])
def test_world_reference_forwards_explicit_profile_without_relabeling(monkeypatch,family):
    from tools import opaque_original_motion_reference as original
    from tools import opaque_inside_angle as inside
    profile={'identity':'explicit test profile'};calls=[]
    class Reached(RuntimeError):pass
    def capture(*args,**kwargs):
        calls.append(kwargs);raise Reached('profile reached selected geometry')
    if family=='front':
        monkeypatch.setattr(original,'prepare',capture);fn=original.reference;piece=0
    elif family=='outside':
        monkeypatch.setattr(original,'angular_material_cover',capture);fn=original.angular_piece_reference;piece=8
    else:
        monkeypatch.setattr(inside,'inside_material_cover',capture);fn=inside.inside_piece_reference;piece=11
    with pytest.raises(Reached):fn({},piece=piece,motion='world_x',anchor=0,world_profile=profile)
    assert len(calls)==1 and calls[0]['world_profile'] is profile and calls[0]['motion']=='world_x' and calls[0]['piece']==piece


def test_original_world_fresh_reference(record_property):
    import os,json,hashlib
    from pathlib import Path
    from tools.opaque_arb_jets import backend
    from tools.opaque_reference_profile import stronger_target
    from tools.opaque_original_motion_reference import reference,angular_piece_reference
    from tools.opaque_inside_angle import inside_piece_reference
    inp=Path(os.environ['GFFX_FULL_INPUT']);input_sha=hashlib.sha256(inp.read_bytes()).hexdigest();assert input_sha=='a15bea40cb5cdead7994d828b57f5dcfac9f44fab0292a06e7d0a7b08719f73a';data=json.loads(inp.read_bytes())
    src=Path(os.environ['GFFX_WORLD_PROFILE']);source_sha=hashlib.sha256(src.read_bytes()).hexdigest();assert source_sha==os.environ['GFFX_WORLD_PROFILE_SHA'];source_profile=json.loads(src.read_bytes())
    case=json.loads(os.environ['GFFX_MOTION_CASE']);assert case['motion'] in ('world_x','camera_x','world_shear_xz') and case['anchor']==0 and 0<=case['piece']<13
    profile=json.loads(os.environ['GFFX_WORLD_REFERENCE_PROFILE']);profile['tolerance']=stronger_target(profile['tolerance']);family='outside' if case['piece'] in (8,9) else 'inside' if case['piece'] in (11,12) else 'front'
    allowed={'front':{'correction_order','conditioned_correction','mixed_curve','transverse_correction','correction_axis','material_inputs','polynomial_inputs','tolerance'},'outside':{'tight_discriminant','algebraic_inverse','mapped_transport','exit_angle','unsquared_inverse','physical_angle','tensor_rule','compiled_series','correction_order','conditioned_correction','contraction_stop','mixed_curve','material_inputs','curved_transverse','polynomial_light','tolerance'},'inside':{'polynomial_light','tolerance'}}
    assert family=='outside' and set(profile)==allowed[family]|{'crossing_support','root_parts','tensor_upgrade_charts'} and profile['tensor_rule'] in ('gauss3x12-local4','gauss4x12-local4') and type(profile['root_parts']) is int and profile['root_parts'] in (2,4) and profile['crossing_support'] is True or family=='outside' and set(profile)==allowed[family]|{'crossing_support','root_parts'} and profile['crossing_support'] is True and type(profile['root_parts']) is int and profile['root_parts'] in (2,4) or set(profile)==allowed[family] or family=='outside' and set(profile)==allowed[family]|{'correction_axis'} and type(profile['correction_axis']) is int and profile['correction_axis'] in (0,1) or family=='front' and set(profile)==allowed[family]|{'line_order'} and type(profile['line_order']) is int and profile['line_order'] in (4,6) or family=='front' and set(profile)==allowed[family]|{'line_order','crossing_support'} and type(profile['line_order']) is int and profile['line_order'] in (4,6) and type(profile['crossing_support']) is bool or family=='front' and set(profile)==allowed[family]|{'line_order','crossing_support','root_parts'} and type(profile['line_order']) is int and profile['line_order'] in (4,6) and type(profile['crossing_support']) is bool and type(profile['root_parts']) is int and profile['root_parts'] in (1,4)
    schedule={}
    if os.environ.get('GFFX_FRONT_FRESH_SCHEDULE'):
        seed=Path(os.environ['GFFX_FRONT_FRESH_SCHEDULE']);assert hashlib.sha256(seed.read_bytes()).hexdigest()==os.environ['GFFX_FRONT_FRESH_SCHEDULE_SHA'];schedule['initial_leaves']=json.loads(seed.read_bytes())['leaves']
    dest=Path(os.environ['GFFX_FULL_REFERENCE']);assert not dest.exists();progress=[];header=dict(case=case,family=family,profile=profile,input_sha256=input_sha,world_profile_sha256=source_sha)
    try:
        with backend(profile='ball',fast_conversion=True):
            result={'front':reference,'outside':angular_piece_reference,'inside':inside_piece_reference}[family](data,world_profile=source_profile,progress=progress.append,**case,**profile,**schedule)
    except Exception as error:
        dest.write_text(json.dumps(dict(status='unresolved',**header,details=getattr(error,'details',repr(error)),progress=progress),default=str,indent=2),encoding='utf8');raise
    dest.write_text(json.dumps(dict(status='qualified',**header,result=result,progress=progress),default=str,indent=2),encoding='utf8');record_property('world_piece_reference',str(dest))


@pytest.mark.parametrize('piece',[8,11])
@pytest.mark.parametrize('motion',['world_x','world_shear_xz'])
@pytest.mark.parametrize('order',[1,4])
def test_original_world_physical_map_keeps_implicit_angle_and_complete_measure(piece,motion,order):
    import json,os,hashlib
    from pathlib import Path
    from tools.opaque_arb_jets import backend
    from tools.opaque_original_motion_reference import angular_material_cover,mapped_inputs
    from tools.opaque_inside_angle import inside_material_cover
    from tools.opaque_physical_angle import physical_angle_chart
    from tests.packaging.test_opaque_motion_extension import PROFILE
    path=Path(os.environ['GFFX_FULL_INPUT']);assert hashlib.sha256(path.read_bytes()).hexdigest()=='a15bea40cb5cdead7994d828b57f5dcfac9f44fab0292a06e7d0a7b08719f73a';data=json.loads(path.read_bytes())
    path=Path(os.environ['GFFX_WORLD_PROFILE']);assert hashlib.sha256(path.read_bytes()).hexdigest()==os.environ['GFFX_WORLD_PROFILE_SHA'];profile=json.loads(path.read_bytes())
    with backend(profile='ball',fast_conversion=True):
        if piece==8:p,curve,cover,charts=angular_material_cover(data,piece=piece,motion=motion,anchor=0,world_profile=profile,**PROFILE,curved_transverse=True,polynomial_light=True)
        else:p,curve,cover,charts=inside_material_cover(data,piece=piece,motion=motion,anchor=0,world_profile=profile,polynomial_light=True)
        index=next(i for i,c in enumerate(cover.charts) if c['low'] is not None and c['high'] is not None)
        t=Jet.variable('.5',order=order);y=Jet('.333',order=order)
        row=physical_angle_chart(p['physical_angle'],cover,index,t,y,lower=lambda s:(1-p['patch'].root(Jet(s[0,0])) if piece==11 else p['patch'].root(Jet(s[0,0])))[0,0]);assert row is not None
        radius=1-row['radius'] if piece==11 else row['radius']
        fields=p['fields'](radius,row['s']);n,v=fields['n'],fields['v']
        dot=lambda a,b:sum((x*y for x,y in zip(a,b)),radius.lift(0))
        nv=dot(n,v)/(dot(n,n)*dot(v,v)).sqrt()
        physical_jac=radius*radius.lift(p['patch'].det[0],tangent=p['patch'].det[1])*row['jacobian']
        assert physical_jac[0,0].lo>0 and not row['radius'][0,1].zero
        for k in range(order+1):
            for j in (0,1):
                a,b=nv[k,j],row['angle'][k,j];assert max(a.lo,b.lo)<=min(a.hi,b.hi),(k,j,a,b)
                assert a.width<I('1e-25').lo and b.width<I('1e-25').lo


@pytest.mark.parametrize('coefficient',[Q(2)**4096,-Q(1,2**4096)])
def test_quadratic_first_motion_product_does_not_construct_discarded_second_motion(coefficient):
    from tools.opaque_quadratic_sqrt import mul
    got=mul([P([1,coefficient])],[P([1,coefficient])])
    assert len(got)==1 and got[0].c==P([1,2*coefficient]).c


def test_quadratic_first_motion_product_retains_complexity_guard_for_consumed_terms():
    from tools.opaque_quadratic_sqrt import mul
    with pytest.raises(ValueError,match='complexity'):
        mul([P(2**4096)],[P(2**4096)])


@pytest.mark.parametrize('motion',[Q(1),-Q(1,2**60)])
@pytest.mark.parametrize('order',[1,4,10])
def test_normalized_quadratic_root_preserves_closed_large_rational_scale(motion,order):
    from tools.opaque_quadratic_sqrt import QuadraticSqrt
    from tools.opaque_arb_jets import backend
    scale=Q(2**700+3,2**701+1)
    coefficients=[P([4*scale**2,12*scale**2*motion]),P([4*scale**2,10*scale**2*motion]),P([scale**2,2*scale**2*motion])]
    with backend(profile='ball',fast_conversion=True):
        root=QuadraticSqrt(coefficients,normalize=True);x=Jet.variable('.3',order=order)+Jet(0,order=order,tangent=motion/7)
        actual=root(x);expected=scale*(2+x+(3+x)*x.lift(0,tangent=motion))
        for i in range(order+1):
            for j in (0,1):
                a,b=actual[i,j],expected[i,j];assert max(a.lo,b.lo)<=min(a.hi,b.hi);assert a.width<I('1e-25').lo
        original=root.range(I('.3'));wanted=I(scale**2)*I('2.3')**2;assert max(original.lo,wanted.lo)<=min(original.hi,wanted.hi)


def test_normalized_quadratic_root_refuses_ambiguous_or_nonpositive_scale():
    from tools.opaque_quadratic_sqrt import QuadraticSqrt
    for bad in (1,'yes',None):
        with pytest.raises(ValueError):QuadraticSqrt([P(1)],normalize=bad)
    with pytest.raises(ValueError):QuadraticSqrt([P(0),P(1)],normalize=True)
    with pytest.raises(ValueError):QuadraticSqrt([P(-1),P(2)],normalize=True)


@pytest.mark.parametrize('normal',[(Q(3,5),0,Q(4,5)),(0,Q(4,5),Q(3,5)),(Q(1,3),Q(2,3),Q(2,3))])
def test_world_shear_normalization_precedes_perspective_interpolation(normal):
    from tools.opaque_projected_motion import aligned_world_motion
    from tools.opaque_arb_jets import backend
    data=dict(scene_sha256='a'*64,selected=dict(original_vertex_ids=[0,1,2]),faces=[dict(vertices=[0,1,2],clip_w=[2,3,5],normal=[normal]*3)])
    profile=dict(status='verified_source_profile',input_sha256='b'*64,scene_sha256='a'*64,camera=dict(fx=2,fy=3,cx=0,cy=0,pose=[[1,0,0,0],[0,1,0,0],[0,0,1,3],[0,0,0,1]]),vertices={str(i):v for i,v in enumerate([[1,0,1],[0,1,0],[-1,0,-2]])})
    r=aligned_world_motion(data,profile,kind='world_shear_xz',input_sha256='b'*64);got=r['faces'][0]['normal_motion'][0]
    with backend(profile='ball',fast_conversion=True):
        theta=Jet(0,order=1,tangent=1);a=[theta.lift(n) for n in normal];a[2]=a[2]-Q(.01)*theta*normal[0]
        length=sum((v*v for v in a),theta.lift(0)).sqrt()
        for v,w in zip(a,got):
            derivative=(v/length)[0,1];assert derivative.lo<=I(w).hi and I(w).lo<=derivative.hi
    assert sum(n*d for n,d in zip(normal,got))==0


@pytest.mark.parametrize('axis',[0,1])
def test_world_angular_correction_direction_reaches_every_chart(monkeypatch,axis):
    import os,json,hashlib
    from pathlib import Path
    import tools.opaque_original_motion_reference as original
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_extension import PROFILE
    inp=Path(os.environ['GFFX_FULL_INPUT']);world=Path(os.environ['GFFX_WORLD_PROFILE']);assert hashlib.sha256(world.read_bytes()).hexdigest()==os.environ['GFFX_WORLD_PROFILE_SHA']
    factory=original.TangentCorrection;axes=[];local=[]
    def observe(*args,**kwargs):
        result=factory(*args,**kwargs);axes.append(result.integrated_axis);local.append(result.local_crossings);return result
    monkeypatch.setattr(original,'TangentCorrection',observe)
    with backend(profile='ball',fast_conversion=True):
        p,curve,cover,charts=original.angular_material_cover(json.loads(inp.read_bytes()),piece=8,motion='world_x',anchor=0,world_profile=json.loads(world.read_bytes()),**PROFILE,curved_transverse=True,polynomial_light=True,correction_axis=axis)
    assert len(axes)==len(charts)==72 and set(axes)=={axis} and p['correction_axis']==axis and all(local)


@pytest.mark.parametrize('axis',[True,-1,2,None])
def test_angular_correction_direction_refuses_invalid_selection(axis):
    from tools.opaque_original_motion_reference import angular_material_cover,angular_piece_reference
    for fn in (angular_material_cover,angular_piece_reference):
        with pytest.raises(ValueError,match='correction axis'):fn({},correction_axis=axis)


@pytest.mark.parametrize('axis',[0,1])
def test_local_crossing_range_excludes_only_provably_absent_knot(axis):
    from types import SimpleNamespace
    from tools.opaque_tangent_peano import TangentCorrection
    from tools.opaque_arb_jets import backend
    with backend(profile='ball',fast_conversion=True):
        extension=SimpleNamespace(cn=[I(0),I('.502'),I(1)],mean=[I(1),I(1)])
        def inputs(x,y):
            z=(x,y)[axis];q=Q(1,2)*(1+z)/(1+z)+z.lift(0,tangent=Q(1,10**30))
            return q,z.lift('.2'),z.lift(1),z.lift(1)
        correction=TangentCorrection(extension,inputs,curved=True,order=4,local_crossings=True)
        correction.ordinary_inputs=lambda u,values:u.lift(0)
        domain=[I(0),I(0)];domain[axis]=I(0,'.01')
        jets=[Jet(v,order=4) for v in domain];jets[axis]=Jet.variable(domain[axis],order=4);raw=inputs(*jets)[0]
        assert raw[0,0].lo<I('.502').lo<raw[0,0].hi and raw[1,0].lo<0<raw[1,0].hi and not raw[0,1].zero
        rows=correction.axis_bounds(tuple(domain),axis)
        assert len(rows)==2 and all(not row['jumps'] for row in rows)
        # A knot at the actual value cannot be ruled out by this argument.
        extension.cn=[I(0),I('.5'),I(1)]
        with pytest.raises(ValueError,match='monotonicity'):correction.axis_bounds(tuple(domain),axis)


@pytest.mark.parametrize('bad',[1,None,'yes'])
def test_local_crossing_range_requires_explicit_boolean(bad):
    from tools.opaque_tangent_peano import TangentCorrection
    with pytest.raises(ValueError,match='local crossing'):TangentCorrection(None,None,local_crossings=bad)


@pytest.mark.parametrize('degree',[1,2,3])
@pytest.mark.parametrize('axis',[0,1])
@pytest.mark.parametrize('reverse',[False,True])
def test_local_crossing_proof_retains_closed_piecewise_integrals(monkeypatch,degree,axis,reverse):
    import tools.opaque_tangent_peano as peano
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_tangent_peano import test_tangent_peano_contains_closed_piecewise_log_integral
    factory=peano.TangentCorrection
    def selected(*args,**kwargs):return factory(*args,**dict(kwargs,order=4,conditioned=True,local_crossings=True))
    monkeypatch.setattr(peano,'TangentCorrection',selected)
    with backend(profile='ball',fast_conversion=True):test_tangent_peano_contains_closed_piecewise_log_integral(degree,axis,reverse)


@pytest.mark.parametrize('degree',[1,2,3])
@pytest.mark.parametrize('axis',[0,1])
@pytest.mark.parametrize('reverse',[False,True])
def test_sixth_line_proof_keeps_closed_log_integrals(monkeypatch,degree,axis,reverse):
    import tools.opaque_tangent_peano as peano
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_tangent_peano import test_tangent_peano_contains_closed_piecewise_log_integral
    factory=peano.TangentCorrection
    def selected(*args,**kwargs):return factory(*args,**dict(kwargs,order=4,conditioned=True,centered=True,transverse=True,line_order=6,integrated_axis=1-axis))
    monkeypatch.setattr(peano,'TangentCorrection',selected)
    with backend(profile='ball',fast_conversion=True):test_tangent_peano_contains_closed_piecewise_log_integral(degree,axis,reverse)


@pytest.mark.parametrize('axis',[0,1])
@pytest.mark.parametrize('degree',[1,2,3,4,5,6,7])
@pytest.mark.parametrize('scale',[Q(1),-Q(1,2**60)])
def test_mixed_fourth_sixth_rule_keeps_all_closed_jumps(axis,degree,scale):
    import math
    from tools.opaque_fourth_tangent import integral
    from tools.opaque_arb_jets import backend
    with backend(profile='ball',fast_conversion=True):
        def field(x,y):
            z=x+y-Q(1,2)
            if z[0,0].hi<0:return [x.lift(0)]*2
            assert z[0,0].lo>0
            return [z**degree,scale*degree*z**(degree-1)]
        def bounds(box,active,order=4):
            other=box[1-active];root=I('.5')-other;lo=max(box[active].lo,root.lo);hi=min(box[active].hi,root.hi);rows=[]
            z=box[0]+box[1]-I('.5');positive=I(max(I(0).lo,z.lo),max(I(0).hi,z.hi))
            for d,factor in [(degree,Q(1)),(degree-1,scale*degree)]:
                derivative=I(0) if d<order else factor*Q(math.factorial(d),math.factorial(d-order))*positive**(d-order)
                if z.lo<0:derivative=I(min(I(0).lo,derivative.lo),max(I(0).hi,derivative.hi))
                values=[I(factor*math.factorial(d)) if k==d else I(0) for k in range(order)]
                rows.append(dict(order=order,derivative=I(derivative.bound),derivative_range=derivative,jumps=[] if lo>hi else [dict(position=I(lo,hi),values=values,required=False)]))
            return rows
        got=integral(field,(I(0,1),I(0,1)),bounds,conditioned=True,direct_conditioned=True,condition_parts=4,integrated_axis=axis,line_bounds=lambda box,active:bounds(box,active,6))
        exact=[(Q(3,2)**(degree+2)-2*Q(1,2)**(degree+2))/((degree+1)*(degree+2)),scale*(Q(3,2)**(degree+1)-2*Q(1,2)**(degree+1))/(degree+1)]
        for actual,value in zip(got,exact):
            wanted=I(value);assert actual.lo<=wanted.lo<=wanted.hi<=actual.hi
        assert not got[1].zero


@pytest.mark.parametrize('kwargs',[dict(line_order=True),dict(line_order=5),dict(line_order=6),dict(order=4,line_order=6),dict(order=4,conditioned=True,line_order=6),dict(order=4,conditioned=True,transverse=True,curved=True,line_order=6)])
def test_sixth_line_proof_rejects_unqualified_profiles(kwargs):
    from tools.opaque_tangent_peano import TangentCorrection
    with pytest.raises(ValueError):TangentCorrection(None,None,**kwargs)


@pytest.mark.parametrize('materials',[([1,1,1],[Q(1,25)]*3),([Q(1,2),Q(1,3),Q(1,2)],[Q(1,25),Q(1,25),Q(1,25)]),([1,1,1],[0,Q(1,2**60),1])])
@pytest.mark.parametrize('omega',[Q(1),-Q(1,2**60)])
def test_identical_rgb_transport_reuses_only_equal_endpoint_proofs(monkeypatch,materials,omega):
    import importlib.util,os,json,hashlib
    from pathlib import Path
    import tools.opaque_knot_transport as current
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_loss_extension import Curve
    path=Path(os.environ['GFFX_FROZEN_TRANSPORT_SOURCE']);assert hashlib.sha256(path.read_bytes()).hexdigest()==os.environ['GFFX_FROZEN_TRANSPORT_SHA']
    spec=importlib.util.spec_from_file_location('old_transport_proof',path);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    a,f=materials;calls=[];original=current.transversal_flux
    def observed(*args,**kwargs):calls.append(1);return original(*args,**kwargs)
    monkeypatch.setattr(current,'transversal_flux',observed)
    with backend(profile='ball',fast_conversion=True):
        curve=Curve();curve.mean=[I(1),I(1)];curve.pi=I(3)
        def field(x,y):return [x/3+x.lift(Q(1,3),tangent=omega),y/4+y.lift(Q(1,3),tangent=-omega)]
        expected=old.material_transport(curve,field,(I(0,1),I(0,1)),a,f)
        actual=current.material_transport(curve,field,(I(0,1),I(0,1)),a,f)
        assert json.dumps(actual,default=str,sort_keys=True)==json.dumps(expected,default=str,sort_keys=True)
        assert len(calls)==2*len(set(zip(a,f)))
        assert all(v.hi>0 for v in actual['bound'])
        calls.clear();overrides=[]
        def override(*args):overrides.append(1);return None
        hook=current.material_transport(curve,field,(I(0,1),I(0,1)),a,f,flux_override=override)
        assert len(overrides)==len(calls)==6
        assert json.dumps(hook,default=str,sort_keys=True)==json.dumps(expected,default=str,sort_keys=True)


@pytest.mark.parametrize('order',[4,6])
def test_front_line_order_reaches_qualified_correction(monkeypatch,order):
    import os,json
    from pathlib import Path
    import tools.opaque_original_motion_reference as original
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    class Reached(Exception):pass
    def selected(*args,**kwargs):
        assert kwargs['line_order']==order
        raise Reached
    monkeypatch.setattr(original,'TangentCorrection',selected)
    with backend(profile='ball',fast_conversion=True),pytest.raises(Reached):
        original.reference(frozen(),piece=3,motion='world_x',anchor=0,world_profile=json.loads(Path(os.environ['GFFX_WORLD_PROFILE']).read_bytes()),correction_order=4,conditioned_correction=True,mixed_curve=True,transverse_correction=True,correction_axis=0,material_inputs=True,polynomial_inputs=True,line_order=order)


@pytest.mark.parametrize('order',[True,3,5,None])
def test_front_line_order_refuses_invalid_selection(order):
    from tools.opaque_original_motion_reference import reference
    with pytest.raises(ValueError,match='line'):reference({},piece=0,motion='world_x',line_order=order)


@pytest.mark.parametrize('order',[0,1,4,6,10])
@pytest.mark.parametrize('axis',[0,1])
@pytest.mark.parametrize('omega',[Q(1),-Q(1,2**60)])
def test_cancelled_polynomial_motion_preserves_complete_coordinate_chain(order,axis,omega):
    from tools.opaque_polynomial_inputs import PolynomialCosines
    from tools.opaque_arb_jets import backend
    z=(Q(0),Q(0));one=(Q(1),Q(0))
    co=dict(n=[[(Q(0),omega),one,one],[one,z,z],[z,z,z]],v=[[(Q(1),omega/3),z,z],[z,z,z],[z,z,z]])
    with backend(profile='ball',fast_conversion=True):
        coords=[Jet(Q(1,5),order=order),Jet(Q(3,10),order=order)];coords[axis]=Jet.variable(coords[axis][0,0],order=order)
        u=coords[0]+coords[0].lift(0,tangent=omega/7);s=coords[1]+coords[1].lift(0,tangent=-omega/11)
        actual=PolynomialCosines(co,cancelled_motion=True)(u,s,material_only=True);old=PolynomialCosines(co)(u,s,material_only=True)
        a=coords[0]*(1+coords[1]);change=omega+omega/7*(1+coords[1])-omega/11*coords[0];den=(1+a*a);expected=[change/(den*den.sqrt()),change*(1-a)/(den*I(6).sqrt()*den.sqrt())]
        for role in (0,1):
            for k in range(order+1):
                assert (actual[role][k,0].lo,actual[role][k,0].hi)==(old[role][k,0].lo,old[role][k,0].hi)
                q,w=actual[role][k,1],expected[role][k,0];assert max(q.lo,w.lo)<=min(q.hi,w.hi);assert q.width<I('1e-25').lo
            assert not actual[role][0,1].zero


@pytest.mark.parametrize('bad',[1,None,'yes'])
def test_cancelled_polynomial_motion_requires_explicit_boolean(bad):
    from tools.opaque_polynomial_inputs import PolynomialCosines
    with pytest.raises(ValueError,match='motion'):PolynomialCosines({},cancelled_motion=bad)


@pytest.mark.parametrize('omega',[Q(1),-Q(1,2**60)])
def test_crossing_support_keeps_every_possible_transverse_root(omega):
    from tools.opaque_tangent_peano import TangentCorrection
    from tools.opaque_loss_extension import LossExtension
    from tests.packaging.test_opaque_loss_extension import Curve
    from tools.opaque_arb_jets import backend
    with backend(profile='ball',fast_conversion=True):
        curve=Curve();curve.mean=[I(1),I(1)];curve.pi=I(3);e=LossExtension(curve);e.indices=[0,1]
        def inputs(x,y):return (x+y)/4+x.lift(Q(1,4),tangent=omega),x.lift(Q(7,10)),x.lift(1),1+y+y*y
        kw=dict(order=4,conditioned=True,centered=True,transverse=True)
        old=TangentCorrection(e,inputs,**kw);new=TangentCorrection(e,inputs,crossing_support=True,**kw)
        box=(I('.2','.3'),I(0,1));before=old.axis_bounds(box,0);after=new.axis_bounds(box,0)
        assert len(after[1]['jumps'])==1
        for j in (0,1):
            a,b=after[j]['jumps'][0],before[j]['jumps'][0]
            assert a['position'].lo==b['position'].lo and a['position'].hi==b['position'].hi
            assert a['values'][0].width<b['values'][0].width
            for y in (Q(3,10),Q(7,20),Q(2,5)):
                reference=old.axis_bounds((box[0],I(y)),0,transverse=False)[j]['jumps']
                for w in reference:
                    for actual,expected in zip(a['values'],w['values']):assert actual.lo<=expected.lo<=expected.hi<=actual.hi


@pytest.mark.parametrize('degree',[1,2,3])
@pytest.mark.parametrize('axis',[0,1])
@pytest.mark.parametrize('reverse',[False,True])
def test_crossing_support_preserves_closed_log_integrals(monkeypatch,degree,axis,reverse):
    import tools.opaque_tangent_peano as peano
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_tangent_peano import test_tangent_peano_contains_closed_piecewise_log_integral
    factory=peano.TangentCorrection
    def selected(*a,**k):return factory(*a,**dict(k,order=4,conditioned=True,centered=True,transverse=True,crossing_support=True))
    monkeypatch.setattr(peano,'TangentCorrection',selected)
    with backend(profile='ball',fast_conversion=True):test_tangent_peano_contains_closed_piecewise_log_integral(degree,axis,reverse)


@pytest.mark.parametrize('bad',[1,None,'yes'])
def test_crossing_support_requires_explicit_boolean(bad):
    from tools.opaque_tangent_peano import TangentCorrection
    with pytest.raises(ValueError,match='crossing support'):TangentCorrection(None,None,crossing_support=bad)


@pytest.mark.parametrize('support',[False,True])
def test_front_crossing_support_reaches_qualified_correction(monkeypatch,support):
    import os,json
    from pathlib import Path
    import tools.opaque_original_motion_reference as original
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    class Reached(Exception):pass
    def selected(*args,**kwargs):
        assert kwargs['crossing_support'] is support
        raise Reached
    monkeypatch.setattr(original,'TangentCorrection',selected)
    with backend(profile='ball',fast_conversion=True),pytest.raises(Reached):
        original.reference(frozen(),piece=3,motion='world_x',anchor=0,world_profile=json.loads(Path(os.environ['GFFX_WORLD_PROFILE']).read_bytes()),correction_order=4,conditioned_correction=True,mixed_curve=True,transverse_correction=True,correction_axis=0,material_inputs=True,polynomial_inputs=True,line_order=6,crossing_support=support)


@pytest.mark.parametrize('support',[1,None,'yes'])
def test_front_crossing_support_refuses_invalid_selection(support):
    from tools.opaque_original_motion_reference import reference
    with pytest.raises(ValueError,match='crossing support'):reference({},piece=0,motion='world_x',crossing_support=support)


def test_front_material_cache_preserves_fresh_region_bounds(record_property):
    import os,json,hashlib,importlib.util
    from pathlib import Path
    from tools.opaque_arb_jets import backend
    from tools.opaque_original_motion_reference import reference
    from tests.packaging.test_opaque_motion_tangent import frozen
    path=Path(os.environ['GFFX_FROZEN_FRONT_SOURCE']);assert hashlib.sha256(path.read_bytes()).hexdigest()==os.environ['GFFX_FROZEN_FRONT_SHA']
    spec=importlib.util.spec_from_file_location('front_material_cache_baseline',path);baseline=importlib.util.module_from_spec(spec);spec.loader.exec_module(baseline)
    profile=json.loads(Path(os.environ['GFFX_WORLD_PROFILE']).read_bytes());data=frozen()
    kw=dict(piece=3,motion='world_x',anchor=0,world_profile=profile,correction_order=4,conditioned_correction=True,mixed_curve=True,transverse_correction=True,correction_axis=0,material_inputs=True,polynomial_inputs=True,line_order=6,crossing_support=True,diagnostic_boxes=[[Q(3,4),Q(49,64),0,Q(1,2)],[Q(49,64),Q(25,32),0,Q(1,2)]])
    with backend(profile='ball',fast_conversion=True):
        expected=baseline.reference(data,**kw);actual=reference(data,**kw)
        assert json.dumps(actual['rows'],default=str,sort_keys=True)==json.dumps(expected['rows'],default=str,sort_keys=True)
        cache=actual['front_material_cache'];assert cache['hits']>0 and 0<cache['entries']<=64
    record_property('cache',json.dumps(cache))


@pytest.mark.parametrize('degree',[1,2,3])
@pytest.mark.parametrize('axis',[0,1])
@pytest.mark.parametrize('omega',[Q(1),-Q(1,2**60)])
@pytest.mark.parametrize('signs',[(1,1),(-1,-1),(1,-1),(-1,1)])
def test_partitioned_crossing_keeps_closed_two_axis_log_integral(monkeypatch,degree,axis,omega,signs):
    import tools.opaque_tangent_transverse as transverse
    selected=transverse.refine
    # Exercise the optional partition proof even when the baseline is tight.
    monkeypatch.setattr(transverse,"refine",lambda *a,**k:selected(*a,**dict(k,refine_small=True)))
    from decimal import Decimal as D,localcontext
    import math
    from tools.opaque_tangent_peano import TangentCorrection
    from tools.opaque_loss_extension import LossExtension
    from tools.opaque_arb_jets import backend
    with backend(profile='ball',fast_conversion=True):
        h=I(2).ln();controls={1:[I(0),h/3,2*h/3,h],2:[I(0),I(0),h*h/3,h*h],3:[I(0),I(0),I(0),h*h*h]}[degree]
        curve=SimpleNamespace(cn=list(map(I,[Q(1,4),Q(1,2),1])),x=[I(Q(1,4)).ln(),I(Q(1,2)).ln(),I(0)],controls=[[[I(0)]*4]*2,[controls]*2],mean=[I(1),I(1)],pi=I(1))
        e=LossExtension(curve);e.indices=[0,1]
        def inputs(x,y):return ((x if signs[0]>0 else 1-x)+(y if signs[1]>0 else 1-y))/4+x.lift(Q(1,4),tangent=omega),x.lift(1),x.lift(1),x.lift(1)
        proof=TangentCorrection(e,inputs,order=4,centered=True,conditioned=True,transverse=True,line_order=6,crossing_support=True,root_parts=4,integrated_axis=axis)
        got=proof((I(0,1),I(0,1)))
        assert proof.root_partition_queries>0
        with localcontext() as c:
            c.prec=90
            # Twice-antiderivative of log(2q)^degree, zero below q=1/2.
            def primitive(q):
                z=(2*q).ln()
                return q*sum(D((-1)**j*math.factorial(degree))/D(math.factorial(degree-j))*(z**(degree-j) if degree!=j else D(1)) for j in range(degree+1))
            # d/dtheta of the square integral is the first primitive at
            # the four corners divided by both spatial slopes.
            half=D('.5');top=D('.75');exact=16*(primitive(top)-primitive(half))*D(2).ln()**degree*D(omega.numerator)/D(omega.denominator)
            assert got[1].lo<=exact<=got[1].hi
        assert not got[1].zero


@pytest.mark.parametrize('bad',[True,0,2,8,None])
def test_root_partition_requires_qualified_fixed_count(bad):
    from tools.opaque_tangent_peano import TangentCorrection
    with pytest.raises(ValueError,match='root'):TangentCorrection(None,None,root_parts=bad)



def test_requalify_saved_tensor_certificate(record_property):
    import os,json,hashlib
    from pathlib import Path
    from tools import opaque_original_motion_reference as original,opaque_inside_angle as inside
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    spec_path=Path(os.environ['GFFX_TENSOR_REQUAL_SPEC']);assert hashlib.sha256(spec_path.read_bytes()).hexdigest()==os.environ['GFFX_TENSOR_REQUAL_SHA'];spec=json.loads(spec_path.read_bytes())
    source=Path(spec['source']);assert hashlib.sha256(source.read_bytes()).hexdigest()==spec['source_sha256'];saved=json.loads(source.read_bytes());assert saved['status']=='qualified'
    case=spec['case'];profile=spec['profile'];leaves=saved['result']['basis']['chart_leaves'];assert 1<=len(leaves)<=128
    world_path=Path(os.environ['GFFX_WORLD_PROFILE']);assert hashlib.sha256(world_path.read_bytes()).hexdigest()==os.environ['GFFX_WORLD_PROFILE_SHA'];world=json.loads(world_path.read_bytes())
    extra=dict(world_profile=world) if case['motion'].startswith('world_') or case['motion']=='camera_x' else {}
    dest=Path(os.environ['GFFX_FULL_REFERENCE']);assert not dest.exists();progress=[];header=dict(case=case,profile=profile,family=spec['family'],source_sha256=spec['source_sha256'],input_sha256=hashlib.sha256(Path(os.environ['GFFX_FULL_INPUT']).read_bytes()).hexdigest(),world_profile_sha256=os.environ['GFFX_WORLD_PROFILE_SHA'] if extra else None,scope='Fresh complete reference; parent supplies geometry/profile only, no cached bounds.')
    fn=dict(front=original.reference,outside=original.angular_piece_reference,inside=inside.inside_piece_reference)[spec['family']]
    try:
        with backend(profile='ball',fast_conversion=True):result=fn(frozen(),**case,**profile,**extra,initial_leaves=leaves,progress=progress.append)
    except Exception as error:
        dest.write_text(json.dumps(dict(status='unresolved',**header,details=getattr(error,'details',repr(error)),progress=progress),default=str,indent=2));raise
    dest.write_text(json.dumps(dict(status='qualified',**header,result=result,progress=progress),default=str,indent=2));record_property('fresh_certificate',str(dest))


@pytest.mark.parametrize('order',[0,1,4,6,10])
@pytest.mark.parametrize('material',[False,True])
def test_polynomial_preparation_omits_only_unconsumed_scalar_work(order,material):
    import importlib.util,os,hashlib
    from pathlib import Path
    import tools.opaque_polynomial_inputs as current
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_polynomial_inputs import pairs,fields
    old_path=Path(os.environ['GFFX_FROZEN_POLYNOMIAL_SOURCE']);assert hashlib.sha256(old_path.read_bytes()).hexdigest()==os.environ['GFFX_FROZEN_POLYNOMIAL_SHA']
    spec=importlib.util.spec_from_file_location('frozen_polynomial_inputs',old_path);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    with backend(profile='ball',fast_conversion=True):
        co=pairs(fields(Q(1,10**30),view=True));before=old.PolynomialCosines(co);after=current.PolynomialCosines(co)
        for u,s in ((I('.2'),I('.3')),(I('.2','.3'),I('.3','.4'))):
            x=Jet.variable(u,order=order) if order else Jet(u);x=x+x.lift(0,tangent=-Q(1,2**60));y=Jet(s,order=order)
            for method in ('__call__','light_highlight'):
                expected=getattr(before,method)(x,y,material_only=material);actual=getattr(after,method)(x,y,material_only=material)
                for a,b in zip(actual,expected):
                    assert {(k,v.lo,v.hi) for k,v in a.c.items()}=={(k,v.lo,v.hi) for k,v in b.c.items()}
        assert sum(p.row_evaluations for pair in after.prepared for p in pair)==0
        if material:assert sum(p.rebuilds for p in after.prepared[4])==0


def test_polynomial_prepare_rebuilds_on_context_changes_without_evaluating():
    import tools.opaque_shading_intervals as scalar
    from tools.opaque_angle_light import PreparedIntervalPolynomial
    p=PreparedIntervalPolynomial({(0,0,0):Q(1,3),(1,1,0):Q(1,7)})
    old=scalar.PRECISION
    try:
        p.prepare();assert p.rebuilds==1 and p.row_evaluations==0
        first=p.rows;p.prepare();assert p.rows is first and p.rebuilds==1
        scalar.PRECISION=47;p.prepare();assert p.rebuilds==2 and p.row_evaluations==0
        got=p(I('.2'),I('.3'));want=I(Q(1,3))+I(Q(1,7))*I('.2')*I('.3');assert got.lo<=want.hi and want.lo<=got.hi
    finally:scalar.PRECISION=old


@pytest.mark.parametrize('order',[1,2,3,4])
@pytest.mark.parametrize('support',[(('0','0'),('1','1')),(('.02','.07'),('.82','.95')),(('.25','.25'),('.75','.75'))])
@pytest.mark.parametrize('coefficient',[('1','2'),('-2','-1'),('-1e-1000','1e-1000'),('1','1'),('0','0')])
def test_weighted_kernel_reuse_preserves_all_frozen_endpoints(monkeypatch,order,support,coefficient):
    import importlib.util,os,hashlib
    from pathlib import Path
    import tools.opaque_high_transport as current
    path=Path(os.environ['GFFX_FROZEN_WEIGHTED_SOURCE']);assert hashlib.sha256(path.read_bytes()).hexdigest()==os.environ['GFFX_FROZEN_WEIGHTED_SHA']
    spec=importlib.util.spec_from_file_location('frozen_weighted_mass',path);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    calls=[];original=current.kernel_integral
    def count(*args):calls.append(1);return original(*args)
    monkeypatch.setattr(current,'kernel_integral',count)
    a,b=[I(*v) for v in support];c=I(*coefficient);expected=old.weighted_mass(c,a,b,order)
    for _ in range(2):
        actual=current.weighted_mass(c,a,b,order);assert (actual.lo,actual.hi)==(expected.lo,expected.hi)
    assert len(calls)==(0 if c.zero else 1)



def test_weighted_kernel_cache_respects_context_and_capacity():
    from tools import opaque_high_transport as m,opaque_shading_intervals as scalar
    from tools.opaque_arb_jets import backend
    with backend(profile='ball',fast_conversion=True):
        old=scalar.PRECISION
        try:
            for precision in (60,47,60):
                scalar.PRECISION=precision
                a,b=I('.13','.17'),I('.81','.89');actual=m.weighted_mass(I(2),a,b,2);expected=I(2)*m.kernel_integral(a,b,2)
                assert (actual.lo,actual.hi)==(expected.lo,expected.hi)
            for k in range(140):m.weighted_mass(I(1,2),I(Q(k,1000)),I('.9'),2)
            assert m._signed_kernel_cached.cache_info().currsize<=128 and m._kernel_range_cached.cache_info().currsize<=128
        finally:scalar.PRECISION=old


@pytest.mark.parametrize('parts',[1,4])
def test_front_root_partition_reaches_qualified_correction(monkeypatch,parts):
    import os,json
    from pathlib import Path
    import tools.opaque_original_motion_reference as original
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    class Reached(Exception):pass
    def selected(*args,**kwargs):
        assert kwargs['root_parts']==parts
        raise Reached
    monkeypatch.setattr(original,'TangentCorrection',selected)
    with backend(profile='ball',fast_conversion=True),pytest.raises(Reached):
        original.reference(frozen(),piece=3,motion='world_x',anchor=0,world_profile=json.loads(Path(os.environ['GFFX_WORLD_PROFILE']).read_bytes()),correction_order=4,conditioned_correction=True,mixed_curve=True,transverse_correction=True,correction_axis=0,material_inputs=True,polynomial_inputs=True,line_order=6,crossing_support=True,root_parts=parts)


@pytest.mark.parametrize('parts',[True,2,8,None])
def test_front_root_partition_refuses_unqualified_profile(parts):
    from tools.opaque_original_motion_reference import reference
    with pytest.raises(ValueError,match='root'):reference({},piece=0,motion='world_x',root_parts=parts)



def test_front_full_samples_share_only_identical_raw_material_fields(monkeypatch):
    import os,json,hashlib,importlib.util
    from pathlib import Path
    import tools.opaque_original_motion_reference as current
    import tools.opaque_polynomial_inputs as polynomial
    from tools.opaque_peano import gauss5
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    path=Path(os.environ['GFFX_FROZEN_FRONT_SOURCE']);assert hashlib.sha256(path.read_bytes()).hexdigest()==os.environ['GFFX_FROZEN_FRONT_SHA']
    spec=importlib.util.spec_from_file_location('frozen_full_sample_reference',path);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    call=polynomial.PolynomialCosines.__call__;counts=[]
    def counted(self,*a,**kw):
        if kw.get('material_only',False):counts.append(1)
        return call(self,*a,**kw)
    monkeypatch.setattr(polynomial.PolynomialCosines,'__call__',counted)
    class Finished(Exception):pass
    def run(module):
        captures=[];values=[];factory=module.TangentCorrection
        def create(*a,**k):q=factory(*a,**k);captures.append(q);return q
        def probe(field,**kw):
            nodes,_=gauss5()
            for aa,bb,cc,dd in ((Q(1,2),Q(17,32),Q(1,2),Q(1)),(Q(17,32),Q(9,16),Q(1,2),Q(1))):
                box=(I(I(aa).lo,I(bb).hi),I(I(cc).lo,I(dd).hi));kw['prepare']((aa+bb)/2,(cc+dd)/2,box)
                points=[(Jet(I(aa)+I(bb-aa)*u),Jet(I(cc)+I(dd-cc)*v)) for u in nodes for v in nodes]
                for x,y in points:values.extend(field(x,y))
                for x,y in points:values.extend(captures[0].ordinary(x,y))
            raise Finished
        monkeypatch.setattr(module,'TangentCorrection',create);monkeypatch.setattr(module,'integrate_square',probe);counts.clear()
        with pytest.raises(Finished):module.reference(frozen(),piece=3,motion='world_x',anchor=0,world_profile=json.loads(Path(os.environ['GFFX_WORLD_PROFILE']).read_bytes()),correction_order=4,conditioned_correction=True,mixed_curve=True,transverse_correction=True,correction_axis=0,material_inputs=True,polynomial_inputs=True)
        return [{k:(v.lo,v.hi) for k,v in q.c.items()} for q in values],len(counts)
    with backend(profile='ball',fast_conversion=True):
        before,nold=run(old);after,nnew=run(current)
        # The centre node already hits the old preparation cache:24 new hits per box.
        assert after==before and nnew<=nold-48,(nold,nnew)


def test_material_seed_freezes_fields_and_rejects_conflicting_values():
    from tools.opaque_prepared_angle import MaterialInputCache
    cache=MaterialInputCache();x=Jet('.2');y=Jet('.3');values=[Jet(1),Jet(2),Jet(0),Jet(4)]
    def forbidden(*args):raise AssertionError('identical seeded tuple must not be recomputed')
    cache.seed('raw',forbidden,x,y,values);values[0].c[0,0]=I(9)
    first=cache('raw',forbidden,x,y);assert first[0][0,0].lo==1;first[0].c[0,0]=I(8)
    assert cache('raw',forbidden,x,y)[0][0,0].lo==1
    with pytest.raises(ValueError):cache.seed('raw',forbidden,x,y,values)


@pytest.mark.parametrize('degree',[1,2,3])
@pytest.mark.parametrize('axis',[0,1])
@pytest.mark.parametrize('reverse',[False,True])
def test_curved_sixth_lines_keep_closed_log_integrals(monkeypatch,degree,axis,reverse):
    import tools.opaque_tangent_peano as peano
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_tangent_peano import test_tangent_peano_contains_closed_piecewise_log_integral
    factory=peano.TangentCorrection
    def selected(*args,**kwargs):return factory(*args,**dict(kwargs,curved=True,curved_transverse=True,order=4,conditioned=True,centered=True,transverse=True,line_order=6,integrated_axis=1-axis))
    monkeypatch.setattr(peano,'TangentCorrection',selected)
    with backend(profile='ball',fast_conversion=True):test_tangent_peano_contains_closed_piecewise_log_integral(degree,axis,reverse)

@pytest.mark.parametrize('axis',[0,1])
@pytest.mark.parametrize('scale',[Q(1),-Q(1,2**60)])
def test_curved_sixth_lines_keep_nonlinear_change_of_variable(axis,scale):
    from decimal import Decimal as D,localcontext
    from tools.opaque_tangent_peano import TangentCorrection
    from tools.opaque_loss_extension import LossExtension
    from tools.opaque_arb_jets import backend
    with backend(profile='ball',fast_conversion=True):
        h=I(2).ln();curve=SimpleNamespace(cn=list(map(I,[Q(1,4),Q(1,2),1])),x=[I(Q(1,4)).ln(),I(Q(1,2)).ln(),I(0)],controls=[[[I(0)]*4]*2,[[I(0),h/3,2*h/3,h]]*2],mean=[I(1),I(1)],pi=I(1))
        e=LossExtension(curve);e.indices=[0,1]
        def inputs(x,y):
            z=x if axis==0 else y
            return z*z/2+z.lift(Q(1,4),tangent=scale),z.lift(1),z.lift(1),z
        box=(I('.5',1),I(0,1)) if axis==0 else (I(0,1),I('.5',1))
        proof=TangentCorrection(e,inputs,curved=True,curved_transverse=True,order=4,conditioned=True,centered=True,transverse=True,line_order=6,integrated_axis=1-axis)
        got=proof(box)
        with localcontext() as c:
            c.prec=90;ln2=D(2).ln();g=D('1.5').ln();value=(D('.75')*(g-1)+D('.5'))*ln2;derivative=g*ln2*D(scale.numerator)/D(scale.denominator)
            assert got[0].lo<=value<=got[0].hi and got[1].lo<=derivative<=got[1].hi
        assert not got[1].zero


@pytest.mark.parametrize('axis',[0,1])
@pytest.mark.parametrize('scale',[Q(1),-Q(1,2**60)])
@pytest.mark.parametrize('parts',[2,4])
def test_curved_partitioned_roots_keep_closed_nonlinear_integral(axis,scale,parts):
    from decimal import Decimal as D,localcontext
    from tools.opaque_tangent_peano import TangentCorrection
    from tools.opaque_loss_extension import LossExtension
    from tools.opaque_arb_jets import backend
    with backend(profile='ball',fast_conversion=True):
        h=I(2).ln();curve=SimpleNamespace(cn=list(map(I,[Q(1,4),Q(1,2),1])),x=[I(Q(1,4)).ln(),I(Q(1,2)).ln(),I(0)],controls=[[[I(0)]*4]*2,[[I(0),h/3,2*h/3,h]]*2],mean=[I(1),I(1)],pi=I(1))
        e=LossExtension(curve);e.indices=[0,1]
        def inputs(x,y):return (x+y*y)/4+x.lift(Q(1,4),tangent=scale),x.lift(1),x.lift(1),2*y
        proof=TangentCorrection(e,inputs,curved=True,curved_transverse=True,order=4,conditioned=True,centered=True,transverse=True,line_order=4,crossing_support=True,root_parts=parts,integrated_axis=axis)
        got=proof((I(0,1),I('.5',1)));assert proof.root_partition_queries>0
        with localcontext() as c:
            c.prec=90
            def F(q):return D(0) if q<=D('.5') else q*((2*q).ln()-1)+D('.5')
            def G(q):return D(0) if q<=D('.5') else q*q*((2*q).ln()/2-D('.75'))+q/2-D('.0625')
            value=16*(G(D('.75'))-G(D('.5625')))*D(2).ln()
            gradient=16*(F(D('.75'))-F(D('.5625')))*D(2).ln()*D(scale.numerator)/D(scale.denominator)
            assert got[0].lo<=value<=got[0].hi and got[1].lo<=gradient<=got[1].hi
        assert not got[1].zero


@pytest.mark.parametrize('same',[True,False])
@pytest.mark.parametrize('omega',[Q(1),-Q(1,2**60)])
def test_explicit_pure_flux_reuse_keeps_every_endpoint_and_tiny_term(same,omega):
    import json
    from tools.opaque_knot_transport import material_transport,transversal_flux
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_loss_extension import Curve
    with backend(profile='ball',fast_conversion=True):
        curve=Curve();curve.mean=[I(1),I(1)];curve.pi=I(3)
        def field(x,y):return [x/3+x.lift(Q(1,3),tangent=omega),y/4+y.lift(Q(1,3),tangent=-omega)]
        calls=[]
        def override(role,box,knots):calls.append(role);return transversal_flux(lambda x,y:field(x,y)[role],box,knots)
        albedo=[1,1,1] if same else [Q(1,2),Q(1,3),Q(1,4)]
        expected=material_transport(curve,field,(I(0,1),I(0,1)),albedo,[Q(1,25)]*3,flux_override=override);assert len(calls)==6;calls.clear()
        actual=material_transport(curve,field,(I(0,1),I(0,1)),albedo,[Q(1,25)]*3,flux_override=override,pure_flux_override=True)
        assert len(calls)==(2 if same else 6)
        assert json.dumps(actual,default=str,sort_keys=True)==json.dumps(expected,default=str,sort_keys=True)
        assert all(v.hi>0 for v in actual['bound'])

@pytest.mark.parametrize('bad',[None,0,1,'yes'])
def test_pure_flux_reuse_refuses_implicit_contract(bad):
    from tools.opaque_knot_transport import material_transport
    with pytest.raises(ValueError,match='pure'):material_transport(None,None,None,[],[],pure_flux_override=bad)
    with pytest.raises(ValueError,match='pure'):material_transport(None,None,None,[],[],pure_flux_override=True)


@pytest.mark.parametrize('method',['angular_material_cover','angular_piece_reference'])
def test_curved_root_profile_reaches_full_checker(monkeypatch,method):
    import tools.opaque_original_motion_reference as original
    from tests.packaging.test_opaque_motion_extension import PROFILE
    from tests.packaging.test_opaque_motion_tangent import frozen
    from tools.opaque_arb_jets import backend
    class Reached(Exception):pass
    def selected(*args,**kwargs):
        assert kwargs['crossing_support'] is True and kwargs['root_parts']==4
        raise Reached
    if method=='angular_material_cover':monkeypatch.setattr(original,'TangentCorrection',selected)
    else:monkeypatch.setattr(original,'angular_material_cover',selected)
    with backend(profile='ball',fast_conversion=True),pytest.raises(Reached):
        getattr(original,method)(frozen(),**dict(PROFILE,tensor_rule='gauss3x8-local4x4'),curved_transverse=True,polynomial_light=True,crossing_support=True,root_parts=4)

@pytest.mark.parametrize('kw',[dict(root_parts=True),dict(root_parts=2),dict(root_parts=4),dict(crossing_support=True),dict(crossing_support=1)])
def test_curved_root_profile_rejects_unqualified_selection(kw):
    from tools.opaque_original_motion_reference import angular_material_cover,angular_piece_reference
    for method in (angular_material_cover,angular_piece_reference):
        with pytest.raises(ValueError):method({},**kw)
