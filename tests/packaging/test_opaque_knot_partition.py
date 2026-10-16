"""Knot-targeted partition guidance and independent integral/transport controls."""
from fractions import Fraction as Q
import pytest
from tools.opaque_shading_intervals import I,Jet,Unresolved
from tools.opaque_square_reference import integrate_square

@pytest.fixture(autouse=True)
def qualified_region_backend():
    import os
    if os.environ.get('GFFX_KNOT_REGION_BACKEND')=='ball':
        from tools.opaque_arb_jets import backend
        with backend(profile='ball'):yield
    else:yield

@pytest.mark.parametrize('axis',[0,1])
@pytest.mark.parametrize('reverse',[False,True])
def test_knot_cut_targets_both_axis_orientations_with_bounded_work(axis,reverse):
    from tools.opaque_knot_partition import knot_cut
    calls=[];box=(Q(0),Q(1),Q(0),Q(1));k=Q(3,8)
    def field(x,y):
        calls.append((x,y));z=(x,y)[axis];return I(1-z if reverse else z)
    result=knot_cut([I(k)],field,box,axis=axis)
    assert result['axis']==axis and result['point']==(1-k if reverse else k)
    assert result['evaluations']==len(calls)<=14 and not result['root_certified']
    assert 0<result['point']<1

def test_knot_guidance_cannot_invent_root_or_swallow_stop():
    from tools.opaque_knot_partition import knot_cut
    assert knot_cut([I('.5')],lambda x,y:I(0),(Q(0),Q(1),Q(0),Q(1)),axis=0) is None
    def stop():raise Unresolved('test stop')
    with pytest.raises(Unresolved,match='test stop'):knot_cut([I('.5')],lambda x,y:I(x),(Q(0),Q(1),Q(0),Q(1)),axis=0,checkpoint=stop)

@pytest.mark.parametrize('axis',[0,1])
@pytest.mark.parametrize('k',[Q(3,8),Q(1,2**60)])
def test_exact_cut_integrates_full_measure_and_moving_jump(axis,k):
    state={};seen=[]
    def prep(x,y,box):
        v=box[axis]
        if v.lo<I(k).lo and v.hi>I(k).hi:raise ValueError('test angular knot')
        state['left']=v.hi<=I(k).hi
    def field(x,y):return [x.lift(2 if state['left'] else 5,tangent=1 if state['left'] else 2)]
    def split(box,reason):seen.append((box,reason));return axis,k
    def flux(box):
        v=box[axis];w=box[1-axis];owns=v.lo==I(k).lo and v.hi>I(k).hi
        return [-3*(I(w.hi)-I(w.lo)) if owns else I(0)]
    result=integrate_square(field,prepare=prep,domain_split=split,tangent_correction_integrator=flux,max_cells=3,max_depth=1)
    for value,exact in [(result['value'][0],5-3*k),(result['derivative'][0],-1-k)]:assert value.lo<=I(exact).lo<=I(exact).hi<=value.hi
    assert result['cells']==3 and result['max_depth']==1 and len(seen)==1
    assert sum((Q(b)-Q(a))*(Q(d)-Q(c)) for a,b,c,d in result['leaves'])==1
    assert any(Q(row[1])-Q(row[0])==k if axis==0 else Q(row[3])-Q(row[2])==k for row in result['leaves'])

@pytest.mark.parametrize('bad',[(2,Q(1,2)),(0,Q(0)),(0,Q(1)),(0,.5)])
def test_invalid_cut_is_refused_without_budget_or_acceptance_bypass(bad):
    def prep(*args):raise ValueError('domain')
    with pytest.raises(Unresolved,match='invalid domain split'):integrate_square(lambda x,y:[x],prepare=prep,domain_split=lambda *args:bad,max_cells=3,max_depth=1)

@pytest.mark.parametrize('cells,depth,reason',[(2,1,'cell budget'),(3,0,'depth budget')])
def test_valid_cut_still_obeys_existing_cell_and_depth_limits(cells,depth,reason):
    def prep(x,y,box):
        if box[0].lo<I('.375').lo and box[0].hi>I('.375').hi:raise ValueError('domain')
    with pytest.raises(Unresolved,match=reason):integrate_square(lambda x,y:[x.lift(1)],prepare=prep,domain_split=lambda *args:(0,Q(3,8)),max_cells=cells,max_depth=depth)


def test_saved_original_knot_cut_diagnostics(record_property):
    import json,os
    from pathlib import Path
    from tools.opaque_original_motion_reference import reference
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    boxes=[(Q(19,64),Q(39,128),Q(0),Q(1,4)),(Q(19,64),Q(39,128),Q(0),Q(1,2)),(Q(1,4),Q(1,2),Q(0),Q(1))]
    with backend(profile='ball'):result=reference(frozen(),piece=8,motion='normal_tilt',anchor=0,horizon_power=8,knot_targeted=True,cut_diagnostic_boxes=boxes)
    assert result['status']=='cut_diagnostic_only' and len(result['rows'])==3
    for row in result['rows']:
        a,b,c,d=map(Q,row['box']);assert len(row['children'])==2
        assert sum((Q(z['box'][1])-Q(z['box'][0]))*(Q(z['box'][3])-Q(z['box'][2])) for z in row['children'])==(b-a)*(d-c)
    path=Path(os.environ['GFFX_FULL_REFERENCE']);assert not path.exists();path.write_text(json.dumps(result,default=str,indent=2),encoding='utf8');record_property('saved_knot_cuts',path.read_text())


@pytest.mark.parametrize('axis',[0,1])
def test_cut_integrates_nonlinear_field_and_complete_moving_measure(axis):
    k=Q(3,8);state={}
    def prep(x,y,box):
        v=box[axis]
        if v.lo<I(k).lo and v.hi>I(k).hi:raise ValueError('test knot')
        state['left']=v.hi<=I(k).hi
    def field(x,y):
        u,v=(x,y) if axis==0 else (y,x)
        height=u.lift(2 if state['left'] else 5,tangent=1 if state['left'] else 2)
        return [height*u**2*v**3*u.lift(1,tangent=Q(1,3))]
    def flux(box):
        u,v=box[axis],box[1-axis]
        return [-3*I(k*k)*(I(v.hi)**4-I(v.lo)**4)/4 if u.lo==I(k).lo and u.hi>I(k).hi else I(0)]
    result=integrate_square(field,prepare=prep,domain_split=lambda *args:(axis,k),tangent_correction_integrator=flux,max_cells=3,max_depth=1)
    value=(5-3*k**3)/12;derivative=(2-k**3)/12+value/3-3*k*k/4
    for got,expected in zip([result['value'][0],result['derivative'][0]],[value,derivative]):assert got.lo<=I(expected).lo<=I(expected).hi<=got.hi
    assert result['cells']==3

def test_original_knot_profile_requires_domain_qualification_before_trial():
    from tools.opaque_original_motion_reference import reference
    with pytest.raises(ValueError,match='qualified domain cover'):reference({},piece=8,motion='normal_tilt',knot_targeted=True)


@pytest.mark.parametrize('sign',[-1,1])
def test_curved_knot_root_and_motion_match_closed_implicit_derivatives(sign):
    from tools.opaque_knot_partition import knot_root
    def field(u,s):return sign*(u*u+s)+u.lift(0,tangent=1)
    result=knot_root(field,Q(1,4),Q(3,4),Q(1,4),I(sign*Q(1,2)))
    for value,exact in [(result['root'],Q(1,2)),(result['parameter_derivative'],-sign),(result['transverse_derivative'],-1)]:assert value.lo<=I(exact).lo<=I(exact).hi<=value.hi
    assert result['unique'] and result['evaluations']<=20

def test_curved_knot_root_refuses_unproved_monotonicity_and_keeps_tiny_root():
    from tools.opaque_knot_partition import knot_root
    with pytest.raises(ValueError,match='monotone'):knot_root(lambda u,s:(u-Q(1,2))**2,Q(0),Q(1),Q(0),I(Q(1,8)))
    result=knot_root(lambda u,s:u+u.lift(0,tangent=Q(1,10**70)),Q(0),Q(1),Q(0),I(Q(1,2**60)))
    assert result['root'].lo<=I(Q(1,2**60)).lo<=result['root'].hi and result['root'].hi>0
    assert result['parameter_derivative'].hi<0


def test_saved_original_knot_contour_slices(record_property):
    import json,os,hashlib,time
    from pathlib import Path
    from tools.opaque_original_motion_reference import prepare,mapped_inputs
    from tools.opaque_principled_reference import PreparedPrincipled
    from tools.opaque_knot_partition import knot_root
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    path=Path(os.environ['GFFX_FULL_REFERENCE']);saved=path.parent/'knot-domain1-reference.json';failure=json.loads(saved.read_bytes());a,b,c,d=map(Q,failure['details']['box']);started=time.monotonic()
    def check():assert time.monotonic()-started<90
    with backend(profile='ball'):
        data=frozen();p=prepare(data,piece=8,motion='normal_tilt',anchor=0,horizon_power=4,checkpoint=check);curve=PreparedPrincipled(data['budget'],1.);field=lambda u,s:mapped_inputs(p,u,s)[0]
        centre=field(Jet((a+b)/2),Jet((c+d)/2))[0,0];target=min(curve.cn,key=lambda k:abs((k.lo+k.hi)-(centre.lo+centre.hi)))
        lo=max(Q(0),a-(b-a));hi=min(Q(1),b+(b-a));rows=[]
        for s0 in [c,(c+d)/2,d]:
            check()
            try:row=knot_root(field,lo,hi,s0,target,checkpoint=check)
            except ValueError as error:row=dict(status='refused',reason=str(error))
            rows.append(dict(s=str(s0),**row))
        qualified=[z for z in rows if z.get('unique')];assert len(qualified)==3
        assert qualified[0]['root'].hi<qualified[2]['root'].lo or qualified[2]['root'].hi<qualified[0]['root'].lo
        result=dict(status='slice_diagnostic_only',failure_sha256=hashlib.sha256(saved.read_bytes()).hexdigest(),failed_box=list(map(str,(a,b,c,d))),expanded_radial_span=list(map(str,(lo,hi))),knot=target,rows=rows,seconds=time.monotonic()-started,scope='Three certified fixed-s slices and implicit first derivatives; no complete curved strip, high-order chart or integral qualification.')
    assert not path.exists();path.write_text(json.dumps(result,default=str,indent=2),encoding='utf8');record_property('original_knot_contour',path.read_text())


@pytest.mark.parametrize('sign',[-1,1])
def test_knot_region_high_mixed_derivatives_match_closed_sqrt(sign):
    from tools.opaque_knot_partition import knot_region
    def field(u,s):return sign*(u*u+s)+u.lift(0,tangent=1)
    s=Jet.variable(Q(1,4),order=10)
    result=knot_region(field,Q(1,4),Q(3,4),s,I(sign*Q(1,2)))
    root=result['root'];coefficient=Q(1,2)
    for i in range(11):
        if i:coefficient*=Q(3-2*i,2*i)*(-4)
        assert root[i,0].lo<=I(coefficient).lo<=I(coefficient).hi<=root[i,0].hi
        derivative=coefficient*(Q(1,2)-i)*(-4)*sign
        assert root[i,1].lo<=I(derivative).lo<=I(derivative).hi<=root[i,1].hi
    assert result['unique'] and result['evaluations']<=34


def test_knot_region_uniform_bracket_and_boundary_refusals():
    from tools.opaque_knot_partition import knot_region
    field=lambda u,s:u*u+s
    result=knot_region(field,Q(1,2),Q(7,8),Jet(I('0','.125'),order=1),I('.5'))
    assert result['unique'] and result['root'][0,0].lo>I('.5').hi and result['root'][0,0].hi<I('.875').lo
    # A midpoint root exists, but at the transverse endpoint it touches/exits.
    with pytest.raises(ValueError,match='uniformly bracketed'):
        knot_region(field,Q(1,2),Q(7,8),Jet(I('0','.25')),I('.5'))
    with pytest.raises(ValueError,match='monotone'):
        knot_region(lambda u,s:(u-Q(1,2))**2,Q(0),Q(1),Jet(0),I('.125'))
    def stop():raise Unresolved('region stop')
    with pytest.raises(Unresolved,match='region stop'):
        knot_region(field,Q(1,2),Q(7,8),Jet(0),I('.5'),checkpoint=stop)


@pytest.mark.parametrize('tiny',[False,True])
def test_knot_region_full_integrals_measure_and_shared_boundary_flux(tiny):
    from tools.opaque_knot_partition import knot_region,knot_strip
    scale=Q(1,2**60) if tiny else Q(1)
    def contour(u,s):return u-scale*(Q(1,4)+s*s/4+s.lift(0,tangent=1)/16*(1+s))
    def field(t,s):
        result=knot_region(contour,Q(0),scale,s,I(0))
        root=result['root'];left,jl=knot_strip(t,s.lift(0),root);right,jr=knot_strip(t,root,s.lift(scale))
        # Polynomial color, unequal sides, plus changing physical area.
        measure=s.lift(1,tangent=Q(1,3))
        return [(2*left*jl+5*right*jr)*measure,(jl+jr)*measure]
    result=integrate_square(field,max_cells=1,max_depth=0)
    # Integral r(s)^2 ds = scale^2 * 7/60; r*r_theta integral = scale^2*25/768.
    expected=scale*scale*Q(93,40)
    deriv=expected/3-scale*scale*Q(25,256)
    for value,exact in [(result['value'][0],expected),(result['derivative'][0],deriv),(result['value'][1],scale),(result['derivative'][1],scale/3)]:
        assert value.lo<=I(exact).lo<=I(exact).hi<=value.hi
    assert result['value'][1].lo>0 and result['cells']==1
    with pytest.raises(ValueError,match='positive strip'):
        knot_strip(Jet(0),Jet(1),Jet(0))



def test_saved_original_whole_knot_region(record_property):
    import json,os,time,hashlib
    from pathlib import Path
    from tools.opaque_original_motion_reference import prepare,mapped_inputs
    from tools.opaque_principled_reference import PreparedPrincipled
    from tools.opaque_knot_partition import knot_region,knot_strip
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    path=Path(os.environ['GFFX_FULL_REFERENCE']);previous=path.parent.parent/'opaque132-20260925'
    saved=previous/'knot-domain1-reference.json';assert hashlib.sha256(saved.read_bytes()).hexdigest()=='64064be2278c40f9b6cf3595f4f6e838e68eb99944afc44a956412f3ba05b545'
    a,b,c,d=map(Q,json.loads(saved.read_bytes())['details']['box']);lo=max(Q(0),a-(b-a));hi=min(Q(1),b+(b-a))
    started=time.monotonic();evaluations=0;cells=0;rows=[];refusals=[]
    def check():
        if time.monotonic()-started>=90:raise Unresolved('region deadline',cells=cells,evaluations=evaluations)
    with backend(profile='ball'):
        data=frozen();p=prepare(data,piece=8,motion='normal_tilt',anchor=0,horizon_power=4,checkpoint=check)
        def field(u,s):
            nonlocal evaluations
            check();evaluations+=1;return mapped_inputs(p,u,s)[0]
        curve=PreparedPrincipled(data['budget'],1.);centre=field(Jet((a+b)/2),Jet((c+d)/2))[0,0]
        target=min(curve.cn,key=lambda k:abs((k.lo+k.hi)-(centre.lo+centre.hi)))
        pending=[(c,d,0)];failure=None
        try:
            while pending:
                check()
                if cells>=256:raise Unresolved('region cell budget',cells=cells)
                left,right,depth=pending.pop();cells+=1
                try:
                    result=knot_region(field,lo,hi,Jet.variable(I(I(left).lo,I(right).hi),order=10),target,checkpoint=check)
                    root=result['root'];u,jac=knot_strip(Jet(Q(1,2),order=10),Jet(lo,order=10),root)
                    assert jac[0,0].lo>0
                    rows.append(dict(s=[str(left),str(right)],depth=depth,root=[[i,j,str(root[i,j].lo),str(root[i,j].hi)] for i in range(11) for j in range(2)],radial_derivative=str(result['radial_derivative']),evaluations=result['evaluations']))
                except ValueError as error:
                    if isinstance(error,Unresolved):raise
                    refusals.append(dict(s=[str(left),str(right)],depth=depth,reason=str(error)))
                    if depth==10:raise Unresolved('region depth budget',cells=cells,last=refusals[-1]) from error
                    mid=(left+right)/2;pending.extend([(mid,right,depth+1),(left,mid,depth+1)])
        except Unresolved as error:failure=error.details
        complete=not pending and failure is None
        if complete:
            ordered=sorted(rows,key=lambda z:Q(z['s'][0]));assert Q(ordered[0]['s'][0])==c and Q(ordered[-1]['s'][1])==d
            assert all(x['s'][1]==y['s'][0] for x,y in zip(ordered,ordered[1:]))
        result=dict(status='whole_region_certified' if complete else 'unresolved',complete_cover=complete,failed_box=list(map(str,(a,b,c,d))),radial_span=list(map(str,(lo,hi))),knot=str(target),cells=cells,evaluations=evaluations,seconds=time.monotonic()-started,rows=rows,refusals=refusals,failure=failure,scope='Only expanded radial span across the saved failed box transverse interval. Uniform root existence/uniqueness and order10 spatial plus first-parameter mixed jets; not whole piece angular-domain cover, shading integral or accuracy acceptance.')
    assert not path.exists();path.write_text(json.dumps(result,indent=2),encoding='utf8');record_property('whole_knot_region',path.read_text())
    assert complete,result



def test_knot_region_varying_physical_jacobian_and_jump_integral():
    from tools.opaque_knot_partition import knot_region,knot_strip
    def contour(u,s):return u-(Q(1,4)+s*s/4+s.lift(0,tangent=1)/16*(1+s))
    def field(t,s):
        root=knot_region(contour,Q(0),Q(1),s,I(0))['root']
        value=t.lift(0)
        for low,high,color in [(s.lift(0),root,2),(root,s.lift(1),5)]:
            u,jac=knot_strip(t,low,high)
            physical=1+u*s+s.lift(0,tangent=1)*(u*u+s)
            value=value+color*u*physical*jac
        return [value]
    result=integrate_square(field,max_cells=1,max_depth=0)
    # Closed antiderivatives: 5/2+5*s/3-3*r^2/2-s*r^3;
    # derivative 5/4+5*s/2-3*r^4/4-3*s*r^2/2-3*r*(1+r*s)*r_theta.
    for value,exact in [(result['value'][0],Q(24031,7680)),(result['derivative'][0],Q(121319,53760))]:
        assert value.lo<=I(exact).lo<=I(exact).hi<=value.hi
    assert result['cells']==1



@pytest.mark.parametrize('reverse',[False,True])
def test_knot_band_proves_width_when_root_ranges_overlap(reverse):
    from tools.opaque_knot_partition import knot_band
    sign=-1 if reverse else 1
    def angle(u,s):return sign*(u-s*s/8-s.lift(0,tangent=Q(1,16)))
    low,high=(I('-0.265625'),I('-.25')) if reverse else (I('.25'),I('.265625'))
    result=knot_band(angle,Q(0),Q(1),Jet(I(0,1),order=10),Jet.variable(I(0,1),order=10),low,high)
    assert result['jacobian'][0,0].lo==result['jacobian'][0,0].hi==I(Q(1,64)).lo
    assert result['coordinate'][0,1].lo<=I(Q(1,16)).lo<=result['coordinate'][0,1].hi
    assert result['angle_range'].lo==low.lo and result['angle_range'].hi==high.hi
    with pytest.raises(ValueError,match='ordered exact knots'):
        knot_band(angle,0,1,Jet(0),Jet(0),high,low)


@pytest.mark.parametrize('profile',['polynomial','logarithmic','rational'])
def test_knot_band_nonlinear_integral_preserves_parameter_conservation(profile):
    from tools.opaque_knot_partition import knot_band
    def angle(u,s):return u*u+s/16+s.lift(0,tangent=Q(1,32))
    def field(t,s):
        band=knot_band(angle,Q(1,4),Q(7,8),t,s,I('.25'),I('.5'))
        u=band['coordinate'];q=angle(u,s);q.c[0,0]=I(max(q[0,0].lo,band['angle_range'].lo),min(q[0,0].hi,band['angle_range'].hi))
        value=q*q if profile=='polynomial' else q.ln() if profile=='logarithmic' else (1+q).reciprocal()
        return [value*2*u*band['jacobian']*(1+s)*s.lift(1,tangent=Q(1,5))]
    result=integrate_square(field,max_cells=31,max_depth=4)
    exact=I(Q(7,128)) if profile=='polynomial' else I(Q(3,2))*(I('.5')*I('.5').ln()-I('.25')*I('.25').ln()-I('.25')) if profile=='logarithmic' else I(Q(3,2))*(I('1.5').ln()-I('1.25').ln())
    for value,exact in [(result['value'][0],exact),(result['derivative'][0],exact/5)]:
        assert value.lo<=I(exact).lo<=I(exact).hi<=value.hi



def test_original_interior_angular_band(record_property):
    import os,json,hashlib
    from pathlib import Path
    from tools.opaque_original_motion_reference import interior_band_reference
    from tools.opaque_principled_reference import PreparedPrincipled
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    path=Path(os.environ['GFFX_FULL_REFERENCE']);prior=path.parent.parent/'opaque133-20260926/original-region1-reference.json'
    assert hashlib.sha256(prior.read_bytes()).hexdigest()=='e85f38e4f52df55b81ee8d7721bb056d94df1bd9e158e15f06fcbd20448201f2'
    saved=json.loads(prior.read_bytes());a,b=map(Q,saved['radial_span']);c,d=map(Q,saved['failed_box'][2:]);data=frozen();failure=None
    with backend(profile='ball'):
        curve=PreparedPrincipled(data['budget'],1.);index=next(i for i,k in enumerate(curve.cn) if str(k)==saved['knot'])
        try:result=interior_band_reference(data,box=(a,b,c,d),knot_index=index,domain_only=os.environ.get('GFFX_BAND_INTEGRATE')!='1')
        except Unresolved as error:failure=error.details;result=dict(status='unresolved',failure=failure)
    assert not path.exists();path.write_text(json.dumps(result,default=str,indent=2),encoding='utf8');record_property('original_angular_band',path.read_text())
    assert failure is None,result



def test_curved_knot_cover_cannot_enable_unqualified_accuracy_trial():
    from tools.opaque_original_motion_reference import reference
    with pytest.raises(ValueError,match='domain-preflight only'):
        reference({},piece=8,motion='normal_tilt',curved_knot=True)


def test_original_curved_knot_domain_cover(record_property):
    import os,json
    from pathlib import Path
    from tools.opaque_original_motion_reference import reference
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    path=Path(os.environ['GFFX_FULL_REFERENCE']);failure=None
    with backend(profile='ball'):
        try:result=reference(frozen(),piece=8,motion='normal_tilt',anchor=0,horizon_power=4,knot_targeted=True,domain_only=True,curved_knot=True)
        except Unresolved as error:failure=error.details;result=dict(status='unresolved',failure=failure)
    assert not path.exists();path.write_text(json.dumps(result,default=str,indent=2),encoding='utf8');record_property('curved_domain_cover',path.read_text())
    assert failure is None,result



@pytest.mark.parametrize('reverse',[False,True])
def test_uniform_knot_endpoints_use_proved_transverse_monotonicity(reverse):
    from tools.opaque_knot_partition import knot_region
    sign=-1 if reverse else 1
    def field(u,s):return sign*(u+s/(1+s))+u.lift(0,tangent=Q(1,16))
    args=(field,Q(15,100),Q(31,100),Jet.variable(I('.25','.5'),order=2),I(sign*Q(1,2)))
    with pytest.raises(ValueError,match='uniformly bracketed'):knot_region(*args)
    result=knot_region(*args,tight_endpoints=True)
    assert result['unique'] and result['interior']
    assert result['root'][0,0].lo<=I(Q(1,6)).lo and result['root'][0,0].hi>=I(Q(3,10)).hi
    assert result['root'][0,1].lo<=I(-sign*Q(1,16)).lo<=result['root'][0,1].hi
    assert result['endpoint_evaluations']>0


def test_uniform_knot_endpoint_refinement_cannot_hide_interior_crossing():
    from tools.opaque_knot_partition import knot_region
    # Equal endpoint values alone hide the minimum at s=1/2.
    with pytest.raises(ValueError,match='uniformly bracketed'):
        knot_region(lambda u,s:u+(s-Q(1,2))**2,Q(1,4),Q(3,4),Jet(I(0,1)),I('.3'),tight_endpoints=True)
    with pytest.raises(ValueError,match='explicit endpoint'):
        knot_region(lambda u,s:u,0,1,Jet(0),I('.5'),tight_endpoints='yes')



def test_saved_boundary_parent_targets(record_property):
    import json,os,time,hashlib
    from pathlib import Path
    from tools.opaque_original_motion_reference import prepare,mapped_inputs
    from tools.opaque_loss_extension import bounded_cosines
    from tools.opaque_principled_reference import PreparedPrincipled
    from tools.opaque_knot_partition import knot_region
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    path=Path(os.environ['GFFX_FULL_REFERENCE']);saved=path.parent/'curved-cover1-reference.json';parent=json.loads(saved.read_bytes())['failure']['guidance']['last_split_parent'];a,b,c,d=map(Q,parent['box']);start=time.monotonic();calls=0
    def check():
        if time.monotonic()-start>=90:raise Unresolved('parent diagnostic deadline')
    with backend(profile='ball'):
        data=frozen();p=prepare(data,piece=8,motion='normal_tilt',anchor=0,horizon_power=4);curve=PreparedPrincipled(data['budget'],1.)
        def field(u,s):
            nonlocal calls
            check();calls+=1;return mapped_inputs(p,u,s)[0]
        value=bounded_cosines(lambda u,s:[field(u,s)],I(I(a).lo,I(b).hi),I(I(c).lo,I(d).hi),monotone=True,partial=True)[0]
        targets=[(i,k) for i,k in enumerate(curve.cn) if value.lo<k.lo and k.hi<value.hi];assert len(targets)<=16;rows=[]
        for i,k in targets:
            try:
                root=knot_region(field,a,b,Jet.variable(I(I(c).lo,I(d).hi),order=1),k,tight_endpoints=True,checkpoint=check)
                row=dict(index=i,status='interior',root=str(root['root'][0,0]))
            except ValueError as error:
                if isinstance(error,Unresolved):raise
                row=dict(index=i,status='refused',reason=str(error))
            rows.append(row)
        result=dict(parent=parent,rows=rows,evaluations=calls,seconds=time.monotonic()-start,saved_sha256=hashlib.sha256(saved.read_bytes()).hexdigest(),scope='Bounded diagnosis of all interior-knot targets on one saved parent; no integral admission.')
    assert not path.exists();path.write_text(json.dumps(result,indent=2),encoding='utf8');record_property('boundary_parent_targets',path.read_text())
    assert rows



def test_knot_edge_crossing_locations_and_parameter_motion():
    from tools.opaque_knot_partition import knot_edge_events
    def field(u,s):return u-s*s+u.lift(0,tangent=Q(1,8))
    result=knot_edge_events(field,(Q(1,4),Q(9,16),Q(1,4),Q(7,8)),I(0))
    assert len(result['events'])==2
    for event,root,motion in zip(result['events'],[Q(1,2),Q(3,4)],[Q(1,8),Q(1,12)]):
        for value,exact in [(event['location'],root),(event['parameter_derivative'],motion)]:assert value.lo<=I(exact).lo<=I(exact).hi<=value.hi
    with pytest.raises(ValueError,match='monotone'):
        knot_edge_events(lambda u,s:u-(s-Q(1,2))**2,(Q(1,4),Q(3,4),Q(0),Q(1)),I(0))
    with pytest.raises(ValueError,match='edge endpoint'):
        knot_edge_events(field,(Q(1,4),Q(9,16),Q(1,2),Q(7,8)),I(0))



def test_saved_original_knot_edge_events(record_property):
    import json,os,time
    from pathlib import Path
    from tools.opaque_original_motion_reference import prepare,mapped_inputs
    from tools.opaque_principled_reference import PreparedPrincipled
    from tools.opaque_knot_partition import knot_edge_events
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    path=Path(os.environ['GFFX_FULL_REFERENCE']);saved=json.loads((path.parent/'curved-cover1-reference.json').read_bytes());box=tuple(map(Q,saved['failure']['guidance']['last_split_parent']['box']));start=time.monotonic()
    def check():
        if time.monotonic()-start>=90:raise Unresolved('edge diagnostic deadline')
    with backend(profile='ball'):
        data=frozen();p=prepare(data,piece=8,motion='normal_tilt',anchor=0,horizon_power=4);curve=PreparedPrincipled(data['budget'],1.)
        result=knot_edge_events(lambda u,s:mapped_inputs(p,u,s)[0],box,curve.cn[70],checkpoint=check)
        result.update(box=list(map(str,box)),knot_index=70,seconds=time.monotonic()-start)
    assert not path.exists();path.write_text(json.dumps(result,default=str,indent=2),encoding='utf8');record_property('original_knot_edge_events',path.read_text())
    assert result['events']



@pytest.mark.parametrize('scale',[Q(1),Q(1,2**60)])
def test_entry_chart_full_cover_and_moving_flux_have_closed_integrals(scale):
    from tools.opaque_knot_partition import KnotEntryChart
    def angle(u,s):return u-scale*s*s+u.lift(0,tangent=scale/8)
    chart=KnotEntryChart(angle,(scale/4,3*scale/4,Q(1,4),Q(3,4)),I(0))
    def field(t,y):
        area=t.lift(0);color=t.lift(0);first=t.lift(0)
        for row in chart.map(t,y):
            u,s,jac=row['u'],row['s'],row['jacobian'];measure=t.lift(1,tangent=Q(1,3))
            area=area+jac*measure;color=color+(2 if row['side']=='below' else 5)*jac*measure;first=first+u*jac*measure
        return [area,color,first]
    result=integrate_square(field,max_cells=1,max_depth=0)
    values=[scale/4,scale*Q(73,64),scale*scale/8]
    tangents=[values[0]/3,values[1]/3+scale*Q(3,32),values[2]/3]
    for rows,expected in [(result['value'],values),(result['derivative'],tangents)]:
        for value,exact in zip(rows,expected):assert value.lo<=I(exact).lo<=I(exact).hi<=value.hi
    assert result['value'][0].lo>0 and result['cells']==1
    mapped=chart.map(Jet(I(0,1),order=10),Jet.variable(I(0,1),order=10))
    assert len(mapped)==3 and all(row['jacobian'][0,0].lo>=0 for row in mapped)
    # The wedge is allowed to collapse only at its certified entry boundary.
    assert chart.map(Jet(Q(1,2)),Jet(0))[1]['jacobian'][0,0].lo<=0


def test_entry_chart_refuses_unsupported_topology_and_outside_chart():
    from tools.opaque_knot_partition import KnotEntryChart
    with pytest.raises(ValueError,match='single lower entry'):
        KnotEntryChart(lambda u,s:u-s*s,(Q(1,4),Q(9,16),Q(1,4),Q(7,8)),I(0))
    chart=KnotEntryChart(lambda u,s:u-s*s,(Q(1,4),Q(3,4),Q(1,4),Q(3,4)),I(0))
    with pytest.raises(ValueError,match='unit chart'):
        chart.map(Jet(2),Jet(0))



def test_entry_chart_nonlinear_inverse_root_integral():
    from tools.opaque_knot_partition import KnotEntryChart
    chart=KnotEntryChart(lambda u,s:u*u-s+u.lift(0,tangent=Q(1,8)),(Q(1,4),Q(3,4),Q(0),Q(1,4)),I(0))
    def field(t,y):
        rows=chart.map(t,y)
        return [sum(((2 if r['side']=='below' else 5)*r['jacobian'] for r in rows),t.lift(0))]
    result=integrate_square(field,max_cells=63,max_depth=6)
    for value,exact in [(result['value'][0],Q(35,64)),(result['derivative'][0],Q(3,32))]:
        assert value.lo<=I(exact).lo<=I(exact).hi<=value.hi


def test_saved_original_entry_chart(record_property):
    import os,json,time,hashlib
    from pathlib import Path
    from tools.opaque_original_motion_reference import prepare,mapped_inputs
    from tools.opaque_loss_extension import bounded_cosines
    from tools.opaque_principled_reference import PreparedPrincipled
    from tools.opaque_knot_partition import KnotEntryChart
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    path=Path(os.environ['GFFX_FULL_REFERENCE']);saved=path.parent.parent/'opaque135-20260928/original-edge1-reference.json';evidence=json.loads(saved.read_bytes());box=tuple(map(Q,evidence['box']));a,b,c,d=box;start=time.monotonic()
    def check():
        if time.monotonic()-start>=90:raise Unresolved('entry map diagnostic deadline')
    with backend(profile='ball'):
        data=frozen();p=prepare(data,piece=8,motion='normal_tilt',anchor=0,horizon_power=4);curve=PreparedPrincipled(data['budget'],1.);knot=curve.cn[evidence['knot_index']]
        chart=KnotEntryChart(lambda u,s:mapped_inputs(p,u,s)[0],box,knot,checkpoint=check)
        bounds=bounded_cosines(lambda u,s:list(mapped_inputs(p,u,s)[:2]),I(I(a).lo,I(b).hi),I(I(c).lo,I(d).hi),monotone=True,partial=True);rows=[]
        count=lambda v:int(v.lo<=curve.cn[0].hi)+sum(v.lo<=curve.cn[i+1].hi and v.hi>=curve.cn[i].lo for i in range(len(curve.cn)-1))
        for axis in (0,1):
            t=Jet.variable(I(0,1),order=10) if axis==0 else Jet(I(0,1),order=10)
            y=Jet.variable(I(0,1),order=10) if axis==1 else Jet(I(0,1),order=10)
            for j,row in enumerate(chart.map(t,y)):
                check();nv,nl,vh,jac=mapped_inputs(p,row['u'],row['s']);physical=jac*row['jacobian']
                angle=I(bounds[0].lo,min(bounds[0].hi,knot.hi)) if row['side']=='below' else I(max(bounds[0].lo,knot.lo),bounds[0].hi)
                rows.append(dict(axis=axis,chart=j,side=row['side'],counts=[count(angle),count(bounds[1])],jacobian=str(row['jacobian'][0,0]),physical_jacobian=str(physical[0,0]),physical_parameter=str(physical[0,1]),spatial10=str(physical[10,0]),mixed10=str(physical[10,1])))
        result=dict(status='local_entry_chart_qualified',box=list(map(str,box)),event=chart.certificate,rows=rows,evaluations=chart.evaluations,seconds=time.monotonic()-start,previous_sha256=hashlib.sha256(saved.read_bytes()).hexdigest(),scope='Three-chart complete cover of one saved parent and high/mixed physical-map enclosures; not complete original-piece shading accuracy.')
    assert not path.exists();path.write_text(json.dumps(result,default=str,indent=2),encoding='utf8');record_property('original_entry_chart',path.read_text())
    assert all(max(row['counts'])<=8 for row in rows)



def test_entry_cover_cannot_enable_unqualified_accuracy():
    from tools.opaque_original_motion_reference import reference
    with pytest.raises(ValueError,match='curved-domain-preflight only'):
        reference({},piece=8,motion='normal_tilt',curved_knot=True,entry_knot=True)


def test_original_entry_following_domain_cover(record_property):
    import json,os
    from pathlib import Path
    from tools.opaque_original_motion_reference import reference
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    path=Path(os.environ['GFFX_FULL_REFERENCE']);failure=None
    with backend(profile='ball'):
        try:result=reference(frozen(),piece=8,motion='normal_tilt',anchor=0,horizon_power=4,knot_targeted=True,domain_only=True,curved_knot=True,entry_knot=True)
        except Unresolved as error:failure=error.details;result=dict(status='unresolved',failure=failure)
    assert not path.exists();path.write_text(json.dumps(result,default=str,indent=2),encoding='utf8');record_property('entry_domain_cover',path.read_text())
    assert failure is None,result
    assert result['cells']<=256 and all(z['depth']<=10 and max(z['counts'])<=8 for z in result['leaves'])



def test_original_angular_cover_cost_audit(record_property):
    import os,json,time
    from pathlib import Path
    from tools.opaque_original_motion_reference import prepare,mapped_inputs
    from tools.opaque_principled_reference import PreparedPrincipled
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    path=Path(os.environ['GFFX_FULL_REFERENCE']);start=time.monotonic();rows=[]
    def check():
        if time.monotonic()-start>=90:raise Unresolved('angular cost audit deadline')
    with backend(profile='ball'):
        data=frozen();p=prepare(data,piece=8,motion='normal_tilt',anchor=0,horizon_power=4,checkpoint=check);curve=PreparedPrincipled(data['budget'],1.)
        for c,d in [(Q(0),Q(1))]+[(Q(i,4),Q(i+1,4)) for i in range(4)]:
            row=dict(transverse=[str(c),str(d)])
            try:
                q=mapped_inputs(p,Jet.variable(I(0,1),order=1),Jet(I(I(c).lo,I(d).hi),order=1),flatten=False)[0]
                row.update(radial_slope=str(q[1,0]),monotone=q[1,0].lo>0,range=str(q[0,0]))
            except ValueError as e:row['refusal']=str(e)
            check();ends=[mapped_inputs(p,Jet(1),Jet(z),flatten=False)[0][0,0] for z in (c,d)]
            row['outer_edge']=[str(v) for v in ends];row['indices']=[sum(k.lo<=v.hi for k in curve.cn) for v in ends];rows.append(row)
        result=dict(status='cost_audit_only',rows=rows,knots=len(curve.cn),seconds=time.monotonic()-start,scope='Whole-radial monotonicity and boundary angular ranges; no coverage or integration acceptance.')
    assert not path.exists();path.write_text(json.dumps(result,indent=2),encoding='utf8');record_property('angular_cover_cost_audit',path.read_text())
    assert len(rows)==5



@pytest.mark.parametrize('scale',[Q(1),Q(1,2**60)])
def test_monotone_cover_closed_measure_jump_and_transport(scale):
    from tools.opaque_knot_partition import MonotoneAngularCover
    cover=MonotoneAngularCover(lambda u,s:scale*(u/(1+s)+u*u.lift(0,tangent=Q(1,8))),[I(scale/4),I(3*scale/4)])
    assert len(cover.charts)==4 and cover.cells==7 and cover.depth<=3
    def field(t,y):
        out=[t.lift(0) for _ in range(3)]
        for i,chart in enumerate(cover.charts):
            row=cover.map(i,t,y);u,jac=row['u'],row['jacobian'];jac=jac*t.lift(scale,tangent=scale/3)
            for j,v in enumerate((1,[2,5,7][chart['band']],u)):out[j]=out[j]+v*jac
        return out
    result=integrate_square(field,max_cells=31,max_depth=4)
    values=[scale,scale*Q(95,24),scale/2];tangents=[scale/3,scale*(Q(95,72)+Q(263,864)),scale/6]
    for rows,exact in [(result['value'],values),(result['derivative'],tangents)]:
        for v,x in zip(rows,exact):assert v.lo<=I(x).lo<=I(x).hi<=v.hi
    assert result['value'][0].lo>0
    for axis in (0,1):
        t=Jet.variable(I(0,1),order=10) if axis==0 else Jet(I(0,1),order=10)
        y=Jet.variable(I(0,1),order=10) if axis==1 else Jet(I(0,1),order=10)
        for i in range(len(cover.charts)):assert cover.map(i,t,y)['jacobian'][0,0].lo>=0


def test_monotone_cover_two_exits_and_closed_nonlinear_measure():
    from tools.opaque_knot_partition import MonotoneAngularCover
    cover=MonotoneAngularCover(lambda u,s:(u+u*u)/(1+s)+u*u.lift(0,tangent=Q(1,8)),[I(Q(3,2)),I(Q(7,4))])
    assert len(cover.charts)==5 and len(cover.events)==2
    def field(t,y):
        rows=[cover.map(i,t,y) for i in range(len(cover.charts))]
        return [sum((r['jacobian'] for r in rows),t.lift(0)),sum((r['u']*r['jacobian'] for r in rows),t.lift(0))]
    result=integrate_square(field,max_cells=127,max_depth=7)
    for v,x in zip(result['value'],[Q(1),Q(1,2)]):assert v.lo<=I(x).lo<=I(x).hi<=v.hi
    for v in result['derivative']:assert v.lo<=0<=v.hi


def test_monotone_cover_refuses_invalid_topology_budget_and_chart():
    from tools.opaque_knot_partition import MonotoneAngularCover
    with pytest.raises(ValueError,match='radial'):
        MonotoneAngularCover(lambda u,s:u*u/(1+s),[I(Q(1,4))])
    with pytest.raises(ValueError,match='horizon'):
        MonotoneAngularCover(lambda u,s:(u+Q(1,8))/(1+s),[I(Q(1,4))])
    with pytest.raises(ValueError,match='ordered'):
        MonotoneAngularCover(lambda u,s:u/(1+s),[I(Q(3,4)),I(Q(1,4))])
    with pytest.raises(ValueError,match='edge'):
        MonotoneAngularCover(lambda u,s:u/(1+s),[I(Q(1,2))])
    with pytest.raises(Unresolved,match='cell'):
        MonotoneAngularCover(lambda u,s:u/(1+s),[I(Q(1,4)),I(Q(3,4))],max_cells=6)
    with pytest.raises(Unresolved,match='depth'):
        MonotoneAngularCover(lambda u,s:u/(1+s),[I(Q(1,4)),I(Q(3,4))],max_depth=1)
    cover=MonotoneAngularCover(lambda u,s:u/(1+s),[I(Q(1,4))])
    with pytest.raises(ValueError,match='unit chart'):cover.map(0,Jet(2),Jet(0))
    with pytest.raises(ValueError,match='chart index'):cover.map(-1,Jet(0),Jet(0))



def test_complete_original_angular_cover(record_property):
    import os,json
    from pathlib import Path
    from tools.opaque_original_motion_reference import angular_cover_reference
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    path=Path(os.environ['GFFX_FULL_REFERENCE']);failure=None
    with backend(profile='ball'):
        try:result=angular_cover_reference(frozen())
        except (Unresolved,ValueError) as error:failure=str(error);result=dict(status='unresolved',failure=getattr(error,'details',str(error)))
    assert not path.exists();path.write_text(json.dumps(result,default=str,indent=2),encoding='utf8');record_property('complete_angular_cover',path.read_text())
    assert failure is None,result
    assert result['cells']<=256 and result['depth']<=10 and all(max(r['counts'])<=8 for r in result['rows'])


def test_angular_cover_refuses_unqualified_accuracy():
    from tools.opaque_original_motion_reference import angular_cover_reference
    with pytest.raises(ValueError,match='domain-preflight only'):angular_cover_reference({},domain_only=False)



def test_monotone_cover_retains_thin_positive_band():
    from tools.opaque_knot_partition import MonotoneAngularCover
    delta=Q(1,2**60);cover=MonotoneAngularCover(lambda u,s:u/(1+s),[I(Q(1,4)),I(Q(1,4)+delta)])
    row=cover.map(1,Jet(I(0,1)),Jet(I(0,1)))
    assert row['jacobian'][0,0].lo>0
    # Integrating an interval range over the unit chart is itself a rigorous
    # enclosure. This checks preservation of a tiny positive band; it does not
    # claim one-cell high-order quadrature resolves cancellation in root jets.
    v=row['jacobian'][0,0];exact=I(delta*Q(3,2));assert 0<v.lo<=exact.lo<=exact.hi<=v.hi


def test_original_angular_chart_integration(record_property):
    import os,json
    from pathlib import Path
    from tools.opaque_original_motion_reference import angular_chart_reference
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    path=Path(os.environ['GFFX_FULL_REFERENCE']);failure=None
    with backend(profile='ball'):
        try:result=angular_chart_reference(frozen(),chart_index=32)
        except ValueError as error:failure=str(error);result=dict(status='unresolved',failure=getattr(error,'details',str(error)))
    assert not path.exists();path.write_text(json.dumps(result,default=str,indent=2),encoding='utf8');record_property('angular_chart_integration',path.read_text())
    assert failure is None,result
    assert result['charged_cells']<=256



def test_saved_angular_chart_stage_diagnostic(record_property):
    import os,json
    from pathlib import Path
    from tools.opaque_original_motion_reference import angular_chart_reference
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    path=Path(os.environ['GFFX_FULL_REFERENCE']);saved=json.loads((path.parent/'chart-integral1-reference.json').read_bytes());box=list(map(Q,saved['failure']['cause']['last']['box']))
    with backend(profile='ball'):result=angular_chart_reference(frozen(),chart_index=32,diagnostic_box=box)
    assert not path.exists();path.write_text(json.dumps(result,default=str,indent=2),encoding='utf8');record_property('angular_chart_stage_diagnostic',path.read_text())
    assert result['status']=='chart_diagnostic_only' and result['rows']



def test_arb_sqrt_keeps_proved_positive_constant_and_mixed_coefficients():
    from tools.opaque_arb_jets import backend
    from fractions import Fraction
    x=I('0.000040032455863408761','0.000150323604019598487')
    for order in (2,10):
        q=Jet(order=order,coeff={(0,0):x,(1,0):I(1),(0,1):I('1e-8')})
        with backend(profile='ball'):result=q.sqrt()
        assert result[0,0].lo>0
        for point in [x.lo,x.hi]:
            v=I(point);root=v.sqrt();choose=Q(1)
            for i in range(order+1):
                exact=I(choose)*root/(v**i)
                tangent=I(choose)*I(Q(1,2)-i)*root/(v**(i+1))*I('1e-8')
                for actual,want in [(result[i,0],exact),(result[i,1],tangent)]:
                    assert actual.lo<=want.lo<=want.hi<=actual.hi
                choose=choose*(Q(1,2)-i)/(i+1)


def test_saved_angular_chart_high_bounds_are_finite(record_property):
    import os,json
    from pathlib import Path
    from tools.opaque_original_motion_reference import angular_chart_reference
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    path=Path(os.environ['GFFX_FULL_REFERENCE']);box=[Q(0),Q(1,4),Q(1,2),Q(1)]
    with backend(profile='ball'):result=angular_chart_reference(frozen(),chart_index=32,diagnostic_box=box)
    assert not path.exists();path.write_text(json.dumps(result,default=str,indent=2),encoding='utf8');record_property('angular_chart_high_bounds',path.read_text())
    assert all(r['status']=='evaluated' for r in result['rows'])


def test_centered_chart_factor_bounds(record_property):
    import os,json
    from pathlib import Path
    from tools.opaque_original_motion_reference import angular_chart_reference
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    path=Path(os.environ['GFFX_FULL_REFERENCE'])
    with backend(profile='ball'):
        result=angular_chart_reference(frozen(),chart_index=32,diagnostic_box=[Q(0),Q(1),Q(7,8),Q(1)])
    result['factor_order']=['view_cosine','light_cosine','half_cosine','physical_jacobian','view_loss','light_loss']
    assert not path.exists();path.write_text(json.dumps(result,default=str,indent=2),encoding='utf8');record_property('centered_factor_bounds',path.read_text())
    assert all(r['status']=='evaluated' for r in result['rows'])


@pytest.mark.parametrize('scale',[Q(1),Q(1,2**60)])
def test_angle_coordinate_closed_measure_and_transport(scale):
    from tools.opaque_knot_partition import MonotoneAngularCover
    field=lambda u,s:scale*(u/(1+s)+u*u.lift(0,tangent=Q(1,8)))
    radial=lambda u,s:scale*(1/(1+s)+u.lift(0,tangent=Q(1,8)))
    cover=MonotoneAngularCover(field,[I(scale/8),I(scale/4)])
    def mapped(t,y):
        row=cover.map_angle(1,t,y,radial)
        return [row['jacobian']*t.lift(1,tangent=Q(1,3)),row['angle']*row['jacobian']]
    result=integrate_square(mapped,max_cells=15,max_depth=3)
    # u=k(1+s) at theta=0; u_theta=-k(1+s)^2/8.
    for got,want in zip(result['value'],[Q(3,16),scale*Q(9,256)]):
        assert got.lo<=I(want).lo<=I(want).hi<=got.hi
    for got,want in zip(result['derivative'],[Q(5,192),-scale*Q(7,1024)]):
        assert got.lo<=I(want).lo<=I(want).hi<=got.hi
    row=cover.map_angle(1,Jet(I(0,1)),Jet(I(0,1)),radial)
    assert row['jacobian'][0,0].lo>0


@pytest.mark.parametrize('axis',[0,1])
def test_angle_coordinate_nonlinear_inverse_high_mixed(axis):
    from tools.opaque_knot_partition import MonotoneAngularCover
    field=lambda u,s:(u+u*u)/(1+s)+u*u.lift(0,tangent=Q(1,8))
    radial=lambda u,s:(1+2*u)/(1+s)+u.lift(0,tangent=Q(1,8))
    cover=MonotoneAngularCover(field,[I(Q(1,8)),I(Q(1,4))])
    for point in [Q(1,4),Q(3,4)]:
        args=[Jet(point,order=10),Jet(point,order=10)];args[axis]=Jet.variable(point,order=10)
        t,y=args;row=cover.map_angle(1,t,y,radial);k=(1+t)/8
        b=1+(1+y)*t.lift(0,tangent=Q(1,8))
        exact=(-b+(b*b+4*k*(1+y)).sqrt())/2
        jac=(1+y)/(8*(b*b+4*k*(1+y)).sqrt())
        for actual,want in [(row['u'],exact),(row['jacobian'],jac)]:
            for i in range(11):
                for j in range(2):
                    # Independent closed inverse versus implicit recurrence.
                    assert actual[i,j].lo<=want[i,j].hi and want[i,j].lo<=actual[i,j].hi
                    assert actual[i,j].width<I('1e-35').lo


def test_angle_coordinate_refuses_exit_and_nonpositive_slope():
    from tools.opaque_knot_partition import MonotoneAngularCover
    cover=MonotoneAngularCover(lambda u,s:u/(1+s),[I(Q(1,4)),I(Q(3,4))])
    with pytest.raises(ValueError,match='interior'):
        cover.map_angle(1,Jet(0),Jet(0),lambda u,s:1/(1+s))
    with pytest.raises(ValueError,match='positive'):
        cover.map_angle(0,Jet(0),Jet(0),lambda u,s:u.lift(-1))


def test_original_radial_angle_derivative_matches_independent_seed():
    from tools.opaque_original_motion_reference import prepare,mapped_inputs,radial_angle_derivative
    from tests.packaging.test_opaque_motion_tangent import frozen
    p=prepare(frozen(),piece=8,motion='normal_tilt',anchor=0,horizon_power=4)
    for x,y in [(Q(1,4),Q(1,8)),(Q(3,4),Q(7,8))]:
        original=mapped_inputs(p,Jet.variable(x,order=1),Jet(y,order=1),flatten=False)[0]
        actual=radial_angle_derivative(p,Jet(x),Jet(y))
        for j in (0,1):
            assert actual[0,j].lo<=original[1,j].hi and original[1,j].lo<=actual[0,j].hi
            assert actual[0,j].width<I('1e-35').lo


def test_angle_coordinate_original_factor_bounds(record_property):
    import os,json
    from pathlib import Path
    from tools.opaque_original_motion_reference import angular_chart_reference
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    path=Path(os.environ['GFFX_FULL_REFERENCE'])
    with backend(profile='ball'):
        result=angular_chart_reference(frozen(),chart_index=32,diagnostic_box=[Q(0),Q(1),Q(7,8),Q(1)],angle_coordinate=True)
    result['factor_order']=['view_cosine','light_cosine','half_cosine','physical_jacobian','view_loss','light_loss']
    assert not path.exists();path.write_text(json.dumps(result,default=str,indent=2),encoding='utf8');record_property('centered_factor_bounds',path.read_text())
    assert all(r['status']=='evaluated' for r in result['rows'])



def test_angle_coordinate_original_chart_integration(record_property):
    import os,json
    from pathlib import Path
    from tools.opaque_original_motion_reference import angular_chart_reference
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    path=Path(os.environ['GFFX_FULL_REFERENCE']);failure=None
    with backend(profile='ball'):
        try:result=angular_chart_reference(frozen(),chart_index=32,angle_coordinate=True)
        except ValueError as error:failure=str(error);result=dict(status='unresolved',failure=getattr(error,'details',str(error)))
    assert not path.exists();path.write_text(json.dumps(result,default=str,indent=2),encoding='utf8');record_property('angular_chart_integration',path.read_text())
    assert failure is None,result
    assert result['charged_cells']<=256




@pytest.mark.parametrize('scale',[Q(1),Q(1,2**60)])
def test_angle_cover_complete_measure_jump_and_transport(scale):
    from tools.opaque_knot_partition import MonotoneAngularCover
    cover=MonotoneAngularCover(lambda u,s:scale*(u/(1+s)+u*u.lift(0,tangent=Q(1,8))),[I(scale/4),I(3*scale/4)])
    radial=lambda u,s:scale*(1/(1+s)+u.lift(0,tangent=Q(1,8)))
    assert len(cover.charts)==4 and cover.cells==7 and cover.depth<=3
    def field(t,y):
        out=[t.lift(0) for _ in range(3)]
        for i,chart in enumerate(cover.charts):
            row=cover.map_angle_cover(i,t,y,radial);u,jac=row['u'],row['jacobian'];jac=jac*t.lift(scale,tangent=scale/3)
            for j,v in enumerate((1,[2,5,7][chart['band']],u)):out[j]=out[j]+v*jac
        return out
    result=integrate_square(field,max_cells=31,max_depth=4)
    values=[scale,scale*Q(95,24),scale/2];tangents=[scale/3,scale*(Q(95,72)+Q(263,864)),scale/6]
    for rows,exact in [(result['value'],values),(result['derivative'],tangents)]:
        for v,x in zip(rows,exact):assert v.lo<=I(x).lo<=I(x).hi<=v.hi
    assert result['value'][0].lo>0
    for axis in (0,1):
        t=Jet.variable(I(0,1),order=10) if axis==0 else Jet(I(0,1),order=10)
        y=Jet.variable(I(0,1),order=10) if axis==1 else Jet(I(0,1),order=10)
        for i in range(len(cover.charts)):assert cover.map_angle_cover(i,t,y,radial)['jacobian'][0,0].lo>=0


def test_angle_cover_two_exits_nonlinear_measure():
    from tools.opaque_knot_partition import MonotoneAngularCover
    cover=MonotoneAngularCover(lambda u,s:(u+u*u)/(1+s)+u*u.lift(0,tangent=Q(1,8)),[I(Q(1,8)),I(Q(1,4)),I(Q(3,2)),I(Q(7,4))])
    radial=lambda u,s:(1+2*u)/(1+s)+u.lift(0,tangent=Q(1,8))
    assert len(cover.charts)==7 and len(cover.events)==2
    assert {cover.map_angle_cover(i,Jet(Q(1,2)),Jet(Q(1,2)),radial)['coordinate_kind'] for i in range(len(cover.charts))}=={'angle','radial'}
    def field(t,y):
        rows=[cover.map_angle_cover(i,t,y,radial) for i in range(len(cover.charts))]
        return [sum((r['jacobian'] for r in rows),t.lift(0)),sum((r['u']*r['jacobian'] for r in rows),t.lift(0))]
    result=integrate_square(field,max_cells=127,max_depth=7)
    for v,x in zip(result['value'],[Q(1),Q(1,2)]):assert v.lo<=I(x).lo<=I(x).hi<=v.hi
    for v in result['derivative']:assert v.lo<=0<=v.hi



def test_complete_original_hybrid_angle_cover(record_property):
    import os,json
    from pathlib import Path
    from tools.opaque_original_motion_reference import angular_cover_reference
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    path=Path(os.environ['GFFX_FULL_REFERENCE']);failure=None
    with backend(profile='ball'):
        try:result=angular_cover_reference(frozen(),angle_coordinate=True)
        except (Unresolved,ValueError) as error:failure=str(error);result=dict(status='unresolved',failure=getattr(error,'details',str(error)))
    assert not path.exists();path.write_text(json.dumps(result,default=str,indent=2),encoding='utf8');record_property('complete_angular_cover',path.read_text())
    assert failure is None,result
    assert result['cells']<=256 and result['depth']<=10 and all(max(r['counts'])<=8 for r in result['rows'])

    assert {r['coordinate_kind'] for r in result['rows']}=={'angle','radial'}


@pytest.mark.parametrize("profile", ["decimal", "ball"])
def test_angle_only_matches_all_saved_coefficients(profile, monkeypatch, record_property):
    from tools.opaque_original_motion_reference import prepare, mapped_inputs, mapped_angle
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    import time
    rows=[]
    from contextlib import nullcontext
    with (nullcontext() if profile=='decimal' else backend(profile=profile)):
        p=prepare(frozen(),piece=8,motion='normal_tilt',anchor=0,horizon_power=4)
        for axis in (0,1):
            for box in [(I('0.25'),I('0.125')),(I('0.5','0.75'),I('0.875',1)),(I(0,'0.01'),I(0,'0.125'))]:
                args=[Jet(v,order=10) for v in box];args[axis]=Jet.variable(box[axis],order=10)
                start=time.monotonic();old=mapped_inputs(p,*args,flatten=False)[0];old_time=time.monotonic()-start
                start=time.monotonic();new=mapped_angle(p,*args);new_time=time.monotonic()-start
                for i in range(11):
                    for j in (0,1):assert (new[i,j].lo,new[i,j].hi)==(old[i,j].lo,old[i,j].hi)
                rows.append(dict(axis=axis,box=list(map(str,box)),old_seconds=old_time,new_seconds=new_time))
        calls=[];original=Jet.sqrt
        def counted(q):calls.append(1);return original(q)
        monkeypatch.setattr(Jet,'sqrt',counted)
        args=[Jet(Q(1,4)),Jet(Q(1,8))]
        mapped_inputs(p,*args,flatten=False);old_calls=len(calls);calls.clear()
        mapped_angle(p,*args);new_calls=len(calls)
        assert new_calls<old_calls
    record_property('angle_only_cost',dict(profile=profile,saved_boxes=rows,old_sqrt_calls=old_calls,new_sqrt_calls=new_calls))


def test_angle_only_refuses_unsupported_profile():
    from tools.opaque_original_motion_reference import mapped_angle
    with pytest.raises(ValueError,match='outside curved'):
        mapped_angle({'curved':False},Jet(0),Jet(0))


@pytest.mark.parametrize('scale',[Q(1),Q(1,2**60)])
def test_shared_charts_sum_complete_measure_and_transport(scale):
    charts=[]
    for left,depth,width in [(Q(0),1,Q(1,2)),(Q(1,2),2,Q(1,4)),(Q(3,4),2,Q(1,4))]:
        def field(t,y,left=left,width=width):
            x=left+width*t;jac=t.lift(scale*width,tangent=scale*width/3)
            return [jac,x*x*jac]
        charts.append(dict(field=field,depth=depth))
    result=integrate_square(None,chart_cover=charts,max_cells=5,max_depth=2)
    assert result['cells']==5 and result['regions']==3 and result['max_depth']==2
    assert result['chart_count']==3 and len(result['chart_leaves'])==3
    for key,want in [('value',[scale,scale/3]),('derivative',[scale/3,scale/9])]:
        for got,x in zip(result[key],want):assert got.lo<=I(x).lo<=I(x).hi<=got.hi
    assert result['value'][0].lo>0


def test_shared_charts_refuse_sum_of_local_passes():
    # Each chart passes by itself, but their combined uncertainty does not.
    charts=[dict(field=lambda t,y:[t.lift(1)],depth=1,
                 tangent_correction_integrator=lambda box:[I('-6e-13','6e-13')]) for _ in range(2)]
    for chart in charts:
        assert integrate_square(chart['field'],tangent_correction_integrator=chart['tangent_correction_integrator'],max_cells=1)['cells']==1
    with pytest.raises(Unresolved) as caught:
        integrate_square(None,chart_cover=charts,max_cells=3,max_depth=1)
    saved=caught.value.details['last_complete_enclosure']
    assert saved['complete_cover'] and saved['cells']==3 and I(saved['widths'][0][1]).lo>I('2e-12').hi


def test_shared_charts_never_reset_work_depth_or_deadline():
    charts=[dict(field=lambda t,y:[1/(1+t)],depth=1) for _ in range(2)]
    with pytest.raises(Unresolved) as caught:integrate_square(None,chart_cover=charts,max_cells=3)
    assert caught.value.details['cells']==3
    with pytest.raises(Unresolved,match='depth budget'):integrate_square(None,chart_cover=charts,max_depth=1)
    visits=[]
    def field(t,y):visits.append(1);return [t.lift(1)]
    def check():
        if len(visits)>=28:raise Unresolved('shared deadline')
    with pytest.raises(Unresolved,match='shared deadline'):
        integrate_square(None,chart_cover=[dict(field=field,depth=1)]*2,checkpoint=check)
    assert len(visits)==28


def test_shared_charts_validate_cover_before_work():
    field=lambda t,y:[t.lift(1)]
    for charts in [[],[dict(field=field,depth=1)],[dict(field=field,depth=0)]*2,[dict(field=field,depth=True)], [dict(field=None,depth=0)]]:
        with pytest.raises(ValueError):integrate_square(None,chart_cover=charts)
    with pytest.raises(ValueError):integrate_square(None,chart_cover=[dict(field=field,depth=1)]*2,max_cells=2)
    with pytest.raises(ValueError):integrate_square(field,chart_cover=[dict(field=field,depth=0)])



def test_shared_charts_keep_hybrid_edges_jumps_and_tiny_measure():
    from tools.opaque_knot_partition import MonotoneAngularCover
    scale=Q(1,2**60)
    cover=MonotoneAngularCover(lambda u,s:u/(1+s)+u*u.lift(0,tangent=Q(1,8)),[I(Q(1,4)),I(Q(3,4))])
    radial=lambda u,s:1/(1+s)+u.lift(0,tangent=Q(1,8))
    charts=[]
    for i,c in enumerate(cover.charts):
        def field(t,y,i=i,band=c['band']):
            row=cover.map_angle_cover(i,t,y,radial);jac=row['jacobian']*t.lift(scale,tangent=scale/3)
            return [jac,[2,5,7][band]*jac,row['u']*jac]
        charts.append(dict(field=field,depth=c['depth']))
    # Absolute tolerance is scaled too: the tiny case must still resolve the
    # same relative mathematical controls, not pass by being below the default.
    result=integrate_square(None,chart_cover=charts,tolerance=I(scale)*I('1e-12'),max_cells=256,max_depth=10)
    for key,want in [('value',[scale,scale*Q(95,24),scale/2]),('derivative',[scale/3,scale*(Q(95,72)+Q(263,864)),scale/6])]:
        for v,x in zip(result[key],want):assert v.lo<=I(x).lo<=I(x).hi<=v.hi
    assert result['cells']==cover.cells+2*(result['accepted']-len(charts))
    assert {r['chart'] for r in result['chart_leaves']}==set(range(len(charts)))
    assert result['value'][0].lo>0


def test_shared_charts_refuse_missing_channel():
    charts=[dict(field=lambda t,y:[t.lift(1),t.lift(2)],depth=1),dict(field=lambda t,y:[t.lift(1)],depth=1)]
    with pytest.raises(Unresolved,match='channel count'):integrate_square(None,chart_cover=charts)


@pytest.mark.parametrize('chart',[0,32,64])
def test_original_hybrid_whole_chart_stage_preflight(chart,record_property):
    import os,json
    from pathlib import Path
    from tools.opaque_original_motion_reference import angular_chart_reference
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    base=Path(os.environ['GFFX_FULL_REFERENCE']);path=base.with_name(base.stem+'-'+str(chart)+'.json')
    with backend(profile='ball'):
        result=angular_chart_reference(frozen(),chart_index=chart,diagnostic_box=[Q(0),Q(1),Q(0),Q(1)],angle_coordinate=(chart==32))
    result['scope']='Bounded initial-chart stage/cost diagnostic only. Refused stages are retained. No whole-piece accuracy or performance acceptance.'
    assert not path.exists();path.write_text(json.dumps(result,default=str,indent=2),encoding='utf8')
    record_property('whole_chart_stage_preflight',path.read_text())
    assert result['status']=='chart_diagnostic_only' and result['rows']
    assert all(r['status'] in ('evaluated','refused') for r in result['rows'])



def test_original_material_chart_state_and_frozen_stage_equivalence(record_property):
    import hashlib,importlib.util,json,os
    from pathlib import Path
    from tools.opaque_original_motion_reference import angular_material_cover
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    old=Path('D:/devspace/gffx/qualification-20260913/m1-continuation-20260913/opaque140-20260928/stage-preflight1-source/tools/opaque_original_motion_reference.py')
    assert hashlib.sha256(old.read_bytes()).hexdigest()=='2afcd449bacf208bb2d78efbab37a4e81c07a2a6ec9e7020c30bd42cd8921b1a'
    spec=importlib.util.spec_from_file_location('opaque140_frozen_motion',old);legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
    rows=[]
    with backend(profile='ball'):
        p,curve,cover,charts=angular_material_cover(frozen())
        assert len(charts)==72 and cover.cells==143
        assert len({id(c['prepare']) for c in charts})==72
        for index in (0,32,64):
            row=charts[index];box=(I(0,1),I(0,1));row['prepare'](Q(1,2),Q(1,2),box)
            def values():
                return [[str(v[0,0]),str(v[0,1]),str(v[v.order,0]),str(v[v.order,1])] for v in row['field'](Jet(Q(1,2)),Jet(Q(1,2)))]
            before=values()
            # Preparing another band must not replace this chart's extension or bounds.
            other=charts[32 if index!=32 else 64];other['prepare'](Q(1,2),Q(1,2),box)
            assert values()==before
            got={
                'center':row['field'](Jet(Q(1,2)),Jet(Q(1,2))),
                'radial10':row['field'](Jet.variable(I(0,1),order=10),Jet(I(0,1),order=10)),
                'transverse10':row['field'](Jet(I(0,1),order=10),Jet.variable(I(0,1),order=10)),
                'residual':row['correction_integrator'](box),
                'transport':row['tangent_correction_integrator'](box)}
            prior=legacy.angular_chart_reference(frozen(),chart_index=index,diagnostic_box=[Q(0),Q(1),Q(0),Q(1)],angle_coordinate=index==32)
            for stage,vals in got.items():
                expected=next(r['result'] for r in prior['rows'] if r['stage']==stage)
                actual=[str(v) if isinstance(v,I) else dict(value=str(v[0,0]),parameter=str(v[0,1]),high=str(v[v.order,0]),mixed=str(v[v.order,1])) for v in vals]
                # Legacy diagnostic residual is the matte value/tangent pair;
                # the integration callback returns specular-zero/matte-value.
                if stage=='residual':expected=['[0,0]',expected[0]]
                assert actual==expected,(index,stage)
            rows.append(dict(chart=index,state_isolated=True,stages_matched=list(got)))
    record_property('original_shared_state',json.dumps(rows))


def test_original_material_shared_driver_keeps_construction_deadline(monkeypatch):
    import types
    import tools.opaque_original_motion_reference as mod
    clock=[0.];entered=[]
    monkeypatch.setattr(mod,'time',types.SimpleNamespace(monotonic=lambda:clock[0]))
    def construct(data,*,checkpoint):clock[0]=91.;checkpoint();raise AssertionError('deadline missed')
    monkeypatch.setattr(mod,'angular_material_cover',construct)
    monkeypatch.setattr(mod,'integrate_square',lambda *a,**kw:entered.append(1))
    with pytest.raises(Unresolved,match='deadline'):mod.angular_piece_reference({})
    assert not entered


def test_original_material_shared_driver_complete_piece(record_property):
    import json,os
    from pathlib import Path
    from tools.opaque_original_motion_reference import angular_piece_reference
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    path=Path(os.environ['GFFX_FULL_REFERENCE']);failure=None;rows=[]
    with backend(profile='ball'):
        try:result=angular_piece_reference(frozen(),progress=rows.append)
        except ValueError as error:failure=str(error);result=dict(status='unresolved',failure=getattr(error,'details',str(error)))
    result['progress']=rows
    assert not path.exists();path.write_text(json.dumps(result,default=str,indent=2),encoding='utf8')
    record_property('shared_original_piece',path.read_text())
    assert failure is None,result
    assert result['basis']['chart_count']==72 and result['basis']['cells']<=256


@pytest.mark.parametrize('chart',[0,27,32])
def test_shared_original_initial_bound_diagnostic(chart,record_property):
    import json,os
    from pathlib import Path
    from tools.opaque_original_motion_reference import angular_chart_reference
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    base=Path(os.environ['GFFX_FULL_REFERENCE']);path=base.with_name(base.stem+'-'+str(chart)+'.json')
    with backend(profile='ball'):
        result=angular_chart_reference(frozen(),chart_index=chart,diagnostic_box=[Q(0),Q(1),Q(0),Q(1)],angle_coordinate=chart!=0)
    result['factor_order']=['view_cosine','light_cosine','half_cosine','physical_jacobian','view_loss','light_loss']
    result['scope']='Initial whole-chart bound diagnosis after the terminal shared trial; no integral acceptance or fresh full trial.'
    assert not path.exists();path.write_text(json.dumps(result,default=str,indent=2),encoding='utf8');record_property('shared_initial_bound_diagnostic',path.read_text())
    assert all(r['status']=='evaluated' for r in result['rows'])


@pytest.mark.parametrize('profile',['decimal','ball'])
def test_quadratic_sqrt_keeps_exact_square_and_moving_composition(profile):
    from contextlib import nullcontext
    from tools.opaque_arb_jets import backend
    from tools.opaque_quadratic_sqrt import QuadraticSqrt
    from tools.opaque_moving_reference import P
    with (nullcontext() if profile=='decimal' else backend(profile='ball')):
        # (2+x+theta*(3+x))^2; all Taylor coefficients are known independently.
        root=QuadraticSqrt([P([4,12]),P([4,10]),P([1,2])])
        for order in (0,1,3,10):
            x=Jet.variable(I(0,1),order=order) if order else Jet(I(0,1))
            x=x+x.lift(0,tangent=Q(1,7))
            actual=root(x);exact=2+x+x.lift(0,tangent=1)*(3+x)
            for i in range(order+1):
                for j in (0,1):
                    assert actual[i,j].lo<=exact[i,j].lo<=exact[i,j].hi<=actual[i,j].hi
                    if i>=2:assert actual[i,j].zero


@pytest.mark.parametrize('profile',['decimal','ball'])
def test_quadratic_sqrt_nonlinear_mixed_matches_point_algebra(profile):
    from contextlib import nullcontext
    from tools.opaque_arb_jets import backend
    from tools.opaque_quadratic_sqrt import QuadraticSqrt
    from tools.opaque_moving_reference import P
    with (nullcontext() if profile=='decimal' else backend(profile='ball')):
        root=QuadraticSqrt([P([2,Q(1,3)]),P([1,Q(1,5)]),P([3,Q(1,7)])])
        broad=root(Jet.variable(I(0,1),order=10))
        for point in (Q(0),Q(1,3),Q(1)):
            x=Jet.variable(point,order=10);x=x+x.lift(0,tangent=Q(1,11))
            actual=root(x);naive=(x.lift(2,tangent=Q(1,3))+x*x.lift(1,tangent=Q(1,5))+x*x*x.lift(3,tangent=Q(1,7))).sqrt()
            fixed=root(Jet.variable(point,order=10))
            for i in range(11):
                for j in (0,1):
                    a,b=actual[i,j],naive[i,j]
                    assert a.lo<=b.hi and b.lo<=a.hi
                    assert a.width<I('1e-30').lo
                    assert broad[i,j].lo<=fixed[i,j].hi and fixed[i,j].lo<=broad[i,j].hi
        # Nonlinear coordinate composition exercises the full nilpotent series.
        x=Jet.variable(Q(1,4),order=10);y=x*x+x.lift(0,tangent=Q(1,9))
        actual=root(y);naive=(y.lift(2,tangent=Q(1,3))+y*y.lift(1,tangent=Q(1,5))+y*y*y.lift(3,tangent=Q(1,7))).sqrt()
        for i in range(11):
            for j in (0,1):assert actual[i,j].lo<=naive[i,j].hi and naive[i,j].lo<=actual[i,j].hi


def test_quadratic_sqrt_refuses_nonpositive_domain_and_keeps_tiny_curvature():
    from tools.opaque_quadratic_sqrt import QuadraticSqrt
    from tools.opaque_moving_reference import P
    with pytest.raises(ValueError):QuadraticSqrt([P(0)]*4)
    with pytest.raises(ValueError,match='positive'):QuadraticSqrt([P(-1),P(0),P(1)])(Jet(I(0,2)))
    eps=Q(1,2**60);root=QuadraticSqrt([P(4+eps),P(4),P(1)])
    actual=root(Jet.variable(I(0,1),order=10))
    assert actual[2,0].lo>0 and actual[2,0].hi<I('1e-18').lo


@pytest.mark.parametrize('positive',[True,False])
def test_tight_discriminant_preserves_closed_curved_area_and_motion(positive):
    from decimal import Decimal as D,localcontext
    from tools.opaque_curved_reference import CurvedTriangle,integrate_curved
    from tools.opaque_quadratic_sqrt import QuadraticSqrt
    from tools.opaque_arb_jets import backend
    with backend(profile='ball'):
        patch=CurvedTriangle([(0,0),(1,0),(0,1)],[0,0,-Q(1,10),1,1,[-Q(1,4),-1]],positive=positive)
        exact=QuadraticSqrt(patch.discriminant);calls=[]
        def selected(x):calls.append(1);return exact(x)
        patch.sqrt_discriminant=selected
        result=integrate_curved(patch,lambda x,y:[x.lift(1)])
        assert calls
    with localcontext() as c:
        c.prec=90;a=D(1)/4;b=D(1)/10;root=2*a/(1+(1-4*a*b).sqrt())
        area=D(1)/2-a*root+root*root/2-b*root**3/3;motion=-root
        if not positive:area=D(1)/2-area;motion=-motion
        for got,want in zip(result['value']+result['derivative'],[area,motion]):
            assert got.lo<=want<=got.hi and got.width<D('2e-12')


def test_tight_original_discriminant_point_and_motion_control():
    from tools.opaque_original_motion_reference import prepare,mapped_inputs
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    with backend(profile='ball'):
        data=frozen();old=prepare(data,piece=8,motion='normal_tilt',anchor=0)
        new=prepare(data,piece=8,motion='normal_tilt',anchor=0,tight_discriminant=True)
        for point in (Q(0),Q(1,2),Q(1)):
            y=Jet.variable(point,order=10)
            for a,b in zip(mapped_inputs(old,Jet(Q(1,2),order=10),y,flatten=False),mapped_inputs(new,Jet(Q(1,2),order=10),y,flatten=False)):
                for i in range(11):
                    for j in (0,1):
                        assert a[i,j].lo<=b[i,j].hi and b[i,j].lo<=a[i,j].hi
                        assert b[i,j].width<I('1e-25').lo


@pytest.mark.parametrize('chart',[0,27,32])
def test_tight_original_initial_bound_diagnostic(chart,record_property):
    import json,os
    from pathlib import Path
    from tools.opaque_original_motion_reference import angular_chart_reference
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    base=Path(os.environ['GFFX_FULL_REFERENCE']);path=base.with_name(base.stem+'-'+str(chart)+'.json')
    with backend(profile='ball'):
        result=angular_chart_reference(frozen(),chart_index=chart,diagnostic_box=[Q(0),Q(1),Q(0),Q(1)],angle_coordinate=chart!=0,tight_discriminant=True)
    result['scope']='Exact discriminant derivative candidate; whole-width bound diagnostic only, not integral accuracy acceptance.'
    assert not path.exists();path.write_text(json.dumps(result,default=str,indent=2),encoding='utf8');record_property('tight_initial_bounds',path.read_text())
    assert all(r['status']=='evaluated' for r in result['rows'])


@pytest.mark.parametrize('chart',[0,27,32])
def test_tight_original_chart_integral(chart,record_property):
    import json,os
    from pathlib import Path
    from tools.opaque_original_motion_reference import angular_chart_reference
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    path=Path(os.environ['GFFX_FULL_REFERENCE']);failure=None
    with backend(profile='ball'):
        try:result=angular_chart_reference(frozen(),chart_index=chart,angle_coordinate=chart!=0,tight_discriminant=True)
        except ValueError as error:failure=str(error);result=dict(status='unresolved',failure=getattr(error,'details',str(error)))
    assert not path.exists();path.write_text(json.dumps(result,default=str,indent=2),encoding='utf8');record_property('tight_local_integral',path.read_text())
    assert failure is None,result
    assert result['status']=='local_angular_chart_integrated' and result['charged_cells']<=256
    assert all(v.width<=I('1e-12').lo for v in result['basis']['value']+result['basis']['derivative'])



def test_tight_shared_cover_matches_frozen_local_stages(record_property):
    import json,hashlib
    from pathlib import Path
    from tools.opaque_original_motion_reference import angular_material_cover
    from tools.opaque_arb_jets import backend
    from tools.opaque_quadratic_sqrt import QuadraticSqrt
    from tests.packaging.test_opaque_motion_tangent import frozen
    root=Path('D:/devspace/gffx/qualification-20260913/m1-continuation-20260913/opaque142-20260928')
    manifest=json.loads((root/'tight-bounds1-source.json').read_bytes())
    for name in ('tools/opaque_quadratic_sqrt.py','tools/opaque_curved_reference.py'):
        expected=next(x['sha256'] for x in manifest if x['path']==name)
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==expected
    rows=[]
    with backend(profile='ball'):
        p,curve,cover,charts=angular_material_cover(frozen(),tight_discriminant=True)
        assert isinstance(p['patch'].sqrt_discriminant.evaluate,QuadraticSqrt)
        assert 'prepared_angle' in p
        assert len(charts)==72 and cover.cells==143
        for index in (0,27,32):
            row=charts[index];box=(I(0,1),I(0,1));row['prepare'](Q(1,2),Q(1,2),box)
            center=lambda:[[str(v[0,0]),str(v[0,1])] for v in row['field'](Jet(Q(1,2)),Jet(Q(1,2)))]
            before=center();charts[32 if index!=32 else 27]['prepare'](Q(1,2),Q(1,2),box);assert center()==before
            got={'center':row['field'](Jet(Q(1,2)),Jet(Q(1,2))),'radial10':row['field'](Jet.variable(I(0,1),order=10),Jet(I(0,1),order=10)),'transverse10':row['field'](Jet(I(0,1),order=10),Jet.variable(I(0,1),order=10)),'residual':row['correction_integrator'](box),'transport':row['tangent_correction_integrator'](box)}
            prior=json.loads((root/f'tight-bounds1-reference-{index}.json').read_bytes())
            for stage,values in got.items():
                expected=next(x['result'] for x in prior['rows'] if x['stage']==stage)
                actual=[str(v) if isinstance(v,I) else dict(value=str(v[0,0]),parameter=str(v[0,1]),high=str(v[v.order,0]),mixed=str(v[v.order,1])) for v in values]
                if stage=='residual':expected=['[0,0]',expected[0]]
                assert actual==expected,(index,stage)
            rows.append(dict(chart=index,state_isolated=True,exact_frozen_stages=list(got)))
    record_property('tight_shared_connection',json.dumps(rows))


def test_tight_shared_driver_keeps_explicit_selection_and_construction_deadline(monkeypatch):
    import types
    import tools.opaque_original_motion_reference as mod
    clock=[0.];calls=[];entered=[]
    monkeypatch.setattr(mod,'time',types.SimpleNamespace(monotonic=lambda:clock[0]))
    def construct(data,*,checkpoint,tight_discriminant=False):
        calls.append(tight_discriminant);clock[0]+=91;checkpoint();raise AssertionError('deadline missed')
    monkeypatch.setattr(mod,'angular_material_cover',construct)
    monkeypatch.setattr(mod,'integrate_square',lambda *a,**kw:entered.append(1))
    for chosen in (False,True):
        with pytest.raises(Unresolved,match='deadline'):mod.angular_piece_reference({},tight_discriminant=chosen)
    assert calls==[False,True] and not entered
    with pytest.raises(ValueError,match='explicit'):mod.angular_piece_reference({},tight_discriminant=1)
    assert calls==[False,True]


def test_tight_shared_original_complete_piece(record_property):
    import json,os
    from pathlib import Path
    from tools.opaque_original_motion_reference import angular_piece_reference
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    path=Path(os.environ['GFFX_FULL_REFERENCE']);failure=None;rows=[]
    with backend(profile='ball'):
        try:result=angular_piece_reference(frozen(),tight_discriminant=True,progress=rows.append)
        except ValueError as error:failure=str(error);result=dict(status='unresolved',failure=getattr(error,'details',str(error)))
    result['progress']=rows
    assert not path.exists();path.write_text(json.dumps(result,default=str,indent=2),encoding='utf8');record_property('tight_shared_original_piece',path.read_text())
    assert failure is None,result
    assert result['tight_discriminant'] is True and result['basis']['chart_count']==72 and result['basis']['cells']<=256
    assert all(v.width<=I('2e-12').lo for v in result['value']+result['derivative'])


@pytest.mark.parametrize('profile',['decimal','ball'])
def test_horizon_cache_exact_keys_ownership_backend_and_bound(profile):
    from contextlib import nullcontext
    from tools.opaque_arb_jets import backend
    from tools.opaque_horizon_cache import HorizonCache
    from tools.opaque_quadratic_sqrt import QuadraticSqrt
    from tools.opaque_moving_reference import P
    import tools.opaque_shading_intervals as scalar
    fn=QuadraticSqrt([P([2,Q(1,3)]),P([1,Q(1,5)]),P([3,Q(1,7)])]);cache=HorizonCache(fn)
    encode=lambda x:{k:(v.lo,v.hi) for k,v in x.c.items()}
    with (nullcontext() if profile=='decimal' else backend(profile='ball')):
        for order in (0,1,10):
            x=Jet.variable(I(0,1),order=order) if order else Jet(I(0,1));x=x+x.lift(0,tangent=Q(1,11))
            want=encode(fn(x));assert encode(cache(x))==want
            before=cache.misses;got=cache(x);assert encode(got)==want and cache.misses==before
            # Mutating either caller-owned object must not poison stored coefficients.
            got.c[0,0].lo=got.c[0,0].lo-1;got.c[0,1]=I(999);assert encode(cache(x))==want
            x.c[0,1]=I(Q(1,13));assert encode(cache(x))==encode(fn(x)) and cache.misses>before
        for i in range(65):cache(Jet(Q(i,128)))
        assert len(cache.entries)==64 and cache.hits>=6
        before=cache.misses;assert encode(cache(Jet(0)))==encode(fn(Jet(0)))
        assert cache.misses==before+1  # Oldest of65 unique scalar inputs was evicted.
        assert encode(cache(Jet(I(0,1))))==encode(fn(Jet(I(0,1))))
        probe=Jet.variable(Q(1,3),order=10);cache(probe)
    # Backend closures are part of identity: don't return a prior context's rounding.
    before=cache.misses
    with backend(profile='endpoint'):
        assert encode(cache(probe))==encode(fn(probe))
        assert cache.misses==before+1
        import flint
        saved=flint.ctx.prec
        try:
            flint.ctx.prec=saved+16
            assert encode(cache(probe))==encode(fn(probe)) and cache.misses==before+2
        finally:flint.ctx.prec=saved
    assert cache.misses==before+2
    with pytest.raises(ValueError):cache(1)
    with pytest.raises(ValueError):HorizonCache(None)
    def refuse(x):raise ValueError('uncertified')
    bad=HorizonCache(refuse)
    with pytest.raises(ValueError):bad(Jet(0))
    assert not bad.entries



def test_tight_shared_initial_cell_cost_diagnostic(record_property):
    import cProfile,pstats,time,json,os
    from pathlib import Path
    from tools.opaque_original_motion_reference import angular_material_cover
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    start=time.monotonic()
    def check():
        if time.monotonic()-start>=90:raise Unresolved('initial-cell diagnostic deadline')
    with backend(profile='ball'):
        p,curve,cover,charts=angular_material_cover(frozen(),tight_discriminant=True,checkpoint=check)
        selected=charts[27];prof=cProfile.Profile();result=None
        prof.enable()
        try:
            integrate_square(selected['field'],prepare=selected['prepare'],correction_integrator=selected['correction_integrator'],tangent_correction_integrator=selected['tangent_correction_integrator'],tolerance='5e-13',max_cells=1,max_depth=10,checkpoint=check)
        except Unresolved as error:result=error.details
        finally:prof.disable()
        assert result is not None and result['reason']=='cell budget' and result['last_complete_enclosure']['complete_cover']
        costs=pstats.Stats(prof).stats
        rows=sorted([dict(file=k[0],line=k[1],function=k[2],primitive_calls=v[0],calls=v[1],self_seconds=v[2],cumulative_seconds=v[3]) for k,v in costs.items()],key=lambda r:r['self_seconds'],reverse=True)
        report=dict(scope='Single chart27 initial-cell cost diagnostic with profiler overhead. Expected one-cell accuracy refusal; not a complete-piece run or performance acceptance.',seconds=time.monotonic()-start,chart=27,result=result,horizon_cache=p['horizon_cache'].report(),self_costs=rows[:35],cumulative_costs=sorted(rows,key=lambda r:r['cumulative_seconds'],reverse=True)[:35])
    path=Path(os.environ['GFFX_FULL_REFERENCE']);assert not path.exists();path.write_text(json.dumps(report,default=str,indent=2),encoding='utf8');record_property('shared_cell_cost',path.read_text())


@pytest.mark.parametrize('motion',['normal_tilt','view_x','screen_x'])
def test_prepared_angle_preserves_original_high_mixed_and_partial(motion,record_property):
    import json
    from tools.opaque_prepared_angle import PreparedAngle
    from tools.opaque_original_motion_reference import prepare,mapped_angle,radial_angle_derivative
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    encode=lambda v:{k:(z.lo,z.hi) for k,z in v.c.items()}
    cases=0
    with backend(profile='ball'):
        p=prepare(frozen(),piece=8,motion=motion,anchor=0,tight_discriminant=True);fast=PreparedAngle(p)
        for order in (0,1,10):
            for axis in (0,1):
                args=[Jet(I('.25','.75'),order=order),Jet(I(0,1),order=order)]
                if order:args[axis]=Jet.variable(args[axis][0,0],order=order)
                args[axis]=args[axis]+args[axis].lift(0,tangent=Q(1,17))
                for call,slow in ((fast,mapped_angle),(fast.radial,radial_angle_derivative)):
                    got=call(*args);want=slow(p,*args);assert encode(got)==encode(want)
                    got.c[0,0].lo-=1;assert encode(call(*args))==encode(want)
                    cases+=1
        for i in range(66):fast(Jet(Q(1,2)),Jet(Q(i,128)))
        assert fast.report()['entries']==64 and fast.report()['hits']>0
    record_property('prepared_angle_exact',json.dumps(dict(motion=motion,cases=cases,cache=fast.report())))


def test_prepared_angle_rejects_mismatched_jets_and_crosses_backend_safely():
    from tools.opaque_prepared_angle import PreparedAngle
    from tools.opaque_original_motion_reference import prepare,mapped_angle
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    p=prepare(frozen(),piece=8,motion='normal_tilt',anchor=0,tight_discriminant=True);fast=PreparedAngle(p)
    encode=lambda v:{k:(z.lo,z.hi) for k,z in v.c.items()}
    u,s=Jet.variable(Q(1,2),order=1),Jet(Q(1,3),order=1)
    assert encode(fast(u,s))==encode(mapped_angle(p,u,s));before=fast.report()['misses']
    with backend(profile='ball'):
        assert encode(fast(u,s))==encode(mapped_angle(p,u,s)) and fast.report()['misses']==before+1
    with pytest.raises(ValueError):fast(Jet(0),Jet(0,order=1))
    with pytest.raises(ValueError):PreparedAngle(dict(curved=False))



def test_prepared_value_only_brackets_match_exact_spatial_endpoints():
    from tools.opaque_prepared_angle import PreparedAngle
    from tools.opaque_original_motion_reference import prepare
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    with backend(profile='ball'):
        fast=PreparedAngle(prepare(frozen(),piece=8,motion='normal_tilt',anchor=0,tight_discriminant=True))
        for order in (0,1):
            for value in (I(0,1),I('.125'),I('.5')):
                u=Jet.variable(value,order=order) if order else Jet(value)
                s=Jet(I(0,1),order=order);before=fast(u,s);after=fast.value_only(u,s)
                for i in range(order+1):assert (after[i,0].lo,after[i,0].hi)==(before[i,0].lo,before[i,0].hi)
                assert all(not j for i,j in after.c)
        with pytest.raises(ValueError):fast.value_only(Jet(0,order=2),Jet(0,order=2))


def test_inverse_value_only_path_keeps_closed_mixed_derivatives():
    from tools.opaque_knot_partition import MonotoneAngularCover
    from tools.opaque_arb_jets import backend
    calls=[]
    field=lambda u,s:(u+u*u)/(1+s)+u*u.lift(0,tangent=Q(1,8))
    radial=lambda u,s:(1+2*u)/(1+s)+u.lift(0,tangent=Q(1,8))
    def values(u,s):
        assert u.order<=1 and not any(j for i,j in u.c) and not any(j for i,j in s.c)
        calls.append(1);return (u+u*u)/(1+s)
    with backend(profile='ball'):
        cover=MonotoneAngularCover(field,[I(Q(1,8)),I(Q(1,4))],value_field=values)
        for axis in (0,1):
            args=[Jet(Q(1,4),order=10),Jet(Q(1,4),order=10)];args[axis]=Jet.variable(Q(1,4),order=10)
            t,y=args;row=cover.map_angle(1,t,y,radial);k=(1+t)/8;b=1+(1+y)*t.lift(0,tangent=Q(1,8))
            exact=(-b+(b*b+4*k*(1+y)).sqrt())/2;jac=(1+y)/(8*(b*b+4*k*(1+y)).sqrt())
            for actual,want in [(row['u'],exact),(row['jacobian'],jac)]:
                for i in range(11):
                    for j in (0,1):
                        assert actual[i,j].lo<=want[i,j].hi and want[i,j].lo<=actual[i,j].hi
                        assert actual[i,j].width<I('1e-30').lo
        assert calls


def test_material_input_cache_keys_ownership_context_and_bound():
    from tools.opaque_prepared_angle import MaterialInputCache
    from tools.opaque_arb_jets import backend
    import tools.opaque_shading_intervals as scalar
    cache=MaterialInputCache();calls=[]
    def field(t,y):
        calls.append(1);return t+y,t*y,(1+t).sqrt(),(2+y).reciprocal()
    encode=lambda rows:[{k:(v.lo,v.hi) for k,v in q.c.items()} for q in rows]
    for order in (0,1,10):
        t=Jet.variable(I('.1','.3'),order=order) if order else Jet(I('.1','.3'))
        t=t+t.lift(0,tangent=Q(1,11));y=Jet(Q(1,2),order=order)
        expected=encode(field(t,y));got=cache(('chart',None),field,t,y);assert encode(got)==expected
        before=len(calls);got[0].c[0,0].lo-=1
        assert encode(cache(('chart',None),field,t,y))==expected and len(calls)==before
        cache(('other_chart',None),field,t,y);assert len(calls)==before+1
        cache(('chart',('bound','changed')),field,t,y);assert len(calls)==before+2
    before=len(calls)
    with backend(profile='ball'):
        assert encode(cache(('chart',None),field,t,y))==encode(field(t,y))
    assert len(calls)==before+2
    old=scalar.PRECISION
    try:
        scalar.PRECISION=old+4;before=len(calls);cache(('chart',None),field,t,y);assert len(calls)==before+1
    finally:scalar.PRECISION=old
    for k in range(66):cache(k,field,Jet(0),Jet(0))
    assert cache.report()['entries']==cache.report()['entry_cap']==64
    assert cache.report()['hits']>=3
    def fail(t,y):raise ValueError('no certificate')
    before=cache.report()['entries']
    with pytest.raises(ValueError,match='no certificate'):cache('fail',fail,Jet(0),Jet(0))
    assert cache.report()['entries']==before
    with pytest.raises(ValueError,match='four matching'):cache('bad',lambda a,b:[1],Jet(0),Jet(0))


def test_material_input_cache_preserves_frozen_chart_after_prepare_and_interleave():
    from tools.opaque_original_motion_reference import angular_material_cover
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    encode=lambda rows:[{k:(v.lo,v.hi) for k,v in q.c.items()} for q in rows]
    with backend(profile='ball'):
        p,curve,cover,charts=angular_material_cover(frozen(),tight_discriminant=True)
        cache=p['material_inputs_cache'];box=(I(0,1),I(0,1));a=charts[27];b=charts[32]
        a['prepare'](Q(1,2),Q(1,2),box);want=encode(a['field'](Jet(Q(1,2)),Jet(Q(1,2))))
        before=cache.report()['hits'];assert encode(a['field'](Jet(Q(1,2)),Jet(Q(1,2))))==want
        assert cache.report()['hits']>before
        b['prepare'](Q(1,2),Q(1,2),box);b['field'](Jet(Q(1,2)),Jet(Q(1,2)))
        assert encode(a['field'](Jet(Q(1,2)),Jet(Q(1,2))))==want
        a['prepare'](Q(1,2),Q(1,2),(I('.25','.75'),I('.25','.75')))
        narrow=encode(a['field'](Jet(Q(1,2)),Jet(Q(1,2))))
        cache.entries.clear();assert encode(a['field'](Jet(Q(1,2)),Jet(Q(1,2))))==narrow
        assert cache.report()['entries']<=64
