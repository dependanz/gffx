"""Bounded exact-work diagnostics; no reference certificate is admitted here."""
from fractions import Fraction as Q
import os,json,time,hashlib,cProfile,pstats
import pytest
from pathlib import Path
from tools.opaque_shading_intervals import I,Jet,Unresolved
from tools.opaque_arb_jets import backend
from tests.packaging.test_opaque_motion_tangent import frozen
from tests.packaging.test_opaque_motion_extension import PROFILE

def test_saved_piece8_cost(record_property):
    from tools.opaque_original_motion_reference import angular_material_cover
    from tools.opaque_tensor_smooth import rule
    spec_path=Path(os.environ['GFFX_CHECKER_COST_SELECTION']);assert hashlib.sha256(spec_path.read_bytes()).hexdigest()==os.environ['GFFX_CHECKER_COST_SELECTION_SHA']
    spec=json.loads(spec_path.read_bytes());old=Path(spec['path']);assert hashlib.sha256(old.read_bytes()).hexdigest()==spec['sha256']
    saved=json.loads(old.read_bytes());assert saved['status']=='qualified'
    assert all(r in saved['result']['basis']['chart_leaves'] for r in spec['regions']) and len(spec['regions'])==3
    start=time.monotonic();rows=[];prof=cProfile.Profile()
    def check():
        if time.monotonic()-start>=90:raise Unresolved('saved-region cost deadline')
    def stage(name,fn):
        check();begin=time.monotonic();prof.enable()
        try:answer=fn()
        finally:prof.disable()
        stages.append(dict(stage=name,seconds=time.monotonic()-begin));check();return answer
    with backend(profile='ball',fast_conversion=True):
        begin=time.monotonic();p,curve,cover,charts=angular_material_cover(frozen(),piece=8,motion='normal_tilt',anchor=0,checkpoint=check,**PROFILE,curved_transverse=True,polynomial_light=True)
        construction=time.monotonic()-begin;directional=rule(checkpoint=check)
        ga=(I(5)+2*I(Q(10,7)).sqrt()).sqrt()/6;gb=(I(5)-2*I(Q(10,7)).sqrt()).sqrt()/6;nodes=[I('.5')-ga,I('.5')-gb,I('.5'),I('.5')+gb,I('.5')+ga]
        for r in spec['regions']:
            a,b,c,d=map(Q,r['box']);box=(I(I(a).lo,I(b).hi),I(I(c).lo,I(d).hi));row=charts[r['chart']];stages=[]
            stage('prepare',lambda:row['prepare']((a+b)/2,(c+d)/2,box))
            nx,ny,orders=(directional[0][0],directional[1][0],directional[2]) if row.get('tensor_rule') else (nodes,nodes,(10,10))
            def samples():
                for x in nx:
                    for y in ny:row['field'](Jet(I(a)+I(b-a)*x),Jet(I(c)+I(d-c)*y));check()
            stage('quadrature',samples)
            ds=stage('radial_remainder',lambda:row['field'](Jet.variable(box[0],order=orders[0]),Jet(box[1],order=orders[0])))
            def transverse():
                count=4 if row.get('tensor_rule') else 1
                return [row['field'](Jet(box[0],order=orders[1]),Jet.variable(I(I(c+(d-c)*Q(k,count)).lo,I(c+(d-c)*Q(k+1,count)).hi),order=orders[1])) for k in range(count)]
            dt=stage('transverse_remainder',transverse)
            ordinary=stage('correction',lambda:row['correction_integrator'](box));tangent=stage('transport',lambda:row['tangent_correction_integrator'](box))
            rows.append(dict(region=r,stages=stages,ordinary=[[str(x.lo),str(x.hi)] for x in ordinary],tangent=[[str(x.lo),str(x.hi)] for x in tangent],radial=[[str(q[orders[0],j].bound) for j in (0,1)] for q in ds],transverse=[[[str(q[orders[1],j].bound) for j in (0,1)] for q in v] for v in dt]))
        stats=pstats.Stats(prof);functions=[dict(file=k[0],line=k[1],function=k[2],primitive=v[0],calls=v[1],own_seconds=v[2],cumulative_seconds=v[3]) for k,v in stats.stats.items()]
        functions.sort(key=lambda r:r['cumulative_seconds'],reverse=True)
        result=dict(status='diagnostic_complete',selection_sha256=os.environ['GFFX_CHECKER_COST_SELECTION_SHA'],construction_seconds=construction,seconds=time.monotonic()-start,rows=rows,profile=functions[:180],physical_inverse=p['physical_angle'].report(),material_cache=p['material_inputs_cache'].report(),scope='Three saved regions, all original samples/correction/transport/remainder evaluations with profiling overhead; no complete-piece or reference acceptance.')
    dest=Path(os.environ['GFFX_FULL_REFERENCE']);assert not dest.exists();dest.write_text(json.dumps(result,indent=2),encoding='utf8');record_property('cost_report',str(dest))


import pytest
@pytest.mark.parametrize('piece',[8,11])
@pytest.mark.parametrize('motion',['normal_tilt','view_x','screen_x'])
@pytest.mark.parametrize('order',[1,4])
def test_chart_cache_keeps_unclipped_values_and_fresh_local_bounds(monkeypatch,piece,motion,order):
    import importlib.util
    import tools.opaque_original_motion_reference as m
    from tools.opaque_inside_angle import inside_material_cover
    from tools.opaque_loss_extension import bounded_cosines
    root=Path(os.environ['GFFX_FROZEN_CHART_SOURCE']);assert hashlib.sha256(root.read_bytes()).hexdigest()==os.environ['GFFX_FROZEN_CHART_SHA']
    spec=importlib.util.spec_from_file_location('frozen_chart_cache_baseline',root);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    captured=[];ctor=m.TangentCorrection
    def capture(*a,**kw):captured.append(a[1]);return ctor(*a,**kw)
    monkeypatch.setattr(m,'TangentCorrection',capture);old.TangentCorrection=capture
    def endpoints(values):return [[(q[i,j].lo,q[i,j].hi) for i in range(q.order+1) for j in (0,1)] for q in values]
    with backend(profile='ball',fast_conversion=True):
        if piece==8:p,curve,cover,charts=m.angular_material_cover(frozen(),piece=piece,motion=motion,anchor=0,**PROFILE,curved_transverse=True,polynomial_light=True)
        else:p,curve,cover,charts=inside_material_cover(frozen(),piece=piece,motion=motion,anchor=0,polynomial_light=True)
        callbacks=list(captured);global_bounds=bounded_cosines(lambda u,s:list(m.mapped_inputs(p,1-u if piece==11 else u,s,flatten=False)[:2]),I(0,1),I(0,1),monotone=True,partial=True)
        for index in (0,next(i for i,c in enumerate(cover.charts) if c['low'] is not None and c['high'] is not None)):
            baseline_p=dict(p);baseline_p.pop('material_inputs_cache')
            baseline=old.angular_material_chart(baseline_p,curve,cover,index,global_bounds);baseline_inputs=captured[-1]
            t=Jet.variable(I(Q(1,2)),order=order);y=Jet(I(Q(1,3)),order=order);selected=callbacks[index];row=charts[index]
            selected(t,y)
            interval=lambda a,b:I(I(a).lo,I(b).hi)
            for domain in [(interval(Q(1,3),Q(2,3)),interval(Q(1,4),Q(1,2))),(interval(Q(2,5),Q(3,5)),interval(Q(1,3),Q(2,5)))]:
                row['prepare'](Q(1,2),Q(1,3),domain);baseline['prepare'](Q(1,2),Q(1,3),domain)
                misses=p['material_inputs_cache'].misses
                current=selected(t,y);assert p['material_inputs_cache'].misses==misses,'region-bound changes must not recalculate coordinate/material jets'
                expected=baseline_inputs(t,y);assert endpoints(current)==endpoints(expected)
                current[0].c.clear();assert endpoints(selected(t,y))==endpoints(expected),'caller mutation must not poison cache'
                assert endpoints(row['field'](t,y))==endpoints(baseline['field'](t,y))
        assert len(p['material_inputs_cache'].entries)<=64


@pytest.mark.parametrize('family',['outside','inside'])
@pytest.mark.parametrize('order',[0,1,4,10])
@pytest.mark.parametrize('omega',[Q(1,100),Q(-1,10**30)])
def test_cancelled_physical_slope_keeps_closed_mixed_coefficients(order,omega,family):
    from math import comb
    from tools.opaque_physical_angle import PhysicalAngleResidual
    z=(Q(0),Q(0));one=(Q(1),Q(0));coeff=dict(n=[[(Q(0),omega),one,z],[one,z,z],[z,z,z]],v=[[(Q(1),Q(1,10**20)),z,z],[z,z,z],[z,z,z]])
    # Positive view rescaling cancels exactly. n=(u+omega*theta,1,0).
    delta=Q(1,10**25)
    def coefficient(i):
        if i%2:return Q(0)
        value=Q(1)
        for k in range(i//2):value*=Q(-3,2)-k;value/=k+1
        return value
    with backend(profile='ball',fast_conversion=True):
        if family=='inside':
            from tools.opaque_inside_angle import InsidePhysicalAngleResidual
            coeff['n'][0]=[(Q(1),omega),(Q(-1),Q(0)),z];field=InsidePhysicalAngleResidual(coeff)
        else:field=PhysicalAngleResidual(coeff)
        u=Jet(order=order,coeff={(0,0):I(0),(0,1):I(delta),**({(1,0):I(1)} if order else {})});s=Jet(0,order=order)
        got=field.radial_polynomial(u,s)
        for i in range(order+1):
            for j,exact in [(0,coefficient(i)),(1,(omega+delta)*(i+1)*coefficient(i+1))]:
                expected=I(exact);assert got[i,j].lo<=expected.lo<=expected.hi<=got[i,j].hi
                if exact:assert not got[i,j].zero


def test_cancelled_physical_slope_removes_interval_self_dependency():
    from tools.opaque_physical_angle import PhysicalAngleResidual
    z=(Q(0),Q(0));one=(Q(1),Q(0));coeff=dict(n=[[z,one,z],[one,z,z],[z,z,z]],v=[[one,z,z],[z,z,z],[z,z,z]])
    with backend(profile='ball',fast_conversion=True):
        got=PhysicalAngleResidual(coeff).radial_polynomial(Jet(I('.25','.75')),Jet(0))[0,0]
        # The exact derivative is (1+u*u)^(-3/2), positive and decreasing here.
        low=(1+I('.75')**2)**3;high=(1+I('.25')**2)**3;expected=I((1/low.sqrt()).lo,(1/high.sqrt()).hi)
        assert got.lo>=expected.lo and got.hi<=expected.hi


@pytest.mark.parametrize('family',['outside','inside'])
@pytest.mark.parametrize('axis',[0,1])
@pytest.mark.parametrize('scale',[Q(1),Q(1,10**30)])
@pytest.mark.parametrize('order',[0,4,10])
def test_shared_inverse_slope_keeps_closed_root_measure_and_mixed_motion(family,axis,scale,order):
    from tools.opaque_physical_angle import PhysicalAngleResidual
    from tools.opaque_inside_angle import InsidePhysicalAngleResidual
    z=(Q(0),Q(0));one=(Q(1),Q(0));factor=(Q(1),scale*Q(1,7))
    n=[[z,factor,factor],[one,z,z],[z,z,z]] if family=='outside' else [[factor,(-Q(1),-scale*Q(1,7)),z],[one,z,z],[z,z,z]]
    coeff=dict(n=n,v=[[one,z,z],[z,z,z],[z,z,z]])
    with backend(profile='ball',fast_conversion=True):
        field=(PhysicalAngleResidual if family=='outside' else InsidePhysicalAngleResidual)(coeff,compiled_series=True,contraction_stop=True)
        for x in (I('.25'),I('.2','.3')):
            args=[Jet(x,order=order),Jet(x,order=order)];args[axis]=Jet.variable(x,order=order);t,s=args
            t=t+t.lift(0,tangent=scale*Q(1,17));s=s+s.lift(0,tangent=-scale*Q(1,19));k=(1+t+s/3)/8
            a=(1+s if family=='outside' else t.lift(1))*(1+t.lift(0,tangent=scale*Q(1,7)))
            exact_radius=k/(a*(1-k*k).sqrt());exact_slope=a*(1-k*k)*(1-k*k).sqrt()
            radius,slope=field.inverse(k,s,lower=I(0),with_radial=True)
            for got,expected in ((radius,exact_radius),(slope,exact_slope),(1/slope,1/exact_slope)):
                for i in range(order+1):
                    for j in (0,1):
                        assert got[i,j].lo<=expected[i,j].hi and expected[i,j].lo<=got[i,j].hi
                        if not x.width:assert got[i,j].width<I('1e-25').lo
            assert not slope[0,1].zero and slope[0,0].lo>0


def test_shared_inverse_slope_rejects_unqualified_query():
    from tools.opaque_physical_angle import PhysicalAngleResidual
    z=(Q(0),Q(0));one=(Q(1),Q(0));coeff=dict(n=[[z,one,z],[one,z,z],[z,z,z]],v=[[one,z,z],[z,z,z],[z,z,z]])
    with backend(profile='ball',fast_conversion=True):
        with pytest.raises(ValueError,match='radial'):PhysicalAngleResidual(coeff).inverse(Jet('.2'),Jet('.2'),lower=I(0),with_radial=True)
        with pytest.raises(ValueError,match='radial'):PhysicalAngleResidual(coeff,compiled_series=True).inverse(Jet('.2'),Jet('.2'),lower=I(0),with_radial=1)


def test_world_saved_piece8_cost(record_property,monkeypatch):
    from tools.opaque_original_motion_reference import angular_material_cover
    axis=int(os.environ.get('GFFX_WORLD_COST_AXIS','1'));assert axis in (0,1)
    import tools.opaque_original_motion_reference as original
    factory=original.TangentCorrection;proofs=[];flux_rows=[]
    original_flux=original.material_transport
    def capture_flux(*args,**kwargs):
        value=original_flux(*args,**kwargs);flux_rows.append(value);return value
    monkeypatch.setattr(original,'material_transport',capture_flux)
    def selected(*args,**kwargs):
        value=factory(*args,**dict(kwargs,integrated_axis=axis,line_order=int(os.environ.get('GFFX_WORLD_COST_LINE_ORDER','4')),crossing_support=os.environ.get('GFFX_WORLD_COST_SUPPORT','0')=='1',root_parts=int(os.environ.get('GFFX_WORLD_COST_ROOT_PARTS','1'))));proofs.append(value);value.diagnostic_refusals=[]
        original_bounds=value.axis_bounds
        def observed(*args,**kwargs):
            try:return original_bounds(*args,**kwargs)
            except ValueError as e:
                box,active=args[:2];jets=[Jet(v,order=value.order) for v in box];jets[active]=Jet.variable(box[active],order=value.order)
                fields=value.inputs(*jets)
                value.diagnostic_refusals.append(dict(box=[str(v) for v in box],axis=active,reason=str(e),fields=[dict(value=str(q[0,0]),slope=str(q[1,0]),active_knots=[str(k) for k in value.e.cn if q[0,0].lo<=k.hi and q[0,0].hi>=k.lo]) for q in fields[:2]]));raise
        value.axis_bounds=observed;return value
    monkeypatch.setattr(original,'TangentCorrection',selected)
    from tools.opaque_tensor_smooth import rule
    spec_path=Path(os.environ['GFFX_CHECKER_COST_SELECTION']);assert hashlib.sha256(spec_path.read_bytes()).hexdigest()==os.environ['GFFX_CHECKER_COST_SELECTION_SHA']
    spec=json.loads(spec_path.read_bytes());old=Path(spec['path']);assert hashlib.sha256(old.read_bytes()).hexdigest()==spec['sha256']
    saved=json.loads(old.read_bytes());assert saved['status']=='unresolved' and saved['case'] in [dict(piece=8,motion=m,anchor=0) for m in ('world_x','world_shear_xz')]
    assert all(r in [{k:r[k] for k in ('chart','box','depth')} for r in saved['details']['cause']['last_complete_enclosure']['error_accounting']['leaf_widths']] for r in spec['regions']) and len(spec['regions'])==3
    world=Path(os.environ['GFFX_WORLD_PROFILE']);assert hashlib.sha256(world.read_bytes()).hexdigest()==os.environ['GFFX_WORLD_PROFILE_SHA'];profile=json.loads(world.read_bytes())
    start=time.monotonic();rows=[];prof=cProfile.Profile()
    def check():
        if time.monotonic()-start>=90:raise Unresolved('saved-region cost deadline')
    def stage(name,fn):
        check();begin=time.monotonic();prof.enable()
        try:answer=fn()
        finally:prof.disable()
        stages.append(dict(stage=name,seconds=time.monotonic()-begin));check();return answer
    with backend(profile='ball',fast_conversion=True,conversion_cache=os.environ.get('GFFX_WORLD_COST_CONVERSION_CACHE')=='1') as arithmetic:
        begin=time.monotonic();p,curve,cover,charts=angular_material_cover(frozen(),**saved['case'],world_profile=profile,checkpoint=check,**PROFILE,curved_transverse=True,polynomial_light=True)
        p['full_light_differential']=os.environ.get('GFFX_WORLD_COST_FULL_LIGHT','0')=='1'
        construction=time.monotonic()-begin;directional=rule(checkpoint=check)
        ga=(I(5)+2*I(Q(10,7)).sqrt()).sqrt()/6;gb=(I(5)-2*I(Q(10,7)).sqrt()).sqrt()/6;nodes=[I('.5')-ga,I('.5')-gb,I('.5'),I('.5')+gb,I('.5')+ga]
        for r in spec['regions']:
            if os.environ.get('GFFX_WORLD_COST_TRANSVERSE_LAYOUT')=='1':
                r=dict(r);a0,b0,c0,d0=map(Q,r['box']);area=(b0-a0)*(d0-c0)
                # A new geometry-only layout diagnostic; not the saved integral.
                r['box']=list(map(str,(Q(0),Q(1),Q(0),area)))
            a,b,c,d=map(Q,r['box']);box=(I(I(a).lo,I(b).hi),I(I(c).lo,I(d).hi));row=charts[r['chart']];stages=[]
            if os.environ.get('GFFX_WORLD_COST_SMOOTH')=='gauss5':row.pop('tensor_rule',None)
            stage('prepare',lambda:row['prepare']((a+b)/2,(c+d)/2,box))
            nx,ny,orders=(directional[0][0],directional[1][0],directional[2]) if row.get('tensor_rule') else (nodes,nodes,(10,10))
            def samples():
                for x in nx:
                    for y in ny:row['field'](Jet(I(a)+I(b-a)*x),Jet(I(c)+I(d-c)*y));check()
            stage('quadrature',samples)
            radial_local=os.environ.get('GFFX_WORLD_COST_RADIAL_LOCAL','0')=='1' and bool(row.get('tensor_rule'))
            if radial_local:
                from tools.opaque_tensor_smooth import radial_quarter_factors
                masses=radial_quarter_factors(checkpoint=check)
                ds=stage('radial_remainder',lambda:[row['field'](Jet.variable(I(I(a+(b-a)*Q(k,4)).lo,I(a+(b-a)*Q(k+1,4)).hi),order=orders[0]),Jet(box[1],order=orders[0])) for k in range(4)])
                radial_bounds=[[str(sum((I(v[ch][orders[0],j].bound)*mass for v,mass in zip(ds,masses)),I(0))) for j in (0,1)] for ch in range(len(ds[0]))]
            else:
                ds=stage('radial_remainder',lambda:row['field'](Jet.variable(box[0],order=orders[0]),Jet(box[1],order=orders[0])))
                radial_bounds=[[str(q[orders[0],j].bound) for j in (0,1)] for q in ds]
            def transverse():
                count=4 if row.get('tensor_rule') else 1
                return [row['field'](Jet(box[0],order=orders[1]),Jet.variable(I(I(c+(d-c)*Q(k,count)).lo,I(c+(d-c)*Q(k+1,count)).hi),order=orders[1])) for k in range(count)]
            dt=stage('transverse_remainder',transverse)
            ordinary=stage('correction',lambda:row['correction_integrator'](box));tangent=stage('transport',lambda:row['tangent_correction_integrator'](box))
            proof=proofs[r['chart']]
            rows.append(dict(flux=[str(v) for v in flux_rows[-1]['bound']],tensor=bool(row.get('tensor_rule')),proof=dict(last=proof.last_proof,fallbacks=proof.range_fallbacks,range_boxes=proof.range_proof_boxes,refusals=proof.diagnostic_refusals),region=r,stages=stages,ordinary=[[str(x.lo),str(x.hi)] for x in ordinary],tangent=[[str(x.lo),str(x.hi)] for x in tangent],radial_local=radial_local,radial=radial_bounds,transverse=[[[str(q[orders[1],j].bound) for j in (0,1)] for q in v] for v in dt]))
        stats=pstats.Stats(prof);functions=[dict(file=k[0],line=k[1],function=k[2],primitive=v[0],calls=v[1],own_seconds=v[2],cumulative_seconds=v[3]) for k,v in stats.stats.items()]
        functions.sort(key=lambda r:r['cumulative_seconds'],reverse=True)
        result=dict(arithmetic_reuse=dict(arithmetic),transverse_layout=os.environ.get('GFFX_WORLD_COST_TRANSVERSE_LAYOUT')=='1',angle_scalar_reuse=dict(hits=p['angle_light_differential'].scalar_hits,misses=p['angle_light_differential'].scalar_misses,entries=len(p['angle_light_differential'].scalar_coefficients)),full_light_differential=p['full_light_differential'],integrated_axis=axis,status='diagnostic_complete',selection_sha256=os.environ['GFFX_CHECKER_COST_SELECTION_SHA'],construction_seconds=construction,seconds=time.monotonic()-start,rows=rows,profile=functions[:180],physical_inverse=p['physical_angle'].report(),material_cache=p['material_inputs_cache'].report(),scope='Three saved regions, all original samples/correction/transport/remainder evaluations with profiling overhead; no complete-piece or reference acceptance.')
    dest=Path(os.environ['GFFX_FULL_REFERENCE']);assert not dest.exists();dest.write_text(json.dumps(result,indent=2),encoding='utf8');record_property('cost_report',str(dest))


def test_world_front_saved_corrections(record_property,monkeypatch):
    import tools.opaque_original_motion_reference as original
    spec_path=Path(os.environ['GFFX_CHECKER_COST_SELECTION']);assert hashlib.sha256(spec_path.read_bytes()).hexdigest()==os.environ['GFFX_CHECKER_COST_SELECTION_SHA']
    spec=json.loads(spec_path.read_bytes());old=Path(spec['path']);assert hashlib.sha256(old.read_bytes()).hexdigest()==spec['sha256']
    saved=json.loads(old.read_bytes());case=saved['case'];assert saved['status'] in ('qualified','unresolved') and case['motion'] in ('world_x','world_shear_xz') and case['anchor']==0 and case['piece'] in (*range(8),10)
    basis=saved['result']['basis'] if saved['status']=='qualified' else saved['details']['cause']['last_complete_enclosure']
    available=basis['error_accounting']['leaf_widths'] if basis is not None else [r for r in saved['details']['cause']['trace'] if r.get('okay') and 'box' in r]
    if 'parents' in spec:
        assert len(spec['parents'])==3
        regions=[]
        for merge in spec['parents']:
            a,b=merge['children'];parent=merge['parent'];assert a in available and b in available
            assert a['chart']==b['chart']==parent['chart'] and a['depth']==b['depth']==parent['depth']+1
            aa,bb,pp=[list(map(Q,r['box'])) for r in (a,b,parent)]
            valid=False
            for axis in (0,1):
                k=2*axis;o=2*(1-axis);left,right=sorted((aa,bb),key=lambda q:q[k])
                valid|=left[o:o+2]==right[o:o+2]==pp[o:o+2] and left[k]==pp[k] and left[k+1]==right[k]==(pp[k]+pp[k+1])/2 and right[k+1]==pp[k+1]
            assert valid;regions.append(parent)
    else:
        regions=spec['regions'];assert len(regions)==3 and all(r in available for r in regions)
    world=Path(os.environ['GFFX_WORLD_PROFILE']);assert hashlib.sha256(world.read_bytes()).hexdigest()==os.environ['GFFX_WORLD_PROFILE_SHA']
    factory=original.TangentCorrection;proofs=[];bound_rows=[]
    local=os.environ.get('GFFX_WORLD_FRONT_LOCAL','0')=='1'
    def selected(*args,**kwargs):
        result=factory(*args,**dict(kwargs,local_crossings=local,line_order=int(os.environ.get('GFFX_WORLD_FRONT_LINE_ORDER','4')),crossing_support=os.environ.get('GFFX_WORLD_FRONT_SUPPORT','0')=='1',root_parts=int(os.environ.get('GFFX_WORLD_ROOT_PARTS','1'))));proofs.append(result)
        original_bounds=result.axis_bounds
        def observed(box,axis,**kw):
            rows=original_bounds(box,axis,**kw);bound_rows.append(dict(box=list(map(str,box)),axis=axis,continuous=kw.get('transverse',True),rows=rows));return rows
        result.axis_bounds=observed;return result
    monkeypatch.setattr(original,'TangentCorrection',selected)
    cancelled=os.environ.get('GFFX_WORLD_FRONT_CANCELLED','0')=='1'
    if cancelled:
        import tools.opaque_polynomial_inputs as polynomial
        original_polynomial=polynomial.PolynomialCosines
        monkeypatch.setattr(polynomial,'PolynomialCosines',lambda *a,**k:original_polynomial(*a,**dict(k,cancelled_motion=True)))
    exact=os.environ.get('GFFX_WORLD_FRONT_EXACT','0')=='1';axis=int(os.environ.get('GFFX_WORLD_COST_AXIS','0'))
    boxes=[list(map(Q,r['box'])) for r in regions]
    rotated=os.environ.get('GFFX_WORLD_FRONT_ROTATE','0')=='1'
    if rotated:
        left,right=sorted(boxes[:2]);a,m,c,d=left;mm,b,cc,dd=right
        assert m==mm and c==cc and d==dd and m-a==b-m
        middle=(c+d)/2;boxes=[[a,b,c,middle],[a,b,middle,d]]
        assert sum((q[1]-q[0])*(q[3]-q[2]) for q in boxes)==(b-a)*(d-c)
    profiler=cProfile.Profile();profile_enabled=os.environ.get('GFFX_WORLD_FRONT_PROFILE')=='1'
    if profile_enabled:profiler.enable()
    with backend(profile='ball',fast_conversion=True):
        result=original.reference(frozen(),**case,world_profile=json.loads(world.read_bytes()),correction_order=4,conditioned_correction=True,mixed_curve=True,transverse_correction=True,correction_axis=axis,material_inputs=True,polynomial_inputs=True,exact_motion=exact,diagnostic_boxes=boxes)
        result.update(rotated=rotated,local_crossings=local,exact_motion=exact,axis=axis,selection_sha256=os.environ['GFFX_CHECKER_COST_SELECTION_SHA'],last_proof=proofs[0].last_proof,cancelled_motion=cancelled,line_order=proofs[0].line_order,crossing_support=proofs[0].crossing_support,root_parts=proofs[0].root_parts,root_partition_queries=proofs[0].root_partition_queries,bound_rows=bound_rows)
    if profile_enabled:
        profiler.disable();stats=pstats.Stats(profiler);result['profile']=sorted([dict(file=k[0],line=k[1],function=k[2],calls=v[1],own=v[2],cumulative=v[3]) for k,v in stats.stats.items()],key=lambda r:r['cumulative'],reverse=True)[:120]
    dest=Path(os.environ['GFFX_FULL_REFERENCE']);assert not dest.exists();dest.write_text(json.dumps(result,default=str,indent=2),encoding='utf8');record_property('saved_corrections',str(dest))



def test_saved_tensor_proof_path_audit(record_property,monkeypatch):
    from tools import opaque_original_motion_reference as original,opaque_inside_angle as inside
    from tools.opaque_tangent_peano import TangentCorrection
    paths=json.loads(os.environ['GFFX_TENSOR_AUDIT_PATHS']);assert 1<=len(paths)<=4
    data=frozen();world_path=Path(os.environ['GFFX_WORLD_PROFILE']);assert hashlib.sha256(world_path.read_bytes()).hexdigest()==os.environ['GFFX_WORLD_PROFILE_SHA'];world=json.loads(world_path.read_bytes())
    start=time.monotonic();reports=[];active_rows=[];leaves=[]
    containment=os.environ.get('GFFX_TENSOR_AUDIT_CONTAINMENT')=='1'
    real_call=TangentCorrection.__call__
    if containment:
        import importlib.util
        import tools.opaque_fourth_tangent as fourth
        old_path=Path(os.environ['GFFX_FROZEN_FOURTH_SOURCE']);assert hashlib.sha256(old_path.read_bytes()).hexdigest()==os.environ['GFFX_FROZEN_FOURTH_SHA']
        spec=importlib.util.spec_from_file_location('frozen_fourth_identity',old_path);old_fourth=importlib.util.module_from_spec(spec);spec.loader.exec_module(old_fourth)
    class Finished(Exception):pass
    def check():
        if time.monotonic()-start>=90:raise Unresolved('tensor path audit deadline')
    def classify(self,box):
        check();row=dict(box=list(map(str,box)),conditioned=self.conditioned,transverse=self.transverse,order=self.order,affected=False)
        if self.conditioned and self.order==4 and self.transverse:
            try:
                mass=(I(box[0].hi)-I(box[0].lo))*(I(box[1].hi)-I(box[1].lo))
                enclosure=[q[0,0]*mass for q in self.ordinary(Jet(box[0]),Jet(box[1]))];width=max(v.width for v in enclosure)
                row.update(range_width=str(width),range_return=width<=I('1e-15').lo,direct=width>I('1e-12').lo and I(box[0].width).hi<=I('.5').lo)
                row['affected']=not row['range_return'] and not row['direct']
            except ValueError as error:
                if isinstance(error,Unresolved):raise
                row.update(affected=True,range_refusal=str(error))
        if row['affected'] and containment:
            new=real_call(self,box);current=fourth.integral
            try:
                fourth.integral=old_fourth.integral;old=real_call(self,box)
            finally:fourth.integral=current
            row.update(old=list(map(str,old)),repaired=list(map(str,new)),enclosure_preserved=all(a.lo<=b.lo<=b.hi<=a.hi for a,b in zip(old,new)))
        active_rows.append(row);return [I(0),I(0)]
    def audit(field,**kwargs):
        for row in leaves:
            check();a,b,c,d=map(Q,row['box']);box=(I(I(a).lo,I(b).hi),I(I(c).lo,I(d).hi))
            chart=kwargs['chart_cover'][row['chart']] if kwargs.get('chart_cover') else kwargs
            chart['prepare']((a+b)/2,(c+d)/2,box);chart['correction_integrator'](box)
        raise Finished
    monkeypatch.setattr(TangentCorrection,'__call__',classify);monkeypatch.setattr(original,'integrate_square',audit);monkeypatch.setattr(inside,'integrate_square',audit)
    with backend(profile='ball',fast_conversion=True):
        for entry in paths:
            path=Path(entry['path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256'];saved=json.loads(path.read_bytes());assert saved['status']=='qualified'
            result=saved['result'];case=saved.get('case') or {k:result[k] for k in ('piece','motion','anchor')};case=dict(case);case.setdefault('anchor',result.get('anchor',0))
            leaves=result['basis'].get('chart_leaves',[dict(box=q,chart=0) for q in result['basis']['leaves']]);assert 1<=len(leaves)<=128;active_rows=[]
            family='outside' if case['piece'] in (8,9) else 'inside' if case['piece'] in (11,12) else 'front'
            allowed=dict(front=['correction_order','conditioned_correction','mixed_curve','transverse_correction','correction_axis','material_inputs','polynomial_inputs','line_order','crossing_support','exact_motion','refinement_axis'],outside=['tight_discriminant','algebraic_inverse','mapped_transport','exit_angle','unsquared_inverse','physical_angle','tensor_rule','compiled_series','correction_order','conditioned_correction','contraction_stop','mixed_curve','material_inputs','curved_transverse','polynomial_light','correction_axis'],inside=['polynomial_light'])
            profile={k:result[k] for k in allowed[family] if k in result}
            extra=dict(world_profile=world) if case['motion'].startswith('world_') or case['motion']=='camera_x' else {}
            fn=dict(front=original.reference,outside=original.angular_piece_reference,inside=inside.inside_piece_reference)[family]
            with pytest.raises(Finished):fn(data,**case,**profile,**extra)
            assert len(active_rows)==len(leaves)
            reports.append(dict(source=entry,case=case,leaf_count=len(leaves),affected_leaves=sum(r['affected'] for r in active_rows),containment_checked=containment,unpreserved_leaves=sum(r.get('enclosure_preserved') is False for r in active_rows),rows=active_rows))
    dest=Path(os.environ['GFFX_FULL_REFERENCE']);assert not dest.exists();dest.write_text(json.dumps(dict(status='path_audit_only',seconds=time.monotonic()-start,reports=reports),indent=2));record_property('audit',str(dest))



def test_world_front_merge_domain_preflight(record_property):
    from tools.opaque_original_motion_reference import prepare
    from tools.opaque_polynomial_inputs import PolynomialCosines
    from tools.opaque_principled_reference import PreparedPrincipled
    from tools.opaque_loss_extension import bounded_cosines
    spec_path=Path(os.environ['GFFX_MERGE_SEED']);assert hashlib.sha256(spec_path.read_bytes()).hexdigest()==os.environ['GFFX_MERGE_SEED_SHA'];spec=json.loads(spec_path.read_bytes());assert len(spec['merged'])<=32
    src=Path(spec['source']);assert hashlib.sha256(src.read_bytes()).hexdigest()==spec['source_sha256'];saved=json.loads(src.read_bytes());case=saved['case'];assert case['piece'] in (*range(8),10) and case['motion']=='world_x' and case['anchor']==0
    world=Path(os.environ['GFFX_WORLD_PROFILE']);assert hashlib.sha256(world.read_bytes()).hexdigest()==os.environ['GFFX_WORLD_PROFILE_SHA'];rows=[]
    with backend(profile='ball',fast_conversion=True):
        data=frozen();p=prepare(data,**case,world_profile=json.loads(world.read_bytes()));model=PolynomialCosines(p['fields'].coefficients);curve=PreparedPrincipled(data['budget'],1.)
        for merge in spec['merged']:
            a,b,c,d=map(Q,merge['parent']['box']);box=(I(I(a).lo,I(b).hi),I(I(c).lo,I(d).hi))
            def fields(u,s):return list(model(u,s,material_only=True)[:2])
            bounds=bounded_cosines(fields,*box,directional=lambda _:None,monotone=False,partial=False)
            counts=[int(q.lo<=curve.cn[0].hi)+sum(q.lo<=curve.cn[k+1].hi and q.hi>=curve.cn[k].lo for k in range(len(curve.cn)-1)) for q in bounds]
            rows.append(dict(parent=merge['parent'],counts=counts,admissible=max(counts)<=8))
    dest=Path(os.environ['GFFX_FULL_REFERENCE']);assert not dest.exists();dest.write_text(json.dumps(dict(status='domain_preflight_only',seed_sha256=os.environ['GFFX_MERGE_SEED_SHA'],rows=rows),indent=2));record_property('merge_domains',str(dest))


def test_saved_shear_cover_natural_range_cost(record_property,monkeypatch):
    """Measure complete ordinary interval integrals; no piece admission."""
    import tools.opaque_original_motion_reference as original
    path=Path(os.environ['GFFX_RANGE_COVER']);assert hashlib.sha256(path.read_bytes()).hexdigest()==os.environ['GFFX_RANGE_COVER_SHA']
    saved=json.loads(path.read_bytes());assert saved['case']['motion']=='world_shear_xz' and saved['case']['piece'] in (8,9)
    leaves=saved['details']['cause']['last_complete_enclosure']['error_accounting']['leaf_widths']
    world=Path(os.environ['GFFX_WORLD_PROFILE']);assert hashlib.sha256(world.read_bytes()).hexdigest()==os.environ['GFFX_WORLD_PROFILE_SHA']
    proofs=[];factory=original.TangentCorrection
    def capture(*a,**k):
        v=factory(*a,**k);proofs.append(v);return v
    monkeypatch.setattr(original,'TangentCorrection',capture)
    start=time.monotonic()
    def check():
        if time.monotonic()-start>=90:raise Unresolved('complete-cover range diagnostic deadline')
    rows=[]
    with backend(profile='ball',fast_conversion=True):
        p,curve,cover,charts=original.angular_material_cover(frozen(),**saved['case'],world_profile=json.loads(world.read_bytes()),checkpoint=check,**PROFILE,curved_transverse=True,polynomial_light=True)
        construction=time.monotonic()-start
        for leaf in leaves:
            check();a,b,c,d=map(Q,leaf['box']);box=(I(I(a).lo,I(b).hi),I(I(c).lo,I(d).hi));row=charts[leaf['chart']];begin=time.monotonic()
            row['prepare']((a+b)/2,(c+d)/2,box);mass=(I(b)-I(a))*(I(d)-I(c));bounds=[q[0,0]*mass for q in proofs[leaf['chart']].ordinary(Jet(box[0]),Jet(box[1]))]
            extra={}
            if os.environ.get('GFFX_RANGE_SMOOTH')=='1':
                full=row['field'](Jet(box[0]),Jet(box[1]));total=[[q[0,j]*mass+(bounds[j] if k==1 else I(0)) for j in (0,1)] for k,q in enumerate(full)]
                extra['smooth_plus_ordinary_widths']=[[str(q.width) for q in ch] for ch in total]
                extra['excludes_boundary_flux']=True
            rows.append(dict(region={k:leaf[k] for k in ('chart','box','depth')},bounds=[[str(q.lo),str(q.hi)] for q in bounds],widths=[str(q.width) for q in bounds],seconds=time.monotonic()-begin,**extra))
    result=dict(status='diagnostic_complete',source=str(path),source_sha256=os.environ['GFFX_RANGE_COVER_SHA'],construction_seconds=construction,seconds=time.monotonic()-start,rows=rows,scope='Complete natural-range ordinary residual enclosures only. Smooth integral, boundary flux and whole-piece acceptance not evaluated.')
    dest=Path(os.environ['GFFX_FULL_REFERENCE']);assert not dest.exists();dest.write_text(json.dumps(result,indent=2));record_property('range_cover',str(dest))


def test_curved_full_samples_seed_identical_raw_material_fields(monkeypatch):
    import importlib.util
    import tools.opaque_original_motion_reference as current
    from tools.opaque_peano import gauss5
    source=Path(os.environ['GFFX_FROZEN_FRONT_SOURCE']);assert hashlib.sha256(source.read_bytes()).hexdigest()==os.environ['GFFX_FROZEN_FRONT_SHA']
    spec=importlib.util.spec_from_file_location('frozen_curved_sample_reference',source);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    world=Path(os.environ['GFFX_WORLD_PROFILE']);assert hashlib.sha256(world.read_bytes()).hexdigest()==os.environ['GFFX_WORLD_PROFILE_SHA']
    def run(module):
        proofs=[];factory=module.TangentCorrection
        def capture(*a,**k):v=factory(*a,**k);proofs.append(v);return v
        monkeypatch.setattr(module,'TangentCorrection',capture)
        p,curve,cover,charts=module.angular_material_cover(frozen(),piece=8,motion='world_shear_xz',anchor=0,world_profile=json.loads(world.read_bytes()),**PROFILE,curved_transverse=True,polynomial_light=True)
        nodes,_=gauss5();values=[];material_misses=0
        for index,(a,b,c,d) in [(0,(Q(0),Q(1),Q(0),Q(1,2))),(54,(Q(0),Q(1),Q(0),Q(1))),(65,(Q(0),Q(1,2),Q(0),Q(1))),(70,(Q(0),Q(1,2),Q(0),Q(1,2)))]:
            row=charts[index];box=(I(I(a).lo,I(b).hi),I(I(c).lo,I(d).hi));row['prepare']((a+b)/2,(c+d)/2,box)
            points=[(Jet(I(a)+I(b-a)*u),Jet(I(c)+I(d-c)*v)) for u in nodes for v in nodes]
            for x,y in points:values.extend(row['field'](x,y))
            before=p['material_inputs_cache'].misses
            for x,y in points:values.extend(proofs[index].ordinary(x,y))
            material_misses+=p['material_inputs_cache'].misses-before
        return [{k:(v.lo,v.hi) for k,v in q.c.items()} for q in values],material_misses
    with backend(profile='ball',fast_conversion=True):
        expected,before=run(old);actual,after=run(current)
        assert actual==expected
        assert after==0 and before>=72,(before,after)


@pytest.mark.parametrize('omega',[Q(1),-Q(1,2**60)])
@pytest.mark.parametrize('uncertain',[False,True])
def test_scalar_implicit_inverse_preserves_closed_motion_without_series(monkeypatch,omega,uncertain):
    import flint
    from tools.opaque_arb_jets import backend
    from tools.opaque_physical_angle import PhysicalAngleResidual
    from tools.opaque_moving_reference import P
    from tools.opaque_compiled_inverse import solve
    coeff={'n':[[P(0),P([1,omega/7]),P([1,omega/7])],[P(0)]*3,[P(1),P(0),P(0)]],'v':[[P(1),P(0),P(0)],[P(0)]*3,[P(0)]*3]}
    with backend(profile='ball',fast_conversion=True):
        field=PhysicalAngleResidual(coeff,compiled_series=True)
        t=Jet(I('.2','.3') if uncertain else I('.25'))+Jet(0,tangent=omega/17)
        s=Jet(I('.2','.3') if uncertain else I('.25'))+Jet(0,tangent=-omega/19)
        k=(1+t)/8;factor=(1+s)*(1+s.lift(0,tangent=omega/7));expected=k/(factor*(1-k*k).sqrt());radial=factor*(1-k*k)*(1-k*k).sqrt()
        root=expected[0,0];B=field.scalar_polynomials[1](root,s[0,0]);slope=B.sqrt()*field.radial(Jet(root),s)[0,0]
        field.coefficients(s)  # Coordinate preparation is separate from the implicit solve.
        old=flint.arb_series;calls=[]
        def spy(*a,**k):calls.append(1);return old(*a,**k)
        monkeypatch.setattr(flint,'arb_series',spy)
        rows=solve(field,root,s,k,slope,with_radial=True)
        assert not calls,'order-zero implicit proof must not construct formal spatial series'
        for got,want in zip(rows,(expected,radial)):
            for j in (0,1):
                a,b=got[0,j],want[0,j];assert a.lo<=b.hi and b.lo<=a.hi
                if not uncertain:assert a.width<I('1e-25').lo
        assert not rows[0][0,1].zero
