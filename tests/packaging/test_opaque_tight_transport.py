from fractions import Fraction as Q
from pathlib import Path
import json,os,time
from tools.opaque_shading_intervals import I,Jet,Unresolved

def test_remaining_combined_terms_diagnostic(record_property,monkeypatch):
    from tools.opaque_arb_jets import backend
    import tools.opaque_original_motion_reference as m
    from tools.opaque_tangent_peano import TangentCorrection
    from tests.packaging.test_opaque_motion_tangent import frozen
    from tools.opaque_tensor_smooth import rule
    start=time.monotonic();rows=[];current={}
    def check():
        if time.monotonic()-start>=85:raise Unresolved('remaining term diagnostic deadline')
    import tools.opaque_tangent_peano as tp
    residual=tp.residual_integral
    def split_errors(*args,**kw):
        return residual(*args,**kw,error_breakdown=lambda v:current.update(breakdown=v))
    monkeypatch.setattr(tp,'residual_integral',split_errors)
    old=TangentCorrection.__call__;oldflux=m.material_transport
    def correction(self,box):
        t=time.monotonic();result=old(self,box);current['ordinary']=dict(value=str(result[0]),motion=str(result[1]),motion_width=str(result[1].width),seconds=time.monotonic()-t,fallbacks=self.range_fallbacks);return result
    def flux(*args,**kw):
        t=time.monotonic();result=oldflux(*args,**kw);current['flux']=dict(result=result,seconds=time.monotonic()-t);return result
    monkeypatch.setattr(TangentCorrection,'__call__',correction);monkeypatch.setattr(m,'material_transport',flux)
    with backend(profile='ball'):
        p,curve,cover,charts=m.angular_material_cover(frozen(),tight_discriminant=True,algebraic_inverse=True,mapped_transport=True,exit_angle=True,unsquared_inverse=True,physical_angle=True,tensor_rule='gauss3x8',checkpoint=check)
        for index,box in [(68,(Q(1,2),1,0,1)),(66,(Q(1,2),1,0,1)),(63,(0,1,Q(1,2),1)),(62,(0,1,Q(1,2),1)),(64,(Q(1,2),Q(3,4),0,1)),(71,(0,1,0,Q(1,4)))]:
            current={};a,b,c,d=map(Q,box);bounds=(I(I(a).lo,I(b).hi),I(I(c).lo,I(d).hi));row=charts[index];row['prepare']((a+b)/2,(c+d)/2,bounds)
            row['correction_integrator'](bounds);answer=row['tangent_correction_integrator'](bounds);rows.append(dict(chart=index,box=list(map(str,box)),terms=current,total_motion=str(answer[1]),width=str(answer[1].width)));check()
        coeffs=[];_,_,orders,factors=rule(checkpoint=check)
        for index in (53,54,55):
            row=charts[index];row['prepare'](Q(1,2),Q(1,2),(I(0,1),I(0,1)));axes=[]
            for axis,order in enumerate(orders):
                args=[Jet(I(0,1),order=order),Jet(I(0,1),order=order)];args[axis]=Jet.variable(I(0,1),order=order);values=row['field'](*args);axes.append([[str(v[order,j]) for j in (0,1)] for v in values]);check()
            coeffs.append(dict(chart=index,axes=axes))
    path=Path(os.environ['GFFX_FULL_REFERENCE']);assert not path.exists();path.write_text(json.dumps(dict(rows=rows,smooth=coeffs,seconds=time.monotonic()-start,scope='Separates ordinary correction, conservative jump-flux uncertainty and smooth derivative bounds. Diagnostic only.'),default=str,indent=2),encoding='utf8');record_property('remaining_terms',str(path))





import pytest
@pytest.mark.parametrize('order',[0,3])
@pytest.mark.parametrize('live',[(False,False),(True,False),(False,True),(True,True)])
def test_ordinary_exact_zero_work_preserves_all_mixed_endpoints(order,live,monkeypatch):
    from types import SimpleNamespace
    import tools.opaque_tangent_peano as m
    from tools.opaque_arb_jets import backend
    with backend(profile='ball'):
        x=Jet.variable(I('.2','.4'),order=order) if order else Jet(I('.2','.4'))
        x=x+x.lift(0,tangent=Q(1,2**60));y=x*Q(2,3)+1;jac=1+x*y
        choices={id(x):[x*Q(1,2**60),x*x*Q(-1,2**62)] if live[0] else [x.lift(0)]*2,id(y):[y*Q(1,2**60),y*y*Q(-1,2**62)] if live[1] else [y.lift(0)]*2}
        calls=[]
        def at(q,i):calls.append(i);return [1+q,2+q*q]
        e=SimpleNamespace(mean=[I(1),I(2)],pi=I(3),indices=[0,1],at=at)
        monkeypatch.setattr(m,'segmented_mixed',lambda e,q,i,**kw:choices[id(q)])
        correction=m.TangentCorrection(e,lambda *a:None)
        dv,dl=choices[id(x)],choices[id(y)];av,bl=at(x,0),at(y,1);mix=correction.mix
        expected=y*(mix(dv)*mix(bl)+mix(av)*mix(dl)+mix(dv)*mix(dl))*jac/(e.pi*correction.mean);calls.clear()
        actual=correction.ordinary_inputs(x,(x,y,x,jac))
        assert all((actual[i,j].lo,actual[i,j].hi)==(expected[i,j].lo,expected[i,j].hi) for i in range(order+1) for j in (0,1))
        assert len(calls)==sum(live)
        if any(live):assert not actual[0,1].zero


def test_light_differential_reuses_only_current_scalar_box(monkeypatch):
    from tools.opaque_angle_light import AngleLightDifferential
    from tools.opaque_arb_jets import backend
    from tools.opaque_original_motion_reference import prepare
    from tests.packaging.test_opaque_motion_tangent import frozen
    import tools.opaque_shading_intervals as scalar
    with backend(profile='ball'):
        p=prepare(frozen(),piece=8,motion='normal_tilt',horizon_power=4,tight_discriminant=True)
        differential=AngleLightDifferential(p['fields'].coefficients);original=differential.prepared;calls=[]
        def counted(fn):
            def f(t,s):calls.append(1);return fn(t,s)
            return f
        differential.prepared=tuple(counted(f) for f in original)
        for precision in (60,75,60):
            scalar.PRECISION=precision
            for a in ('.5','.6'):
                t=Jet.variable(I(a),order=3);s=Jet(I('.2'),order=3)+t.lift(0,tangent=Q(1,2**60));target=t*Q(1,4)+s*Q(1,9)+t.lift(0,tangent=Q(3,11));start=len(calls)
                for key in ((1,0),(0,1)):
                    N,V,Pt,Qt,Hs,Htheta=[f(t[0,0],s[0,0]) for f in original]
                    expected=Pt*(V**3).sqrt()/(I(6).sqrt()*Qt)*target[key]+Hs/((6*N**3).sqrt()*Qt)*s[key]
                    if key==(0,1):expected=expected+Htheta/((6*N**3).sqrt()*Qt)
                    got=differential.bound(t,s,target,key);assert (got.lo,got.hi)==(expected.lo,expected.hi)
                assert len(calls)-start==6


def test_localized_tensor_contains_independent_exact_controls(monkeypatch):
    import tools.opaque_square_reference as m
    from tests.packaging.test_opaque_tensor_smooth import test_tensor_smooth_compiled_exact_controls
    old=m.integrate_square
    def local(*args,**kw):
        if kw.get('tensor_rule')=='gauss3x8':kw['tensor_rule']='gauss3x8-local2'
        if kw.get('chart_cover'):kw['chart_cover']=[dict(row,**({'tensor_rule':'gauss3x8-local2'} if row.get('tensor_rule') else {})) for row in kw['chart_cover']]
        return old(*args,**kw)
    monkeypatch.setattr(m,'integrate_square',local);test_tensor_smooth_compiled_exact_controls()


def test_localized_tensor_saved_smooth_regions(record_property):
    from tools.opaque_original_motion_reference import angular_material_cover
    from tools.opaque_square_reference import integrate_square
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    start=time.monotonic();rows=[]
    def check():
        if time.monotonic()-start>=60:raise Unresolved('localized tensor regions deadline')
    with backend(profile='ball'):
        p,curve,cover,charts=angular_material_cover(frozen(),tight_discriminant=True,algebraic_inverse=True,mapped_transport=True,exit_angle=True,unsquared_inverse=True,physical_angle=True,tensor_rule='gauss3x8-local2',checkpoint=check)
        for index in (53,54,55):
            row=charts[index];t=time.monotonic();result=integrate_square(row['field'],prepare=row['prepare'],correction_integrator=row['correction_integrator'],tangent_correction_integrator=row['tangent_correction_integrator'],tensor_rule=row['tensor_rule'],tolerance='5e-13',max_cells=1,checkpoint=check)
            rows.append(dict(chart=index,seconds=time.monotonic()-t,result=result))
    path=Path(os.environ['GFFX_FULL_REFERENCE']);assert not path.exists();path.write_text(json.dumps(dict(rows=rows,seconds=time.monotonic()-start,scope='Three complete one-cell local checks; no piece acceptance.'),default=str,indent=2),encoding='utf8');record_property('localized_tensor_regions',str(path))


def test_localized_tensor_fresh_complete_piece(record_property):
    from tools.opaque_original_motion_reference import angular_piece_reference
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    failure=None;rows=[]
    with backend(profile='ball'):
        try:result=angular_piece_reference(frozen(),tight_discriminant=True,algebraic_inverse=True,mapped_transport=True,exit_angle=True,unsquared_inverse=True,physical_angle=True,tensor_rule='gauss3x8-local2',progress=rows.append)
        except ValueError as error:failure=str(error);result=dict(status='unresolved',failure=getattr(error,'details',str(error)))
    result['progress']=rows;path=Path(os.environ['GFFX_FULL_REFERENCE']);assert not path.exists();path.write_text(json.dumps(result,default=str,indent=2),encoding='utf8');record_property('complete_piece',str(path))
    assert failure is None,result
    assert result['basis']['chart_count']==72 and result['basis']['cells']<=256
    assert all(v.width<=I('2e-12').lo for v in result['value']+result['derivative'])


def test_redundant_work_controlled_cost_and_exact_piece_endpoints(record_property,monkeypatch):
    import importlib.util
    import tools.opaque_tangent_peano as tp
    import tools.opaque_angle_light as light
    from tools.opaque_original_motion_reference import angular_material_cover
    from tools.opaque_square_reference import integrate_square
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    base=Path(os.environ['GFFX_FULL_REFERENCE']).parent.parent/'opaque148-20260929/complete-piece1-source/tools';old=[]
    for n in ('opaque_tangent_peano','opaque_angle_light'):
        spec=importlib.util.spec_from_file_location('frozen_'+n,base/(n+'.py'));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);old.append(module)
    new=(tp.TangentCorrection.ordinary_inputs,light.AngleLightDifferential.bound);start=time.monotonic();rows=[]
    def check():
        if time.monotonic()-start>=60:raise Unresolved('controlled reuse timing deadline')
    with backend(profile='ball'):
        for selected in (False,True,True,False):
            monkeypatch.setattr(tp.TangentCorrection,'ordinary_inputs',new[0] if selected else old[0].TangentCorrection.ordinary_inputs)
            monkeypatch.setattr(light.AngleLightDifferential,'bound',new[1] if selected else old[1].AngleLightDifferential.bound)
            p,curve,cover,charts=angular_material_cover(frozen(),tight_discriminant=True,algebraic_inverse=True,mapped_transport=True,exit_angle=True,unsquared_inverse=True,physical_angle=True,tensor_rule='gauss3x8-local2',checkpoint=check)
            row=charts[63];t=time.monotonic()
            try:
                q=integrate_square(row['field'],prepare=row['prepare'],correction_integrator=row['correction_integrator'],tangent_correction_integrator=row['tangent_correction_integrator'],tensor_rule=row['tensor_rule'],max_cells=1,tolerance='5e-13',checkpoint=check)
                result=dict(value=[[str(v.lo),str(v.hi)] for v in q['value']],derivative=[[str(v.lo),str(v.hi)] for v in q['derivative']])
            except Unresolved as error:
                assert error.details['reason']=='cell budget';q=error.details['last_complete_enclosure'];result=dict(value=q['value'],derivative=q['derivative'])
            rows.append(dict(reuse=selected,seconds=time.monotonic()-t,result=result))
    assert all(r['result']==rows[0]['result'] for r in rows)
    path=Path(os.environ['GFFX_FULL_REFERENCE']);assert not path.exists();path.write_text(json.dumps(dict(rows=rows,seconds=time.monotonic()-start,scope='Fresh A-B-B-A full local evaluation including both correction and smooth work; identical outward endpoints. Not a whole-piece runtime guarantee.'),indent=2),encoding='utf8');record_property('controlled_reuse',str(path))


def test_gradient_correction_selects_axis_without_changing_acceptance():
    from tools.opaque_square_reference import integrate_square
    def tangent(box):
        h=[I(v.hi)-I(v.lo) for v in box];error=I('1e-10')*h[0]*h[1]**4
        return [I(-error.hi,error.hi)]
    tangent.preferred_split_axis=1
    result=integrate_square(lambda x,y:[x.lift(0)],tangent_correction_integrator=tangent,max_cells=3,tolerance='2e-11')
    assert result['cells']==3 and result['derivative'][0].width<=I('4e-11').lo
    assert all(row[:2]==['0','1'] for row in result['leaves'])
    assert all(row[2:] in (['0','1/2'],['1/2','1']) for row in result['leaves'])
    # No hint, malformed hint and an exhausted budget cannot waive the bound.
    for hint in (None,True,2):
        tangent.preferred_split_axis=hint
        with pytest.raises(Unresolved):integrate_square(lambda x,y:[x.lift(0)],tangent_correction_integrator=tangent,max_cells=3,tolerance='2e-11')
    tangent.preferred_split_axis=1
    with pytest.raises(Unresolved):integrate_square(lambda x,y:[x.lift(0)],tangent_correction_integrator=tangent,max_cells=1,tolerance='2e-11')


def test_tangent_centered_remainder_uses_zero_kernel_moment():
    from tools.opaque_peano import residual_integral
    from tools.opaque_arb_jets import backend
    with backend(profile='ball'):
        for sign in (-1,1):
            for scale in (Q(1),Q(1,2**60)):
                def f(x,y):return [sign*scale*x**3]
                def bounds(box,axis):
                    d=I(6*sign*scale if axis==0 else 0)
                    return [dict(third=I(d.bound),third_range=d,jumps=[I(0)]*3)]
                q=residual_integral(f,(I(0,1),I(0,1)),bounds)[0];want=I(sign*scale/4)
                assert q.lo<=want.lo<=want.hi<=q.hi
                assert q.width<(I(scale)*I('1e-40')).hi
        def bad(box,axis):return [dict(third=I(1),third_range=I(-2,2),jumps=[I(0)]*3)]
        with pytest.raises(ValueError):residual_integral(lambda x,y:[x],(I(0,1),I(0,1)),bad)


def test_tangent_centered_remainder_keeps_closed_log_motion(monkeypatch):
    import tools.opaque_tangent_peano as m
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_tangent_peano import test_tangent_peano_contains_closed_piecewise_log_integral,test_tangent_peano_preserves_zero_parameter_direction
    old=m.TangentCorrection.__init__
    def init(self,*a,**kw):kw['centered']=True;return old(self,*a,**kw)
    monkeypatch.setattr(m.TangentCorrection,'__init__',init)
    with backend(profile='ball'):
        for degree in (1,2,3):
            for axis in (0,1):
                for reverse in (False,True):test_tangent_peano_contains_closed_piecewise_log_integral(degree,axis,reverse)
        test_tangent_peano_preserves_zero_parameter_direction()


def test_centered_tangent_original_saved_cells(record_property,monkeypatch):
    import tools.opaque_tangent_peano as m
    from tools.opaque_original_motion_reference import angular_material_cover
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    init=m.TangentCorrection.__init__;rows=[];start=time.monotonic()
    def check():
        if time.monotonic()-start>=60:raise Unresolved('centered correction comparison deadline')
    with backend(profile='ball'):
        for centered in (False,True):
            def selected(self,*a,**kw):kw['centered']=centered;return init(self,*a,**kw)
            monkeypatch.setattr(m.TangentCorrection,'__init__',selected)
            p,curve,cover,charts=angular_material_cover(frozen(),tight_discriminant=True,algebraic_inverse=True,mapped_transport=True,exit_angle=True,unsquared_inverse=True,physical_angle=True,tensor_rule='gauss3x8-local2',checkpoint=check)
            for index,box in [(68,(Q(1,2),1,0,1)),(66,(Q(1,2),1,0,1)),(63,(0,1,Q(1,2),1)),(62,(0,1,Q(1,2),1)),(64,(Q(1,2),Q(3,4),0,1)),(71,(0,1,0,Q(1,4)))]:
                a,b,c,d=map(Q,box);bounds=(I(I(a).lo,I(b).hi),I(I(c).lo,I(d).hi));row=charts[index];row['prepare']((a+b)/2,(c+d)/2,bounds);t=time.monotonic()
                result=[row['correction_integrator'](bounds)[1],row['tangent_correction_integrator'](bounds)[1]]
                rows.append(dict(centered=centered,chart=index,box=list(map(str,box)),seconds=time.monotonic()-t,result=[[str(v.lo),str(v.hi)] for v in result],widths=[str(v.width) for v in result]));check()
    from decimal import Decimal as D
    for old,new in zip(rows[:6],rows[6:]):
        assert old['chart']==new['chart']
        assert all(D(a)<=D(c)<=D(d)<=D(b) for (a,b),(c,d) in zip(old['result'],new['result']))
    path=Path(os.environ['GFFX_FULL_REFERENCE']);assert not path.exists();path.write_text(json.dumps(dict(rows=rows,seconds=time.monotonic()-start,scope='Correction only, every jump retained, same samples; no complete-piece acceptance.'),indent=2),encoding='utf8');record_property('centered_correction',str(path))


def test_direct_tangent_range_saved_cells(record_property,monkeypatch):
    import tools.opaque_tangent_peano as m
    from tools.opaque_original_motion_reference import angular_material_cover
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    original=m.TangentCorrection.__call__;rows=[];current={};start=time.monotonic()
    def check():
        if time.monotonic()-start>=70:raise Unresolved('direct range comparison deadline')
    def compared(self,box):
        t=time.monotonic();raw=self.ordinary(Jet(box[0]),Jet(box[1]));mass=(I(box[0].hi)-I(box[0].lo))*(I(box[1].hi)-I(box[1].lo));natural=[v[0,0]*mass for v in raw];nt=time.monotonic()-t
        t=time.monotonic();prior=original(self,box);pt=time.monotonic()-t
        current.update(natural=[[str(v.lo),str(v.hi)] for v in natural],prior=[[str(v.lo),str(v.hi)] for v in prior],natural_widths=[str(v.width) for v in natural],prior_widths=[str(v.width) for v in prior],natural_seconds=nt,prior_seconds=pt)
        assert all(max(a.lo,b.lo)<=min(a.hi,b.hi) for a,b in zip(natural,prior));return prior
    monkeypatch.setattr(m.TangentCorrection,'__call__',compared)
    with backend(profile='ball'):
        p,curve,cover,charts=angular_material_cover(frozen(),tight_discriminant=True,algebraic_inverse=True,mapped_transport=True,exit_angle=True,unsquared_inverse=True,physical_angle=True,tensor_rule='gauss3x8-local2',checkpoint=check)
        cases=[(i,(0,1,0,1)) for i in range(60,72)]+[(68,(Q(1,2),1,0,1)),(66,(Q(1,2),1,0,1)),(63,(0,1,Q(1,2),1)),(62,(0,1,Q(1,2),1)),(64,(Q(1,2),Q(3,4),0,1)),(71,(0,1,0,Q(1,4)))]
        for index,box in cases:
            current={};a,b,c,d=map(Q,box);bounds=(I(I(a).lo,I(b).hi),I(I(c).lo,I(d).hi));row=charts[index];row['prepare']((a+b)/2,(c+d)/2,bounds);row['correction_integrator'](bounds);rows.append(dict(chart=index,box=list(map(str,box)),**current));check()
    path=Path(os.environ['GFFX_FULL_REFERENCE']);assert not path.exists();path.write_text(json.dumps(dict(rows=rows,seconds=time.monotonic()-start,scope='Intersection/cost diagnostic only; no integral acceptance.'),indent=2),encoding='utf8');record_property('direct_tangent_range',str(path))


def test_fresh_refinement_schedule_checks_cover_and_charges_ancestors():
    from tools.opaque_refinement_schedule import prepare_schedule
    charts=[dict(depth=1),dict(depth=1)]
    leaves=[dict(chart=0,box=['0','1/2','0','1'],depth=2),dict(chart=0,box=['1/2','1','0','1'],depth=2),dict(chart=1,box=['0','1','0','1'],depth=1)]
    pending,count=prepare_schedule(charts,leaves,max_cells=5,max_depth=2)
    assert count==2 and len(pending)==3 and count+len(pending)==5
    for bad in (leaves[:-1],leaves+[leaves[0]], [dict(leaves[0],depth=1),*leaves[1:]], [dict(leaves[0],box=['0','2/3','0','1']),*leaves[1:]], [dict(leaves[0],chart=True),*leaves[1:]]):
        with pytest.raises(ValueError):prepare_schedule(charts,bad,max_cells=5,max_depth=2)
    with pytest.raises(ValueError):prepare_schedule(charts,leaves,max_cells=4,max_depth=2)
    with pytest.raises(ValueError):prepare_schedule(charts,leaves,max_cells=5,max_depth=1)


def test_fresh_refinement_schedule_recomputes_all_fields_and_remainders():
    from tools.opaque_square_reference import integrate_square
    calls=[]
    def f(x,y):calls.append(x.order);return [(x*x+y)*(1+x.lift(0,tangent=Q(-2,3)))+Q(1,2**60)]
    charts=[dict(field=f,depth=0)]
    leaves=[dict(chart=0,box=['0','1/2','0','1'],depth=1),dict(chart=0,box=['1/2','1','0','1'],depth=1)]
    q=integrate_square(None,chart_cover=charts,initial_leaves=leaves,max_cells=3,max_depth=1)
    assert q['cells']==3 and q['regions']==2 and q['quadrature_samples']==50
    assert calls.count(0)==50 and calls.count(10)==4
    for got,want in zip([q['value'][0],q['derivative'][0]],[I(Q(5,6)+Q(1,2**60)),I(Q(-5,9))]):assert got.lo<=want.lo<=want.hi<=got.hi
    with pytest.raises(ValueError):integrate_square(None,chart_cover=charts,initial_leaves=leaves,max_cells=2)


def test_fresh_scheduled_complete_piece(record_property,monkeypatch):
    import hashlib
    import tools.opaque_original_motion_reference as m
    path=Path(os.environ['GFFX_FULL_REFERENCE']).parent/'seed-layout1.json'
    assert hashlib.sha256(path.read_bytes()).hexdigest()=="c3e10ab8a5ed5391ce43fc483b0485af63b69d67a27a78d80149b8bb5c0390ed"
    leaves=json.loads(path.read_bytes())['leaves'];original=m.angular_piece_reference
    def scheduled(*a,**kw):return original(*a,**kw,initial_leaves=leaves)
    monkeypatch.setattr(m,'angular_piece_reference',scheduled)
    test_localized_tensor_fresh_complete_piece(record_property)


@pytest.mark.parametrize('order',[0,3,10])
@pytest.mark.parametrize('axis',[0,1])
@pytest.mark.parametrize('scale',[Q(1),Q(1,2**60)])
def test_full_light_differential_keeps_closed_mixed_coefficients(order,axis,scale):
    from tools.opaque_angle_light import AngleLightDifferential
    from tools.opaque_moving_reference import P
    from tools.opaque_arb_jets import backend
    coeff={'n':[[P(0),P([1,scale*Q(1,7)]),P([1,scale*Q(1,7)])],[P(0)]*3,[P(1),P(0),P(0)]],'v':[[P(1),P(0),P(0)],[P(0)]*3,[P(0)]*3]}
    with backend(profile='ball'):
        f=AngleLightDifferential(coeff)
        t,s=[Jet(I('.25'),order=order),Jet(I('.2'),order=order)]
        if order:
            if axis==0:t=Jet.variable(I('.25'),order=order)
            else:s=Jet.variable(I('.2'),order=order)
        t+=t.lift(0,tangent=scale*Q(1,17));s+=s.lift(0,tangent=-scale*Q(1,19))
        z=t*(1+s)*(1+t.lift(0,tangent=scale*Q(1,7)));target=z/(1+z*z).sqrt();expected=(z+2)/((1+z*z).sqrt()*I(6).sqrt())
        rows=f.coefficient_bounds(t,s,target)
        assert set(rows)=={(i,j) for i in range(order+1) for j in (0,1) if i+j}
        for key,got in rows.items():
            want=expected[key];assert max(got.lo,want.lo)<=min(got.hi,want.hi)
            assert got.width<I('1e-35').lo
        assert not rows[0,1].zero


def test_full_light_original_cells(record_property):
    from tools.opaque_original_motion_reference import angular_material_cover
    from tools.opaque_square_reference import integrate_square
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    from decimal import Decimal as D
    start=time.monotonic();rows=[]
    def check():
        if time.monotonic()-start>=70:raise Unresolved('full differential comparison deadline')
    with backend(profile='ball'):
        for index in (46,53,57,63,64,68):
            for enabled in (False,True):
                p,c,cover,charts=angular_material_cover(frozen(),tight_discriminant=True,algebraic_inverse=True,mapped_transport=True,exit_angle=True,unsquared_inverse=True,physical_angle=True,tensor_rule='gauss3x8-local2',checkpoint=check);p['full_light_differential']=enabled;row=charts[index];t=time.monotonic()
                try:
                    z=integrate_square(row['field'],prepare=row['prepare'],correction_integrator=row['correction_integrator'],tangent_correction_integrator=row['tangent_correction_integrator'],tensor_rule=row.get('tensor_rule'),tolerance='5e-13',max_cells=1,checkpoint=check);q=dict(value=[[str(v.lo),str(v.hi)] for v in z['value']],derivative=[[str(v.lo),str(v.hi)] for v in z['derivative']],accounting=z['error_accounting'])
                except Unresolved as error:
                    assert error.details['reason']=='cell budget';z=error.details['last_complete_enclosure'];assert z is not None;q=dict(value=z['value'],derivative=z['derivative'],accounting=z['error_accounting'])
                rows.append(dict(chart=index,enabled=enabled,seconds=time.monotonic()-t,**q))
    for a,b in zip(rows[::2],rows[1::2]):
        for key in ('value','derivative'):
            assert all(max(D(x),D(z))<=min(D(y),D(w)) for (x,y),(z,w) in zip(a[key],b[key]))
    path=Path(os.environ['GFFX_FULL_REFERENCE']);assert not path.exists();path.write_text(json.dumps(dict(rows=rows,seconds=time.monotonic()-start,scope='Closed controls are independent; original old/new interval overlap is only consistency and cost evidence.'),indent=2),encoding='utf8');record_property('full_differential',str(path))


def test_compiled_inverse_keeps_independent_mixed_roots(monkeypatch):
    from tools.opaque_compiled_inverse import solve
    from tools.opaque_physical_angle import PhysicalAngleResidual
    from tests.packaging.test_opaque_algebraic_angle_inverse import test_physical_angle_inverse_keeps_closed_moving_radius_and_jacobian
    original=PhysicalAngleResidual.implicit_coefficients;calls=[]
    def compiled(self,root,s,k,slope):calls.append(1);return solve(self,root,s,k,slope)
    monkeypatch.setattr(PhysicalAngleResidual,'implicit_coefficients',compiled)
    for axis in (0,1):
        for scale in (Q(1),Q(1,2**60)):test_physical_angle_inverse_keeps_closed_moving_radius_and_jacobian(axis,scale)
    assert len(calls)==8


def test_compiled_inverse_keeps_complete_moving_measure(monkeypatch):
    from tools.opaque_compiled_inverse import solve
    from tools.opaque_physical_angle import PhysicalAngleResidual
    from tests.packaging.test_opaque_algebraic_angle_inverse import test_physical_angle_cover_retains_complete_moving_measure
    monkeypatch.setattr(PhysicalAngleResidual,'implicit_coefficients',lambda self,r,s,k,d:solve(self,r,s,k,d))
    for scale in (Q(1),Q(1,2**60)):test_physical_angle_cover_retains_complete_moving_measure(scale)


def test_compiled_original_cells(record_property):
    from tools.opaque_original_motion_reference import angular_material_cover
    from tools.opaque_square_reference import integrate_square
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    from decimal import Decimal as D
    start=time.monotonic();rows=[]
    def check():
        if time.monotonic()-start>=70:raise Unresolved('compiled inverse comparison deadline')
    with backend(profile='ball'):
        for index in (46,57,63,68):
            for enabled in (False,True):
                p,c,cover,charts=angular_material_cover(frozen(),tight_discriminant=True,algebraic_inverse=True,mapped_transport=True,exit_angle=True,unsquared_inverse=True,physical_angle=True,tensor_rule='gauss3x8-local2',compiled_series=enabled,checkpoint=check);row=charts[index];t=time.monotonic()
                try:
                    z=integrate_square(row['field'],prepare=row['prepare'],correction_integrator=row['correction_integrator'],tangent_correction_integrator=row['tangent_correction_integrator'],tensor_rule=row.get('tensor_rule'),tolerance='5e-13',max_cells=1,checkpoint=check);q=dict(value=[[str(v.lo),str(v.hi)] for v in z['value']],derivative=[[str(v.lo),str(v.hi)] for v in z['derivative']],accounting=z['error_accounting'])
                except Unresolved as error:
                    assert error.details['reason']=='cell budget';z=error.details['last_complete_enclosure'];assert z is not None;q=dict(value=z['value'],derivative=z['derivative'],accounting=z['error_accounting'])
                rows.append(dict(chart=index,enabled=enabled,seconds=time.monotonic()-t,**q))
    for a,b in zip(rows[::2],rows[1::2]):
        for key in ('value','derivative'):
            assert all(max(D(x),D(z))<=min(D(y),D(w)) for (x,y),(z,w) in zip(a[key],b[key]))
    path=Path(os.environ['GFFX_FULL_REFERENCE']);assert not path.exists();path.write_text(json.dumps(dict(rows=rows,seconds=time.monotonic()-start,scope='Closed controls are independent; original old/new interval overlap is only consistency and cost evidence.'),indent=2),encoding='utf8');record_property('compiled_inverse_comparison',str(path))



def test_fourth_tangent_keeps_independent_log_motion(monkeypatch):
    import tools.opaque_tangent_peano as m
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_tangent_peano import test_tangent_peano_contains_closed_piecewise_log_integral,test_tangent_peano_preserves_zero_parameter_direction
    old=m.TangentCorrection.__init__
    def selected(self,*a,**kw):kw.update(centered=True,order=4);return old(self,*a,**kw)
    monkeypatch.setattr(m.TangentCorrection,'__init__',selected)
    with backend(profile='ball'):
        for degree in (1,2,3):
            for axis in (0,1):
                for reverse in (False,True):test_tangent_peano_contains_closed_piecewise_log_integral(degree,axis,reverse)
        test_tangent_peano_preserves_zero_parameter_direction()


@pytest.mark.parametrize('degree',range(6))
@pytest.mark.parametrize('sign',[-1,1])
def test_fourth_remainder_keeps_every_jump_order(degree,sign,amplitude=None):
    from tools.opaque_fourth_tangent import integral
    from tools.opaque_arb_jets import backend
    import math
    with backend(profile='ball'):
        tiny=Q(sign,2**60) if amplitude is None else sign*amplitude;k=I('.37')
        def field(x,y):
            if x[0,0].hi<k.lo:return [x.lift(0)]
            assert x[0,0].lo>k.hi
            return [tiny*(x-k)**degree]
        def bounds(box,axis):
            jumps=[]
            if axis==0 and degree<4:
                values=[I(0)]*4;values[degree]=I(tiny*math.factorial(degree))
                jumps=[dict(position=k,values=values,required=True)]
            d=I(0)
            if axis==0 and degree>=4:d=I(abs(tiny)*Q(math.factorial(degree),math.factorial(degree-4)))*I('.63')**(degree-4)
            return [dict(order=4,derivative=d,jumps=jumps)]
        got=integral(field,(I(0,1),I(0,1)),bounds)[0];exact=I(tiny)*I('.63')**(degree+1)/(degree+1)
        assert got.lo<=exact.lo<=exact.hi<=got.hi


def test_fourth_tangent_original_saved_cells(record_property,monkeypatch):
    import tools.opaque_tangent_peano as m
    from tools.opaque_original_motion_reference import angular_material_cover
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    init=m.TangentCorrection.__init__;rows=[];start=time.monotonic()
    def check():
        if time.monotonic()-start>=60:raise Unresolved('fourth correction comparison deadline')
    with backend(profile='ball'):
        for centered in (False,True):
            def selected(self,*a,**kw):kw.update(centered=True,order=4 if centered else 3);return init(self,*a,**kw)
            monkeypatch.setattr(m.TangentCorrection,'__init__',selected)
            p,curve,cover,charts=angular_material_cover(frozen(),tight_discriminant=True,algebraic_inverse=True,mapped_transport=True,exit_angle=True,unsquared_inverse=True,physical_angle=True,tensor_rule='gauss3x8-local2',compiled_series=True,checkpoint=check)
            for index,box in [(68,(Q(1,2),1,0,1)),(66,(Q(1,2),1,0,1)),(63,(0,1,Q(1,2),1)),(62,(0,1,Q(1,2),1)),(64,(Q(1,2),Q(3,4),0,1)),(71,(0,1,0,Q(1,4)))]:
                a,b,c,d=map(Q,box);bounds=(I(I(a).lo,I(b).hi),I(I(c).lo,I(d).hi));row=charts[index];row['prepare']((a+b)/2,(c+d)/2,bounds);t=time.monotonic()
                result=[row['correction_integrator'](bounds)[1],row['tangent_correction_integrator'](bounds)[1]]
                rows.append(dict(centered=centered,chart=index,box=list(map(str,box)),seconds=time.monotonic()-t,result=[[str(v.lo),str(v.hi)] for v in result],widths=[str(v.width) for v in result]));check()
    from decimal import Decimal as D
    for old,new in zip(rows[:6],rows[6:]):
        assert old['chart']==new['chart']
        assert all(max(D(a),D(c))<=min(D(b),D(d)) for (a,b),(c,d) in zip(old['result'],new['result']))
    path=Path(os.environ['GFFX_FULL_REFERENCE']);assert not path.exists();path.write_text(json.dumps(dict(rows=rows,seconds=time.monotonic()-start,scope='Correction only, every jump retained, same samples; no complete-piece acceptance.'),indent=2),encoding='utf8');record_property('fourth_correction',str(path))



def test_quarter_localized_smooth_keeps_independent_controls(monkeypatch):
    import tools.opaque_square_reference as m
    old=m.integrate_square
    def quarter(*a,**kw):
        if kw.get('tensor_rule') in ('gauss3x8','gauss3x8-local2'):kw['tensor_rule']='gauss3x8-local4'
        if kw.get('chart_cover'):kw['chart_cover']=[dict(row,**({'tensor_rule':'gauss3x8-local4'} if row.get('tensor_rule') else {})) for row in kw['chart_cover']]
        return old(*a,**kw)
    monkeypatch.setattr(m,'integrate_square',quarter)
    from tests.packaging.test_opaque_tensor_smooth import test_tensor_smooth_compiled_exact_controls
    test_tensor_smooth_compiled_exact_controls()


def test_compiled_scheduled_complete_piece(record_property,monkeypatch):
    import hashlib
    import tools.opaque_original_motion_reference as m
    path=Path(os.environ['GFFX_FULL_REFERENCE']).parent/'seed-layout1.json'
    assert hashlib.sha256(path.read_bytes()).hexdigest()=="c3e10ab8a5ed5391ce43fc483b0485af63b69d67a27a78d80149b8bb5c0390ed"
    leaves=json.loads(path.read_bytes())['leaves'];original=m.angular_piece_reference
    def scheduled(*a,**kw):kw['tensor_rule']='gauss3x8-local4';return original(*a,**kw,initial_leaves=leaves,compiled_series=True,correction_order=4)
    monkeypatch.setattr(m,'angular_piece_reference',scheduled)
    test_localized_tensor_fresh_complete_piece(record_property)



def test_compiled_and_fourth_contract_preserves_refusal(monkeypatch):
    import tools.opaque_original_motion_reference as m
    from tools.opaque_compiled_inverse import solve
    from tools.opaque_arb_jets import backend
    for kw in (dict(compiled_series=True),dict(compiled_series=1),dict(correction_order=4),dict(correction_order=True),dict(correction_order=5)):
        with pytest.raises(ValueError):m.angular_material_cover({},**kw)
        with pytest.raises(ValueError):m.angular_piece_reference({},**kw)
    with pytest.raises(ValueError,match='context'):solve(None,I('.5'),Jet('.5'),Jet('.5'),I(1))
    with backend(profile='ball'):
        for root,s,k,slope in ((I(-1,1),Jet('.5'),Jet('.5'),I(1)),(I('.5'),Jet('.5'),Jet('.5'),I(-1,1)),(I('.5'),Jet('.5'),Jet(0),I(1)),(I('.5'),Jet('.5',order=1),Jet('.5'),I(1))):
            with pytest.raises(ValueError):solve(None,root,s,k,slope)


def test_conditioned_fourth_keeps_independent_controls(monkeypatch):
    import tools.opaque_fourth_tangent as m
    from tools.opaque_arb_jets import backend
    old=m.integral
    def selected(*a,**kw):kw['conditioned']=True;return old(*a,**kw)
    monkeypatch.setattr(m,'integral',selected)
    # Each called test owns the arithmetic context and independent exact answer.
    for degree in range(6):
        for sign in (-1,1):test_fourth_remainder_keeps_every_jump_order(degree,sign)
    test_fourth_tangent_keeps_independent_log_motion(monkeypatch)


def test_conditioned_fourth_original_saved_cells(record_property,monkeypatch):
    import tools.opaque_tangent_peano as m
    from tools.opaque_original_motion_reference import angular_material_cover
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    init=m.TangentCorrection.__init__;rows=[];start=time.monotonic()
    def check():
        if time.monotonic()-start>=60:raise Unresolved('fourth correction comparison deadline')
    with backend(profile='ball'):
        for centered in (False,True):
            def selected(self,*a,**kw):kw.update(centered=True,order=4,conditioned=centered);return init(self,*a,**kw)
            monkeypatch.setattr(m.TangentCorrection,'__init__',selected)
            p,curve,cover,charts=angular_material_cover(frozen(),tight_discriminant=True,algebraic_inverse=True,mapped_transport=True,exit_angle=True,unsquared_inverse=True,physical_angle=True,tensor_rule='gauss3x8-local2',compiled_series=True,checkpoint=check)
            for index,box in [(68,(Q(1,2),1,0,1)),(66,(Q(1,2),1,0,1)),(63,(0,1,Q(1,2),1)),(62,(0,1,Q(1,2),1)),(64,(Q(1,2),Q(3,4),0,1)),(71,(0,1,0,Q(1,4)))]:
                a,b,c,d=map(Q,box);bounds=(I(I(a).lo,I(b).hi),I(I(c).lo,I(d).hi));row=charts[index];row['prepare']((a+b)/2,(c+d)/2,bounds);t=time.monotonic()
                result=[row['correction_integrator'](bounds)[1],row['tangent_correction_integrator'](bounds)[1]]
                rows.append(dict(centered=centered,chart=index,box=list(map(str,box)),seconds=time.monotonic()-t,result=[[str(v.lo),str(v.hi)] for v in result],widths=[str(v.width) for v in result]));check()
    from decimal import Decimal as D
    for old,new in zip(rows[:6],rows[6:]):
        assert old['chart']==new['chart']
        assert all(max(D(a),D(c))<=min(D(b),D(d)) for (a,b),(c,d) in zip(old['result'],new['result']))
    path=Path(os.environ['GFFX_FULL_REFERENCE']);assert not path.exists();path.write_text(json.dumps(dict(rows=rows,seconds=time.monotonic()-start,scope='Correction only, every jump retained, same samples; no complete-piece acceptance.'),indent=2),encoding='utf8');record_property('conditioned_correction',str(path))




def test_cached_arb_conversions_keep_exact_endpoints_and_context():
    import tools.opaque_arb_jets as m
    import flint
    original=(Jet.__mul__,Jet.sqrt,flint.ctx.prec)
    answers=[]
    for cached in (False,True):
        with m.backend(profile='ball',conversion_cache=cached) as report:
            rows=[]
            for precision in (256,300,256):
                flint.ctx.prec=precision
                x=Jet.variable(I('.2','.21'),order=10)+Jet(I(0),tangent=Q(1,2**60),order=10)
                for repeat in range(3):
                    z=(1+x*x).sqrt()+((2+x).ln()/(3-x))
                    rows.append([(str(z[i,j].lo),str(z[i,j].hi)) for i in range(11) for j in (0,1)])
            assert report['cache_capacity']==(4096 if cached else 0)
            if cached:assert report['cache_hits']>0 and report['cache_size']<=4096
            answers.append(rows)
    assert answers[0]==answers[1]
    assert (Jet.__mul__,Jet.sqrt,flint.ctx.prec)==original
    assert m._ball_converter is None
    with pytest.raises(ValueError):
        with m.backend(profile='ball',conversion_cache=1):pass
    with pytest.raises(RuntimeError):
        with m.backend(profile='ball',conversion_cache=True):raise RuntimeError('stop')
    assert m._ball_converter is None and (Jet.__mul__,Jet.sqrt,flint.ctx.prec)==original


def test_conditioned_correction_keeps_samples_domains_and_deadline():
    from tools.opaque_fourth_tangent import integral
    import tools.opaque_original_motion_reference as m
    from tools.opaque_arb_jets import backend
    for kw in (dict(conditioned_correction=True),dict(conditioned_correction=1)):
        with pytest.raises(ValueError):m.angular_material_cover({},**kw)
        with pytest.raises(ValueError):m.angular_piece_reference({},**kw)
    with backend(profile='ball'):
        calls=[];domains=[]
        def f(x,y):calls.append(1);return [1+x+y*Q(1,2**60)]
        def b(box,axis):domains.append((box,axis));return [dict(order=4,derivative=I(0),jumps=[])]
        z=integral(f,(I(0,1),I(0,1)),b,conditioned=True)[0]
        exact=I(Q(3,2)+Q(1,2**61));assert z.lo<=exact.lo<=exact.hi<=z.hi
        assert len(calls)==25 and [a for _,a in domains]==[0,1]
        def stop():raise Unresolved('conditioned deadline')
        with pytest.raises(Unresolved,match='conditioned deadline'):integral(f,(I(0,1),I(0,1)),b,conditioned=True,checkpoint=stop)


def test_cached_conditioned_original_cost(record_property,monkeypatch):
    import tools.opaque_arb_jets as m
    old=m.backend;rows=[]
    def selected(**kw):return old(**kw,conversion_cache=True)
    monkeypatch.setattr(m,'backend',selected)
    test_conditioned_fourth_original_saved_cells(record_property,monkeypatch)


def test_localized_kernel_mass_encloses_independent_exact_moments():
    from tools.opaque_tensor_smooth import quarter_factors,rule
    from tools.opaque_gauss8 import rule as gauss8
    from tools.opaque_arb_jets import backend
    import math
    with backend(profile='ball'):
        nodes,weights=gauss8();data=tuple((x.lo,x.hi,w.lo,w.hi) for x,w in zip(nodes,weights))
        factors=quarter_factors();assert len(factors)==4 and all(x.lo>0 for x in factors)
        # Fubini: integral t^p K10(t) dt = p!/(p+10)! *
        # (Q[x^(p+10)] - integral x^(p+10)dx), independent moment identity.
        for power in range(9):
            moment=sum((w*x**10*x**power for x,w in zip(nodes,weights)),I(0))-I(Q(1,power+11))
            exact=moment*Q(math.factorial(power)*math.factorial(10),math.factorial(power+10))
            upper=sum((f*Q(i+1,4)**power for i,f in enumerate(factors)),I(0))
            assert exact.bound<=upper.hi
        total=sum(factors,I(0));assert total.hi<rule()[3][1].hi
        assert factors[0].hi<total.hi/4 and factors[3].hi<total.hi/4


def test_weighted_quarter_smooth_keeps_independent_controls(monkeypatch):
    test_quarter_localized_smooth_keeps_independent_controls(monkeypatch)


def test_all_scheduled_correction_ranges(record_property,monkeypatch):
    import tools.opaque_tangent_peano as m
    from tools.opaque_original_motion_reference import angular_material_cover
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    rows=[];start=time.monotonic();current={}
    def check():
        if time.monotonic()-start>=75:raise Unresolved('range selection diagnostic deadline')
    def range_only(self,box):
        result=[v[0,0]*(I(box[0].hi)-I(box[0].lo))*(I(box[1].hi)-I(box[1].lo)) for v in self.ordinary(Jet(box[0]),Jet(box[1]))]
        current.update(widths=[str(v.width) for v in result],result=[[str(v.lo),str(v.hi)] for v in result]);return result
    monkeypatch.setattr(m.TangentCorrection,'__call__',range_only)
    path=Path(os.environ['GFFX_FULL_REFERENCE']).parent/'seed-layout1.json';leaves=json.loads(path.read_bytes())['leaves']
    with backend(profile='ball'):
        p,curve,cover,charts=angular_material_cover(frozen(),tight_discriminant=True,algebraic_inverse=True,mapped_transport=True,exit_angle=True,unsquared_inverse=True,physical_angle=True,tensor_rule='gauss3x8-local4',compiled_series=True,correction_order=4,checkpoint=check)
        for leaf in leaves:
            current={};a,b,c,d=map(Q,leaf['box']);box=(I(I(a).lo,I(b).hi),I(I(c).lo,I(d).hi));row=charts[leaf['chart']];t=time.monotonic()
            try:row['prepare']((a+b)/2,(c+d)/2,box);row['correction_integrator'](box)
            except ValueError as e:
                if isinstance(e,Unresolved):raise
                current=dict(refused=str(e))
            rows.append(dict(**leaf,seconds=time.monotonic()-t,**current))
    path=Path(os.environ['GFFX_FULL_REFERENCE']);assert not path.exists();path.write_text(json.dumps(dict(rows=rows,seconds=time.monotonic()-start,scope='Independent range candidates only; no integral or piece acceptance.'),indent=2),encoding='utf8');record_property('range_selection',str(path))


def test_conditioned_range_selection_retains_tiny_signed_enclosures(monkeypatch):
    from tools.opaque_tangent_peano import TangentCorrection
    import tools.opaque_fourth_tangent as module
    from types import SimpleNamespace
    from tools.opaque_arb_jets import backend
    with backend(profile='ball'):
        tiny=Q(1,2**60);e=SimpleNamespace(mean=[I(1),I(1)])
        c=TangentCorrection(e,None,order=4,conditioned=True)
        c.ordinary=lambda x,y:[tiny*(x+2*y),-tiny*(1+x*y)]
        def forbidden(*a,**kw):raise AssertionError('range certificate should avoid quadrature')
        monkeypatch.setattr(module,'integral',forbidden)
        values=c((I(0,1),I(0,1)))
        for got,want in zip(values,[I(tiny*Q(3,2)),I(-tiny*Q(5,4))]):assert got.lo<=want.lo<=want.hi<=got.hi and not got.zero
        assert c.last_proof=='range' and c.preferred_split_axis is None
        observed=[]
        def chosen(*a,**kw):observed.append(kw.get('conditioned',False));return [I(1),I(2)]
        monkeypatch.setattr(module,'integral',chosen)
        for scale in (Q(1,10**12),Q(1,10**9)):
            c.ordinary=lambda x,y:[scale*x,scale*y];c((I(0,1),I(0,1)))
        assert observed==[True,True]
        def stop():raise Unresolved('range check deadline')
        c.checkpoint=stop
        with pytest.raises(Unresolved,match='range check deadline'):c((I(0,1),I(0,1)))


def test_conditioned_scheduled_complete_piece(record_property,monkeypatch):
    import hashlib
    import tools.opaque_original_motion_reference as m
    path=Path(os.environ['GFFX_FULL_REFERENCE']).parent/'seed-layout1.json'
    assert hashlib.sha256(path.read_bytes()).hexdigest()=="c3e10ab8a5ed5391ce43fc483b0485af63b69d67a27a78d80149b8bb5c0390ed"
    leaves=json.loads(path.read_bytes())['leaves'];original=m.angular_piece_reference
    def scheduled(*a,**kw):kw['tensor_rule']='gauss3x8-local4';return original(*a,**kw,initial_leaves=leaves,compiled_series=True,correction_order=4,conditioned_correction=True)
    monkeypatch.setattr(m,'angular_piece_reference',scheduled)
    test_localized_tensor_fresh_complete_piece(record_property)




def test_conditioned_proof_reuses_samples_and_escalates_only_if_needed():
    from tools.opaque_fourth_tangent import integral
    from tools.opaque_arb_jets import backend
    with backend(profile='ball'):
        for derivative,jobs in ((0,2),(1,9)):
            samples=[];proof=[]
            def f(x,y):samples.append(1);return [1+x+y*Q(1,2**60)]
            def b(box,axis):proof.append(axis);return [dict(order=4,derivative=I(derivative),jumps=[])]
            z=integral(f,(I(0,1),I(0,1)),b,conditioned=True)[0]
            exact=I(Q(3,2)+Q(1,2**61));assert z.lo<=exact.lo<=exact.hi<=z.hi
            assert len(samples)==25 and len(proof)==jobs


def test_certified_inverse_contraction_stop_keeps_closed_controls(monkeypatch):
    from tools.opaque_physical_angle import PhysicalAngleResidual
    old=PhysicalAngleResidual.__init__
    def selected(self,*a,**kw):kw['contraction_stop']=True;return old(self,*a,**kw)
    monkeypatch.setattr(PhysicalAngleResidual,'__init__',selected)
    test_compiled_inverse_keeps_independent_mixed_roots(monkeypatch)
    test_compiled_inverse_keeps_complete_moving_measure(monkeypatch)


def test_inverse_contraction_compares_actual_whole_cells(record_property,monkeypatch):
    from tools.opaque_original_motion_reference import angular_material_cover
    from tools.opaque_square_reference import integrate_square
    from tools.opaque_physical_angle import PhysicalAngleResidual
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    from decimal import Decimal as D
    old=PhysicalAngleResidual.__init__;rows=[];start=time.monotonic()
    def check():
        if time.monotonic()-start>=75:raise Unresolved('inverse contraction cost deadline')
    with backend(profile='ball'):
        for index in (46,57,63,68):
            for enabled in ((False,True) if index%2==0 else (True,False)):
                def selected(self,*a,**kw):kw['contraction_stop']=enabled;return old(self,*a,**kw)
                monkeypatch.setattr(PhysicalAngleResidual,'__init__',selected)
                p,c,cover,charts=angular_material_cover(frozen(),tight_discriminant=True,algebraic_inverse=True,mapped_transport=True,exit_angle=True,unsquared_inverse=True,physical_angle=True,tensor_rule='gauss3x8-local4',compiled_series=True,correction_order=4,conditioned_correction=True,checkpoint=check);row=charts[index];t=time.monotonic()
                try:
                    z=integrate_square(row['field'],prepare=row['prepare'],correction_integrator=row['correction_integrator'],tangent_correction_integrator=row['tangent_correction_integrator'],tensor_rule=row.get('tensor_rule'),tolerance='5e-13',max_cells=1,checkpoint=check);q=dict(value=[[str(v.lo),str(v.hi)] for v in z['value']],derivative=[[str(v.lo),str(v.hi)] for v in z['derivative']],accounting=z['error_accounting'])
                except Unresolved as error:
                    assert error.details['reason']=='cell budget';z=error.details['last_complete_enclosure'];assert z is not None;q=dict(value=z['value'],derivative=z['derivative'],accounting=z['error_accounting'])
                rows.append(dict(chart=index,enabled=enabled,seconds=time.monotonic()-t,inverse=p['physical_angle'].report(),**q))
    for a,b in zip(rows[::2],rows[1::2]):
        for key in ('value','derivative'):assert all(max(D(x),D(z))<=min(D(y),D(w)) for (x,y),(z,w) in zip(a[key],b[key]))
    path=Path(os.environ['GFFX_FULL_REFERENCE']);assert not path.exists();path.write_text(json.dumps(dict(rows=rows,seconds=time.monotonic()-start,scope='Alternating fresh whole-cell cost and enclosure consistency; independent closed controls establish correctness.'),indent=2),encoding='utf8');record_property('inverse_contraction_cost',str(path))


def test_remaining_complete_calculation_profile(record_property,monkeypatch):
    import cProfile,pstats
    from tools.opaque_original_motion_reference import angular_material_cover
    from tools.opaque_square_reference import integrate_square
    from tools.opaque_physical_angle import PhysicalAngleResidual
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    old=PhysicalAngleResidual.__init__
    def selected(self,*a,**kw):kw['contraction_stop']=True;return old(self,*a,**kw)
    monkeypatch.setattr(PhysicalAngleResidual,'__init__',selected);start=time.monotonic()
    def check():
        if time.monotonic()-start>=60:raise Unresolved('calculation profile deadline')
    with backend(profile='ball'):
        p,c,cover,charts=angular_material_cover(frozen(),tight_discriminant=True,algebraic_inverse=True,mapped_transport=True,exit_angle=True,unsquared_inverse=True,physical_angle=True,tensor_rule='gauss3x8-local4',compiled_series=True,correction_order=4,conditioned_correction=True,checkpoint=check);row=charts[57]
        from tools.opaque_tensor_smooth import quarter_factors,rule
        quarter_factors(checkpoint=check);rule(checkpoint=check)
        profile=cProfile.Profile();profile.enable()
        try:result=integrate_square(row['field'],prepare=row['prepare'],correction_integrator=row['correction_integrator'],tangent_correction_integrator=row['tangent_correction_integrator'],tensor_rule=row.get('tensor_rule'),tolerance='5e-13',max_cells=1,checkpoint=check)
        except Unresolved as e:
            assert e.details['reason']=='cell budget';result=e.details['last_complete_enclosure'];assert result is not None
        finally:profile.disable()
    path=Path(os.environ['GFFX_FULL_REFERENCE']);assert not path.exists();profile_path=path.with_suffix('.pstats');assert not profile_path.exists();profile.dump_stats(str(profile_path));stats=pstats.Stats(profile)
    rows=[dict(file=key[0],line=key[1],function=key[2],primitive_calls=v[0],calls=v[1],own_seconds=v[2],cumulative_seconds=v[3]) for key,v in stats.stats.items()]
    path.write_text(json.dumps(dict(chart=57,seconds=time.monotonic()-start,total_profile_seconds=stats.total_tt,top_cumulative=sorted(rows,key=lambda r:r['cumulative_seconds'],reverse=True)[:35],top_own=sorted(rows,key=lambda r:r['own_seconds'],reverse=True)[:35],scope='Profile instrumented and not a runtime qualification; complete local correction paths included.'),indent=2),encoding='utf8');record_property('profile',str(path))


def test_fast_arb_conversion_keeps_independent_formulas_and_tiny_terms(monkeypatch):
    import tools.opaque_arb_jets as m
    import tests.packaging.test_opaque_arb_jets as checks
    old=m.backend
    def selected(**kw):kw.update(profile='ball',fast_conversion=True);return old(**kw)
    monkeypatch.setattr(m,'backend',selected)
    for name in ('reciprocal','sqrt','ln'):
        for order in (2,3,6,10):checks.test_compiled_primal_and_tangent_coefficients_match_independent_formulas(name,order)
    for order in (2,6,10):checks.test_compiled_convolution_contains_exact_rational_dual_polynomial(order)
    checks.test_compiled_interval_and_tiny_tangent_coefficients_are_not_dropped()
    checks.test_compiled_context_restores_methods_settings_and_cancellation()
    test_compiled_inverse_keeps_independent_mixed_roots(monkeypatch)
    test_compiled_inverse_keeps_complete_moving_measure(monkeypatch)
    with old(profile='ball',fast_conversion=True):
        tiny=Jet(2,order=4,tangent='1e-1000');result=tiny.ln();assert result[0,1].lo<=I('5e-1001').lo<=result[0,1].hi and not result[0,1].zero
    assert m._ball_converter is None and m._interval_converter is None


def test_fast_conversion_complete_local_cost(record_property,monkeypatch):
    import tools.opaque_arb_jets as m
    old=m.backend
    def selected(**kw):return old(**kw,fast_conversion=True)
    monkeypatch.setattr(m,'backend',selected)
    test_inverse_contraction_compares_actual_whole_cells(record_property,monkeypatch)


def test_selected_candidate_contracts_and_construction_deadline(monkeypatch):
    import tools.opaque_original_motion_reference as m
    import tools.opaque_arb_jets as arithmetic
    for kw in (dict(contraction_stop=True),dict(contraction_stop=1)):
        with pytest.raises(ValueError):m.angular_material_cover({},**kw)
        with pytest.raises(ValueError):m.angular_piece_reference({},**kw)
    with pytest.raises(ValueError):
        with arithmetic.backend(profile='ball',fast_conversion=1):pass
    calls=[]
    def delayed(data,*,checkpoint,**kw):calls.append(kw);monkeypatch.setattr(m.time,'monotonic',lambda:100.);checkpoint()
    monkeypatch.setattr(m,'angular_material_cover',delayed);monkeypatch.setattr(m.time,'monotonic',lambda:0.)
    flags=dict(tight_discriminant=True,algebraic_inverse=True,mapped_transport=True,exit_angle=True,unsquared_inverse=True,physical_angle=True,tensor_rule='gauss3x8-local4',compiled_series=True,correction_order=4,conditioned_correction=True,contraction_stop=True)
    with pytest.raises(Unresolved,match='shared original reference deadline'):m.angular_piece_reference({},**flags)
    assert calls[0]['contraction_stop'] and calls[0]['conditioned_correction']


def test_selected_fresh_complete_piece(record_property,monkeypatch):
    import tools.opaque_arb_jets as arithmetic
    import tools.opaque_original_motion_reference as m
    old=arithmetic.backend;original=m.angular_piece_reference
    def fast(**kw):kw['fast_conversion']=True;return old(**kw)
    def contracted(*a,**kw):return original(*a,**kw,contraction_stop=True)
    monkeypatch.setattr(arithmetic,'backend',fast);monkeypatch.setattr(m,'angular_piece_reference',contracted)
    test_conditioned_scheduled_complete_piece(record_property,monkeypatch)


def test_selected_candidate_all_fifteen_smooth_regions(record_property):
    from tools.opaque_original_motion_reference import angular_material_cover
    from tools.opaque_square_reference import integrate_square
    from tools.opaque_arb_jets import backend
    from tools.opaque_tensor_smooth import quarter_factors,rule
    from tests.packaging.test_opaque_motion_tangent import frozen
    rows=[];start=time.monotonic()
    def check():
        if time.monotonic()-start>=85:raise Unresolved('fifteen selected smooth regions deadline')
    with backend(profile='ball',fast_conversion=True):
        quarter_factors(checkpoint=check);rule(checkpoint=check)
        for index in [*range(40,54),67]:
            p,c,cover,charts=angular_material_cover(frozen(),tight_discriminant=True,algebraic_inverse=True,mapped_transport=True,exit_angle=True,unsquared_inverse=True,physical_angle=True,tensor_rule='gauss3x8-local4',compiled_series=True,correction_order=4,conditioned_correction=True,contraction_stop=True,checkpoint=check);row=charts[index];t=time.monotonic()
            try:
                z=integrate_square(row['field'],prepare=row['prepare'],correction_integrator=row['correction_integrator'],tangent_correction_integrator=row['tangent_correction_integrator'],tensor_rule=row.get('tensor_rule'),tolerance='5e-13',max_cells=1,checkpoint=check);q=dict(status='locally_verified',widths=[[str(v.width),str(d.width)] for v,d in zip(z['value'],z['derivative'])])
            except Unresolved as e:
                assert e.details['reason']=='cell budget';z=e.details['last_complete_enclosure'];q=dict(status='retained_uncertainty',widths=z['widths'])
            rows.append(dict(chart=index,seconds=time.monotonic()-t,**q))
    path=Path(os.environ['GFFX_FULL_REFERENCE']);assert not path.exists();path.write_text(json.dumps(dict(rows=rows,seconds=time.monotonic()-start,scope='Fresh actual15region local checks; no combined piece or installed comparison acceptance.'),indent=2),encoding='utf8');record_property('selected_fifteen',str(path))
    assert len(rows)==15


def test_selected_proof_rules_in_fast_arithmetic(monkeypatch):
    import tools.opaque_arb_jets as m
    old=m.backend
    def selected(**kw):kw['fast_conversion']=True;return old(**kw)
    monkeypatch.setattr(m,'backend',selected)
    test_weighted_quarter_smooth_keeps_independent_controls(monkeypatch)
    test_conditioned_fourth_keeps_independent_controls(monkeypatch)
    test_certified_inverse_contraction_stop_keeps_closed_controls(monkeypatch)
    test_fresh_refinement_schedule_checks_cover_and_charges_ancestors()
    test_fresh_refinement_schedule_recomputes_all_fields_and_remainders()


def test_reverse_conditioned_identity_keeps_independent_controls(monkeypatch):
    import tools.opaque_fourth_tangent as m
    old=m.integral
    def reverse(*a,**kw):kw['integrated_axis']=0;return old(*a,**kw)
    monkeypatch.setattr(m,'integral',reverse)
    test_conditioned_fourth_keeps_independent_controls(monkeypatch)

    # Unit-amplitude cases must exercise escalation, unlike tiny cases which
    # may already have a sufficient ordinary enclosure.
    for degree in range(6):
        for sign in (-1,1):test_fourth_remainder_keeps_every_jump_order(degree,sign,amplitude=Q(1))


def test_both_conditioned_directions_on_remaining_regions(record_property,monkeypatch):
    import tools.opaque_fourth_tangent as m
    from tools.opaque_original_motion_reference import angular_material_cover
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    from decimal import Decimal as D
    old=m.integral;rows=[];start=time.monotonic()
    def check():
        if time.monotonic()-start>=75:raise Unresolved('remaining direction diagnostic deadline')
    with backend(profile='ball',fast_conversion=True):
        for index,box in [(64,(Q(3,4),1,Q(1,2),1)),(68,(0,Q(1,4),0,Q(1,2))),(71,(0,1,Q(1,4),Q(3,8)))]:
            for axis in (1,0):
                def selected(*a,**kw):kw['integrated_axis']=axis;return old(*a,**kw)
                monkeypatch.setattr(m,'integral',selected)
                p,c,cover,charts=angular_material_cover(frozen(),tight_discriminant=True,algebraic_inverse=True,mapped_transport=True,exit_angle=True,unsquared_inverse=True,physical_angle=True,tensor_rule='gauss3x8-local4',compiled_series=True,correction_order=4,conditioned_correction=True,contraction_stop=True,checkpoint=check);row=charts[index];a,b,c,d=map(Q,box);bounds=(I(I(a).lo,I(b).hi),I(I(c).lo,I(d).hi));row['prepare']((a+b)/2,(c+d)/2,bounds);t=time.monotonic()
                result=[row['correction_integrator'](bounds)[1],row['tangent_correction_integrator'](bounds)[1]]
                rows.append(dict(chart=index,box=list(map(str,box)),axis=axis,seconds=time.monotonic()-t,result=[[str(v.lo),str(v.hi)] for v in result],widths=[str(v.width) for v in result]));check()
    for old,new in zip(rows[::2],rows[1::2]):assert all(max(D(a),D(c))<=min(D(b),D(d)) for (a,b),(c,d) in zip(old['result'],new['result']))
    path=Path(os.environ['GFFX_FULL_REFERENCE']);assert not path.exists();path.write_text(json.dumps(dict(rows=rows,seconds=time.monotonic()-start,scope='Both complete correction identities on the three largest saved contributors; no piece acceptance.'),indent=2),encoding='utf8');record_property('both_directions',str(path))


def test_repaired_selection_scheduled_complete_piece(record_property,monkeypatch):
    import hashlib
    import tools.opaque_arb_jets as arithmetic
    import tools.opaque_original_motion_reference as m
    path=Path(os.environ['GFFX_FULL_REFERENCE']).parent/'seed-layout2.json'
    assert hashlib.sha256(path.read_bytes()).hexdigest()=="f9aafffad528e6b6398be32184a1fb39f84a726a7f232567afa1ac1972fa230d"
    leaves=json.loads(path.read_bytes())['leaves'];original=m.angular_piece_reference;old=arithmetic.backend
    def fast(**kw):kw['fast_conversion']=True;return old(**kw)
    def scheduled(*a,**kw):
        kw['tensor_rule']='gauss3x8-local4'
        return original(*a,**kw,initial_leaves=leaves,compiled_series=True,correction_order=4,conditioned_correction=True,contraction_stop=True)
    monkeypatch.setattr(arithmetic,'backend',fast);monkeypatch.setattr(m,'angular_piece_reference',scheduled)
    test_localized_tensor_fresh_complete_piece(record_property)


def test_native_series_reuse_keeps_independent_controls_and_mutation(monkeypatch):
    import tools.opaque_arb_jets as m
    old=m.backend
    def selected(**kw):kw.update(fast_conversion=True,series_reuse=True);return old(**kw)
    monkeypatch.setattr(m,'backend',selected)
    test_fast_arb_conversion_keeps_independent_formulas_and_tiny_terms(monkeypatch)
    with old(profile='ball',fast_conversion=True,series_reuse=True) as report:
        x=Jet.variable(I('.2','.21'),order=6)+Jet(0,order=6,tangent=Q(1,2**60));q=(1+x*x).sqrt();z=q*q
        assert report['series_hits']>0 and report['series_size']<=128
        misses=report['series_misses'];q.c[0,0]=I(2,3);z=q*q
        assert z[0,0].lo<=4<=9<=z[0,0].hi and report['series_misses']>misses
        assert not z[0,1].zero
        misses=report['series_misses'];q.c[0,0]=I(q[0,0].lo,q[0,0].hi);z=q*q
        assert report['series_misses']>misses
    assert m._series_converter is None
    with pytest.raises(ValueError):
        with old(profile='ball',series_reuse=True):pass


def test_native_series_complete_local_cost(record_property,monkeypatch):
    import tools.opaque_arb_jets as m
    old=m.backend
    def selected(**kw):kw.update(fast_conversion=True,series_reuse=True);return old(**kw)
    monkeypatch.setattr(m,'backend',selected)
    test_inverse_contraction_compares_actual_whole_cells(record_property,monkeypatch)


@pytest.mark.parametrize('bounds',[(0,1,0,1),('0.1','0.3','0.2','0.5')])
def test_partitioned_range_fallback_keeps_complete_measure_and_tiny_motion(bounds):
    from tools.opaque_tangent_peano import TangentCorrection
    from types import SimpleNamespace
    from tools.opaque_arb_jets import backend
    with backend(profile='ball',fast_conversion=True):
        e=SimpleNamespace(mean=[I(1),I(1)]);c=TangentCorrection(e,None,curved=True,range_fallback=True,partitioned_range=True)
        tiny=Q(1,2**60)
        c.ordinary=lambda x,y:[x*(1-x)+y*y+tiny,-tiny*(1+x*y)]
        def unresolved(box,axis):raise ValueError('curved angular crossings need certified monotonicity')
        a,b,d,f=map(Q,bounds);box=(I(I(a).lo,I(b).hi),I(I(d).lo,I(f).hi));mass=(b-a)*(f-d)
        c.axis_bounds=unresolved;got=c(box)
        value=mass*((a+b)/2-(a*a+a*b+b*b)/3+(d*d+d*f+f*f)/3+tiny);motion=-tiny*mass*(1+(a+b)*(d+f)/4)
        whole=c.ordinary(Jet(box[0]),Jet(box[1]))[0][0,0]*I(mass)
        for z,exact in zip(got,[I(value),I(motion)]):assert z.lo<=exact.lo<=exact.hi<=z.hi
        assert got[0].width<whole.width and not got[1].zero and c.range_proof_boxes==4
        assert c.range_fallbacks==1 and c.preferred_split_axis is None
        calls=[]
        def stop():calls.append(1);raise Unresolved('partitioned range deadline')
        c.checkpoint=stop
        with pytest.raises(Unresolved,match='partitioned range deadline'):c.range_integral((I(0,1),I(0,1)),parts=2)
        for kw in (dict(partitioned_range=True),dict(partitioned_range=1,curved=True,range_fallback=True)):
            with pytest.raises(ValueError):TangentCorrection(e,None,**kw)


def test_partitioned_range_original_remaining_box(record_property,monkeypatch):
    import tools.opaque_tangent_peano as m
    from tools.opaque_original_motion_reference import angular_material_cover
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    old=m.TangentCorrection.__init__;rows=[];start=time.monotonic();instances=[]
    def check():
        if time.monotonic()-start>=45:raise Unresolved('partitioned original range deadline')
    with backend(profile='ball',fast_conversion=True):
        for enabled in (False,True):
            def selected(self,*a,**kw):kw['partitioned_range']=enabled;old(self,*a,**kw);instances.append(self)
            monkeypatch.setattr(m.TangentCorrection,'__init__',selected)
            p,c,cover,charts=angular_material_cover(frozen(),tight_discriminant=True,algebraic_inverse=True,mapped_transport=True,exit_angle=True,unsquared_inverse=True,physical_angle=True,tensor_rule='gauss3x8-local4',compiled_series=True,correction_order=4,conditioned_correction=True,contraction_stop=True,checkpoint=check);row=charts[68];box=(I(0,'.25'),I(0,'.5'));row['prepare'](Q(1,8),Q(1,4),box);t=time.monotonic();result=[row['correction_integrator'](box)[1],row['tangent_correction_integrator'](box)[1]]
            rows.append(dict(enabled=enabled,seconds=time.monotonic()-t,result=[[str(v.lo),str(v.hi)] for v in result],widths=[str(v.width) for v in result],fallbacks=sum(q.range_fallbacks for q in instances),proof_boxes=sum(q.range_proof_boxes for q in instances)));instances.clear()
    from decimal import Decimal as D
    assert all(D(b)<=D(a) for a,b in zip(rows[0]['widths'],rows[1]['widths']))
    assert rows[1]['fallbacks']==1 and rows[1]['proof_boxes']==4
    path=Path(os.environ['GFFX_FULL_REFERENCE']);assert not path.exists();path.write_text(json.dumps(dict(rows=rows,scope='Complete range fallback comparison only; no piece acceptance.'),indent=2),encoding='utf8');record_property('partitioned_range',str(path))


@pytest.mark.parametrize('order',[0,1,4,10])
def test_mixed_loss_polynomial_contains_independent_coefficients(order):
    from types import SimpleNamespace
    from tools.opaque_tangent_peano import MixedLoss
    from tools.opaque_full_material_reference import HorizonExtension
    from tools.opaque_arb_jets import backend
    # Cubics in log(q), generated independently from their power coefficients.
    coeff=[[Q(1),Q(2),Q(-3),Q(4)],[Q(2),Q(-1),Q(2),Q(-2)]]
    import math
    def controls(row):return [sum(row[k]*Q(math.comb(i,k),math.comb(3,k)) for k in range(i+1)) for i in range(4)]
    curve=SimpleNamespace(x=[I(0),I(1)],cn=[I(1),I(3)],controls=[[list(map(I,controls(row))) for row in coeff]],horizon=[I(Q(1,3)),I(Q(1,7))],values=[[I(1),I(2)]])
    with backend(profile='ball',fast_conversion=True):
        e=HorizonExtension(curve);mixed=MixedLoss(e,I(Q(1,25)))
        for tiny in (Q(1,2**60),Q(-1,2**60)):
            # q=exp(x); this proof's field is a polynomial in x. Supply exact
            # independent exp Taylor coefficients rather than the implementation's log.
            q=Jet(order=order,coeff={(i,j):I((tiny if j else Q(1))/math.factorial(i)) for i in range(order+1) for j in (0,1)})
            # At x=0, q=1. Parameter derivative is tiny*d/dx of the cubic.
            got=mixed.at(q,0)
            powers=[(1-Q(1,25))*a+Q(1,25)*b for a,b in zip(*coeff)]
            for i in range(order+1):
                for j in (0,1):
                    exact=(powers[i] if i<4 else 0) if not j else (tiny*(i+1)*powers[i+1] if i<3 else 0)
                    z=I(exact);assert got[i,j].lo<=z.lo<=z.hi<=got[i,j].hi
            assert not got[0,1].zero
            h=mixed.at(q,-1);slope=(1-Q(1,25))*(1-Q(1,3))+Q(1,25)*(2-Q(1,7))
            for i in range(order+1):
                for j in (0,1):
                    exact=slope*(tiny if j else Q(1))/math.factorial(i)
                    if i==0 and not j:exact+=(1-Q(1,25))*Q(1,3)+Q(1,25)*Q(1,7)
                    z=I(exact);assert h[i,j].lo<=z.lo<=z.hi<=h[i,j].hi
            assert not h[0,1].zero
        with pytest.raises(ValueError):MixedLoss(e,I(-1))


def test_mixed_loss_keeps_actual_segment_jumps_and_tiny_motion():
    from tools.opaque_tangent_peano import MixedLoss,segmented_mixed
    from tools.opaque_full_material_reference import HorizonExtension
    from tools.opaque_principled_reference import PreparedPrincipled
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    with backend(profile='ball',fast_conversion=True):
        e=HorizonExtension(PreparedPrincipled(frozen()['budget'],1.));m=MixedLoss(e,I(.04))
        for index in (1,30,70):
            for actual in (index-1,index,index+1):
                q=Jet.variable(e.cn[index],order=4)+Jet(0,order=4,tangent=Q(1,2**60))
                from tools.opaque_peano_material import difference
                expected=difference(e,q,actual,index);expected=expected[0]*(1-I(.04))+expected[1]*I(.04)
                got=m.difference(q,actual,index)
                for i in range(5):
                    for j in (0,1):assert max(got[i,j].lo,expected[i,j].lo)<=min(got[i,j].hi,expected[i,j].hi)
                if actual==index:assert not got.c
        q=Jet.variable(I(e.cn[29].lo,e.cn[32].hi),order=4)+Jet(0,order=4,tangent=Q(1,2**60))
        got=m.segmented(q,30,horizon=True)
        assert not got[0,1].zero
        # Every one-sided segment is contained, not only the midpoint branch.
        for index in range(29,32):
            c=dict(q.c);c[0,0]=I(e.cn[index].lo,e.cn[index+1].hi);part=m.difference(Jet(order=4,coeff=c),index,30)
            for i in range(5):
                for j in (0,1):assert got[i,j].lo<=part[i,j].lo<=part[i,j].hi<=got[i,j].hi



def test_mixed_material_actual_complete_local_cost(record_property,monkeypatch):
    import tools.opaque_original_motion_reference as m
    from tools.opaque_square_reference import integrate_square
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    rows=[];start=time.monotonic()
    def check():
        if time.monotonic()-start>=80:raise Unresolved('mixed material local comparison deadline')
    with backend(profile='ball',fast_conversion=True):
        for index,bounds in [(55,(0,1,0,1)),(64,(Q(3,4),1,Q(1,2),1)),(68,(0,Q(1,4),0,Q(1,2))),(71,(0,1,0,Q(1,4)))]:
            for enabled in (False,True):
                p,c,cover,charts=m.angular_material_cover(frozen(),tight_discriminant=True,algebraic_inverse=True,mapped_transport=True,exit_angle=True,unsquared_inverse=True,physical_angle=True,tensor_rule='gauss3x8-local4',compiled_series=True,correction_order=4,conditioned_correction=True,contraction_stop=True,mixed_curve=enabled,checkpoint=check);row=charts[index]
                a,b,c,d=map(Q,bounds);t=time.monotonic();hx,hy=I(b-a),I(d-c)
                def mapped_box(box):return (I(a)+hx*box[0],I(c)+hy*box[1])
                def field(x,y):return [v*hx*hy for v in row['field'](x*hx+I(a),y*hy+I(c))]
                def prepare(x,y,box):return row['prepare'](a+(b-a)*x,c+(d-c)*y,mapped_box(box))
                def correction(box):return row['correction_integrator'](mapped_box(box))
                def tangent(box):return row['tangent_correction_integrator'](mapped_box(box))
                try:z=integrate_square(field,prepare=prepare,correction_integrator=correction,tangent_correction_integrator=tangent,tensor_rule=row.get('tensor_rule'),tolerance='5e-13',max_cells=1,checkpoint=check)
                except Unresolved as error:
                    assert error.details['reason']=='cell budget';z=error.details['last_complete_enclosure'];assert z is not None
                    z={**z,'value':[I(*v) for v in z['value']],'derivative':[I(*v) for v in z['derivative']]}
                rows.append(dict(chart=index,enabled=enabled,seconds=time.monotonic()-t,box=list(map(str,(a,b,c,d))),values=[[str(v.lo),str(v.hi)] for v in z['value']],derivatives=[[str(v.lo),str(v.hi)] for v in z['derivative']],widths=[[str(v.width),str(w.width)] for v,w in zip(z['value'],z['derivative'])]));check()
    from decimal import Decimal as D
    for old,new in zip(rows[::2],rows[1::2]):
        for a,b in zip(old['values']+old['derivatives'],new['values']+new['derivatives']):assert max(D(a[0]),D(b[0]))<=min(D(a[1]),D(b[1]))
    path=Path(os.environ['GFFX_FULL_REFERENCE']);assert not path.exists();path.write_text(json.dumps(dict(rows=rows,seconds=time.monotonic()-start,scope='Fresh full local mathematical enclosures and runtime; not a whole-piece or production acceptance.'),indent=2),encoding='utf8');record_property('mixed_local',str(path))


def test_mixed_material_complete_piece(record_property,monkeypatch):
    import tools.opaque_original_motion_reference as m
    old=m.angular_piece_reference
    def selected(*a,**kw):kw['mixed_curve']=True;return old(*a,**kw)
    monkeypatch.setattr(m,'angular_piece_reference',selected)
    test_repaired_selection_scheduled_complete_piece(record_property,monkeypatch)



def test_mixed_material_contracts_and_construction_deadline(monkeypatch):
    import tools.opaque_original_motion_reference as m
    for value in (True,1):
        with pytest.raises(ValueError):m.angular_material_cover({},mixed_curve=value)
        with pytest.raises(ValueError):m.angular_piece_reference({},mixed_curve=value)
    old=m.angular_piece_reference
    def selected(*a,**kw):kw['mixed_curve']=True;return old(*a,**kw)
    monkeypatch.setattr(m,'angular_piece_reference',selected)
    test_selected_candidate_contracts_and_construction_deadline(monkeypatch)



def test_material_only_inputs_preserve_all_consumed_endpoints(monkeypatch):
    import tools.opaque_original_motion_reference as m
    from tools.opaque_tangent_peano import TangentCorrection
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    captures=[];old=TangentCorrection.__init__
    def capture(self,*a,**kw):old(self,*a,**kw);captures.append(self)
    monkeypatch.setattr(TangentCorrection,'__init__',capture)
    with backend(profile='ball',fast_conversion=True):
        common=dict(tight_discriminant=True,algebraic_inverse=True,mapped_transport=True,exit_angle=True,unsquared_inverse=True,physical_angle=True,tensor_rule='gauss3x8-local4',compiled_series=True,correction_order=4,conditioned_correction=True,contraction_stop=True,mixed_curve=True)
        m.angular_material_cover(frozen(),**common);original=list(captures);captures.clear()
        m.angular_material_cover(frozen(),**common,material_inputs=True);selected=list(captures)
        for index in (0,55,64,68,71):
            for order in (0,4,10):
                x=Jet.variable(I('.3','.4'),order=order) if order else Jet(I('.3','.4'));y=Jet(I('.2','.3'),order=order)
                a=original[index].inputs(x,y);b=selected[index].inputs(x,y)
                for channel in (0,1,3):
                    assert all((a[channel][i,j].lo,a[channel][i,j].hi)==(b[channel][i,j].lo,b[channel][i,j].hi) for i in range(order+1) for j in (0,1))
                assert any(not b[k][0,1].zero for k in (0,1,3))
        with pytest.raises(ValueError):m.angular_material_cover({},material_inputs=1)
        with pytest.raises(ValueError):m.angular_piece_reference({},material_inputs=True)


def test_material_only_actual_complete_local_cost(record_property,monkeypatch):
    import tools.opaque_original_motion_reference as m
    old=m.angular_material_cover
    def selected(*a,**kw):
        enabled=kw.pop('mixed_curve');kw.update(mixed_curve=True,material_inputs=enabled);return old(*a,**kw)
    monkeypatch.setattr(m,'angular_material_cover',selected)
    test_mixed_material_actual_complete_local_cost(record_property,monkeypatch)


def test_material_only_complete_piece(record_property,monkeypatch):
    import tools.opaque_original_motion_reference as m
    old=m.angular_piece_reference
    def selected(*a,**kw):kw['material_inputs']=True;return old(*a,**kw)
    monkeypatch.setattr(m,'angular_piece_reference',selected)
    test_mixed_material_complete_piece(record_property,monkeypatch)



def test_targeted_schedule_validates_complete_depth_and_charges():
    import hashlib
    from tools.opaque_refinement_schedule import prepare_schedule
    from tools.opaque_original_motion_reference import angular_material_cover
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    path=Path(os.environ['GFFX_FULL_REFERENCE']).parent/'seed-layout3.json'
    assert hashlib.sha256(path.read_bytes()).hexdigest()=='df95ef2baa6f4b3545cd21247cfccc1ead8096aae9bcebb700a18a7472956f44'
    leaves=json.loads(path.read_bytes())['leaves']
    with backend(profile='ball',fast_conversion=True):
        p,c,cover,charts=angular_material_cover(frozen(),tight_discriminant=True,algebraic_inverse=True,mapped_transport=True,exit_angle=False)
        pending,count=prepare_schedule(charts,leaves,max_cells=256,max_depth=10,checkpoint=lambda:None)
        assert len(pending)==111 and count==110 and count+len(pending)==221
    test_fresh_refinement_schedule_checks_cover_and_charges_ancestors()
    test_fresh_refinement_schedule_recomputes_all_fields_and_remainders()


def test_targeted_material_complete_piece(record_property,monkeypatch):
    import hashlib
    import tools.opaque_original_motion_reference as m
    path=Path(os.environ['GFFX_FULL_REFERENCE']).parent/'seed-layout3.json'
    assert hashlib.sha256(path.read_bytes()).hexdigest()=='df95ef2baa6f4b3545cd21247cfccc1ead8096aae9bcebb700a18a7472956f44'
    leaves=json.loads(path.read_bytes())['leaves'];old=m.angular_piece_reference
    def selected(*a,**kw):kw['initial_leaves']=leaves;return old(*a,**kw)
    monkeypatch.setattr(m,'angular_piece_reference',selected)
    test_material_only_complete_piece(record_property,monkeypatch)



def test_direct_quarter_conditioned_proof_keeps_independent_integrals(monkeypatch):
    import tools.opaque_fourth_tangent as m
    old=m.integral
    def selected(*a,**kw):kw.update(conditioned=True,direct_conditioned=True,condition_parts=4);return old(*a,**kw)
    monkeypatch.setattr(m,'integral',selected)
    test_conditioned_fourth_keeps_independent_controls(monkeypatch)
    for degree in range(6):
        for sign in (-1,1):test_fourth_remainder_keeps_every_jump_order(degree,sign,amplitude=Q(1))
    samples=[];proof=[]
    def field(x,y):samples.append(1);return [1+x+y*Q(1,2**60)]
    def bound(box,axis):proof.append(axis);return [dict(order=4,derivative=I(0),jumps=[])]
    z=old(field,(I(0,1),I(0,1)),bound,conditioned=True,direct_conditioned=True,condition_parts=4)[0]
    exact=I(Q(3,2)+Q(1,2**61));assert z.lo<=exact.lo<=exact.hi<=z.hi
    assert len(samples)==25 and len(proof)==9
    for options in (dict(direct_conditioned=True),dict(condition_parts=3),dict(conditioned=True,direct_conditioned=1)):
        with pytest.raises(ValueError):old(field,(I(0,1),I(0,1)),bound,**options)


def test_direct_quarter_actual_complete_local_cost(record_property,monkeypatch):
    import tools.opaque_fourth_tangent as f
    import tools.opaque_original_motion_reference as m
    original=m.angular_material_cover;integral=f.integral
    def selected(*a,**kw):
        enabled=kw.pop('mixed_curve');kw.update(mixed_curve=True,material_inputs=True)
        def chosen(*args,**options):
            if not enabled:options.update(direct_conditioned=False,condition_parts=2)
            return integral(*args,**options)
        monkeypatch.setattr(f,'integral',chosen);return original(*a,**kw)
    monkeypatch.setattr(m,'angular_material_cover',selected)
    test_mixed_material_actual_complete_local_cost(record_property,monkeypatch)
