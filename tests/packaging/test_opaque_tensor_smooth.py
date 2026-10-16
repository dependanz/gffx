from fractions import Fraction as Q
import json,os,time,math
from pathlib import Path
from tools.opaque_shading_intervals import I,Jet,Unresolved

def test_directional_tensor_remainder_diagnostic(record_property):
    from tools.opaque_gauss8 import rule
    from tools.opaque_high_peano import _mass
    from tools.opaque_original_motion_reference import angular_material_cover
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    start=time.monotonic();rows=[]
    def check():
        if time.monotonic()-start>=60:raise Unresolved('directional smooth diagnostic deadline')
    with backend(profile='ball'):
        nodes,weights=rule(checkpoint=check);data=tuple((x.lo,x.hi,w.lo,w.hi) for x,w in zip(nodes,weights));mass8=I(math.factorial(10))*I(_mass(10,data));mass3=I(Q(math.factorial(3)**4,7*math.factorial(6)**2));mass5=I(Q(math.factorial(5)**4,11*math.factorial(10)**2))
        p,curve,cover,charts=angular_material_cover(frozen(),tight_discriminant=True,algebraic_inverse=True,mapped_transport=True,exit_angle=True,unsquared_inverse=True,physical_angle=True,checkpoint=check)
        for index in (40,46,53,67):
            row=charts[index];row['prepare'](Q(1,2),Q(1,2),(I(0,1),I(0,1)));started=time.monotonic()
            dx=row['field'](Jet.variable(I(0,1),order=6),Jet(I(0,1),order=6));dy=row['field'](Jet(I(0,1),order=10),Jet.variable(I(0,1),order=10))
            bound=[[str(2*(I(x[6,j].bound)*mass3+I(y[10,j].bound)*mass8)) for j in (0,1)] for x,y in zip(dx,dy)]
            rows.append(dict(chart=index,width=bound,seconds=time.monotonic()-started))
    path=Path(os.environ['GFFX_FULL_REFERENCE']);assert not path.exists();path.write_text(json.dumps(dict(rows=rows,mass3=str(mass3),mass5=str(mass5),mass8=str(mass8),seconds=time.monotonic()-start,scope='Directional derivative and Peano bounds only; no integral acceptance.'),indent=2),encoding='utf8');record_property('directional_bounds',str(path))


import pytest
@pytest.mark.parametrize('px,py',[(0,0),(3,9),(5,15),(6,10),(8,15)])
@pytest.mark.parametrize('tiny',[False,True])
def test_tensor_smooth_exact_integral_motion_and_tiny_terms(px,py,tiny):
    from tools.opaque_square_reference import integrate_square
    scale=Q(1,2**60) if tiny else Q(1)
    def f(x,y):return [scale*x**px*y**py*(1+x.lift(0,tangent=Q(-2,3))),scale*(x**2+y**3)*(1+x.lift(0,tangent=Q(4,7)))]
    result=integrate_square(f,tensor_rule='gauss3x8',tolerance=str((I(scale)*I('1e-12')).lo))
    for v,d,exact,factor in zip(result['value'],result['derivative'],[scale/Q((px+1)*(py+1)),scale*Q(7,12)],[Q(-2,3),Q(4,7)]):
        a,b=I(exact),I(exact*factor);assert v.lo<=a.lo<=a.hi<=v.hi;assert d.lo<=b.lo<=b.hi<=d.hi;assert v.lo>0
    assert result['work_charge']==1


def test_tensor_smooth_uses_24_samples_and_all_corrections():
    from tools.opaque_square_reference import integrate_square
    calls=[]
    def field(x,y):
        if x.order==0:calls.append(1)
        return [x+y+x.lift(0,tangent=Q(2,3))]
    result=integrate_square(field,tensor_rule='gauss3x8',max_cells=1,correction_integrator=lambda box:[I(Q(1,2**60))],tangent_correction_integrator=lambda box:[I(Q(-1,2**60))])
    assert len(calls)==24 and result['cells']==1 and result['quadrature_samples']==24
    for got,want in zip([result['value'][0],result['derivative'][0]],[I(1+Q(1,2**60)),I(Q(2,3)-Q(1,2**60))]):assert got.lo<=want.lo<=want.hi<=got.hi


def test_tensor_smooth_rational_integral_and_moving_denominator():
    from tools.opaque_square_reference import integrate_square
    def field(x,y):return [(1+x)/(2-y-y.lift(0,tangent=1))]
    result=integrate_square(field,tensor_rule='gauss3x8')
    for got,want in zip([result['value'][0],result['derivative'][0]],[I(2).ln()*Q(3,2),I(Q(3,4))]):assert got.lo<=want.lo<=want.hi<=got.hi


def test_tensor_smooth_contract_deadline_and_mixed_chart_cover():
    from tools.opaque_square_reference import integrate_square
    f=lambda x,y:[x+y];row=dict(field=f,depth=1)
    for bad in ('gauss8x8',24,True):
        with pytest.raises(ValueError):integrate_square(f,tensor_rule=bad)
    with pytest.raises(ValueError):integrate_square(f,tensor_rule='gauss3x8',remainder_partition=(1,4))
    with pytest.raises(ValueError):integrate_square(f,tensor_rule='gauss3x8',experimental_rule='gauss8')
    result=integrate_square(None,chart_cover=[dict(row,tensor_rule='gauss3x8'),row],max_cells=3)
    assert result['cells']==3 and result['quadrature_samples']==49
    assert result['value'][0].lo<=2<=result['value'][0].hi
    def stop():raise Unresolved('tensor construction deadline')
    with pytest.raises(Unresolved,match='tensor construction deadline'):integrate_square(f,tensor_rule='gauss3x8',checkpoint=stop)


def test_tensor_gauss8_independent_moments_through_fifteen():
    from tools.opaque_gauss8 import rule
    nodes,weights=rule()
    for power in range(16):
        moment=sum((w*x**power for x,w in zip(nodes,weights)),I(0));exact=I(Q(1,power+1));assert moment.lo<=exact.lo<=exact.hi<=moment.hi
    assert all(w.lo>0 for w in weights)


def test_tensor_smooth_compiled_exact_controls():
    from tools.opaque_arb_jets import backend
    with backend(profile='ball'):
        for pair in ((0,0),(3,9),(5,15),(6,10),(8,15)):
            for tiny in (False,True):test_tensor_smooth_exact_integral_motion_and_tiny_terms(*pair,tiny)
        test_tensor_smooth_uses_24_samples_and_all_corrections()
        test_tensor_smooth_rational_integral_and_moving_denominator()
        test_tensor_smooth_contract_deadline_and_mixed_chart_cover()
        test_tensor_gauss8_independent_moments_through_fifteen()


def test_tensor_smooth_all_fifteen_original_regions(record_property):
    from tools.opaque_original_motion_reference import angular_material_cover
    from tools.opaque_square_reference import integrate_square
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    from decimal import Decimal as D
    start=time.monotonic();rows=[]
    def check():
        if time.monotonic()-start>=85:raise Unresolved('fifteen-region comparison deadline')
    with backend(profile='ball'):
        for index in [*range(40,54),67]:
            pairs=[]
            for chosen in ((False,True) if index%2==0 else (True,False)):
                p,curve,cover,charts=angular_material_cover(frozen(),tight_discriminant=True,algebraic_inverse=True,mapped_transport=True,exit_angle=True,unsquared_inverse=True,physical_angle=True,checkpoint=check)
                row=charts[index];options={'tensor_rule':'gauss3x8'} if chosen else {'remainder_partition':(1,4)} if index!=67 else {};started=time.monotonic()
                try:
                    result=integrate_square(row['field'],prepare=row['prepare'],correction_integrator=row['correction_integrator'],tangent_correction_integrator=row['tangent_correction_integrator'],max_cells=1,tolerance='5e-13',checkpoint=check,**options)
                    complete=dict(value=[[str(v.lo),str(v.hi)] for v in result['value']],derivative=[[str(v.lo),str(v.hi)] for v in result['derivative']],error_accounting=result['error_accounting']);status='locally_verified'
                except Unresolved as error:
                    assert error.details['reason']=='cell budget';complete=error.details['last_complete_enclosure'];status='retained_uncertainty'
                pairs.append(dict(directional=chosen,seconds=time.monotonic()-started,complete=complete,status=status))
            for key in ('value','derivative'):
                for a,b in zip(pairs[0]['complete'][key],pairs[1]['complete'][key]):assert D(a[0])<=D(b[1]) and D(b[0])<=D(a[1])
            rows.append(dict(chart=index,cases=pairs))
    path=Path(os.environ['GFFX_FULL_REFERENCE']);assert not path.exists();path.write_text(json.dumps(dict(rows=rows,seconds=time.monotonic()-start,scope='Fresh one-cell local comparisons of every saved smooth contributor. Alternating order and full corrections; not a combined-piece certificate.'),default=str,indent=2),encoding='utf8');record_property('all_smooth_regions',str(path))
    assert len(rows)==15


@pytest.mark.parametrize('index',[53,63,65,67])
def test_tensor_smooth_complete_local_region(index,record_property):
    from tools.opaque_original_motion_reference import angular_material_cover
    from tools.opaque_square_reference import integrate_square
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    start=time.monotonic();failure=None
    def check():
        if time.monotonic()-start>=90:raise Unresolved('complete local tensor deadline')
    with backend(profile='ball'):
        p,curve,cover,charts=angular_material_cover(frozen(),tight_discriminant=True,algebraic_inverse=True,mapped_transport=True,exit_angle=True,unsquared_inverse=True,physical_angle=True,checkpoint=check)
        row=charts[index]
        try:result=integrate_square(row['field'],prepare=row['prepare'],correction_integrator=row['correction_integrator'],tangent_correction_integrator=row['tangent_correction_integrator'],tensor_rule='gauss3x8',max_cells=256,max_depth=10,tolerance='5e-13',checkpoint=check)
        except Unresolved as error:failure=str(error);result=dict(status='unresolved',details=error.details)
    path=Path(os.environ['GFFX_FULL_REFERENCE']);assert not path.exists();path.write_text(json.dumps(dict(chart=index,seconds=time.monotonic()-start,result=result,scope='Complete local region with all corrections; not a complete-piece certificate.'),default=str,indent=2),encoding='utf8');record_property('tensor_complete_local',str(path))
    assert failure is None,result
    assert all(v.width<=I('1e-12').lo for v in result['value']+result['derivative'])


def test_tensor_smooth_chart_connection_and_construction_deadline(monkeypatch):
    import tools.opaque_original_motion_reference as m
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    with pytest.raises(ValueError):m.angular_piece_reference({},tensor_rule='gauss3x8')
    with pytest.raises(ValueError):m.angular_material_cover({},tensor_rule='bad')
    with backend(profile='ball'):
        flags=dict(tight_discriminant=True,algebraic_inverse=True,mapped_transport=True,exit_angle=True,unsquared_inverse=True,physical_angle=True,tensor_rule='gauss3x8')
        with pytest.raises(ValueError):m.angular_material_cover(frozen(),smooth_partition=(1,4),**flags)
        p,curve,cover,charts=m.angular_material_cover(frozen(),**flags)
        selected=[i for i,row in enumerate(charts) if row.get('tensor_rule')]
        expected=[i for i,c in enumerate(cover.charts) if c['low'] is not None and c['high'] is not None]
        assert selected==expected and all(i in selected for i in [*range(40,54),63,65,67])
        assert 68 not in selected and 0 not in selected
        assert len(charts)==72 and cover.cells==143
        assert not any(c.get('remainder_partition') for c in charts)
        calls=[]
        def delayed(data,*,checkpoint,**options):
            calls.append(options);monkeypatch.setattr(m.time,'monotonic',lambda:100.);checkpoint()
        monkeypatch.setattr(m,'angular_material_cover',delayed);monkeypatch.setattr(m.time,'monotonic',lambda:0.)
        with pytest.raises(Unresolved) as error:m.angular_piece_reference({},**flags)
        assert 'shared original reference deadline' in str(error.value) and calls[0]['tensor_rule']=='gauss3x8'


def test_tensor_smooth_fresh_complete_original_piece(record_property):
    from tools.opaque_original_motion_reference import angular_piece_reference
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    failure=None;rows=[]
    with backend(profile='ball'):
        try:result=angular_piece_reference(frozen(),tight_discriminant=True,algebraic_inverse=True,mapped_transport=True,exit_angle=True,unsquared_inverse=True,physical_angle=True,tensor_rule='gauss3x8',progress=rows.append)
        except ValueError as error:failure=str(error);result=dict(status='unresolved',failure=getattr(error,'details',str(error)))
    result['progress']=rows;path=Path(os.environ['GFFX_FULL_REFERENCE']);assert not path.exists();path.write_text(json.dumps(result,default=str,indent=2),encoding='utf8');record_property('tensor_whole_piece',str(path))
    assert failure is None,result
    assert result['basis']['chart_count']==72 and result['basis']['cells']<=256
    assert all(v.width<=I('2e-12').lo for v in result['value']+result['derivative'])


@pytest.mark.parametrize('backend_name',['decimal','ball'])
def test_radial_quarter_peano_masses_enclose_independent_antiderivative(backend_name):
    from contextlib import nullcontext
    from tools.opaque_arb_jets import backend
    from tools.opaque_tensor_smooth import radial_quarter_factors
    with (backend(profile='ball',fast_conversion=True) if backend_name=='ball' else nullcontext()):
        root=I(Q(3,5)).sqrt()/2;nodes=[I('.5')-root,I('.5'),I('.5')+root];weights=[I(Q(5,18)),I(Q(4,9)),I(Q(5,18))]
        got=radial_quarter_factors();exact=[]
        def positive(q):return I(max(I(0).lo,q.lo),max(I(0).hi,q.hi))
        for k in range(4):
            a,b=I(Q(k,4)),I(Q(k+1,4))
            # Independent integrated K6 times6!, not the Bernstein routine.
            mass=((1-a)**7-(1-b)**7)/7-sum((w*(positive(x-a)**6-positive(x-b)**6) for x,w in zip(nodes,weights)),I(0))
            assert mass.lo>0 and got[k].hi>=mass.hi
            exact.append(mass)
        total=sum(exact,I(0));want=I(Q(math.factorial(3)**4,7*math.factorial(6)**2));assert total.lo<=want.lo<=want.hi<=total.hi
        assert sum(got,I(0)).hi<=2*want.hi


def test_biaxis_local_tensor_preserves_closed_integrals_and_all_contributions(monkeypatch):
    from tools.opaque_arb_jets import backend
    import tools.opaque_square_reference as square
    original=square.integrate_square
    def selected(*args,**kwargs):return original(*args,**dict(kwargs,tensor_rule='gauss3x8-local4x4'))
    monkeypatch.setattr(square,'integrate_square',selected)
    with backend(profile='ball',fast_conversion=True):
        for pair in ((0,0),(3,9),(5,15),(6,10),(8,15)):
            for tiny in (False,True):test_tensor_smooth_exact_integral_motion_and_tiny_terms(*pair,tiny)
        test_tensor_smooth_uses_24_samples_and_all_corrections()
        test_tensor_smooth_rational_integral_and_moving_denominator()


@pytest.mark.parametrize("engine",["decimal","ball"])
def test_gauss12_positive_rule_independent_moments_and_kernels(engine):
    from contextlib import nullcontext
    from tools.opaque_arb_jets import backend
    from tools.opaque_gauss12 import rule
    from tools.opaque_tensor_smooth import quarter_factors
    with (backend(profile='ball',fast_conversion=True) if engine=='ball' else nullcontext()):
        nodes,weights=rule();assert len(nodes)==len(weights)==12
        assert all(0<x.lo<=x.hi<1 and w.lo>0 for x,w in zip(nodes,weights))
        powers=[I(1) for x in nodes]
        for degree in range(24):
            exact=I(Q(1,degree+1));v=sum((w*x for w,x in zip(weights,powers)),I(0))
            assert v.lo<=exact.lo<=exact.hi<=v.hi
            powers=[v*x for v,x in zip(powers,nodes)]
        old=quarter_factors();new=quarter_factors(points=12)
        assert sum(v.hi for v in new)<sum(v.lo for v in old)/10
        # Independent antiderivatives bound the signed Peano10 mass on each quarter.
        fact=math.factorial(10)
        for k,mass in enumerate(new):
            a,b=I(Q(k,4)),I(Q(k+1,4))
            def power(v,n):
                q=I(1)
                for _ in range(n):q=q*v
                return q
            def primitive(t):
                return -sum((w*power(I(max(I(0).lo,(x-t).lo),max(I(0).hi,(x-t).hi)),10) for x,w in zip(nodes,weights)),I(0))/fact+power(1-t,11)/math.factorial(11)
            signed=(primitive(b)-primitive(a))*fact
            assert signed.bound<=mass.hi

@pytest.mark.parametrize("tiny",[False,True])
def test_tensor36_complete_integrals_corrections_and_work_charge(tiny):
    from tools.opaque_square_reference import integrate_square
    from tools.opaque_arb_jets import backend
    scale=Q(1,2**60) if tiny else Q(1)
    with backend(profile='ball',fast_conversion=True):
        def field(x,y):return [scale*(x**5*y**9+x**2)*(1+x.lift(0,tangent=-Q(2,3)))]
        kw=dict(tensor_rule='gauss3x12-local4',tolerance=str((I(scale)*I('1e-12')).lo),correction_integrator=lambda box:[I(scale/Q(2**40))],tangent_correction_integrator=lambda box:[I(-scale/Q(2**41))])
        r=integrate_square(field,max_cells=2,**kw)
        value=scale*(Q(1,60)+Q(1,3)+Q(1,2**40));grad=scale*(-Q(2,3)*(Q(1,60)+Q(1,3))-Q(1,2**41))
        for got,want in zip([r['value'][0],r['derivative'][0]],[I(value),I(grad)]):assert got.lo<=want.lo<=want.hi<=got.hi
        assert r['cells']==2 and r['quadrature_samples']==36 and r['work_charge']==2
        with pytest.raises(Unresolved,match='cell budget'):integrate_square(field,max_cells=1,**kw)
        row=dict(field=lambda x,y:[x+y],depth=1)
        r=integrate_square(None,chart_cover=[dict(row,tensor_rule='gauss3x12-local4'),row],max_cells=4)
        assert r['cells']==4 and r['quadrature_samples']==61
        calls=[]
        def fail(x,y):calls.append(1);raise Unresolved('domain')
        with pytest.raises(Unresolved) as caught:integrate_square(fail,tensor_rule='gauss3x12-local4',max_cells=2)
        assert caught.value.details['cells']==2 and len(calls)==1


def test_tensor36_saved_shear_regions_combined_bounds(record_property):
    import hashlib
    from tools.opaque_arb_jets import backend
    from tools.opaque_original_motion_reference import angular_material_cover
    from tools.opaque_square_reference import integrate_square
    from tests.packaging.test_opaque_motion_tangent import frozen
    out=Path(os.environ['GFFX_FULL_REFERENCE']).parent
    saved=json.loads((out/'shear8-r11-preview.json').read_bytes());profile=json.loads((out/'world-profile1-values.json').read_bytes())
    options=dict(saved['profile']);options.pop('tolerance');rows=[];start=time.monotonic()
    def check():
        if time.monotonic()-start>=90:raise Unresolved('saved tensor36 comparison deadline')
    with backend(profile='ball',fast_conversion=True):
        for selected in ('gauss3x8-local4',os.environ.get('GFFX_TENSOR_STRONG_RULE','gauss3x12-local4')):
            options['tensor_rule']=selected
            p,curve,cover,charts=angular_material_cover(frozen(),**saved['case'],world_profile=profile,checkpoint=check,**options)
            for index,coords in [(54,(0,1,0,1)),(65,(0,Q(1,2),0,1)),(70,(0,Q(1,2),0,Q(1,2)))]:
                a,b,c,d=coords;row=charts[index];mass=(b-a)*(d-c);begin=time.monotonic()
                def mapped(box):return (I(a)+(b-a)*box[0],I(c)+(d-c)*box[1])
                def field(x,y):return [q*mass for q in row['field'](a+(b-a)*x,c+(d-c)*y)]
                def prepare(x,y,box):row['prepare'](a+(b-a)*x,c+(d-c)*y,mapped(box))
                kw=dict(prepare=prepare,correction_integrator=lambda box:row['correction_integrator'](mapped(box)),tangent_correction_integrator=lambda box:row['tangent_correction_integrator'](mapped(box)),tensor_rule=row.get('tensor_rule'),checkpoint=check,max_cells=2 if row.get('tensor_rule') in ('gauss3x12-local4','gauss4x12-local4') else 1,tolerance='5e-13')
                try:
                    r=integrate_square(field,**kw);result={k:[[str(v.lo),str(v.hi)] for v in r[k]] for k in ('value','derivative')};result['error_accounting']=r['error_accounting'];samples=r['quadrature_samples']
                except Unresolved as e:
                    assert e.details['reason']=='cell budget';result=e.details['last_complete_enclosure'];samples=result['quadrature_samples']
                rows.append(dict(rule=selected,chart=index,box=list(map(str,coords)),seconds=time.monotonic()-begin,samples=samples,result=result))
        for x,y in zip(rows[:3],rows[3:]):
            for key in ('value','derivative'):
                for left,right in zip(x['result'][key],y['result'][key]):assert I(*left).lo<=I(*right).hi and I(*right).lo<=I(*left).hi
    path=Path(os.environ['GFFX_FULL_REFERENCE']);assert not path.exists();path.write_text(json.dumps(dict(rows=rows,seconds=time.monotonic()-start,tensor_charts=[i for i,r in enumerate(charts) if r.get('tensor_rule')],scope='Same complete saved regions, all smooth/correction/transport terms, no complete piece acceptance.'),indent=2),encoding='utf8');record_property('comparison',str(path))


@pytest.mark.parametrize('chosen',['gauss3x12-local4','gauss4x12-local4'])
def test_tensor36_chart_selection_is_explicit_and_bound(monkeypatch,chosen):
    import tools.opaque_original_motion_reference as m
    from tests.packaging.test_opaque_motion_tangent import frozen
    from tools.opaque_arb_jets import backend
    selected=[40,54,65];flags=dict(tight_discriminant=True,algebraic_inverse=True,mapped_transport=True,exit_angle=True,unsquared_inverse=True,physical_angle=True,tensor_rule=chosen,tensor_upgrade_charts=selected)
    with backend(profile='ball',fast_conversion=True):
        p,curve,cover,charts=m.angular_material_cover(frozen(),**flags)
        assert [i for i,r in enumerate(charts) if r.get('tensor_rule')==chosen]==selected
        assert charts[1]['tensor_rule']=='gauss3x8-local4' and charts[0].get('tensor_rule') is None
        for bad in ([True],[40,40],[-1],[999],[0]):
            with pytest.raises(ValueError,match='upgrade'):m.angular_material_cover(frozen(),**dict(flags,tensor_upgrade_charts=bad))
        class Reached(Exception):pass
        def bound(*a,**k):assert k['tensor_upgrade_charts']==selected;raise Reached
        monkeypatch.setattr(m,'angular_material_cover',bound)
        with pytest.raises(Reached):m.angular_piece_reference(frozen(),**flags)


def test_tensor48_radial_moments_full_integrals_and_work():
    from tools.opaque_tensor_smooth import rule
    from tools.opaque_square_reference import integrate_square
    from tools.opaque_arb_jets import backend
    with backend(profile='ball',fast_conversion=True):
        (nodes,weights),_,orders,_=rule(points=12,radial=4)
        assert orders==(8,10)
        for degree in range(8):
            got=sum((w*x**degree for x,w in zip(nodes,weights)),I(0));want=I(Q(1,degree+1));assert got.lo<=want.lo<=want.hi<=got.hi
        tiny=Q(1,2**60)
        def field(x,y):return [(x**7*y**9+tiny)*(1+x.lift(0,tangent=-tiny))]
        r=integrate_square(field,tensor_rule='gauss4x12-local4',max_cells=2,tolerance='5e-13',correction_integrator=lambda b:[I(tiny)],tangent_correction_integrator=lambda b:[I(-tiny)])
        for got,want in zip((r['value'][0],r['derivative'][0]),(I(Q(1,80)+2*tiny),I(-tiny*(Q(1,80)+tiny)-tiny))):assert got.lo<=want.lo<=want.hi<=got.hi
        assert r['cells']==2 and r['quadrature_samples']==48 and r['work_charge']==2
        with pytest.raises(Unresolved,match='cell budget'):integrate_square(field,tensor_rule='gauss4x12-local4',max_cells=1)
