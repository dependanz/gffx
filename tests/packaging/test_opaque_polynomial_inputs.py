"""Independent closed Taylor obligations for exact polynomial cosine inputs."""
from fractions import Fraction as Q
from decimal import Decimal as D,localcontext
from math import comb
import pytest
from tools.opaque_shading_intervals import I,Jet
from tools.opaque_arb_jets import backend


def fields(omega,view=False):
    if view:return {'n':[[Q(0)]*3,[Q(0)]*3,[[Q(1),0],Q(0),Q(0)]],'v':[[[0,omega],Q(1),Q(0)],[Q(0)]*3,[Q(1),Q(0),Q(0)]]}
    return {'n':[[Q(1),Q(0),Q(0)],[Q(0),Q(1),Q(0)],[[0,-omega],Q(0),Q(0)]],'v':[[Q(0)]*3,[Q(0)]*3,[Q(1),Q(0),Q(0)]]}


def pairs(co):
    # The radial-field interface is an exact first-parameter pair per coefficient.
    return {name:[[v if isinstance(v,list) else [v,Q(0)] for v in row] for row in rows] for name,rows in co.items()}


def binomial(i):return Q((-1)**(i//2)*comb(i,i//2),4**(i//2)) if i>=0 and i%2==0 else Q(0)

def contains_closed(interval,rational,*,sqrt6=False):
    with localcontext() as c:
        c.prec=120;v=D(rational.numerator)/D(rational.denominator)
        if sqrt6:v/=D(6).sqrt()
        assert interval.lo<=v<=interval.hi,(interval,v)


@pytest.mark.parametrize('order',[0,1,4,10])
@pytest.mark.parametrize('omega',[Q(1,100),Q(1,10**30)])
def test_rotating_normal_has_closed_full_taylor_and_tiny_motion(order,omega):
    from tools.opaque_polynomial_inputs import PolynomialCosines
    with backend(profile='ball',fast_conversion=True):
        model=PolynomialCosines(pairs(fields(omega)));u=Jet.variable(I(0),order=order) if order else Jet(0);s=Jet(0,order=order);nv,nl,vh=model(u,s)
        for i in range(order+1):
            contains_closed(nv[i,0],Q(0));contains_closed(nv[i,1],-omega*binomial(i))
            contains_closed(nl[i,0],binomial(i)+binomial(i-1),sqrt6=True);contains_closed(nl[i,1],-2*omega*binomial(i),sqrt6=True)
        assert nv[0,1].hi<0 and nl[0,1].hi<0
        assert not any(k[2] for k in model.normal_norm) # exact intrinsic rotational norm cancellation


@pytest.mark.parametrize('order',[0,4,10])
def test_moving_view_has_closed_full_taylor(order):
    from tools.opaque_polynomial_inputs import PolynomialCosines
    delta=Q(1,10**20)
    with backend(profile='ball',fast_conversion=True):
        model=PolynomialCosines(pairs(fields(delta,view=True)));u=Jet.variable(I(0),order=order) if order else Jet(0);nv,nl,vh=model(u,Jet(0,order=order))
        for i in range(order+1):
            contains_closed(nv[i,0],binomial(i));contains_closed(nv[i,1],delta*(i+1)*binomial(i+1))
            contains_closed(nl[i,0],Q(2) if i==0 else Q(0),sqrt6=True);contains_closed(nl[i,1],Q(0))


def test_coordinate_motion_retains_all_mixed_terms_after_norm_cancellation():
    from tools.opaque_polynomial_inputs import PolynomialCosines
    omega=Q(1,100);delta=Q(1,10**20);order=10
    with backend(profile='ball',fast_conversion=True):
        model=PolynomialCosines(pairs(fields(omega)));u=Jet.variable(I(0),order=order);u.c[0,1]=I(delta);nv,nl,vh=model(u,Jet(0,order=order))
        for i in range(order+1):
            contains_closed(nv[i,1],-omega*binomial(i))
            contains_closed(nl[i,1],delta*(i+1)*(binomial(i+1)+binomial(i))-2*omega*binomial(i),sqrt6=True)


@pytest.mark.parametrize('bad',['normal','view','orders','shape'])
def test_polynomial_inputs_refuse_invalid_norms_or_jets(bad):
    from tools.opaque_polynomial_inputs import PolynomialCosines
    co=pairs(fields(Q(1,100)))
    if bad in ('normal','view'):co['n' if bad=='normal' else 'v']=[[[Q(0),Q(0)]]*3]*3
    if bad=='shape':co['n']=co['n'][:2]
    with backend(profile='ball',fast_conversion=True):
        with pytest.raises(ValueError):PolynomialCosines(co)(Jet(0,order=1),Jet(0,order=0 if bad=='orders' else 1))


@pytest.mark.parametrize('motion',['normal_tilt','view_x','screen_x'])
@pytest.mark.parametrize('order',[0,1,4,10])
def test_original_front_inputs_keep_all_coefficients_and_jacobian(motion,order):
    from tools.opaque_polynomial_inputs import PolynomialCosines
    from tools.opaque_original_motion_reference import prepare,mapped_inputs
    from tests.packaging.test_opaque_motion_tangent import frozen
    with backend(profile='ball',fast_conversion=True):
        p=prepare(frozen(),piece=3,motion=motion,anchor=0);model=PolynomialCosines(p['fields'].coefficients)
        u=Jet.variable(I('.65','.66'),order=order) if order else Jet(I('.65','.66'));s=Jet(I('.5','.6'),order=order)
        old=mapped_inputs(p,u,s);new=model(u,s)
        for a,b in zip(old[:3],new):
            for i in range(order+1):
                for j in (0,1):assert max(a[i,j].lo,b[i,j].lo)<=min(a[i,j].hi,b[i,j].hi)
        jac=u*u.lift(p['det'][0],tangent=p['det'][1])
        for i in range(order+1):
            for j in (0,1):assert (jac[i,j].lo,jac[i,j].hi)==(old[3][i,j].lo,old[3][i,j].hi)


@pytest.mark.parametrize('profile',[dict(polynomial_inputs=1),dict(polynomial_inputs=True),dict(polynomial_inputs=True,conditioned_correction=True,correction_order=4)])
def test_polynomial_reference_rejects_invalid_or_unqualified_profile(profile):
    from tools.opaque_original_motion_reference import reference
    with pytest.raises(ValueError,match='polynomial inputs require'):reference({},piece=3,motion='normal_tilt',**profile)


def test_polynomial_reference_uses_full_model_and_original_jacobian(monkeypatch):
    import tools.opaque_original_motion_reference as m
    from tools.opaque_polynomial_inputs import PolynomialCosines
    from tests.packaging.test_opaque_motion_tangent import frozen
    called=[];old=PolynomialCosines.__call__
    def model(self,*a,**kw):called.append(kw);return old(self,*a,**kw)
    monkeypatch.setattr(PolynomialCosines,'__call__',model)
    monkeypatch.setattr(m,'mapped_inputs',lambda *a,**kw:pytest.fail('selected front model invoked old input evaluator'))
    def integrate(field,**kw):
        chart=kw['chart_cover'][0] if 'chart_cover' in kw else dict(kw,field=field);box=(I('.65'),I('.55'));chart['prepare'](Q(65,100),Q(55,100),box)
        r=chart['field'](Jet.variable(box[0],order=1),Jet(box[1],order=1));assert len(r)==2 and all(isinstance(v,Jet) for v in r)
        assert kw['max_cells']==256 and kw['max_depth']==10 and kw['tolerance']=='5e-13'
        return dict(value=[I(0)]*2,derivative=[I(0)]*2)
    monkeypatch.setattr(m,'integrate_square',integrate)
    with backend(profile='ball',fast_conversion=True):r=m.reference(frozen(),piece=3,motion='normal_tilt',anchor=0,polynomial_inputs=True,correction_order=4,conditioned_correction=True,mixed_curve=True,transverse_correction=True,correction_axis=0,material_inputs=True)
    assert r['polynomial_inputs'] is True and called and r['piece']==3 and r['motion']=='normal_tilt'


def test_polynomial_reference_refuses_unqualified_curved_mapping():
    from tools.opaque_original_motion_reference import reference
    from tests.packaging.test_opaque_motion_tangent import frozen
    with backend(profile='ball',fast_conversion=True):
        with pytest.raises(ValueError,match='polynomial inputs require noncurved'):reference(frozen(),piece=9,motion='normal_tilt',polynomial_inputs=True,correction_order=4,conditioned_correction=True,mixed_curve=True,transverse_correction=True,correction_axis=0,material_inputs=True)



def test_saved_front_polynomial_input_cost_and_bounds(record_property):
    import os,json,hashlib,time
    from pathlib import Path
    from tools.opaque_original_motion_reference import reference
    from tools.opaque_shading_intervals import Unresolved
    from tests.packaging.test_opaque_motion_tangent import frozen
    pin=Path(os.environ['GFFX_FRONT_DIAGNOSTIC']);assert hashlib.sha256(pin.read_bytes()).hexdigest()==os.environ['GFFX_FRONT_DIAGNOSTIC_SHA']
    spec=json.loads(pin.read_bytes());assert len(spec['cases'])==3;start=time.monotonic();rows=[]
    profile=dict(correction_order=4,conditioned_correction=True,mixed_curve=True,transverse_correction=True,correction_axis=0,material_inputs=True)
    with backend(profile='ball',fast_conversion=True):
        for index,row in enumerate(spec['cases']):
            src=Path(row['path']);assert hashlib.sha256(src.read_bytes()).hexdigest()==row['sha256'];j=json.loads(src.read_bytes());r=j['result']
            assert j['status']=='qualified' and r['piece']==row['piece'] and r['motion']=='normal_tilt' and r['anchor']==0 and not r['curved']
            boxes=[tuple(map(Q,v['box'])) for v in r['basis']['error_accounting']['largest_leaves'][:3]];assert len(boxes)==3
            for selected in ([False,True] if index%2==0 else [True,False]):
                if time.monotonic()-start>=90:raise Unresolved('saved polynomial diagnostic deadline')
                before=time.monotonic();v=reference(frozen(),piece=row['piece'],motion='normal_tilt',anchor=0,diagnostic_boxes=boxes,polynomial_inputs=selected,**profile)
                assert v['status']=='diagnostic_only' and v['piece']==row['piece'];rows.append(dict(piece=row['piece'],polynomial_inputs=selected,seconds=time.monotonic()-before,result=v))
                if time.monotonic()-start>=90:raise Unresolved('saved polynomial diagnostic deadline')
    p=Path(os.environ['GFFX_FULL_REFERENCE']);assert not p.exists();p.write_text(json.dumps(dict(status='diagnostic_complete',selection_sha256=os.environ['GFFX_FRONT_DIAGNOSTIC_SHA'],seconds=time.monotonic()-start,rows=rows,scope='Alternating local cost and correction bounds on three saved regions each for pieces2/3/10. Independent closed Taylor controls qualify the input algebra; separate component intervals need not represent the same chosen extension. No complete-piece or runtime admission.'),default=str,indent=2),encoding='utf8');record_property('polynomial_input_diagnostic',str(p))


@pytest.mark.parametrize('order',[0,4,10])
def test_light_only_keeps_closed_coordinate_chain_without_unused_view_work(monkeypatch,order):
    from tools.opaque_polynomial_inputs import PolynomialCosines
    omega=Q(1,100);delta=Q(1,10**20);co=pairs(fields(omega));co['v']=[[[Q(0),Q(0)]]*3]*3
    with backend(profile='ball',fast_conversion=True):
        model=PolynomialCosines(co);allowed=[p for index in (0,3) for p in model.prepared[index]];old=model._evaluate
        def evaluate(p,u,s):
            assert any(p is a for a in allowed),'unused view/dot polynomial evaluated';return old(p,u,s)
        monkeypatch.setattr(model,'_evaluate',evaluate);u=Jet.variable(I(0),order=order) if order else Jet(0);u.c[0,1]=I(delta)
        nl,vh=model.light_highlight(u,Jet(0,order=order),material_only=True)
        for i in range(order+1):
            contains_closed(nl[i,0],binomial(i)+binomial(i-1),sqrt6=True)
            contains_closed(nl[i,1],delta*(i+1)*(binomial(i+1)+binomial(i))-2*omega*binomial(i),sqrt6=True)
        assert not vh.c

@pytest.mark.parametrize('bad',['orders','flag','norm'])
def test_light_highlight_rejects_invalid_inputs(bad):
    from tools.opaque_polynomial_inputs import PolynomialCosines
    co=pairs(fields(Q(1,100)))
    if bad=='norm':co['n']=[[[Q(0),Q(0)]]*3]*3
    with backend(profile='ball',fast_conversion=True):
        with pytest.raises(ValueError):PolynomialCosines(co).light_highlight(Jet(0,order=1),Jet(0,order=0 if bad=='orders' else 1),material_only=1 if bad=='flag' else True)

@pytest.mark.parametrize('motion',['normal_tilt','view_x','screen_x'])
@pytest.mark.parametrize('order',[0,1,4,10])
def test_physical_light_highlight_keeps_original_full_coefficients(motion,order):
    from tools.opaque_polynomial_inputs import PolynomialCosines
    from tools.opaque_original_motion_reference import prepare,mapped_inputs
    from tests.packaging.test_opaque_motion_tangent import frozen
    with backend(profile='ball',fast_conversion=True):
        p=prepare(frozen(),piece=8,motion=motion,anchor=0);model=PolynomialCosines(p['fields'].coefficients)
        u=Jet.variable(I('.65','.66'),order=order) if order else Jet(I('.65','.66'));s=Jet(I('.5','.6'),order=order)
        t,jac=p['patch'].radial_map(u,s);old=mapped_inputs(p,u,s,flatten=False);new=model.light_highlight(t,s)
        for a,b in zip(old[1:3],new):
            for i in range(order+1):
                for j in (0,1):assert max(a[i,j].lo,b[i,j].lo)<=min(a[i,j].hi,b[i,j].hi)



@pytest.mark.parametrize('order',[0,1,4,10])
def test_polynomial_row_reuse_keeps_frozen_endpoints_and_private_values(order):
    from tools.opaque_angle_light import PreparedIntervalPolynomial
    from tools.opaque_polynomial_inputs import PolynomialCosines
    def frozen(p,u,s):
        p.prepare();value=u.lift(0)
        for row in p.rows:
            q=s.lift(0)
            for c in row:q=q*s+c
            value=value*u+q
        return value
    encode=lambda q:tuple((k,v.lo,v.hi) for k,v in q.c.items())
    with backend(profile='ball',fast_conversion=True):
        p=PreparedIntervalPolynomial({(2,2,0):Q(3,7),(1,1,0):Q(-1,11),(0,0,0):Q(1,10**30)})
        s=Jet(I('.4','.41'),order=order,tangent=Q(1,10**20))
        if order:s.c[1,0]=I(Q(1,3))
        for x in ('.2','.3','.25'):
            u=Jet(I(x),order=order,tangent=Q(-1,10**25))
            if order:u.c[1,0]=I(1)
            got=PolynomialCosines._evaluate(p,u,s);want=frozen(p,u,s)
            assert encode(got)==encode(want)
            got.c[0,0]=I(999)
        assert p.jet_row_evaluations==1 and p.jet_row_hits==2
        assert len(p.jet_rows)==1


def test_polynomial_row_reuse_bounds_storage_and_rebuilds_after_context_change():
    from tools.opaque_angle_light import PreparedIntervalPolynomial
    from tools.opaque_polynomial_inputs import PolynomialCosines
    p=PreparedIntervalPolynomial({(1,1,0):Q(1,7),(0,0,0):Q(1,10**30)})
    with backend(profile='ball',fast_conversion=True):
        for i in range(80):PolynomialCosines._evaluate(p,Jet('.2',order=1),Jet(Q(i,100),order=1))
        assert len(p.jet_rows)==64
        before=p.jet_row_evaluations
    with backend(profile='ball',fast_conversion=True):
        PolynomialCosines._evaluate(p,Jet('.2',order=1),Jet(Q(79,100),order=1))
        assert p.jet_row_evaluations==before+1



def test_polynomial_horner_omits_only_leading_exact_zero_products(monkeypatch):
    from tools.opaque_angle_light import PreparedIntervalPolynomial
    from tools.opaque_polynomial_inputs import PolynomialCosines
    with backend(profile='ball',fast_conversion=True):
        original=Jet.__mul__;empty=[]
        def observed(a,b):
            if not a.c:empty.append(1)
            return original(a,b)
        monkeypatch.setattr(Jet,'__mul__',observed)
        p=PreparedIntervalPolynomial({(2,2,0):Q(3,7),(1,1,0):Q(-1,11),(0,0,0):Q(1,10**30)})
        u=Jet.variable(I('.2'),order=4);s=Jet(I('.4'),order=4,tangent=Q(1,10**30))
        got=PolynomialCosines._evaluate(p,u,s)
        assert not empty and got[0,1].hi<0


def test_angle_differential_reuses_only_identical_scalar_domains(monkeypatch):
    from tools.opaque_angle_light import AngleLightDifferential,PreparedIntervalPolynomial
    import tools.opaque_shading_intervals as scalar
    co=pairs({'n':[[1,-1,0],[0,0,0],[1,0,0]],'v':[[0,0,0],[0,0,0],[1,0,0]]})
    model=AngleLightDifferential(co);calls=[];original=PreparedIntervalPolynomial.__call__
    def counted(self,*a):calls.append(1);return original(self,*a)
    monkeypatch.setattr(PreparedIntervalPolynomial,'__call__',counted)
    def run(u):
        return model.bound(Jet(I(u),order=1),Jet(I('.3'),order=1),Jet.variable(I('.7'),order=1)+Jet(0,order=1,tangent=-Q(1,10**40)),(1,0))
    first=run('.2');second=run('.3');before=len(calls);got=run('.2')
    assert len(calls)==before
    assert (got.lo,got.hi)==(first.lo,first.hi)
    with localcontext() as c:
        c.prec=100;exact=D('.75')/D(6).sqrt();assert got.lo<=exact<=got.hi
    tangent=model.bound(Jet(I('.2'),order=1),Jet(I('.3'),order=1),Jet(I('.7'),order=1,tangent=-Q(1,10**40)),(0,1))
    with localcontext() as c:
        c.prec=100;exact=-D('.75')/D(6).sqrt()*D('1e-40');assert tangent.lo<=exact<=tangent.hi and not tangent.zero
    got.lo=D(-99);again=run('.2');assert (again.lo,again.hi)==(first.lo,first.hi)
    for i in range(1,80):run(Q(i,100))
    assert len(model.scalar_coefficients)<=64
    before=len(calls);monkeypatch.setattr(scalar,'PRECISION',61);run('.2');assert len(calls)>before


@pytest.mark.parametrize('order',[0,6,10])
@pytest.mark.parametrize('precision',[40,60,80])
def test_jet_addition_shares_context_without_changing_frozen_endpoints(monkeypatch,order,precision):
    import os,hashlib,importlib.util
    from pathlib import Path
    import tools.opaque_shading_intervals as current
    source=Path(os.environ['GFFX_FROZEN_INTERVAL_SOURCE']);assert hashlib.sha256(source.read_bytes()).hexdigest()==os.environ['GFFX_FROZEN_INTERVAL_SHA']
    spec=importlib.util.spec_from_file_location('frozen_interval_addition',source);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    old.PRECISION=precision;monkeypatch.setattr(current,'PRECISION',precision)
    pairs=[((i,j),(-Q(i+1,3),Q(i+1,7)) if (i+j)%2 else (Q(1,10**70),Q(1,10**69))) for i in range(order+1) for j in (0,1)]
    def jets(module):
        a=module.Jet(order=order,coeff={k:module.I(module.I(lo).lo,module.I(hi).hi) for k,(lo,hi) in pairs})
        b=module.Jet(order=order,coeff={k:module.I(module.I(Q(1,13)).lo,module.I(Q(2,13)).hi) for k,_ in pairs if k[1]})
        return a,b
    a,b=jets(old);expected=a+b;a,b=jets(current);calls=[];lc=current.localcontext
    def counted(*a,**k):calls.append(1);return lc(*a,**k)
    monkeypatch.setattr(current,'localcontext',counted);actual=a+b
    encode=lambda q:{k:(v.lo,v.hi) for k,v in q.c.items()}
    assert encode(actual)==encode(expected)
    assert len(calls)<=1
    assert actual[0,0].lo>0
    before=encode(a);actual.c[0,0].lo=D(-99);assert encode(a)==before


@pytest.mark.parametrize('precision',[40,60,80])
def test_prepared_scalar_horner_shares_context_and_keeps_frozen_endpoints(monkeypatch,precision):
    import os,hashlib,importlib.util
    from pathlib import Path
    import tools.opaque_shading_intervals as scalar
    from tools.opaque_angle_light import PreparedIntervalPolynomial
    path=Path(os.environ['GFFX_FROZEN_POLYNOMIAL_SOURCE']);assert hashlib.sha256(path.read_bytes()).hexdigest()==os.environ['GFFX_FROZEN_POLYNOMIAL_SHA']
    spec=importlib.util.spec_from_file_location('scalar_horner_frozen',path);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    monkeypatch.setattr(scalar,'PRECISION',precision)
    poly={(4,3,0):Q(7,13),(2,2,0):-Q(3,7),(1,0,0):Q(1,10**80),(0,1,0):Q(11,19),(0,0,0):-Q(1,3)}
    oldp=old.PreparedIntervalPolynomial(poly);newp=PreparedIntervalPolynomial(poly)
    domains=[I('-2','-1'),I('-1','1'),I('0','1'),I('1','2'),I(0),I(Q(1,10**70))]
    expected={(i,j):oldp(a,b) for i,a in enumerate(domains) for j,b in enumerate(domains)}
    newp.prepare();calls=[];multiply=I.__mul__
    def track(a,b):calls.append(1);return multiply(a,b)
    monkeypatch.setattr(I,'__mul__',track)
    for i,a in enumerate(domains):
        for j,b in enumerate(domains):
            got=newp(a,b);want=expected[i,j];assert (got.lo,got.hi)==(want.lo,want.hi)
            saved=(got.lo,got.hi);got.lo=I(-100).lo;again=newp(a,b);assert (again.lo,again.hi)==saved
    assert not calls,'scalar Horner should share directed context without interval temporary products'
