"""A bounded table candidate must earn value, derivative and energy acceptance."""
import importlib
import inspect
import math
import sys
from pathlib import Path

import pytest
torch=pytest.importorskip('torch')
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.append(str(ROOT))
from tools import material_budget_reference as reference
from tools import shading_energy_probe as oracle


def module():
    return importlib.import_module('tools.material_budget_table')


@pytest.fixture(autouse=True)
def bounded_test_threads():
    previous=torch.get_num_threads();torch.set_num_threads(1)
    try:yield
    finally:torch.set_num_threads(previous)


@pytest.fixture(scope='module')
def tables():
    previous=torch.get_num_threads();torch.set_num_threads(1)
    try:
        mod=module()
        yield (mod.generate(roughness_nodes=17,cosine_nodes=129,resolution=128),
               mod.generate(roughness_nodes=33,cosine_nodes=257,resolution=256))
    finally: torch.set_num_threads(previous)


def t(value,grad=False):
    return torch.tensor(value,dtype=torch.float64,requires_grad=grad)


@pytest.mark.parametrize('c,r,f',[(.17,.37,.11),(.43,.63,.29),(.79,.87,.73)])
def test_table_held_out_values_and_derivatives(tables,c,r,f,request):
    coarse,fine=tables
    args=tuple(t(x,True) for x in (c,r,f))
    result=fine.evaluate(*args)['directional']
    expected=reference.directional(*args,resolution=256)
    grad=torch.autograd.grad(result,args)
    exact_grad=torch.autograd.grad(expected,args)
    report=dict(point=[c,r,f],value=result.item(),reference=expected.item(),
                gradient=[g.item() for g in grad],reference_gradient=[g.item() for g in exact_grad],
                value_refinement=abs(result.item()-coarse.evaluate(*args)['directional'].item()))
    plugin=request.config.pluginmanager.get_plugin('tools.testing_dashboard.pytest_plugin')
    if plugin: plugin.Evidence(request.node.nodeid).record('table-value-and-gradient',report)
    assert abs(result.item()-expected.item()) < 5e-4
    assert all(abs(a.item()-b.item()) <= .01+.02*abs(b.item()) for a,b in zip(grad,exact_grad))
    assert report['value_refinement'] < .001


def test_table_average_is_its_exact_cosine_weighted_mean(tables):
    table=tables[1]
    # Two Gauss nodes integrate a quadratic curve times cosine exactly.
    c=table.cosines; lo,hi=c[:-1],c[1:]; mid=(lo+hi)/2; half=(hi-lo)/2
    nodes=torch.stack((mid-half/math.sqrt(3),mid+half/math.sqrt(3)),-1)
    r,f=t(.633,True),t(.29,True)
    values=table.evaluate(nodes,r.expand_as(nodes),f.expand_as(nodes))
    actual=(values['directional']*nodes*half[:,None]*2).sum()
    expected=table.evaluate(t(.4),r,f)['average']
    assert torch.allclose(actual,expected,atol=1e-13,rtol=1e-13)
    ga=torch.autograd.grad(actual,(r,f));gb=torch.autograd.grad(expected,(r,f))
    assert all(torch.allclose(a,b,atol=1e-12,rtol=1e-12) for a,b in zip(ga,gb))


def test_table_keeps_the_existing_independent_furnace_gate(tables,request):
    coarse,fine=tables; rows=[]
    mu=(torch.arange(8192,dtype=torch.float64)+.5)/8192
    for r in (.2,.5,1.):
        for f in (0.,.04,.5,1.):
            light_budgets=[table.evaluate(mu,t(r).expand_as(mu),t(f).expand_as(mu)) for table in tables]
            for c in (.1,.5,1.):
                spec=oracle.furnace(c,r,f0=f,model='specular',resolution=2048)
                for a in (0.,.5,1.):
                    values=[]
                    for table,light in zip(tables,light_budgets):
                        view=table.evaluate(t(c),t(r),t(f))
                        matte=oracle.coupled_diffuse(t(a),view['directional'],light['directional'],view['average'])
                        values.append(spec+float((matte*mu).mean()*2*math.pi))
                    rows.append(dict(r=r,f0=f,c=c,albedo=a,coarse=values[0],fine=values[1],delta=abs(values[1]-values[0])))
    plugin=request.config.pluginmanager.get_plugin('tools.testing_dashboard.pytest_plugin')
    if plugin: plugin.Evidence(request.node.nodeid).record('table-independent-furnace',dict(rows=rows))
    assert len(rows)==108
    assert max(row['delta'] for row in rows) < .001
    assert all(0 <= row['fine'] <= 1.001 for row in rows)
    assert all(abs(row['fine']-1) < .001 for row in rows if row['albedo']==1)


def test_table_preserves_tensor_shapes_fresnel_linearity_and_gradcheck(tables):
    table=tables[1]
    c=t([[.43,.57],[.63,.77]],True);r=t(.63,True);f=t(.29,True)
    def evaluate(c,r,f):
        values=table.evaluate(c,r,f)
        return torch.stack((values['directional'],values['average']))
    assert torch.autograd.gradcheck(evaluate,(c,r,f),eps=1e-6,atol=1e-5,rtol=1e-3)
    result=table.evaluate(c,r,f)
    zero=table.evaluate(c,r,t(0.));one=table.evaluate(c,r,t(1.))
    for key in result:
        assert result[key].shape==c.shape
        assert torch.allclose(result[key],zero[key]+f*(one[key]-zero[key]),atol=1e-13)
    assert table.evaluate(c.float(),r.float(),f.float())['directional'].dtype==torch.float32
    assert table.evaluate(c[:,:0],r,f)['directional'].shape==(2,0)


def test_table_range_errors_never_silently_clamp(tables):
    table=tables[1]
    for c,r,f in ((-.1,.5,.2),(1.1,.5,.2),(.5,.19,.2),(.5,1.1,.2),(.5,.5,-.1),(.5,.5,float('nan'))):
        with pytest.raises(ValueError):table.evaluate(t(c),t(r),t(f))
    with pytest.raises(ValueError):table.evaluate(t([.4,.5]),t([.5,.6,.7]),t(.2))
    with pytest.raises(TypeError):table.evaluate(t(.4).float(),t(.5),t(.2))
    with pytest.raises(ValueError):table.evaluate(t(.4).to('meta'),t(.5),t(.2))
    value=table.evaluate(t([0.,1.]),t(.2),t(.04))
    assert all(torch.isfinite(x).all() for x in value.values())


def test_table_generation_is_bounded_and_reproducible(monkeypatch):
    mod=module()
    for kwargs in (dict(roughness_nodes=True),dict(roughness_nodes=10000),dict(cosine_nodes=1),dict(resolution=1024)):
        with pytest.raises(ValueError):mod.generate(**kwargs)
    # Deterministic guard test, not dependent on eigensolver roundoff direction.
    with monkeypatch.context() as patch:
        patch.setattr(mod,'directional_albedo_endpoints',lambda *args,**kwargs:(.1,1.01))
        with pytest.raises(ValueError,match='energy'):
            mod.generate(roughness_nodes=3,cosine_nodes=5,resolution=16)
    a=mod.generate(roughness_nodes=3,cosine_nodes=5,resolution=128)
    b=mod.generate(roughness_nodes=3,cosine_nodes=5,resolution=128)
    assert torch.equal(a.energy,b.energy)
    assert not a.energy.requires_grad
    assert a.metadata['roughness_min']==.2
    assert a.metadata['sample_count']==3*5*128*256


def test_table_average_agrees_with_independent_differentiable_reference(tables):
    r,f=t(.633,True),t(.29,True)
    actual=tables[1].evaluate(t(.4),r,f)['average']
    expected=reference.average(r,f,resolution=128,view_resolution=64)
    assert abs(actual.item()-expected.item()) < 5e-4
    ga=torch.autograd.grad(actual,(r,f));ge=torch.autograd.grad(expected,(r,f))
    assert all(abs(a.item()-b.item()) <= .01+.02*abs(b.item()) for a,b in zip(ga,ge))


def test_table_demo_exposes_value_and_derivative_approximation(tables):
    demo=importlib.import_module('tools.testing_dashboard.material_table_demo')
    data=demo.examples(tables[1])
    assert len(data['rows'])==41
    assert len(data['grazing'])==6
    for row in data['grazing']:
        assert abs(row['table']-row['reference']) < 5e-4
        assert abs(row['table_gradient']-row['reference_gradient']) <= .01+.02*abs(row['reference_gradient'])
    assert data['table_bytes']==138544
    assert max(abs(row['table']-row['reference']) for row in data['rows']) < .001
    assert all(math.isfinite(row['table_gradient']) for row in data['rows'])
    assert 'piecewise' in data['scope']
    from tools.testing_dashboard.build_lessons import table_section
    assert 'not zero roughness' in table_section()


def evidence(request, name, report):
    plugin=request.config.pluginmanager.get_plugin('tools.testing_dashboard.pytest_plugin')
    if plugin: plugin.Evidence(request.node.nodeid).record(name,report)


def value_and_gradient(function, args):
    value=function(*args)
    gradient=torch.autograd.grad(value,args)
    return value.item(),torch.cat([g.reshape(-1) for g in gradient]).detach()


def comparison(actual, expected, lower_reference):
    value,gradient=actual; exact,exact_gradient=expected
    lower,lower_gradient=lower_reference
    return dict(value=value,reference=exact,gradient=gradient.tolist(),
        reference_gradient=exact_gradient.tolist(),value_error=abs(value-exact),
        gradient_error=(gradient-exact_gradient).abs().tolist(),
        reference_value_change=abs(exact-lower),
        reference_gradient_change=(exact_gradient-lower_gradient).abs().tolist(),
        reference_converged=abs(exact-lower)<=5e-5 and bool(
            ((exact_gradient-lower_gradient).abs()<=.001+.002*exact_gradient.abs()).all()),
        candidate_matches=abs(value-exact)<5e-4 and bool(
            ((gradient-exact_gradient).abs()<=.01+.02*exact_gradient.abs()).all()))


@pytest.mark.parametrize('c',[.025,.11,.43,.79,.97,.013,.019,.031,.047,.073],
                         ids=['grazing','low','mid','high','near-normal','extra-013','extra-019','extra-031','extra-047','extra-073'])
@pytest.mark.parametrize('r',[.213,.337,.613,.937],ids=['sharp-edge','glossy','rough','very-rough'])
def test_table_expanded_directional_accuracy(tables,c,r,request):
    rows=[]
    for f in (0.,.04,1.):
        args=tuple(t(x,True) for x in (c,r,f))
        actual=value_and_gradient(lambda *xs:tables[1].evaluate(*xs)['directional'],args)
        exact=value_and_gradient(lambda *xs:reference.directional(*xs,resolution=512),args)
        lower=value_and_gradient(lambda *xs:reference.directional(*xs,resolution=256),args)
        row=comparison(actual,exact,lower)
        row.update(cosine=c,roughness=r,f0=f,
            table_refinement=abs(actual[0]-tables[0].evaluate(*args)['directional'].item()))
        rows.append(row)
    evidence(request,'expanded-directional',dict(rows=rows,
        derivative_order=['view cosine','roughness','F0'],
        scope='Reference convergence first; candidate screening, not public admission.'))
    assert all(row['reference_converged'] for row in rows), 'REFERENCE UNRESOLVED: '+str(rows)
    assert all(row['candidate_matches'] for row in rows), 'TABLE ACCURACY: '+str(rows)
    assert all(row['table_refinement']<.001 for row in rows), 'TABLE REFINEMENT: '+str(rows)


@pytest.mark.parametrize('axis',['cosine','roughness'])
@pytest.mark.parametrize('position',['lower','first-interior','middle','upper'])
def test_table_knot_one_sided_derivatives(tables,axis,position,request):
    table=tables[1]; knots=table.cosines if axis=='cosine' else table.roughness
    index={'lower':0,'first-interior':1,'middle':len(knots)//2,'upper':len(knots)-1}[position]
    coordinate=knots[index].item(); last=index==len(knots)-1
    gap=(knots[index]-knots[index-1] if last else knots[index+1]-knots[index]).item()
    h=gap*1e-4; sign=-1 if last else 1
    def query(x):
        return table.evaluate(x,t(.613),t(.29)) if axis=='cosine' else table.evaluate(t(.43),x,t(.29))
    x=t(coordinate,True); outputs=query(x); shifted=query(t(coordinate+sign*h))
    twice=query(t(coordinate+2*sign*h))
    rows=[]
    for name,value in outputs.items():
        if axis=='cosine' and name=='average':
            # The cosine-weighted mean depends on r/F0, never on a query cosine.
            # Do not globally allow_unused: missing required gradients must fail.
            assert not value.requires_grad
            assert torch.equal(value,shifted[name])
            grad=torch.zeros_like(x)
        else:
            grad=torch.autograd.grad(value,x,retain_graph=True)[0]
        estimate=(-3*value.detach()+4*shifted[name]-twice[name])/(2*sign*h)
        assert torch.allclose(grad,estimate,atol=2e-7,rtol=2e-6)
        rows.append(dict(output=name,value=value.item(),gradient=grad.item(),one_sided=estimate.item()))
    # Continuity, not a claim that the two neighboring slopes match.
    if 0<index<len(knots)-1:
        left_gap=(knots[index]-knots[index-1]).item()
        tiny=min(gap,left_gap)*1e-7
        left=query(t(coordinate-tiny));right=query(t(coordinate+tiny))
        assert all(abs(left[key].item()-right[key].item())<1e-6 for key in outputs)
    evidence(request,'knot-derivatives',dict(axis=axis,position=position,rows=rows,
        scope='Derivative of the interpolant, not agreement with physical endpoint derivatives.'))


@pytest.mark.parametrize('c,r,f',[(.025,.213,.04),(.43,.613,.4),(.97,.937,1.)])
def test_table_float32_values_and_gradients(tables,c,r,f,request):
    args=tuple(t(x,True) for x in (c,r,f))
    low=tuple(x.detach().float().requires_grad_() for x in args)
    rows=[]
    for name in ('directional','average'):
        if name=='directional':
            expected=value_and_gradient(lambda *xs:tables[1].evaluate(*xs)[name],args)
            actual=value_and_gradient(lambda *xs:tables[1].evaluate(*xs)[name],low)
            order=['cosine','roughness','F0']
        else:
            expected=value_and_gradient(lambda *xs:tables[1].evaluate(args[0].detach(),*xs)[name],args[1:])
            actual=value_and_gradient(lambda *xs:tables[1].evaluate(low[0].detach(),*xs)[name],low[1:])
            order=['roughness','F0']
        assert math.isclose(actual[0],expected[0],abs_tol=2e-5,rel_tol=2e-4)
        assert torch.allclose(actual[1].double(),expected[1],atol=2e-5,rtol=2e-4)
        rows.append(dict(output=name,float32=actual[0],float64=expected[0],
            derivative_order=order,
            float32_gradient=actual[1].tolist(),float64_gradient=expected[1].tolist()))
    evidence(request,'dtype-comparison',dict(rows=rows,scope='Same interpolant at two dtypes, not physical accuracy.'))


@pytest.mark.parametrize('roughness',[.01,.05,.1,.199])
def test_table_sharper_materials_are_explicitly_unsupported(tables,roughness):
    with pytest.raises(ValueError,match='roughness outside explicit table domain'):
        tables[1].evaluate(t(.43),t(roughness),t(.04))


def coupled_brdf(args,table=None,*,resolution=512,mean_resolution=128,detach_budget=False):
    """Scalar-colour interior fixture, not a public material implementation.

    Both paths use the same direct specular expression: this test isolates the
    budget approximation and its chain rule, not independent lobe correctness.
    """
    normal,view,light,r,f,a=args
    n=normal/normal.norm();v=view/view.norm();l=light/light.norm()
    h=(v+l)/(v+l).norm(); cv=n@v;cl=n@l;nh=n@h;vh=v@h
    alpha2=r.pow(4)
    distribution=alpha2/(math.pi*((1-nh)*(1+nh)+alpha2*nh.square()).square())
    visibility=.5/(cl*torch.sqrt(alpha2+(1-alpha2)*cv.square())+
                    cv*torch.sqrt(alpha2+(1-alpha2)*cl.square()))
    specular=distribution*visibility*(f+(1-f)*(1-vh).pow(5))
    if table is None:
        ev=reference.directional(cv,r,f,resolution=resolution)
        el=reference.directional(cl,r,f,resolution=resolution)
        avg=reference.average(r,f,resolution=mean_resolution,view_resolution=64)
    else:
        budgets=table.evaluate(torch.stack((cv,cl)),r,f)
        ev,el=budgets['directional'].unbind();avg=budgets['average'][0]
    if detach_budget: ev,el,avg=(x.detach() for x in (ev,el,avg))
    return specular+a*(1-ev)*(1-el)/(math.pi*(1-avg))


@pytest.mark.parametrize('r,f,a',[(.337,.04,.2),(.613,.4,.6),(.937,.8,1.)])
def test_table_full_material_gradients(tables,r,f,a,request):
    args=tuple(t(x,True) for x in ([.1,-.05,1.],[.8,.2,.6],[-.3,.4,.8],r,f,a))
    actual=value_and_gradient(lambda *xs:coupled_brdf(xs,tables[1]),args)
    exact=value_and_gradient(lambda *xs:coupled_brdf(xs),args)
    lower=value_and_gradient(lambda *xs:coupled_brdf(xs,resolution=256,mean_resolution=96),args)
    detached=value_and_gradient(lambda *xs:coupled_brdf(xs,tables[1],detach_budget=True),args)
    report=comparison(actual,exact,lower)
    report.update(roughness=r,f0=f,albedo=a,
        derivative_order=['Nx','Ny','Nz','Vx','Vy','Vz','Lx','Ly','Lz','roughness','F0','albedo'],
        detached_roughness_gradient=detached[1][9].item(),
        detached_gradient_difference=abs(actual[1][9].item()-detached[1][9].item()))
    evidence(request,'full-material-gradient',report)
    assert torch.autograd.gradcheck(lambda *xs:coupled_brdf(xs,tables[1]),args,eps=1e-6,atol=1e-5,rtol=1e-3)
    assert report['detached_gradient_difference']>1e-5, 'Detached-budget negative control not discriminating'
    assert report['reference_converged'], 'REFERENCE UNRESOLVED: '+str(report)
    assert report['candidate_matches'], 'TABLE ACCURACY: '+str(report)


@pytest.mark.parametrize('c,r',[(0.,.2),(1.,.2),(.013,.213),(.025,.213),(.43,.337),(.97,.937),(0.,1.),(1.,1.)])
def test_table_paired_endpoints_match_original_integrator(c,r):
    pair=module().directional_albedo_endpoints(c,r,resolution=128)
    assert len(pair)==2
    for f,value in enumerate(pair):
        assert value==pytest.approx(oracle.directional_albedo(c,r,f0=float(f),resolution=128),abs=1e-12,rel=0.)


def test_table_paired_generation_counts_actual_work(monkeypatch):
    mod=module()
    assert inspect.signature(mod.generate).parameters['cosine_nodes'].default==257
    for c,r,n in ((-.1,.2,128),(1.1,.2,128),(.5,0.,128),(.5,float('nan'),128),(.5,.2,True),(.5,.2,513)):
        with pytest.raises(ValueError):mod.directional_albedo_endpoints(c,r,resolution=n)
    calls=[]
    def pair(c,r,*,resolution):
        calls.append((c,r,resolution));return (.1,.8)
    monkeypatch.setattr(mod,'directional_albedo_endpoints',pair)
    result=mod.generate(roughness_nodes=3,cosine_nodes=5,resolution=16)
    assert len(calls)==15
    assert result.metadata['sample_count']==15*16*32
    assert result.metadata['endpoint_outputs_per_sample']==2
    assert result.metadata['sample_count_basis']=='shared geometric quadrature samples'
    with pytest.raises(ValueError,match='1200000000'):
        mod.generate(roughness_nodes=65,cosine_nodes=257,resolution=256)
    assert len(calls)==15, 'Oversized generation must fail before integration'


def test_table_retains_old_grid_gradient_failure(tables,request):
    old=module().generate(roughness_nodes=33,cosine_nodes=129,resolution=256,roughness_grid='uniform',interpolation='linear')
    args=tuple(t(x,True) for x in (.025,.213,1.))
    old_value,old_gradient=value_and_gradient(lambda *xs:old.evaluate(*xs)['directional'],args)
    new_value,new_gradient=value_and_gradient(lambda *xs:tables[1].evaluate(*xs)['directional'],args)
    ref_value,ref_gradient=value_and_gradient(lambda *xs:reference.directional(*xs,resolution=512),args)
    limit=.01+.02*abs(ref_gradient[0].item())
    evidence(request,'old-new-grid',dict(old_value=old_value,new_value=new_value,reference=ref_value,
        old_dc=old_gradient[0].item(),new_dc=new_gradient[0].item(),reference_dc=ref_gradient[0].item(),
        permitted_error=limit,scope='Legacy negative control; complete sweep remains mandatory.'))
    assert abs(old_gradient[0].item()-ref_gradient[0].item())>limit
    assert abs(new_gradient[0].item()-ref_gradient[0].item())<=limit


def test_table_roughness_grid_is_explicit_and_concentrates_sharp_materials(monkeypatch):
    mod=module()
    assert inspect.signature(mod.generate).parameters['roughness_grid'].default=='uniform'
    monkeypatch.setattr(mod,'directional_albedo_endpoints',lambda *args,**kwargs:(.1,.8))
    fine=mod.generate(roughness_nodes=33,cosine_nodes=5,resolution=16,roughness_grid='sharp-dense')
    coarse=mod.generate(roughness_nodes=17,cosine_nodes=5,resolution=16,roughness_grid='sharp-dense')
    old=mod.generate(roughness_nodes=33,cosine_nodes=5,resolution=16,roughness_grid='uniform')
    assert torch.equal(coarse.roughness,fine.roughness[::2])
    assert fine.roughness[0].item()==.2 and fine.roughness[-1].item()==1.
    assert fine.roughness[8].item()==pytest.approx(.3,abs=1e-15)
    assert torch.allclose(fine.roughness[1:9]-fine.roughness[:8],torch.full((8,),.0125,dtype=torch.float64))
    assert torch.equal(old.roughness,torch.linspace(.2,1.,33,dtype=torch.float64))
    assert fine.metadata['roughness_grid']=='sharp-dense' and old.metadata['roughness_grid']=='uniform'
    with pytest.raises(ValueError,match='roughness_grid'):mod.generate(roughness_grid='automatic')


def test_table_quadratic_reproduces_known_values_gradients_and_mean(monkeypatch):
    mod=module()
    def endpoints(c,r,**kwargs):
        zero=.1+.1*r+.05*r*r+.04*c+.02*c*c+.01*r*c
        return zero,zero+.3
    monkeypatch.setattr(mod,'directional_albedo_endpoints',endpoints)
    table=mod.generate(roughness_nodes=5,cosine_nodes=9,resolution=16,interpolation='quadratic')
    c,r,f=(t(x,True) for x in ([.13,.37,.83],[.27,.53,.89],.2))
    values=table.evaluate(c,r,f)
    expected=.1+.1*r+.05*r*r+.04*c+.02*c*c+.01*r*c+.3*f
    average=.1+.1*r+.05*r*r+(.04+.01*r)*2/3+.01+.3*f
    assert torch.allclose(values['directional'],expected,atol=1e-12,rtol=0)
    assert torch.allclose(values['average'],average,atol=1e-12,rtol=0)
    for key,exact in (('directional',expected),('average',average)):
        inputs=(c,r,f) if key=='directional' else (r,f)
        actual_grad=torch.autograd.grad(values[key].sum(),inputs,retain_graph=True)
        exact_grad=torch.autograd.grad(exact.sum(),inputs,retain_graph=True)
        assert all(torch.allclose(a,b,atol=1e-12,rtol=0) for a,b in zip(actual_grad,exact_grad))
    assert table.metadata['interpolation']=='quadratic'
    with pytest.raises(ValueError,match='interpolation'):mod.generate(interpolation='automatic')


def test_table_quadratic_dense_range_screen(tables,request):
    c=torch.linspace(0.,1.,513,dtype=torch.float64)[None,:].expand(65,-1)
    r=torch.linspace(.2,1.,65,dtype=torch.float64)[:,None].expand_as(c)
    rows=[]
    for f in (0.,1.):
        values=tables[1].evaluate(c,r,t(f))
        for key,value in values.items():
            rows.append(dict(f0=f,output=key,minimum=value.min().item(),maximum=value.max().item()))
            assert torch.isfinite(value).all() and (value>=-1e-12).all() and (value<=1+1e-12).all()
    evidence(request,'quadratic-range',dict(rows=rows,queries_per_f0=65*513,
        scope='Finite sampled range check with roundoff allowance, not an analytic shape-preservation guarantee.'))
