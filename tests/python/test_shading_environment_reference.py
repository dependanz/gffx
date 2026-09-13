"""Known-answer and failure controls for the independent environment measuring tool."""
import importlib
import math

import pytest


def module():
    return importlib.import_module('tools.shading_environment_reference')


def constant(rgb=(1., 1., 1.)):
    return [[math.sqrt(4*math.pi)*x for x in rgb]]+[[0., 0., 0.] for _ in range(8)]


def lambert(n, v, l):
    return [.2/math.pi, .4/math.pi, .6/math.pi]


def call(**kwargs):
    return module().environment_reference([0.,0.,1.], [0.,0.,1.], constant(), brdf=lambert, **kwargs)


@pytest.mark.parametrize('order', [2, 8, 32, 128])
def test_gauss_rule_has_positive_weights_and_exact_polynomial_moments(order):
    nodes, weights = module().gauss_legendre(order)
    assert len(nodes) == len(weights) == order
    assert all(0 < x < 1 for x in nodes)
    assert all(w > 0 for w in weights)
    assert nodes == sorted(nodes)
    for power in range(min(2*order, 10)):
        assert math.fsum(w*x**power for x,w in zip(nodes,weights)) == pytest.approx(1/(power+1), abs=2e-14, rel=0)


@pytest.mark.parametrize('rgb', [(1.,1.,1.), (-2.,.5,3.), (0.,0.,0.), (1e-12,-1e-12,1e-12)])
def test_constant_environment_preserves_units_sign_and_near_zero(rgb):
    result=module().environment_reference([0.,0.,3.], [0.,0.,2.], constant(rgb), brdf=lambert)
    assert result['status']=='converged'
    assert result['radiance']==pytest.approx([a*b for a,b in zip((.2,.4,.6),rgb)], abs=1e-14, rel=1e-12)
    assert len(result['history'])==3
    assert result['evaluations']==5376
    assert result['history'][-1]['streak']==2


def test_all_sh_bands_match_analytic_lambert_at_tilted_normals():
    m=module()
    # Expected irradiance uses the independently written closed SH convolution,
    # not the implementation's radiance helper or its quadrature.
    for normal in ([0.,0.,1.],[1.,0.,0.],[.3,-.4,math.sqrt(.75)]):
        x,y,z=normal
        basis=[1/math.sqrt(4*math.pi), math.sqrt(3/(4*math.pi))*y,
               math.sqrt(3/(4*math.pi))*z, math.sqrt(3/(4*math.pi))*x,
               math.sqrt(15/(4*math.pi))*x*y, math.sqrt(15/(4*math.pi))*y*z,
               math.sqrt(5/(16*math.pi))*(3*z*z-1), math.sqrt(15/(4*math.pi))*x*z,
               math.sqrt(15/(16*math.pi))*(x*x-y*y)]
        coefficients=[[(i+1)*.1,(-1)**i*.2,(i-4)*.15] for i in range(9)]
        factors=[math.pi]+[2*math.pi/3]*3+[math.pi/4]*5
        expected=[a/math.pi*sum(coefficients[i][k]*basis[i]*factors[i] for i in range(9)) for k,a in enumerate((.2,.4,.6))]
        result=m.environment_reference(normal,normal,coefficients,brdf=lambert)
        assert result['radiance']==pytest.approx(expected,abs=1e-12,rel=1e-12)


def test_classic_overreflection_remains_visible():
    from tools.shading_independent_evaluator import classic
    result=module().environment_reference([0.,0.,1.],[0.,0.,1.],constant(),
        brdf=lambda n,v,l:classic(n,v,l,albedo=0.,specular=1.,exponent=2.))
    assert result['radiance']==pytest.approx([25/24]*3,abs=1e-12,rel=1e-12)
    assert min(result['radiance'])>1


@pytest.mark.parametrize('slope', [0., .5])
def test_complete_principled_exact_budget_matches_independent_moments(slope):
    from tools.shading_independent_evaluator import principled
    class ExactRoughOne:
        r=(.001,1.)
        def loss(self,c,r,f):
            assert r==f==1
            return c*math.log1p(1/c) if c else 0.
        def mean(self,r,f):
            assert r==f==1
            return (4*math.log(2)-1)/3
    c=.4; n=[0.,0.,1.]; v=[math.sqrt(1-c*c),0.,c]
    coefficients=constant()
    coefficients[2]=[slope/math.sqrt(3/(4*math.pi))]*3
    result=module().environment_reference(n,v,coefficients,
        brdf=lambda n,v,l:principled(n,v,l,albedo=[0.,.4,1.],roughness=1.,f0=1.,budget=ExactRoughOne()))
    energy=1-c*math.log1p(1/c)
    i2=2*math.log(2)/3-1/6
    expected=[energy+slope*(.5-c+c*c*math.log1p(1/c))+a*(1-energy)*(1+slope*(5/24)/i2) for a in (0.,.4,1.)]
    assert result['radiance']==pytest.approx(expected,abs=1e-9,rel=1e-9)
    if slope:
        shortcut=[(energy+a*(1-energy))*(1+2*slope/3) for a in (0.,.4,1.)]
        assert max(abs(a-b) for a,b in zip(shortcut,expected))>.001


def test_refinement_requires_two_transitions_and_is_repeatable():
    first=call(); second=call()
    assert first==second
    assert [(x['n_mu'],x['n_phi']) for x in first['history']]==[(8,16),(16,32),(32,64)]
    assert [x['streak'] for x in first['history']]==[0,1,2]
    assert first['profile']=='env-sh2-v1'
    for row in first['history'][1:]:
        assert all(d<=t for d,t in zip(row['change'],row['threshold']))


def test_shifted_azimuth_breaks_false_coarse_grid_agreement():
    m=module()
    def aliased(n,v,l):
        angle=math.atan2(l[1],l[0])
        return [(1+.5*math.cos(64*angle))/math.pi]*3
    with pytest.raises(m.ReferenceConvergenceError) as caught:
        m.environment_reference([0.,0.,1.],[0.,0.,1.],constant(),brdf=aliased,max_levels=3)
    report=caught.value.report
    assert report['status']=='unresolved' and report['reason']=='level_limit'
    assert max(report['history'][-1]['offset_change'])>.1
    assert report['history'][-1]['streak']==0


def test_work_budget_stops_before_starting_an_unaffordable_level():
    m=module(); calls=[]
    with pytest.raises(m.ReferenceConvergenceError) as caught:
        m.environment_reference([0.,0.,1.],[0.,0.,1.],constant(),brdf=lambda *args:calls.append(1) or [1.]*3,max_evaluations=256)
    assert len(calls)==caught.value.report['evaluations']==256
    assert caught.value.report['reason']=='work_limit'
    assert len(caught.value.report['history'])==1


def test_cooperative_deadline_withholds_unfinished_reference(monkeypatch):
    m=module(); clock=[0.]; calls=[]
    monkeypatch.setattr(m,'monotonic',lambda:clock[0])
    def slow(*args):
        calls.append(1); clock[0]=31.; return [1.]*3
    with pytest.raises(m.ReferenceConvergenceError) as caught:
        m.environment_reference([0.,0.,1.],[0.,0.,1.],constant(),brdf=slow)
    assert len(calls)==caught.value.report['evaluations']==1
    assert caught.value.report['reason']=='time_limit'
    assert not caught.value.report['history']


@pytest.mark.parametrize('bad', ['normal','view','horizon','coeff_shape','coeff_nan','coeff_bool','negative_brdf','nan_brdf','rgb_shape','overflow','levels','level_bool','work','deadline'])
def test_invalid_inputs_and_nonrepresentable_results_fail_explicitly(bad):
    m=module(); n=[0.,0.,1.]; v=n; cs=constant(); kwargs={'brdf':lambert}
    if bad=='normal':n=[0.,0.,0.]
    elif bad=='view':v=[0.,0.,float('inf')]
    elif bad=='horizon':v=[1.,0.,0.]
    elif bad=='coeff_shape':cs=cs[:8]
    elif bad=='coeff_nan':cs[1][0]=float('nan')
    elif bad=='coeff_bool':cs[1][0]=True
    elif bad=='negative_brdf':kwargs['brdf']=lambda *args:[-1.,0.,0.]
    elif bad=='nan_brdf':kwargs['brdf']=lambda *args:[float('nan')]*3
    elif bad=='rgb_shape':kwargs['brdf']=lambda *args:[1.]
    elif bad=='overflow':cs=constant([1e307]*3);kwargs['brdf']=lambda *args:[1e307]*3
    elif bad=='levels':kwargs['max_levels']=6
    elif bad=='level_bool':kwargs['max_levels']=True
    elif bad=='work':kwargs['max_evaluations']=87297
    elif bad=='deadline':kwargs['deadline_seconds']=31.
    with pytest.raises(ValueError):m.environment_reference(n,v,cs,**kwargs)


def test_reference_is_independent_and_standard_library_only():
    import ast
    from pathlib import Path
    tree=ast.parse(Path(module().__file__).read_text())
    imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
    imports += [a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names]
    assert set(imports)<={'math','time'}


def test_huge_integer_coefficients_raise_representability_value_error():
    cs=constant(); cs[0][0]=10**10000
    with pytest.raises(ValueError,match='representable|finite'):
        module().environment_reference([0.,0.,1.],[0.,0.,1.],cs,brdf=lambert)


def test_input_coefficients_are_copied_and_callback_directions_are_unit():
    m=module(); cs=constant(); directions=[]
    def observe(n,v,l):
        cs[0][0]=0.  # A caller mutation must not change the captured environment.
        if len(directions)<2:directions.append((n,v,l))
        assert abs(math.hypot(*l)-1)<2e-15
        assert 0<sum(a*b for a,b in zip(n,l))<1
        return [.2/math.pi]*3
    result=m.environment_reference([.3,.4,.5],[.3,.4,.5],cs,brdf=observe)
    assert result['radiance']==pytest.approx([.2]*3,abs=1e-12)
    assert directions and all(isinstance(x,tuple) for x in directions[0])


def test_environment_demo_retains_actual_histories_and_failed_alias_case():
    demo=importlib.import_module('tools.testing_dashboard.environment_reference_demo')
    data=demo.study()
    assert data['profile']=='env-sh2-v1'
    assert [x['id'] for x in data['cases']]==['diffuse','classic_overreflection','aliased_sampling']
    assert data['cases'][0]['report']['radiance']==pytest.approx([.2,.4,.6],abs=1e-12)
    assert data['cases'][1]['report']['radiance']==pytest.approx([25/24]*3,abs=1e-12)
    assert data['cases'][2]['report']['status']=='unresolved'
    assert data['cases'][2]['report']['reason']=='level_limit'
    assert all(len(x['report']['history'])==3 for x in data['cases'])
    assert 'not a rendered scene' in data['limits']
