"""Internal tensor resource: scalar parity and safe loading before public admission."""
import copy
import json
import pytest

torch = pytest.importorskip('torch')


def t(value, grad=False):
    return torch.tensor(value, dtype=torch.float64, requires_grad=grad)


def scalar():
    from tools.material_loss_smooth import SmoothLossResource
    r, c = t([.001, .01, .1, 1.]), t([0., .001, .1, .5, 1.])
    v = torch.stack([torch.stack([torch.stack((.8*x**.1, x*0))]+[
        torch.stack(((.6+.05*y)*x**.1, (.2+.03*y)*x**.1)) for y in c[1:]]) for x in r])
    return SmoothLossResource(r, c, v)


def candidate():
    from tools.material_loss_tensor import TensorLossResource
    return TensorLossResource.from_resource(scalar(), evidence_id='synthetic-unit-control')


@pytest.fixture(autouse=True)
def one_thread():
    prior = torch.get_num_threads(); torch.set_num_threads(1)
    try: yield
    finally: torch.set_num_threads(prior)


@pytest.mark.parametrize('size', [1, 8, 32])
def test_tensor_batch_values_and_all_gradients_match_frozen_scalar(size):
    ref, resource = scalar(), candidate()
    c = torch.linspace(.02, .91, size, dtype=torch.float64).requires_grad_()
    r = torch.linspace(.023, .73, size, dtype=torch.float64).requires_grad_()
    f = torch.linspace(.12, .82, size, dtype=torch.float64).requires_grad_()
    actual = resource.evaluate(c, r, f)
    expected = [ref.evaluate(*point) for point in zip(c, r, f)]
    for key in actual:
        assert torch.allclose(actual[key], torch.stack([x[key] for x in expected]), atol=1e-12, rtol=1e-11)
    a = torch.autograd.grad((actual['loss']+actual['mean_loss']).sum(), (c, r, f))
    b = torch.autograd.grad(sum(x['loss']+x['mean_loss'] for x in expected), (c, r, f))
    assert all(torch.allclose(x, y, atol=1e-12, rtol=1e-11) for x, y in zip(a, b))


def test_shared_fields_strided_shapes_and_gradcheck():
    resource = candidate()
    c = t([[.02,.8,.4,.9],[.2,.7,.3,.6]], True)[:,::2]
    r, f = t(.037, True), t(.42, True)
    assert torch.autograd.gradcheck(lambda c,r,f: resource.evaluate(c,r,f)['loss'], (c,r,f))
    mean = resource.mean_loss(r, f)
    assert mean.shape == ()
    assert torch.allclose(resource.mean_loss(r.expand(2,2),f), mean.expand(2,2))
    assert torch.equal(resource.directional_loss(c,r,f), resource.evaluate(c,r,f)['loss'])


def test_exact_mean_and_tiny_loss_stay_connected():
    from tools.material_loss_smooth import SmoothLossResource
    from tools.material_loss_tensor import TensorLossResource
    ref = scalar()
    ref = SmoothLossResource(t(ref.roughness_nodes), t(ref.cosine_nodes), t(ref.losses)*1e-25)
    resource = TensorLossResource.from_resource(ref, evidence_id='tiny-synthetic')
    r,f=t(.037,True),t(.41,True)
    a=resource.mean_loss(r,f); b=ref.integrate_for_test(r,f)
    assert a > 0 and resource.evaluate(t(.33),r,f)['mean_energy'] == 1
    assert torch.allclose(a,b,atol=1e-36,rtol=1e-11)
    ga=torch.autograd.grad(a,(r,f)); gb=torch.autograd.grad(b,(r,f))
    assert all(torch.allclose(x,y,atol=1e-36,rtol=1e-11) for x,y in zip(ga,gb))


@pytest.mark.parametrize('c', [0., .001, .1, .5, 1.])
def test_knot_and_horizon_side_conventions_match_scalar(c):
    ref, resource = scalar(), candidate()
    x,r,f=t(c,True),t(.037,True),t(1.,True)
    a=resource.evaluate(x,r,f); b=ref.evaluate(x,r,f)
    assert a['loss'].item() == pytest.approx(b['loss'].item(),abs=1e-13)
    ga=torch.autograd.grad(a['loss'],(x,r,f)); gb=torch.autograd.grad(b['loss'],(x,r,f))
    assert all(torch.allclose(u,v,atol=1e-12,rtol=1e-11) for u,v in zip(ga,gb))


def test_two_node_resource_and_empty_queries():
    from tools.material_loss_smooth import SmoothLossResource
    from tools.material_loss_tensor import TensorLossResource
    ref=SmoothLossResource(t([.001,1.]),t([0.,1.]),t([[[.8,0.],[.6,.3]],[[.9,0.],[.7,.4]]]))
    resource=TensorLossResource.from_resource(ref,evidence_id='two-node')
    assert torch.equal(resource.evaluate(t(1.),t(.3),t(.4))['loss'],ref.evaluate(t(1.),t(.3),t(.4))['loss'])
    c=t([],True); r,f=t(.03,True),t(.4,True)
    output=resource.evaluate(c,r,f)
    assert all(x.shape == (0,) for x in output.values())
    gradients=torch.autograd.grad(sum(x.sum() for x in output.values()),(c,r,f))
    assert gradients[0].numel()==0 and gradients[1]==0 and gradients[2]==0
    with pytest.raises(ValueError): resource.evaluate(c,t(0.),f)


@pytest.mark.parametrize('bad', ['dtype','number','shape','nan','domain','size','device'])
def test_query_errors_before_curve_allocation(monkeypatch,bad):
    resource=candidate(); c,r,f=t([.2,.4]),t(.03),t(.4)
    if bad=='dtype': r=r.float()
    elif bad=='number': r=.03
    elif bad=='shape': r=t([.03])
    elif bad=='nan': f=t(float('nan'))
    elif bad=='domain': r=t(.0001)
    elif bad=='size': c=t([.4]*33)
    elif bad=='device': r=torch.empty((),device='meta',dtype=torch.float64)
    monkeypatch.setattr(type(resource),'_curve',lambda *args:pytest.fail('allocated curve before validation'))
    with pytest.raises((TypeError,ValueError)): resource.evaluate(c,r,f)


def test_workspace_limit_and_storage_inspection_cannot_mutate_resource():
    resource=candidate()
    snapshot=resource.inspect()
    before=resource.evaluate(t(.2),t(.03),t(.4))['loss']
    for value in snapshot['tensors'].values(): value.zero_()
    assert torch.equal(before,resource.evaluate(t(.2),t(.03),t(.4))['loss'])
    estimate=resource.workspace(32,backward=True)
    assert 0 < estimate <= 64*1024**2
    with pytest.raises(ValueError,match='workspace'):
        resource.evaluate(t(.2),t(.03),t(.4),workspace_limit_bytes=1)
    with pytest.raises(ValueError): resource.workspace(33,backward=True)


def test_loading_roundtrip_and_caller_mutation_isolation(tmp_path):
    from tools.material_loss_tensor import TensorLossResource
    resource=candidate(); envelope=resource.to_envelope()
    path=tmp_path/'budget.json';path.write_text(json.dumps(envelope))
    loaded=TensorLossResource.load(path)
    envelope['payload']['losses'][0][1][0]=0
    assert torch.equal(loaded.evaluate(t(.2),t(.03),t(.4))['loss'], resource.evaluate(t(.2),t(.03),t(.4))['loss'])
    assert loaded.metadata['evidence_id']=='synthetic-unit-control'
    loaded.metadata['evidence_id']='changed'
    assert loaded.metadata['evidence_id']=='synthetic-unit-control'


@pytest.mark.parametrize('field', ['format_version','model','representation','dtype','domain','payload_sha256','payload','unknown','generator','evidence_id'])
def test_bad_envelopes_rejected_before_tensor_allocation(monkeypatch,field):
    from tools.material_loss_tensor import TensorLossResource
    value=candidate().to_envelope()
    value[field]='' if field=='evidence_id' else 'invalid'
    monkeypatch.setattr(torch,'tensor',lambda *args,**kwargs:pytest.fail('allocated before metadata/shape validation'))
    with pytest.raises(ValueError): TensorLossResource.from_envelope(value)


def test_loader_byte_limit_duplicate_keys_and_missing_file(tmp_path):
    from tools.material_loss_tensor import TensorLossResource
    path=tmp_path/'budget.json'
    with pytest.raises(FileNotFoundError): TensorLossResource.load(path)
    path.write_bytes(b' '*(8*1024**2+1))
    with pytest.raises(ValueError,match='byte'): TensorLossResource.load(path)
    path.write_text('{"format_version":1,"format_version":1}')
    with pytest.raises(ValueError,match='duplicate'): TensorLossResource.load(path)


def test_queries_do_not_construct_constants_read_files_or_integrate(monkeypatch):
    from pathlib import Path
    import tools.material_loss_smooth as smooth
    resource=candidate(); c,r,f=t(.2,True),t(.03,True),t(.4,True)
    def forbidden(*args,**kwargs): pytest.fail('query rebuilt constants or performed I/O/quadrature')
    monkeypatch.setattr(torch,'tensor',forbidden)
    monkeypatch.setattr(Path,'open',forbidden)
    monkeypatch.setattr(smooth,'_gauss_unit',forbidden)
    first=resource.evaluate(c,r,f)
    second=resource.evaluate(c,r,f)
    assert all(torch.equal(first[k],second[k]) for k in first)
    torch.autograd.grad(first['loss']+first['mean_loss'],(c,r,f))


def test_tensor_mean_matches_independent_segment_quadrature():
    resource=candidate(); r,f=t(.037,True),t(.41,True)
    a=resource.mean_loss(r,f); b=resource.integrate_for_test(r,f,order=32)
    assert torch.allclose(a,b,atol=1e-11,rtol=1e-11)
    ga=torch.autograd.grad(a,(r,f)); gb=torch.autograd.grad(b,(r,f))
    assert all(torch.allclose(x,y,atol=1e-11,rtol=1e-11) for x,y in zip(ga,gb))


def test_probe_uses_real_batch_and_accounts_saved_graph_storage():
    from tools.material_loss_tensor_probe import query
    resource=candidate()
    args=(t([.2,.3],True),t([.03,.04],True),t([.4,.5],True))
    result=query(resource,args,backward=True,account=True)
    assert result['saved_tensor_bytes']>0
    assert result['saved_nonconstant_bytes']<=resource.workspace(2,backward=True)
    assert len(result['gradient_sums'])==3


def test_supervisor_can_launch_the_bounded_tensor_probe(tmp_path):
    from tools.material_loss_cost import worker_command
    command=worker_command(tmp_path,tmp_path/'report.json',worker_module='tools.material_loss_tensor_probe',worker_args=('--mode','physical'))
    assert 'tools.material_loss_tensor_probe' in command[2]
    assert command[-2:]==['--mode','physical']
