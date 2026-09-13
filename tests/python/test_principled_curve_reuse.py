"""Call-local reuse must preserve RGB, point ownership and the current graph."""
import pytest
import torch


def t(x, grad=False):
    return torch.tensor(x, dtype=torch.float64, requires_grad=grad)


@pytest.fixture(scope='module')
def budget():
    from gffx.torch.material import PrincipledBudget
    return PrincipledBudget.bundled('ggx-smith-loss-cpu-f64-v1', device='cpu', dtype=torch.float64)


@pytest.mark.parametrize('cosines', [[.013, .77], [0., 1.]])
def test_call_local_losses_match_three_queries_and_all_partials(budget, cosines):
    r=t([.213, .827], True); f=t([[.07,.47,.93],[1.,.04,0.]], True)
    v=t(cosines, True); l=t([.027,.92,.3,.013], True)
    ids=torch.tensor([1,0,1,0])
    rr=r[:,None].expand_as(f)
    expected=(budget.directional_loss(v[:,None].expand_as(f),rr,f),
              budget.mean_loss(rr,f),
              budget.directional_loss(l[:,None].expand(4,3),rr[ids],f[ids]))
    actual=budget._material_losses(v,l,r,f,ids)
    for x,y in zip(actual,expected):
        torch.testing.assert_close(x,y,atol=1e-12,rtol=1e-12)
    weights=[torch.arange(x.numel(),dtype=torch.float64).reshape(x.shape)+1 for x in expected]
    loss=lambda xs:sum((x*w).sum() for x,w in zip(xs,weights))
    ga=torch.autograd.grad(loss(actual),(r,f,v,l))
    gb=torch.autograd.grad(loss(expected),(r,f,v,l))
    for x,y in zip(ga,gb):torch.testing.assert_close(x,y,atol=1e-10,rtol=1e-9)


def test_material_builds_one_curve_per_point_without_retaining_graph(budget,monkeypatch):
    from gffx.torch import material
    calls=[]; original=type(budget)._curve
    def observed(self,r):
        calls.append(tuple(r.shape)); return original(self,r)
    monkeypatch.setattr(type(budget),'_curve',observed)
    n=t([[0.,0.,1.],[0.,0.,1.]])
    l=t([[.1,.2,1.],[-.2,.1,1.],[.1,-.1,-1.]])
    r=t([.213,.827],True); f=t([[.07,.47,.93],[.93,.47,.07]],True)
    outputs=[]
    for _ in range(2):
        out=material.principled(n,n,l,albedo=t([.2,.4,.6]),roughness=r,f0=f,budget=budget)
        outputs.append(out.detach().clone())
        g,=torch.autograd.grad(out.sum(),(r,))
        assert torch.isfinite(g).all()
        with torch.no_grad():r.add_(.0001)
    assert calls==[(2,),(2,)]
    assert not torch.equal(*outputs)


def test_material_rejects_old_limit_before_any_curve(budget,monkeypatch):
    from gffx.torch import material
    monkeypatch.setattr(type(budget),'_curve',lambda *a:pytest.fail('curve allocated before rejection'))
    n=t([[0.,0.,1.]]*8,True)
    with pytest.raises(ValueError,match='workspace'):
        material.principled(n,n,t([[0.,0.,1.]]),albedo=t([.2,.4,.6]),roughness=t(1.),f0=t(.04),budget=budget)


def test_curve_reuse_preserves_two_node_resource():
    from gffx.torch.material import PrincipledBudget
    from tools.testing_dashboard.complete_material_demo import synthetic_envelope
    envelope=synthetic_envelope()
    payload=envelope['payload']; payload['cosine_nodes']=[0.,1.]
    payload['losses']=[[row[0],row[-1]] for row in payload['losses']]
    from gffx.torch._principled_budget import _digest
    envelope['payload_sha256']=_digest(payload)
    b=PrincipledBudget.from_envelope(envelope,device='cpu',dtype=torch.float64,allow_unverified=True)
    r=t([.4]); f=t([[.1,.4,.8]]); v=t([.3]); l=t([0.,.7,1.]); ids=torch.tensor([0,0,0])
    a=b._material_losses(v,l,r,f,ids)
    expected=(b.directional_loss(v[:,None].expand_as(f),r[:,None].expand_as(f),f),b.mean_loss(r[:,None].expand_as(f),f),
              b.directional_loss(l[:,None].expand(3,3),r.expand(3,3),f.expand(3,3)))
    for x,y in zip(a,expected):torch.testing.assert_close(x,y,atol=1e-12,rtol=1e-12)


def test_cost_probe_stops_at_expired_deadline(budget):
    from tools.principled_reuse_probe import costs
    with pytest.raises(TimeoutError):costs(budget,0.)


def test_horizon_partial_matches_linear_segment_without_cancellation(budget):
    r=t([.213]); f=t([[.07,.47,.93]]); c=t([0.],True)
    cn,values,horizon,_=budget._curve(r)
    slopes=(values[:,0]-horizon)/cn[1]
    weights=t([[1.,2.,3.]])
    expected=(((1-f)*slopes[:,0,None]+f*slopes[:,1,None])*weights).sum()
    loss,_,_=budget._material_losses(c,t([.1]),r,f,torch.tensor([0]))
    gradient,=torch.autograd.grad((loss*weights).sum(),(c,))
    torch.testing.assert_close(gradient[0],expected,atol=1e-10,rtol=1e-12)


def test_cost_probe_reports_memory_categories_separately(budget):
    import time
    from tools.principled_reuse_probe import costs
    rows=costs(budget,time.perf_counter()+30)
    assert [(x['points'],x['lights'],x['backward']) for x in rows]==[(1,10,False),(8,1,False),(4,1,True)]
    for row in rows:
        assert len(row['durations_seconds'])==3 and len(row['working_set_bytes'])==12
        assert max(row['working_set_bytes'])<1024**3
        assert 0<row['declared_material_workspace_bytes']<=64*1024**2
        assert (row['saved_unique_storage_bytes']>0)==row['backward']


@pytest.mark.parametrize('name',['baseline-costs','optimized-costs','optimized-environment'])
def test_saved_reuse_measurements_retain_exact_work_and_limits(name):
    import os
    from tools.testing_dashboard.curve_reuse_demo import read
    root=os.environ.get('GFFX_REUSE_REPORTS')
    if not root:pytest.skip('explicit supervised reuse reports required')
    row=read(root,name)
    assert row['supervisor']['elapsed_seconds']<180
    if name=='optimized-environment':
        assert row['result']['calls']==6554 and row['result']['directions']==65536
        assert row['result']['elapsed_seconds']<120
        assert all(error<=limit for errors in row['errors'] for error,limit in zip(errors,row['limits']))
    else:
        assert len(row['costs'])==3
        assert all(len(c['working_set_bytes'])==12 and c['declared_material_workspace_bytes']<=64*1024**2 for c in row['costs'])


def test_reuse_demo_refuses_changed_measurements(tmp_path):
    from tools.testing_dashboard.curve_reuse_demo import read
    (tmp_path/'optimized-costs.json').write_text('{}',encoding='utf-8')
    with pytest.raises(ValueError,match='identity'):read(tmp_path,'optimized-costs')
