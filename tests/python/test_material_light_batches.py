"""Larger light batches preserve explicit ownership, old query limits and gradients."""
import pytest
import torch


def t(value,grad=False):
    return torch.tensor(value,dtype=torch.float64,requires_grad=grad)


@pytest.fixture(scope='module')
def budget():
    from gffx.torch.material import PrincipledBudget
    return PrincipledBudget.bundled('ggx-smith-loss-cpu-f64-v1',device='cpu',dtype=torch.float64)


def run(budget,points,lights,grad=False,**kwargs):
    from gffx.torch.material import principled
    n=t([[0.,0.,1.]]*points,grad)
    return principled(n,n,t([[.2,.3,1.]]*lights),albedo=t([.2,.4,.6]),roughness=t(.827),f0=t(.04),budget=budget,**kwargs)


@pytest.mark.parametrize('points,lights,grad',[(1,1024,False),(8,128,False),(1,1024,True),(4,256,True)])
def test_exact_material_pair_capacity_is_accepted(budget,points,lights,grad):
    out=run(budget,points,lights,grad)
    assert out.shape==(points,lights,3) and torch.isfinite(out).all()
    assert out.requires_grad==grad


@pytest.mark.parametrize('points,lights,grad',[(1,1025,False),(8,129,False),(9,1,False),(5,1,True),(8,1,True)])
def test_material_capacity_rejects_before_mask_or_curve(budget,monkeypatch,points,lights,grad):
    from gffx.torch import material
    old=material._geometry
    def guard(*a,**k):
        assert k.get('_layout_only'), 'constructed mask before capacity rejection'
        return old(*a,**k)
    monkeypatch.setattr(material,'_geometry',guard)
    with pytest.raises(ValueError,match='workspace'):run(budget,points,lights,grad)


@pytest.mark.parametrize('grad',[False,True])
def test_material_workspace_exact_byte_boundary(budget,grad):
    points=4 if grad else 8;lights=1024//points
    needed=3*budget.workspace(3*points,backward=grad)+8*1024*(192 if grad else 64)
    with pytest.raises(ValueError,match='workspace'):run(budget,points,lights,grad,workspace_limit_bytes=needed-1)
    assert torch.isfinite(run(budget,points,lights,grad,workspace_limit_bytes=needed)).all()


def test_standalone_resource_query_limit_remains_32(budget):
    with pytest.raises(ValueError,match='limit'):
        budget.directional_loss(torch.full((33,),.3,dtype=torch.float64),t(.4),t(.04))
    assert budget.metadata['supported_query_entries']==32


@pytest.mark.parametrize('empty',['points','lights'])
def test_empty_material_keeps_zero_gradients(budget,empty):
    from gffx.torch.material import principled
    n=t([] if empty=='points' else [[0.,0.,1.]]*9).reshape(-1,3).requires_grad_()
    v=n.detach().clone().requires_grad_()
    l=t([] if empty=='lights' else [[.2,.3,1.]]).reshape(-1,3).requires_grad_()
    a,r,f=t([.2,.4,.6],True),t(.827,True),t(.04,True)
    out=principled(n,v,l,albedo=a,roughness=r,f0=f,budget=budget)
    assert out.shape==(len(n),len(l),3)
    assert all(torch.count_nonzero(g)==0 for g in torch.autograd.grad(out.sum(),(n,v,l,a,r,f)))


def test_inactive_large_batch_keeps_zero_gradients_and_counts_capacity(budget):
    from gffx.torch.material import principled
    n=t([[0.,0.,1.]],True);v=n.detach().clone().requires_grad_()
    l=t([[0.,0.,-1.]]*1024,True);a,r,f=t([.2,.4,.6],True),t(.827,True),t(.04,True)
    out=principled(n,v,l,albedo=a,roughness=r,f0=f,budget=budget)
    assert torch.count_nonzero(out)==0
    assert all(torch.count_nonzero(g)==0 for g in torch.autograd.grad(out.sum(),(n,v,l,a,r,f)))
    with pytest.raises(ValueError,match='workspace'):
        principled(n,v,torch.cat((l,l[:1])),albedo=a,roughness=r,f0=f,budget=budget)


def test_batched_strided_large_material_matches_small_tiles_and_all_gradients(budget):
    from gffx.torch.material import principled
    def leaf(x):return x.repeat_interleave(2,dim=-1)[...,::2].detach().requires_grad_()
    n=leaf(t([[[0.,0.,1.],[.1,0.,1.]],[[0.,.1,1.],[.1,.1,1.]]]))
    v=leaf(t([[[.3,0.,1.],[-.2,0.,1.]],[[0.,.2,1.],[.2,.3,1.]]]))
    angles=torch.arange(256,dtype=torch.float64)/256
    directions=torch.stack((.3*angles.sin(),.2*angles.cos(),torch.where(torch.arange(256)%11==0,-1.,1.)),dim=-1)
    l=leaf(directions.expand(2,2,256,3))
    a=leaf(t([.2,.4,.6]).expand(2,2,3));r=t([[.213,.827],[.463,.67]],True)
    f=leaf(t([.07,.47,.93]).expand(2,2,3));args=(n,v,l,a,r,f)
    weight=torch.linspace(.1,1.,4*256*3,dtype=torch.float64).reshape(2,2,256,3)
    out=principled(n,v,l,albedo=a,roughness=r,f0=f,budget=budget)
    actual=out.detach();gradients=torch.autograd.grad((out*weight).sum(),args)
    expected=torch.empty_like(actual);reference=[torch.zeros_like(x) for x in args]
    for b in range(2):
        for p in range(2):
            for start in range(0,256,8):
                sl=slice(start,start+8)
                tile=principled(n[b,p:p+1],v[b,p:p+1],l[b,p,sl],albedo=a[b,p],roughness=r[b,p],f0=f[b,p],budget=budget)
                expected[b,p,sl]=tile.detach()[0]
                gs=torch.autograd.grad((tile[0]*weight[b,p,sl]).sum(),args)
                for total,g in zip(reference,gs):total.add_(g)
    torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)
    for x,y in zip(gradients,reference):torch.testing.assert_close(x,y,atol=1e-10,rtol=1e-9)


@pytest.mark.parametrize('chunk',[0,True,1025])
def test_environment_sampler_refuses_unqualified_chunk(chunk):
    import time
    from tools.environment_candidate_point import sample
    with pytest.raises(ValueError,match='chunk'):
        sample(.4,model='classic',order=8,deadline=time.perf_counter()+30,chunk_size=chunk)


def test_environment_chunking_preserves_samples_and_weights():
    import time
    from tools.environment_candidate_point import sample
    small=sample(.4,model='classic',order=8,deadline=time.perf_counter()+30)
    large=sample(.4,model='classic',order=8,deadline=time.perf_counter()+30,chunk_size=1024)
    assert small['directions']==large['directions']==256
    assert small['calls']==26 and large['calls']==2
    torch.testing.assert_close(t(small['estimates']),t(large['estimates']),atol=1e-12,rtol=1e-12)


def test_large_cost_probe_reports_all_four_capacity_corners(budget):
    import time
    from tools.principled_reuse_probe import costs
    rows=costs(budget,time.perf_counter()+30,large=True)
    assert [(x['points'],x['lights'],x['backward']) for x in rows]==[(1,1024,False),(8,128,False),(1,1024,True),(4,256,True)]
    assert all(x['declared_material_workspace_bytes']<=64*1024**2 and len(x['working_set_bytes'])==12 for x in rows)


def test_no_grad_uses_forward_point_capacity(budget):
    with torch.no_grad():out=run(budget,8,128,True)
    assert not out.requires_grad and out.shape==(8,128,3)


def test_point_capacity_counts_across_batches(budget):
    from gffx.torch.material import principled
    n=t([[[0.,0.,1.]]*5]*2)
    with pytest.raises(ValueError,match='workspace'):
        principled(n,n,t([[0.,0.,1.]]),albedo=t([.2,.4,.6]),roughness=t(.827),f0=t(.04),budget=budget)


@pytest.mark.parametrize('name',['costs','environment'])
def test_saved_large_batch_evidence_keeps_limits_and_accuracy(name):
    import os
    from tools.testing_dashboard.light_batch_demo import read
    root=os.environ.get('GFFX_LIGHT_BATCH_REPORTS')
    if not root:pytest.skip('explicit measured large-batch reports required')
    row=read(root,name)
    assert row['supervisor']['elapsed_seconds']<180
    if name=='costs':
        assert len(row['costs'])==4
        assert all(x['declared_material_workspace_bytes']<=64*1024**2 and len(x['working_set_bytes'])==12 for x in row['costs'])
    else:
        assert row['result']['directions']==65536 and row['result']['calls']==64
        assert all(x<=limit for error in row['errors'] for x,limit in zip(error,row['limits']))


def test_batch_demo_rejects_changed_report(tmp_path):
    from tools.testing_dashboard.light_batch_demo import read
    (tmp_path/'costs.json').write_text('{}',encoding='utf-8')
    with pytest.raises(ValueError,match='identity'):read(tmp_path,'costs')
