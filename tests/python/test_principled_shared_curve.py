"""One immutable curve row must match dense copies without changing group limits."""
import os,torch,pytest
from test_principled_groups import grouped_fixture,same

def inputs(points=17,lights=2,device='cpu'):
    budget,dense=grouped_fixture(points,lights)
    for k in (2,3,4,5,10):dense[k]=dense[k][:1].expand((points,)+dense[k].shape[1:]).clone()
    dense=[t.to(device) for t in dense];shared=list(dense)
    for k in (2,3,4,5):shared[k]=dense[k][:1].clone()
    return dense,shared

@pytest.mark.parametrize('points,lights',[(5,1),(17,3),(128,8)])
@pytest.mark.parametrize('derivatives',[False,True])
def test_shared_curve_matches_dense_values_jacobian_and_vjp(points,lights,derivatives):
    dev=os.environ.get('GFFX_KNOT_DEVICE','cpu');dense,shared=inputs(points,lights,dev)
    y,j=torch.ops.gffx.shading_principled_groups(*dense,False,derivatives)
    cy,cj=torch.ops.gffx.shading_principled_shared_groups(*shared,False,derivatives)
    same(y,cy);same(j,cj)
    if derivatives:
        g=(torch.arange(points*lights*3,device=dev,dtype=torch.float64).reshape(points,lights,3)%11-5)/17
        for a,b in zip(torch.ops.gffx.shading_principled_groups_backward(j,g),torch.ops.gffx.shading_principled_groups_backward(cj,g)):same(a,b)

def test_shared_curve_rejects_mixed_shapes_invalid_values_and_old_pair_scope():
    dev=os.environ.get('GFFX_KNOT_DEVICE','cpu');dense,shared=inputs(5,1,dev)
    # First require the new valid shape,so the old implementation fails this test.
    torch.ops.gffx.shading_principled_shared_groups(*shared,False,True)
    for slot in (2,3,4,5):
        bad=list(shared);bad[slot]=dense[slot]
        with pytest.raises(RuntimeError):torch.ops.gffx.shading_principled_shared_groups(*bad,False,True)
        bad=list(shared);bad[slot]=shared[slot].clone();bad[slot].flatten()[-1]=float('nan')
        with pytest.raises(RuntimeError):torch.ops.gffx.shading_principled_shared_groups(*bad,False,True)
    for p,q in ((129,1),(128,9)):
        _,bad=inputs(p,q,dev)
        with pytest.raises(RuntimeError):torch.ops.gffx.shading_principled_shared_groups(*bad,False,True)
    _,bad=inputs(4,1,dev)
    with pytest.raises(RuntimeError):torch.ops.gffx.shading_principled_pairs(*bad,False,True)

def test_shared_curve_storage_bound_is_explicit_and_retains_point_pair_charges():
    from gffx.torch.material import PrincipledBudget
    from gffx.torch._principled_groups import workspace_bound
    b=PrincipledBudget.bundled('ggx-smith-loss-cpu-f64-v1',device='cpu',dtype=torch.float64)
    old=workspace_bound(b,128,8,True);shared=workspace_bound(b,128,8,True,shared_curve=True)
    assert shared<old and shared>=8*128*8*192
    assert workspace_bound(b,128,8,True,shared_curve=False)==old


def test_shared_curve_empty_outputs_still_validate_curve_domain():
    dev=os.environ.get('GFFX_KNOT_DEVICE','cpu');_,shared=inputs(5,1,dev)
    for k in range(6,12):shared[k]=shared[k][:0]
    y,j=torch.ops.gffx.shading_principled_shared_groups(*shared,False,True)
    assert y.shape==(0,1,3) and j.shape==(0,1,3,16)
    for slot in (2,3,4,5):
        bad=list(shared);bad[slot]=shared[slot].clone();bad[slot].flatten()[-1]=float('nan')
        with pytest.raises(RuntimeError):torch.ops.gffx.shading_principled_shared_groups(*bad,False,True)


def test_old_provider_refuses_shared_operation_before_reading_row():
    if os.environ.get('GFFX_OLD_PROVIDER_CONTROL')!='1':pytest.skip('Requires explicitly selected preserved old provider in a fresh process')
    _,dense=inputs(5,1,'cuda');shared=list(dense)
    # This fixture's first result has one curve row; old dense groups still work.
    for k in (2,3,4,5):dense[k]=dense[k][:1].expand((5,)+dense[k].shape[1:]).clone();shared[k]=dense[k][:1].clone()
    y,j=torch.ops.gffx.shading_principled_groups(*dense,False,True)
    assert torch.isfinite(y).all() and torch.isfinite(j).all()
    with pytest.raises(RuntimeError,match='provider lacks shared Principled groups'):
        torch.ops.gffx.shading_principled_shared_groups(*shared,False,True)
