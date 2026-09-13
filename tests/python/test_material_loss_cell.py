"""Local log-loss interpolation trial; no full-table or furnace acceptance."""
import math
import pytest
torch=pytest.importorskip('torch')
from tools.testing_dashboard.pytest_plugin import Evidence


def t(value, grad=False):return torch.tensor(value,dtype=torch.float64,requires_grad=grad)


@pytest.fixture(autouse=True)
def serial():
    old=torch.get_num_threads();torch.set_num_threads(1)
    try:yield
    finally:torch.set_num_threads(old)


def test_log_loss_cell_exact_field_gradients_and_independent_storage():
    from tools.material_loss_cell import LossCell
    rn=t([.01,.02]);cn=t([.1,.2])
    data=rn[:,None,None]**2*cn[None,:,None]*t([.5,.2])[None,None,:]
    cell=LossCell(rn,cn,data)
    data.fill_(1);rn.fill_(.9);cn.fill_(.9)
    c,r,f=t(.13,True),t(.013,True),t(.4,True)
    result=cell.evaluate(c,r,f)
    expected=r*r*c*(.5*(1-f)+.2*f)
    assert torch.allclose(result['loss'],expected,rtol=1e-13,atol=0)
    assert result['energy']+result['loss']==1
    assert torch.autograd.gradcheck(lambda c,r,f:cell.evaluate(c,r,f)['loss'],(c,r,f),eps=1e-6,atol=1e-10)
    # A tiny signal remains represented even when its energy rounds to one.
    tiny=LossCell(t([.01,.02]),t([.1,.2]),torch.full((2,2,2),1e-25,dtype=torch.float64))
    assert tiny.evaluate(c,r,f)['loss']>0 and tiny.evaluate(c,r,f)['energy']==1


def test_loss_cell_rejects_invalid_data_and_outside_queries():
    from tools.material_loss_cell import LossCell, generate
    rn=t([.01,.02]);cn=t([.1,.2]);values=torch.full((2,2,2),.3,dtype=torch.float64)
    for bad in (values*0,values*4,values.float(),values[:1],values*float('nan')):
        with pytest.raises(ValueError):LossCell(rn,cn,bad)
    with pytest.raises(ValueError):LossCell(rn.flip(0),cn,values)
    with pytest.raises(ValueError):LossCell(rn,t([0,.2]),values)
    cell=LossCell(rn,cn,values)
    for args in ((t(.09),t(.015),t(.4)),(t(.15),t(.03),t(.4)),(t(.15),t(.015),t(1.1)),(t([.15]),t(.015),t(.4))):
        with pytest.raises(ValueError):cell.evaluate(*args)
    with pytest.raises(ValueError):generate(rn,cn,resolution=1024)


def test_loss_cell_demo_publishes_finite_comparisons_without_acceptance_claim(tmp_path,monkeypatch):
    from tools.testing_dashboard import loss_cell_demo as demo
    from tools.testing_dashboard.server import TestingServer
    data=dict(rows=[dict(roughness=.001,loss=1e-12,reference=1e-12,energy=1-1e-12,slope=4e-9,reference_slope=4e-9)],scope='fixture')
    monkeypatch.setattr(demo,'measure',lambda:data)
    with TestingServer(tmp_path,port=0,allow_loopback_for_tests=True) as server:
        demo.publish('http://'+server.address,'cell-demo')
        page=server.index.page('shading/loss-cell-demo')
        assert page['evidence']['state']=='finished' and len(page['artifacts'])==5
        assert 'does not admit' in server.state.get('loss-cell/read-me',run='cell-demo')['value']


@pytest.mark.parametrize('roughness,cosine',[(.001,.43),(.01,.43),(.05,.43),(.199,.43),(.01,.0001),(.05,.0025)])
def test_local_log_loss_cell_reference_value_gradient_and_refinement(roughness,cosine,request):
    from tools.material_loss_cell import generate
    from tools import material_loss_reference as reference
    cells=[generate(t([roughness*math.exp(-w),roughness*math.exp(w)]),
                    t([cosine*math.exp(-w),cosine*math.exp(w)]),resolution=512) for w in (.01,.005)]
    rows=[]
    for f0 in (0.,.04,1.):
        args=t(cosine,True),t(roughness,True),t(f0,True)
        expected=reference.directional(*args,resolution=512,method='log')
        grads=torch.autograd.grad(expected,args)
        coarse,fine=[cell.evaluate(*args)['loss'] for cell in cells]
        actual_grads=torch.autograd.grad(fine,args)
        rows.append(dict(roughness=roughness,cosine=cosine,f0=f0,reference=expected.item(),loss=fine.item(),
                         relative_error=abs((fine/expected-1).item()),energy_error=abs((fine-expected).item()),
                         refinement=abs((coarse/fine-1).item()),
                         gradients=[g.item() for g in grads],
                         gradient_errors=[abs((a-b).item()) for a,b in zip(actual_grads,grads)]))
    evidence=Evidence(request.node.nodeid)
    evidence.record('local-loss-cell',dict(rows=rows,scope='Six local cells, not a global resource, integrated mean, furnace or performance acceptance.'))
    with evidence.check('Local value and residual accuracy'):
        assert all(r['relative_error']<=.001 and r['energy_error']<=5e-4 for r in rows),rows
    with evidence.check('Local partial derivative accuracy'):
        assert all(e<=.01+.02*abs(g) for r in rows for e,g in zip(r['gradient_errors'],r['gradients'])),rows
    with evidence.check('Local cell refinement'):
        assert all(r['refinement']<=.001 and 0<r['loss']<=1 for r in rows),rows
