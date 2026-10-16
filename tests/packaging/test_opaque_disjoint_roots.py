"""Independent physical roots share one enclosure and budget, not a fictitious tree."""
import pytest
from tools.opaque_square_reference import integrate_square
from tools.opaque_shading_intervals import I,Jet,Unresolved

def root(value):
    return dict(field=lambda s,t:[s.lift(value,tangent=2*value)*(s+t)],depth=0)

def test_independent_roots_preserve_closed_integral_tangent_and_actual_work():
    result=integrate_square(None,chart_cover=[root(2),root(3)],disjoint_roots=True,max_cells=2,max_depth=0)
    assert result['cells']==2
    for key,value in [('value',5),('derivative',10)]:
        q=result[key][0]
        assert q.lo<=value<=q.hi
        assert q.width<I('1e-12').lo

def test_independent_roots_never_accept_partial_cover_or_reset_budget():
    with pytest.raises(ValueError,match='root.*budget'):
        integrate_square(None,chart_cover=[root(2),root(3)],disjoint_roots=True,max_cells=1)
    calls=[]
    def unresolved(s,t):
        calls.append(1)
        raise Unresolved('deliberately unresolved second physical root')
    with pytest.raises(Unresolved):
        integrate_square(None,chart_cover=[root(2),dict(field=unresolved,depth=0)],disjoint_roots=True,max_cells=2)
    assert calls

@pytest.mark.parametrize('kwargs',[
    dict(disjoint_roots=1),dict(disjoint_roots=True),
    dict(disjoint_roots=True,chart_cover=[dict(root(2),depth=1)]),
    dict(disjoint_roots=True,chart_cover=[root(2)],initial_leaves=[]),
])
def test_independent_root_contract_rejects_ambiguous_accounting(kwargs):
    with pytest.raises(ValueError,match='disjoint'):
        integrate_square(None,**kwargs)

def test_default_chart_partition_still_requires_complete_binary_depths():
    with pytest.raises(ValueError,match='binary'):
        integrate_square(None,chart_cover=[root(2),root(3)])
    result=integrate_square(None,chart_cover=[dict(root(2),depth=1),dict(root(3),depth=1)],max_cells=3)
    assert result['cells']==3
