"""Stronger private proof targets retain both curved families' original caps."""
from types import SimpleNamespace
import pytest
from tools.opaque_shading_intervals import I

@pytest.mark.parametrize('family',['outside','inside'])
@pytest.mark.parametrize('bad',[None,True,1,'0','-1','NaN','Infinity','6e-13'])
def test_curved_reference_refuses_invalid_or_looser_proof_targets(family,bad):
    from tools.opaque_original_motion_reference import angular_piece_reference
    from tools.opaque_inside_angle import inside_piece_reference
    method=angular_piece_reference if family=='outside' else inside_piece_reference
    with pytest.raises(ValueError,match='stronger reference target'):method({},piece=8 if family=='outside' else 11,motion='normal_tilt',tolerance=bad)

@pytest.mark.parametrize('family',['outside','inside'])
@pytest.mark.parametrize('target',['5e-13','3e-13','1e-13'])
def test_curved_reference_forwards_stronger_target_without_budget_reset(monkeypatch,family,target):
    import tools.opaque_original_motion_reference as outside
    import tools.opaque_inside_angle as inside
    module=outside if family=='outside' else inside;cover='angular_material_cover' if family=='outside' else 'inside_material_cover';method=module.angular_piece_reference if family=='outside' else module.inside_piece_reference
    seen=[]
    monkeypatch.setattr(module,cover,lambda *a,**kw:({'face':5742,'prepared_angle':SimpleNamespace(report=lambda:{}),'horizon_cache':SimpleNamespace(report=lambda:{})},None,SimpleNamespace(cells=1,depth=0,evaluations=0),[]))
    def integrate(field,**kw):
        assert kw['tolerance']==target and kw['max_cells']==256 and kw['max_depth']==10
        kw['checkpoint']();seen.append(True);return dict(value=[I(0)]*2,derivative=[I(0)]*2,tolerance=target)
    monkeypatch.setattr(module,'integrate_square',integrate)
    r=method({'material':{'albedo':[.2,.4,.6]}},piece=8 if family=='outside' else 11,motion='normal_tilt',tolerance=target)
    assert seen==[True] and r['tolerance']==target and r['basis']['tolerance']==target
