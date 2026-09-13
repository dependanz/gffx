"""Finite recovery-presentation boundaries; numerical capture runs only under its worker lease."""


def test_recovery_presentation_profile_is_fixed_and_honestly_scoped():
    from tools.testing_dashboard.image_recovery_demo import PROFILE,GUIDE
    assert PROFILE['size']==32 and PROFILE['steps']==100 and PROFILE['learning_rate']==.08
    assert PROFILE['residual_full_scale']==.1 and PROFILE['display_scale']==8
    assert 'CI-07 integration readiness' in GUIDE and 'not a consumer' in GUIDE
    assert '32' in GUIDE and 'enlarged' in GUIDE


def test_recovery_presentation_metadata_does_not_import_tensor_framework():
    import ast
    from pathlib import Path
    path=Path(__file__).parents[2]/'tools/testing_dashboard/image_recovery_demo.py'
    tree=ast.parse(path.read_text(encoding='utf-8'))
    imports=[node for node in tree.body if isinstance(node,(ast.Import,ast.ImportFrom))]
    names=[node.module or '' for node in imports if isinstance(node,ast.ImportFrom)]
    names += [alias.name for node in imports if isinstance(node,ast.Import) for alias in node.names]
    assert not any(name.startswith(('torch','numpy','gffx.torch','PIL')) for name in names)


def test_recovery_capture_preserves_qualified_fixture_and_measured_history():
    """Capture real initial/final/error pixels and optimizer states, not reconstructed or fabricated history."""
    import torch
    from tools.testing_dashboard.image_recovery_demo import capture,board,history_chart
    checks=[]
    value=capture(checkpoint=lambda:checks.append(True))
    assert len(checks)==102
    assert len(value['loss_history'])==len(value['parameter_history'])==101
    assert value['loss_history'][-1]<=value['loss_history'][0]*.01
    assert value['source_unchanged'] and value['background_unchanged']
    for name in ('target','before','after'):
        assert value[name].shape==(32,32,4) and not value[name].requires_grad
    assert torch.equal(value['residual_after'],(value['after'][...,:3]-value['target'][...,:3]).abs())
    display=board(value)
    assert display.shape==(800,608,3) and display.dtype==torch.uint8
    # Error is fixed-scale, not normalized by the observed maximum for visual drama.
    error=(value['residual_after'].amax(-1)/.1).clamp(0,1)
    expected=(error.unsqueeze(-1)*torch.tensor([255.,104.,34.],dtype=error.dtype)).round().to(torch.uint8)
    expected=expected.repeat_interleave(8,0).repeat_interleave(8,1)
    assert torch.equal(display[450:706,320:576],expected)
    chart=history_chart(value)
    assert chart.shape==(280,608,3) and chart.dtype==torch.uint8


def test_recovery_probe_rejects_incomplete_or_nonfinite_histories():
    """A cached guide must not present missing optimizer states or nonfinite values as evidence."""
    import copy
    import pytest
    from tools.image_recovery_probe import validate_metadata
    from tools.testing_dashboard.image_recovery_demo import PROFILE
    value={'profile':dict(PROFILE),'loss_history':[1.]*100+[.001],
           'parameter_history':[[.4,-.3,.8] for _ in range(101)],
           'source_unchanged':True,'background_unchanged':True,
           'independent_target':'scalar-bilinear-over-v1','target_parameters':[.4,-.3,.8]}
    validate_metadata(value)
    for key,replacement in (('loss_history',[1.]),('parameter_history',[[float('nan'),0.,.8]]*101),
                            ('source_unchanged',False),('target_parameters',[.5,-.3,.8])):
        invalid=copy.deepcopy(value);invalid[key]=replacement
        with pytest.raises(ValueError): validate_metadata(invalid)
