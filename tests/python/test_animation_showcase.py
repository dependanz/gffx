"""Finite first-party rig presentation; its image is not conformance evidence."""
import math

import torch

from tools.testing_dashboard import animation_showcase as demo


def test_animation_showcase_uses_all_five_public_operations(monkeypatch):
    import gffx
    calls = []
    for module, name in ((gffx.torch.mesh, 'blend_shapes'), (gffx.torch.transforms, 'rodrigues'),
                         (gffx.torch.skeleton, 'forward_kinematics'), (gffx.torch.mesh, 'skin_lbs'),
                         (gffx.torch.mesh, 'evaluate_barycentric')):
        original = getattr(module, name)
        def wrapped(*args, _function=original, _name=name, **kwargs):
            calls.append(_name)
            return _function(*args, **kwargs)
        monkeypatch.setattr(module, name, wrapped)
    result = demo.frame(.4)
    assert calls == ['blend_shapes', 'rodrigues', 'forward_kinematics', 'skin_lbs', 'evaluate_barycentric']
    assert result['max_native_reference_error'] <= 2e-6
    assert result['markers'].shape == (3, 3)
    assert bool(torch.isfinite(result['vertices']).all())
    assert bool(torch.isfinite(result['normals']).all())


def test_animation_showcase_has_finite_loop_stationary_plinth_and_nonrigid_motion():
    source = demo.fixture()
    a, b, closed = demo.frame(0., source), demo.frame(math.pi / 2, source), demo.frame(math.tau, source)
    count = len(source['template'])
    assert demo.FRAMES == 24 and demo.JOINTS == 5
    assert count < 2000 and len(source['faces']) < 3000
    torch.testing.assert_close(a['vertices'][count:], b['vertices'][count:], atol=0, rtol=0)
    torch.testing.assert_close(a['vertices'], closed['vertices'], atol=2e-6, rtol=2e-5)
    assert float((a['vertices'][:count] - b['vertices'][:count]).abs().max()) > .05
    assert float((a['joint_positions'] - b['joint_positions']).abs().max()) > .05
    torch.testing.assert_close(source['skin_weights'].sum(-1), torch.ones(count), atol=0, rtol=0)
    assert torch.count_nonzero(source['inverse_bind'][..., :3, 3]) == demo.JOINTS
