"""Every public animation entry point rejects invalid tensor contracts explicitly."""
import pytest
import torch


@pytest.mark.parametrize('name', ['blend_shapes', 'rodrigues', 'forward_kinematics', 'skin_lbs', 'evaluate_barycentric'])
@pytest.mark.parametrize('damage', ['dtype', 'device', 'layout', 'shape', 'nonfinite'])
def test_animation_each_public_entry_rejects_invalid_tensor_contract(name, damage):
    from test_animation_stream import case
    function, original = case(name)
    inputs = list(original)
    first = inputs[0]
    if damage == 'dtype':
        inputs[0] = first.to(torch.float16)
    elif damage == 'device':
        inputs[0] = first.to('meta')
    elif damage == 'layout':
        storage = torch.empty((*first.shape[:-1], first.shape[-1] * 2), dtype=first.dtype)
        inputs[0] = storage[..., ::2]
        inputs[0].copy_(first)
    elif damage == 'shape':
        inputs[0] = first[..., :2].contiguous()
    else:
        inputs[0] = first.clone()
        inputs[0].reshape(-1)[0] = float('nan')
    with pytest.raises(TypeError if damage == 'dtype' else ValueError):
        function(*inputs)
    # Deliberately malformed caller copies never mutate the known-good fixture.
    for value in original:
        if value.is_floating_point():
            assert bool(torch.isfinite(value).all())
