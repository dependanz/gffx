"""Finite-value validation has an explicit temporary-storage bound."""
import gffx
import pytest
import torch

CHUNK = 65536


def check_chunked_finite(monkeypatch, dtype, device, bad_index=None, empty=False):
    # Test the validator directly: no numerical kernel may hide its allocation.
    from gffx.torch import _animation
    original = torch.isfinite
    observed = []
    def bounded(value):
        observed.append((value.numel(), value.device))
        return original(value)
    monkeypatch.setattr(torch, 'isfinite', bounded)
    value = torch.ones(0 if empty else CHUNK * 2 + 1, dtype=dtype, device=device)
    if bad_index is not None:
        value[bad_index] = float('nan')
        with pytest.raises(ValueError, match='finite'):
            _animation._floating(value, 'fixture')
    else:
        _animation._floating(value, 'fixture')
    assert all(size <= CHUNK and where == value.device for size, where in observed)
    if empty:
        assert observed == []
    elif bad_index is None:
        assert [size for size, _ in observed] == [CHUNK, CHUNK, 1]
    else:
        assert len(observed) == bad_index // CHUNK + 1


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_animation_finite_validation_bounds_temporary_chunks(monkeypatch, dtype):
    check_chunked_finite(monkeypatch, dtype, 'cpu')


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('index', [0, CHUNK, CHUNK * 2])
def test_animation_finite_validation_rejects_nonfinite_in_every_chunk(monkeypatch, dtype, index):
    check_chunked_finite(monkeypatch, dtype, 'cpu', bad_index=index)


def test_animation_finite_validation_empty_input_needs_no_mask(monkeypatch):
    check_chunked_finite(monkeypatch, torch.float64, 'cpu', empty=True)
