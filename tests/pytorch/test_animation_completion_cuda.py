"""Real-device counterparts of the focused animation closeout controls."""
import pytest
import torch
from test_animation_completion import (NAMES, check_repeated_backward,
    check_inverse_bind_chain, check_overflow, check_storage_errors)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires real CUDA")


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_animation_cuda_repeated_backward_is_byte_stable(name, dtype):
    check_repeated_backward(name, dtype, "cuda")


def test_animation_cuda_nonidentity_inverse_bind_chain_all_inputs():
    check_inverse_bind_chain("cuda")


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_animation_cuda_finite_arithmetic_overflow_is_explicit(name, dtype):
    check_overflow(name, dtype, "cuda")


@pytest.mark.parametrize("name", NAMES)
def test_animation_cuda_each_out_rejects_alias_scratch_and_gradients(name):
    check_storage_errors(name, "cuda")
