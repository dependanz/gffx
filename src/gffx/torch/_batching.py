"""Packed-offset validation shared by the batched operations.

GFFX batches by packing elements end to end and marking boundaries with an int32 offset tensor,
rather than by padding to a common size. `API_CONTRACT_V0_1.md` section 5 allows an eager adapter
to synthesise the offsets for the unbatched case, which is what `resolve_offsets` does: a caller
with one mesh should not have to write `torch.tensor([0, V], dtype=torch.int32)` by hand.
"""

from __future__ import annotations

from typing import Optional

import torch


def check_offsets(offsets: torch.Tensor, total: int, name: str) -> torch.Tensor:
    """Validate a packed-offset tensor against the packed length it partitions."""
    if not isinstance(offsets, torch.Tensor):
        raise TypeError("%s must be a torch.Tensor, not %s" % (name, type(offsets).__name__))
    if offsets.device.type != "cpu":
        raise ValueError("%s must be on the cpu device; received %s" % (name, offsets.device))
    if offsets.dtype != torch.int32:
        raise TypeError(
            "%s must be int32, the packed-offset dtype; received %s. Convert explicitly with "
            "offsets.to(torch.int32)." % (name, offsets.dtype)
        )
    if offsets.dim() != 1 or offsets.numel() < 2:
        raise ValueError(
            "%s must be a 1-D tensor of at least two entries, holding B+1 boundaries; received "
            "shape %s" % (name, tuple(offsets.shape))
        )
    if not offsets.is_contiguous():
        raise ValueError("%s must be dense and C-contiguous" % (name,))
    # Checked here rather than left to the kernel because the message can name the mismatch,
    # which is the single most common packing mistake.
    if int(offsets[0]) != 0 or int(offsets[-1]) != total:
        raise ValueError(
            "%s must start at 0 and end at the packed length %d; received %d..%d"
            % (name, total, int(offsets[0]), int(offsets[-1]))
        )
    return offsets


def resolve_offsets(
    offsets: Optional[torch.Tensor], total: int, name: str
) -> torch.Tensor:
    """Return the caller's offsets, or synthesise the single-element case.

    Synthesising is permitted for the eager surface only. The streaming surface takes offsets
    explicitly, because allocating one there would be exactly the hidden allocation section 11
    forbids.
    """
    if offsets is None:
        return torch.tensor([0, total], dtype=torch.int32)
    return check_offsets(offsets, total, name)
