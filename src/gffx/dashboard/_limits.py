"""Accounted-payload limits, not a promise about process RSS or hard real-time behavior."""
from dataclasses import dataclass, fields
from ._framing import ArrayView


@dataclass(frozen=True)
class ResourceLimits:
    history_entries: int = 2048
    text_entries: int = 64
    path_bytes: int = 16 * 1024 * 1024
    run_bytes: int = 64 * 1024 * 1024
    paths: int = 128
    runs: int = 8
    log_bytes: int = 64 * 1024 * 1024
    total_log_bytes: int = 256 * 1024 * 1024
    page_bytes: int = 8 * 1024 * 1024

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("%s must be a positive integer" % field.name)


def value_bytes(value):
    if isinstance(value, ArrayView):
        return len(value.data) + 64
    if isinstance(value, str):
        return len(value.encode("utf-8")) + 64
    if isinstance(value, (bytes, memoryview)):
        return len(value) + 64
    if isinstance(value, dict):
        return 64 + sum(value_bytes(k) + value_bytes(v) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return 64 + sum(value_bytes(v) for v in value)
    return 32
