"""Isolated serializer prototype; never imported by a numerical worker.

Only temporary identity routing and DFS frontier storage change. The immutable
input graph and every unique original object remain strongly held throughout
preflight. SQLite contains exact unsigned64 address keys and reference ordinals,
never scientific values. Its cache is a target, not a total-process memory cap.
Strong-reference pages remain O(unique objects); caller histories are not freed.
An index/validation refusal happens before destination writes. Writer errors may
leave partial new output, as in the original codec. Scratch is disposable.
"""
from dataclasses import dataclass
from fractions import Fraction
import json
from pathlib import Path
import sqlite3
import struct
import sys
import tempfile

MAX_ENTRIES = (1 << 63) - 1
MAX_ADDRESS = (1 << 64) - 1


@dataclass(frozen=True)
class IndexLimits:
    reference_page_slots: int = 4096
    cache_kib: int = 32768
    commit_entries: int = 4096
    max_entries: int = MAX_ENTRIES
    max_database_bytes: int = 16 * 1024**3
    max_scratch_bytes: int = 20 * 1024**3

    def __post_init__(self):
        for name, lower, upper in (
            ("reference_page_slots", 1, 65536),
            ("cache_kib", 1, 1048576),
            ("commit_entries", 1, 65536),
            ("max_entries", 1, MAX_ENTRIES),
            ("max_database_bytes", 8192, (1 << 32) * 4096 - 8192),
            ("max_scratch_bytes", 8192, MAX_ENTRIES),
        ):
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError("ordinary exact index limit")
            if not lower <= value <= upper:
                raise ValueError("finite exact index limit")
        if self.max_scratch_bytes < self.max_database_bytes:
            raise ValueError("scratch must cover the database")


class DiskHeldIndex:
    """Exact addresses route to append-only pages of original Python references."""

    def __init__(self, path, limits=None):
        if struct.calcsize("P") != 8:
            raise ValueError("qualified unsigned64 process layout")
        self.limits = limits if limits is not None else IndexLimits()
        if type(self.limits) is not IndexLimits:
            raise TypeError("ordinary exact index limits")
        self.path = Path(path)
        self._pages = []
        self._count = self._pending = 0
        self._failed = self._closed = False
        self._connection = None
        self._peak_scratch_bytes = 0
        # Never attach to or replace an existing database/evidence file.
        with self.path.open("xb"):
            pass
        try:
            self._connection = sqlite3.connect(str(self.path))
            connection = self._connection
            connection.execute("PRAGMA page_size=4096")
            connection.execute("PRAGMA journal_mode=DELETE")
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("PRAGMA temp_store=FILE")
            connection.execute("PRAGMA mmap_size=0")
            connection.execute("PRAGMA cache_size=-" + str(self.limits.cache_kib))
            connection.execute("PRAGMA cache_spill=ON")
            connection.execute("PRAGMA max_page_count=" +
                               str(self.limits.max_database_bytes // 4096))
            connection.execute(
                "CREATE TABLE held(address BLOB PRIMARY KEY "
                "CHECK(length(address)=8), ordinal INTEGER NOT NULL "
                "CHECK(ordinal>0)) WITHOUT ROWID"
            )
            connection.commit()
            self._check_scratch()
        except BaseException:
            self._failed = True
            if self._connection is not None:
                self._connection.close()
            self._closed = True
            raise

    @staticmethod
    def _key(key):
        if type(key) is not int:
            raise TypeError("ordinary exact address")
        if not 1 <= key <= MAX_ADDRESS:
            raise ValueError("qualified full unsigned64 address")
        return key.to_bytes(8, "big")

    def _usable(self):
        if self._failed or self._closed:
            raise ValueError("whole exact index failed or closed")

    def _ordinal(self, encoded):
        self._usable()
        try:
            row = self._connection.execute(
                "SELECT ordinal FROM held WHERE address=?", (encoded,)
            ).fetchone()
        except sqlite3.Error:
            self._failed = True
            raise
        return None if row is None else row[0]

    def _original(self, ordinal):
        if type(ordinal) is not int or not 1 <= ordinal <= self._count:
            self._failed = True
            raise ValueError("valid held exact reference ordinal required")
        page, slot = divmod(ordinal - 1, self.limits.reference_page_slots)
        return self._pages[page][slot]

    def _check_scratch(self):
        # Includes the rollback journal before commit. SQLite max_page_count is
        # a separate hard database-file guard; this guard is checked each batch.
        size = sum(path.stat().st_size for path in (
            self.path, Path(str(self.path) + "-journal")
        ) if path.exists())
        self._peak_scratch_bytes = max(self._peak_scratch_bytes, size)
        if size > self.limits.max_scratch_bytes:
            self._failed = True
            raise OSError("whole exact index scratch forecast exceeded")

    def flush(self):
        self._usable()
        try:
            self._check_scratch()
            self._connection.commit()
            self._pending = 0
            self._check_scratch()
        except BaseException:
            self._failed = True
            raise

    def remember(self, value):
        """Return True for a new node; every reuse verifies the original with is."""
        encoded = self._key(id(value))
        ordinal = self._ordinal(encoded)
        if ordinal is not None:
            if self._original(ordinal) is not value:
                self._failed = True
                raise ValueError("held exact object identity required")
            return False
        if self._count >= self.limits.max_entries:
            raise MemoryError("whole exact index entry refusal")
        try:
            page, slot = divmod(self._count, self.limits.reference_page_slots)
            if slot == 0:
                self._pages.append([None] * self.limits.reference_page_slots)
            self._pages[page][slot] = value
            self._count += 1
            self._connection.execute(
                "INSERT INTO held(address,ordinal) VALUES(?,?)",
                (encoded, self._count),
            )
            self._pending += 1
            if self._pending >= self.limits.commit_entries:
                self.flush()
        except BaseException:
            # A partly inserted node never authorizes continued validation.
            # Its original remains strongly retained until this index dies.
            self._failed = True
            raise
        return True

    def __contains__(self, key):
        encoded = self._key(key)
        ordinal = self._ordinal(encoded)
        if ordinal is None:
            return False
        if id(self._original(ordinal)) != key:
            self._failed = True
            raise ValueError("held exact object address required")
        return True

    def __getitem__(self, key):
        encoded = self._key(key)
        ordinal = self._ordinal(encoded)
        if ordinal is None:
            raise KeyError(key)
        original = self._original(ordinal)
        if id(original) != key:
            self._failed = True
            raise ValueError("held exact object address required")
        return original

    def __setitem__(self, key, value):
        self._key(key)
        if id(value) != key:
            raise ValueError("supplied address never authorizes foreign object")
        self.remember(value)

    def __len__(self):
        return self._count

    def metrics(self):
        slots = len(self._pages) * self.limits.reference_page_slots
        return {
            "reference_entries": self._count,
            "reference_pages": len(self._pages),
            "reference_slots": slots,
            "reference_pointer_payload_bytes": slots * struct.calcsize("P"),
            "reference_list_shallow_bytes": sum(
                sys.getsizeof(page) for page in self._pages
            ) + sys.getsizeof(self._pages),
            "cache_target_bytes": self.limits.cache_kib * 1024,
            "peak_observed_scratch_bytes": self._peak_scratch_bytes,
            "failed": self._failed,
        }

    def close(self):
        if not self._closed:
            self._connection.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        try:
            self.close()
        except BaseException:
            if kind is None:
                raise


def validate_exact_graph(root, *, scratch_root=None, limits=None, telemetry=None):
    """Same unique-node/edge counts and rejection order as the frozen validator."""
    with tempfile.TemporaryDirectory(prefix="exact-index-", dir=scratch_root) as work:
        with DiskHeldIndex(Path(work) / "held.sqlite", limits) as held:
            active = set()
            frames = []
            nodes = edges = maximum_frames = 0
            value = root
            try:
                while True:
                    identity = id(value)
                    if identity in active:
                        raise ValueError("acyclic original JSON state required")
                    if held.remember(value):
                        nodes += 1
                        kind = type(value)
                        if value is None or kind in (int, bool, str, Fraction):
                            pass
                        elif kind is dict:
                            if any(type(key) is not str for key in value):
                                raise TypeError("ordinary original string JSON keys")
                            edges += len(value)
                            active.add(identity)
                            frames.append((identity, iter(value.values())))
                        elif kind in (list, tuple):
                            edges += len(value)
                            active.add(identity)
                            frames.append((identity, iter(value)))
                        else:
                            raise TypeError("original exact JSON state types")
                        maximum_frames = max(maximum_frames, len(frames))
                    # One iterator per ancestor, never a tuple for every sibling.
                    while frames:
                        parent, children = frames[-1]
                        try:
                            value = next(children)
                            break
                        except StopIteration:
                            frames.pop()
                            active.remove(parent)
                    else:
                        held.flush()
                        return {"validated_unique_objects": nodes,
                                "validated_container_edges": edges}
            finally:
                if telemetry is not None:
                    telemetry.update(held.metrics())
                    telemetry.update(maximum_active_frames=maximum_frames,
                                     visited_unique_objects=nodes,
                                     visited_container_edges=edges)


def dump_exact(value, out, *, scratch_root=None, limits=None, telemetry=None):
    stats = validate_exact_graph(
        value, scratch_root=scratch_root, limits=limits, telemetry=telemetry
    )
    fractions = 0

    def exact_default(item):
        nonlocal fractions
        if type(item) is not Fraction:
            raise TypeError("ordinary exact Fraction only")
        fractions += 1
        return {"rational_hex": (format(item.numerator, "x"),
                                 format(item.denominator, "x"))}

    # Deliberately identical to the frozen codec's encoding call and options.
    json.dump(value, out, default=exact_default, separators=(",", ":"),
              allow_nan=False, ensure_ascii=True)
    stats["serialized_fraction_occurrences"] = fractions
    return stats
