import gc
import importlib.util
import io
import json
from pathlib import Path
import random
import sqlite3
import tempfile
import unittest
import weakref
from fractions import Fraction as Q
from unittest.mock import patch

from disk_exact_json import DiskHeldIndex, IndexLimits, dump_exact, validate_exact_graph
from fixture_forecast import make_fixture, reference_dump, resource_forecast

LATEST_FORECAST = None

FROZEN = Path("D:/devspace/gffit/qualification-20261006/boundary-sh28/gradient136-source20-checkpoint2")
spec = importlib.util.spec_from_file_location("frozen_legacy312", FROZEN / "streamed_exact_json312.py")
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)


class CodecTests(unittest.TestCase):
    def compare(self, value):
        old, new, oracle = io.StringIO(), io.StringIO(), io.StringIO()
        expected = legacy.dump_exact(value, old)
        observed = dump_exact(value, new, limits=IndexLimits(cache_kib=256))
        reference_dump(value, oracle)
        self.assertEqual(new.getvalue(), old.getvalue())
        self.assertEqual(new.getvalue(), oracle.getvalue())
        self.assertEqual(observed, expected)
        return observed

    def test_scalar_hex_unicode_and_key_order(self):
        self.compare({"b": [None, True, False, 0, -4, "é😀\n", Q(0), Q(-1,7)],
                      "a": ((), {}, Q(2**16000+1,2**16001+3))})

    def test_aliases_distinct_equal_and_counts(self):
        shared = [Q(1,7), "x"]
        left, right = [Q(2,9)], [Q(2,9)]
        value = [shared, shared, left, right, {"shared": shared}]
        self.compare(value)
        self.assertIs(value[0], value[1])
        self.assertIsNot(left, right)

    def test_complete_all_named_histories(self):
        value = make_fixture(64)
        stats = self.compare(value)
        self.assertEqual(stats["serialized_fraction_occurrences"], 8*64+2)
        self.assertIs(value["coordinator_history"][5], value["pixel_traces"][5])

    def test_deterministic_random_dags(self):
        rng = random.Random(341342)
        for case in range(12):
            pool = [None, True, Q(-case,17), "é"]
            for number in range(20):
                choices = [rng.choice(pool) for _ in range(rng.randrange(4))]
                value = choices if number % 2 else {"k"+str(i): x for i,x in enumerate(choices)}
                pool.append(value)
            with self.subTest(case=case):
                self.compare(pool)

    def test_cycles_at_root_and_between_branches(self):
        a = []; a.append(a)
        b = {}; b["self"] = b
        c, d = [], []; c.append(d); d.append(c)
        for value in (a,b,{"left": c, "right": d}):
            out = io.StringIO()
            with self.assertRaisesRegex(ValueError, "acyclic"):
                dump_exact(value, out)
            self.assertEqual(out.getvalue(), "")

    def test_bad_types_and_keys_before_output(self):
        class Int(int): pass
        class Str(str): pass
        class FractionSubclass(Q): pass
        for value in (1.0, float("nan"), object(), {1: Q(1)}, {Str("x"): 1},
                      Int(1), FractionSubclass(1,7), {"valid": [1], "late": object()}):
            out = io.StringIO()
            with self.assertRaises(TypeError):
                dump_exact(value, out)
            self.assertEqual(out.getvalue(), "")

    def test_no_arbitrary_hash_or_equality_hooks(self):
        class Poison:
            def __eq__(self, other): raise AssertionError("no equality")
            def __hash__(self): raise AssertionError("no hash")
        item = Poison()
        with tempfile.TemporaryDirectory() as root:
            with DiskHeldIndex(Path(root)/"index.sqlite") as index:
                self.assertTrue(index.remember(item))
                self.assertFalse(index.remember(item))
                self.assertIs(index[id(item)], item)
        with self.assertRaises(TypeError):
            validate_exact_graph([item])

    def test_wide_dfs_frontier_is_one_frame(self):
        value = list(range(10000))
        telemetry = {}
        stats = validate_exact_graph(value, telemetry=telemetry)
        self.assertEqual(stats, legacy.validate_exact_graph(value))
        self.assertEqual(telemetry["maximum_active_frames"], 1)
        self.assertEqual(telemetry["reference_pages"], 3)

    def test_deep_iterative_validation(self):
        value = Q(1,7)
        for _ in range(2000): value = [value]
        telemetry = {}
        self.assertEqual(validate_exact_graph(value, telemetry=telemetry),
                         legacy.validate_exact_graph(value))
        self.assertEqual(telemetry["maximum_active_frames"], 2000)

    def test_entry_refusal_before_output(self):
        out = io.StringIO()
        with self.assertRaises(MemoryError):
            dump_exact([Q(1,7),Q(2,9)], out, limits=IndexLimits(max_entries=2))
        self.assertEqual(out.getvalue(), "")

    def test_database_full_before_output(self):
        out = io.StringIO()
        with self.assertRaises(sqlite3.Error):
            dump_exact(list(range(1024)), out,
                       limits=IndexLimits(max_database_bytes=8192, max_scratch_bytes=8192))
        self.assertEqual(out.getvalue(), "")

    def test_disk_failure_before_output(self):
        out = io.StringIO()
        failure = OSError("injected scratch failure")
        with patch.object(DiskHeldIndex, "flush", side_effect=failure):
            with self.assertRaises(OSError) as caught:
                dump_exact([Q(1,7)], out)
        self.assertIs(caught.exception, failure)
        self.assertEqual(out.getvalue(), "")

    def test_writer_exception_identity(self):
        failure = OSError("exact fixture writer refusal")
        class Fail:
            def write(self, value): raise failure
        with self.assertRaises(OSError) as caught: dump_exact([Q(1)], Fail())
        self.assertIs(caught.exception, failure)

    def test_original_encoder_recursion_failure_retained(self):
        value = 1
        for _ in range(2000): value = [value]
        for writer in (legacy.dump_exact, dump_exact):
            with self.assertRaises(RecursionError): writer(value, io.StringIO())

    def test_scratch_and_input_remain_intact(self):
        value = make_fixture(64)
        before = io.StringIO(); reference_dump(value,before)
        with tempfile.TemporaryDirectory() as root:
            self.compare(value)
            dump_exact(value, io.StringIO(), scratch_root=root)
            self.assertEqual(list(Path(root).iterdir()), [])
        after = io.StringIO(); reference_dump(value,after)
        self.assertEqual(before.getvalue(),after.getvalue())

    def test_small_fixture_count_formula(self):
        first = self.compare(make_fixture(64))
        second = self.compare(make_fixture(128))
        self.assertEqual(second["validated_unique_objects"]-first["validated_unique_objects"], 5*64)
        self.assertEqual(second["validated_container_edges"]-first["validated_container_edges"], 8*64)

    def test_forecast_is_only_small_sample_and_has_reserves(self):
        global LATEST_FORECAST
        forecast = LATEST_FORECAST = resource_forecast()
        self.assertEqual(forecast["proposed_rows"],16_777_216)
        self.assertGreater(forecast["predicted_unique_objects"],67_108_865)
        self.assertEqual(forecast["minimum_fresh_available_RAM_bytes"] -
                         forecast["host_working_set_forecast_bytes"],4*1024**3)
        self.assertEqual(forecast["minimum_fresh_free_disk_bytes"] -
                         forecast["scratch_and_two_output_forecast_bytes"],1024**3)

    def test_large_fixture_cannot_be_launched_by_helper(self):
        with self.assertRaisesRegex(ValueError, "large runs"):
            make_fixture(16_777_216)


class IndexTests(unittest.TestCase):
    def index(self, **kwargs):
        root = tempfile.TemporaryDirectory()
        self.addCleanup(root.cleanup)
        index = DiskHeldIndex(Path(root.name)/"held.sqlite", IndexLimits(**kwargs))
        self.addCleanup(index.close)
        return index

    def test_many_reference_pages_and_exact_lookup(self):
        index = self.index(reference_page_slots=16, commit_entries=17)
        values = [[i] for i in range(1024)]
        for value in values: index[id(value)] = value
        for value in values:
            self.assertIn(id(value), index)
            self.assertIs(index[id(value)], value)
        self.assertEqual(len(index),1024)
        self.assertEqual(index.metrics()["reference_pages"],64)

    def test_foreign_key_refused_without_mutation(self):
        index = self.index()
        a,b = [],[]
        index[id(a)] = a
        with self.assertRaises(ValueError): index[id(a)] = b
        self.assertEqual(len(index),1)
        self.assertIs(index[id(a)],a)

    def test_address_type_range_and_missing_key(self):
        index = self.index()
        for key in (True,False,"1",1.0):
            with self.assertRaises(TypeError): key in index
        for key in (0,-1,1<<64):
            with self.assertRaises(ValueError): key in index
        self.assertNotIn(1,index)
        with self.assertRaises(KeyError): index[1]

    def test_strong_original_refs_survive_entry_refusal(self):
        class Value: pass
        index = self.index(max_entries=1)
        value = Value(); ref = weakref.ref(value); key = id(value)
        index.remember(value); del value; gc.collect()
        self.assertIs(ref(), index[key])
        with self.assertRaises(MemoryError): index.remember(Value())
        self.assertIs(ref(), index[key])

    def test_corrupt_ordinal_never_authorizes_another_object(self):
        index = self.index()
        a,b = [],[]; index.remember(a); index.remember(b)
        index._connection.execute("UPDATE held SET ordinal=2 WHERE address=?",(id(a).to_bytes(8,"big"),))
        with self.assertRaisesRegex(ValueError,"identity"):
            index.remember(a)
        with self.assertRaises(ValueError): index[id(b)]

    def test_existing_file_never_replaced(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/"held.sqlite"; path.write_bytes(b"original evidence")
            with self.assertRaises(FileExistsError): DiskHeldIndex(path)
            self.assertEqual(path.read_bytes(),b"original evidence")

    def test_limits_types_and_ranges(self):
        for values in ({"max_entries":True},{"cache_kib":1.0},{"commit_entries":0},
                       {"reference_page_slots":0},{"max_database_bytes":1},
                       {"max_database_bytes":8192,"max_scratch_bytes":8191}):
            with self.assertRaises((TypeError,ValueError)): IndexLimits(**values)

    def test_sqlite_pragmas_and_close_refusal(self):
        index = self.index(cache_kib=256,commit_entries=7)
        self.assertEqual(index._connection.execute("PRAGMA mmap_size").fetchone()[0],0)
        self.assertEqual(index._connection.execute("PRAGMA cache_size").fetchone()[0],-256)
        self.assertEqual(index._connection.execute("PRAGMA journal_mode").fetchone()[0],"delete")
        value=[]; index.remember(value); index.flush(); index.close()
        with self.assertRaises(ValueError): index.remember(value)


if __name__ == "__main__": unittest.main()
