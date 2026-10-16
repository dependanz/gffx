"""Finite storage-observer regressions; no graph decoding or science imports."""
import argparse
from collections import Counter
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import unittest
from unittest.mock import patch

import owned_storage_observer as storage

FIXTURE_PARENT = None


class ObserverTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="observer-case-", dir=FIXTURE_PARENT)
        self.root = Path(self.temp.name)
        self.index = self.root / "scratch" / "exact-index-abcdefgh"
        self.index.mkdir(parents=True)
        self.db = self.index / "held.sqlite"
        self.journal = self.index / "held.sqlite-journal"
        self.output = self.root / "legacy312.json"
        self.db.write_bytes(b"database")
        self.journal.write_bytes(b"journal")
        self.output.write_bytes(b"output")
        self.observer = storage.StorageObserver(self.root)
        self.calls = Counter()
        self.real_stat = storage._entry_stat
        self.real_scan = storage._scandir

    def tearDown(self):
        self.temp.cleanup()

    def stat_once(self, entry):
        self.calls[Path(entry.path)] += 1
        return self.real_stat(entry)

    def disappear_at_stat(self, target, *, directory=False):
        def hook(entry):
            path = Path(entry.path)
            self.calls[path] += 1
            if path == target:
                if directory:
                    shutil.rmtree(path)
                else:
                    path.unlink()
            return self.real_stat(entry)
        return hook

    def test_journal_deletion_between_enumeration_and_stat(self):
        with patch.object(storage, "_entry_stat", self.disappear_at_stat(self.journal)):
            self.assertEqual(self.observer.bytes_used(), 14)
        self.assertEqual(self.calls[self.journal], 1)
        self.assertTrue(all(n == 1 for n in self.calls.values()))
        self.assertEqual(self.observer.allowed_disappearances, 1)

    def test_journal_deletion_between_samples(self):
        self.assertEqual(self.observer.bytes_used(), 21)
        self.journal.unlink()
        self.assertEqual(self.observer.bytes_used(), 14)

    def test_database_normal_cleanup_at_stat(self):
        with patch.object(storage, "_entry_stat", self.disappear_at_stat(self.db)):
            self.assertEqual(self.observer.bytes_used(), 13)
        self.assertEqual(self.calls[self.db], 1)

    def test_owned_index_directory_cleanup_at_stat(self):
        with patch.object(storage, "_entry_stat", self.disappear_at_stat(self.index, directory=True)):
            self.assertEqual(self.observer.bytes_used(), 6)
        self.assertEqual(self.calls[self.index], 1)

    def test_owned_index_directory_cleanup_after_stat_before_scandir(self):
        def scan(path):
            if path == self.index:
                shutil.rmtree(path)
            return self.real_scan(path)
        with patch.object(storage, "_scandir", scan), patch.object(storage, "_entry_stat", self.stat_once):
            self.assertEqual(self.observer.bytes_used(), 6)
        self.assertEqual(self.calls[self.index], 1)

    def test_owned_scratch_directory_cleanup(self):
        self.assertEqual(self.observer.bytes_used(), 21)
        scratch = self.root / "scratch"
        def scan(path):
            if path == scratch:
                shutil.rmtree(path)
            return self.real_scan(path)
        with patch.object(storage, "_scandir", scan), patch.object(storage, "_entry_stat", self.stat_once):
            self.assertEqual(self.observer.bytes_used(), 6)
        self.assertEqual(self.calls[scratch], 1)
        self.assertEqual(self.observer.bytes_used(), 6)

    def test_missing_permanent_output_at_stat_refuses(self):
        with patch.object(storage, "_entry_stat", self.disappear_at_stat(self.output)):
            with self.assertRaises(FileNotFoundError):
                self.observer.bytes_used()
        self.assertEqual(self.calls[self.output], 1)

    def test_missing_permanent_output_between_samples_refuses(self):
        self.observer.bytes_used()
        self.output.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "permanent"):
            self.observer.bytes_used()

    def test_required_permanent_output_never_created_refuses(self):
        observer = storage.StorageObserver(self.root, required=("prototype.json",))
        with self.assertRaisesRegex(FileNotFoundError, "prototype.json"):
            observer.bytes_used()

    def test_unknown_disappearing_scratch_file_refuses(self):
        unknown = self.index / "unowned.json"
        unknown.write_bytes(b"x")
        with patch.object(storage, "_entry_stat", self.disappear_at_stat(unknown)):
            with self.assertRaises(FileNotFoundError):
                self.observer.bytes_used()
        self.assertEqual(self.calls[unknown], 1)

    def test_unknown_disappearing_scratch_directory_refuses(self):
        unknown = self.root / "scratch" / "unowned-directory"
        unknown.mkdir()
        with patch.object(storage, "_entry_stat", self.disappear_at_stat(unknown, directory=True)):
            with self.assertRaises(FileNotFoundError):
                self.observer.bytes_used()
        self.assertEqual(self.calls[unknown], 1)

    def test_permission_error_on_temporary_journal_refuses(self):
        def hook(entry):
            self.calls[Path(entry.path)] += 1
            if Path(entry.path) == self.journal:
                raise PermissionError(13, "denied", str(self.journal))
            return self.real_stat(entry)
        with patch.object(storage, "_entry_stat", hook):
            with self.assertRaises(PermissionError):
                self.observer.bytes_used()
        self.assertEqual(self.calls[self.journal], 1)

    def test_permission_error_on_scratch_scandir_refuses(self):
        def scan(path):
            if path == self.index:
                raise PermissionError(13, "denied", str(path))
            return self.real_scan(path)
        with patch.object(storage, "_scandir", scan):
            with self.assertRaises(PermissionError):
                self.observer.bytes_used()

    def test_combined_storage_excess_refuses(self):
        observer = storage.StorageObserver(self.root, limit=20)
        with self.assertRaisesRegex(OSError, "combined"):
            observer.bytes_used()

    def test_reparse_point_refuses(self):
        class Reparse:
            st_mode = 0o100644
            st_size = 7
            st_file_attributes = 0x400
        def hook(entry):
            if Path(entry.path) == self.journal:
                return Reparse()
            return self.real_stat(entry)
        with patch.object(storage, "_entry_stat", hook):
            with self.assertRaisesRegex(OSError, "reparse"):
                self.observer.bytes_used()

    def test_stable_entries_are_statted_exactly_once(self):
        with patch.object(storage, "_entry_stat", self.stat_once):
            self.assertEqual(self.observer.bytes_used(), 21)
        self.assertEqual(len(self.calls), 5)
        self.assertTrue(all(n == 1 for n in self.calls.values()))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(root):
    global FIXTURE_PARENT
    started = time.perf_counter()
    cpu = time.process_time()
    root.resolve().relative_to(Path("D:/devspace/gffit/qualification-20261006").resolve())
    root.mkdir()
    FIXTURE_PARENT = root / "scratch"
    FIXTURE_PARENT.mkdir()
    files = (Path(__file__), Path(storage.__file__),
             Path(__file__).with_name("serializer_saved_pixel486_benchmark_observer2.py"))
    sources = {str(path): sha(path) for path in files}
    with (root / "launch.json").open("x") as target:
        json.dump({"scope": "focused observer regressions only; one finite run",
                   "pid": os.getpid(), "sources": sources}, target, indent=2)
    log = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ObserverTests)
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    (root / "tests.log").write_text(log.getvalue())
    elapsed = time.perf_counter() - started
    report = {"status": "passed" if result.wasSuccessful() else "stopped",
              "tests": result.testsRun, "failures": len(result.failures),
              "errors": len(result.errors), "skips": len(result.skipped),
              "process_CPU_seconds": time.process_time() - cpu,
              "conservative_worker_seconds": elapsed, "supervised_seconds": elapsed,
              "pid": os.getpid(), "sources": sources,
              "source_hashes_unchanged": all(sha(path) == digest for path, digest in sources.items()),
              "scratch_output_bytes": sum(p.stat().st_size for p in root.rglob("*") if p.is_file()),
              "no_benchmark_graph_loaded": True, "no_retry": True}
    with (root / "result.json").open("x") as target:
        json.dump(report, target, indent=2)
    print(json.dumps(report))
    return 0 if report["status"] == "passed" and report["source_hashes_unchanged"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    raise SystemExit(main(parser.parse_args().root))
