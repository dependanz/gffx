"""One finite focused engineering run. No science imports or benchmark launch."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
HOST_LIMIT = 1024**3
TIME_LIMIT = 120
OUTPUT_LIMIT = 64*1024**2


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def child(output):
    import unittest
    import test_disk_exact_json as tests
    started = time.perf_counter()
    with (output/"tests.log").open("x",encoding="utf-8") as log:
        result = unittest.TextTestRunner(stream=log,verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromModule(tests))
    data = {"tests": result.testsRun, "failures": len(result.failures),
            "errors": len(result.errors), "skips": len(result.skipped),
            "worker_seconds": time.perf_counter()-started, "worker_pid": os.getpid(),
            "forecast": tests.LATEST_FORECAST,
            "status": "passed" if result.wasSuccessful() else "failed"}
    with (output/"child-results.json").open("x",encoding="utf-8") as target:
        json.dump(data,target,indent=2)
    return 0 if result.wasSuccessful() else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--frozen-before",type=Path)
    parser.add_argument("--child",action="store_true")
    args = parser.parse_args()
    if args.child:
        return child(args.output_dir)
    import psutil
    started = time.perf_counter()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError("new evidence directory required")
    output.mkdir()
    scratch = output/"scratch"
    scratch.mkdir()
    before = json.loads(args.frozen_before.read_text(encoding="utf-8-sig"))
    for row in before:
        if sha(row["path"]) != row["sha256"]:
            raise ValueError("frozen evidence changed before focused tests")
    snapshot = {"available_RAM_bytes": psutil.virtual_memory().available,
                "free_disk_bytes": psutil.disk_usage(str(output)).free}
    if snapshot["available_RAM_bytes"] < HOST_LIMIT+4*1024**3:
        raise MemoryError("focused host forecast plus4GiB reserve")
    if snapshot["free_disk_bytes"] < OUTPUT_LIMIT+1024**3:
        raise OSError("focused output forecast plus1GiB reserve")
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["TEMP"] = env["TMP"] = str(scratch)
    peak_rss = peak_private = peak_os_wset = peak_os_commit = samples = 0
    refusal = None
    with (output/"child-console.log").open("x",encoding="utf-8") as console:
        process = subprocess.Popen(
            [getattr(sys,"_base_executable",sys.executable),"-B",str(ROOT/"run_focused.py"),"--child",
             "--output-dir",str(output)],cwd=ROOT,env=env,
            stdout=console,stderr=subprocess.STDOUT)
        worker = psutil.Process(process.pid)
        while process.poll() is None:
            try:
                memory = worker.memory_info()
                samples += 1
                peak_rss = max(peak_rss,memory.rss)
                peak_private = max(peak_private,getattr(memory,"private",0))
                peak_os_wset = max(peak_os_wset,getattr(memory,"peak_wset",0))
                peak_os_commit = max(peak_os_commit,getattr(memory,"peak_pagefile",0))
                if memory.rss > HOST_LIMIT or getattr(memory,"private",0) > HOST_LIMIT:
                    refusal = "focused host limit exceeded"
            except psutil.NoSuchProcess:
                break
            except (psutil.AccessDenied, OSError) as error:
                refusal = "memory sampling failed: "+repr(error)
            if time.perf_counter()-started > TIME_LIMIT:
                refusal = "focused timeout"
            if refusal:
                process.kill()
                break
            time.sleep(0.01)
        exit_code = process.wait()
    checks = [{"path":row["path"],"before_sha256":row["sha256"],
               "after_sha256":sha(row["path"])} for row in before]
    frozen_equal = all(row["before_sha256"] == row["after_sha256"] for row in checks)
    source_files = [ROOT/name for name in (
        "disk_exact_json.py","fixture_forecast.py","test_disk_exact_json.py","run_focused.py")]
    result_path = output/"child-results.json"
    results = json.loads(result_path.read_text()) if result_path.exists() else {}
    identity_equal = results.get("worker_pid") == process.pid
    if not identity_equal and refusal is None:
        refusal = "monitored PID differs from the declared Python worker"
    observed_bytes = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    data = {"status": "passed" if exit_code==0 and refusal is None and frozen_equal else "stopped",
            "scope": "focused non-numerical serializer engineering only; not campaign qualification",
            "observed_at_UTC": datetime.now(timezone.utc).isoformat(),
            "tests":results,"supervised_seconds":time.perf_counter()-started,
            "exit_code":exit_code,"refusal":refusal,
            "sample_interval_seconds":0.01,"samples":samples,
            "monitored_pid":process.pid,"declared_worker_pid":results.get("worker_pid"),
            "actual_worker_identity_matches":identity_equal,
            "sampled_worker_peak_working_set_bytes":peak_rss,
            "sampled_worker_peak_private_bytes":peak_private,
            "observed_OS_peak_working_set_bytes":peak_os_wset,
            "observed_OS_peak_commit_bytes":peak_os_commit,
            "fresh_admission_snapshot":snapshot,
            "focused_limits":{"host_bytes":HOST_LIMIT,"seconds":TIME_LIMIT,
                              "output_forecast_bytes":OUTPUT_LIMIT},
            "observed_evidence_bytes_before_summary":observed_bytes,
            "frozen_evidence_hashes_equal":frozen_equal,"frozen_checks":checks,
            "sources":[{"path":str(path),"sha256":sha(path),"bytes":path.stat().st_size}
                       for path in source_files],
            "historical_campaign_ledger_modified":False,
            "numerical_render_integration_or_large_benchmark_calls":0}
    if observed_bytes > OUTPUT_LIMIT:
        data["status"] = "stopped"
        data["refusal"] = "focused output forecast exceeded"
    with (output/"focused-results.json").open("x",encoding="utf-8") as report:
        json.dump(data,report,indent=2)
    if results.get("forecast") is not None:
        with (output/"benchmark-forecast.json").open("x",encoding="utf-8") as report:
            json.dump(results["forecast"],report,indent=2)
    print(json.dumps({key:data[key] for key in ("status","tests","supervised_seconds",
                                               "sampled_worker_peak_working_set_bytes",
                                               "frozen_evidence_hashes_equal")},indent=2))
    return 0 if data["status"]=="passed" else 1


if __name__ == "__main__": raise SystemExit(main())
