"""Serial M1 test worker using existing resource supervision and wheel identity."""
import argparse
import os
from pathlib import Path
import time

from tools import material_loss_cost as cost, shading_cuda_closeout as artifact


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    for name in ("fixtures", "output", "wheel"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--test", action="append", required=True)
    args = parser.parse_args()
    if not args.worker:
        if args.output.exists() or args.output.with_suffix(".xml").exists():
            raise ValueError("report path already exists; preserve prior evidence")
        if args.cuda:
            cost.PROFILE["sampled_working_set_limit_bytes"] = 2 * 1024**3
        extra = ["--wheel", str(args.wheel), "--sha256", args.sha256]
        if args.cuda:
            extra.append("--cuda")
        for test in args.test:
            extra.extend(("--test", test))
        return cost.supervise(args.fixtures, args.output,
                              worker_module="tools.m1_worker_probe", worker_args=extra)
    row = {"status": "working", "worker_pid": os.getpid(), "cuda": args.cuda}
    cost.write_report(args.output, row)
    deadline = time.perf_counter() + 120
    try:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        row["wheel"] = artifact.identity(args.wheel, args.sha256)
        os.environ['GFFX_TEST_ARTIFACT_SHA256'] = row['wheel']['sha256']
        row["driver_sha256"] = artifact.digest(__file__)
        row["tests"] = {test.split("::")[0]: artifact.digest(test.split("::")[0])
                        for test in args.test}
        if args.cuda:
            if not torch.cuda.is_available():
                raise ValueError("CUDA required")
            torch.cuda.reset_peak_memory_stats()
        import pytest
        code = pytest.main(["-q", "-x", "-p", "tools.testing_dashboard.pytest_plugin",
                            *args.test, "--junitxml=" + str(args.output.with_suffix(".xml"))])
        row["pytest_exit_code"] = int(code)
        if args.cuda:
            row["gpu_peak_bytes"] = torch.cuda.max_memory_allocated()
            if row["gpu_peak_bytes"] > 64 * 1024**2:
                raise MemoryError("GPU allocation cap")
        cost.check_deadline(deadline)
        if code:
            raise ValueError("M1 tests did not pass")
        row["status"] = "passed"
    except Exception as error:
        row.update(status="stopped", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        cost.write_report(args.output, row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
