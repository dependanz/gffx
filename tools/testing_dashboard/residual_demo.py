"""Finite diagnostic: why tiny remaining-light budgets need independent checks."""
import argparse
import hashlib
import subprocess
from pathlib import Path

import torch
from gffx.dashboard import connect
from tools import material_residual_probe as probe
from .reporter import Reporter


def publish(address, run):
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        data = probe.study()
    finally:
        torch.set_num_threads(old)
    source = b"".join(Path(p).read_bytes() for p in (
        __file__, probe.__file__, probe.sharp.__file__,
        probe.loss.__file__,
        Path(probe.__file__).with_name("shading_energy_probe.py")))
    reporter = Reporter(address, run)
    def send(event, **values):
        if not reporter.send(event, **values):
            raise RuntimeError(reporter.error)
    send("start", kind="demo", provenance=dict(
        revision=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        dirty=bool(subprocess.check_output(["git","status","--porcelain"],text=True)),
        artifact_sha256=hashlib.sha256(source).hexdigest()))
    with connect(address, run=run, wait_acks=True) as dash:
        dash.text("residual/read-me",
            "Six pairs, increasing roughness 0.001, 0.003, 0.01, each with F0=0.99999999 then 1. "
            "The first plot magnifies the remaining-light fraction by one trillion. "
            "The next plots show relative refinement and independent disagreement as percentages; "
            "the declared residual-refinement target is 0.1 percent. "
            "Legacy subtracts a mean from one; direct sums the leftover at each regular view; "
            "transformed redistributes views toward the horizon; independent uses a different VNDF integrator. "
            "Loss-sinh and loss-log instead integrate lost light directly and account for below-surface reflections analytically. "
            "Their finite acceptance is backed by exact controls, direct tail integration and gradient tests. "
            "The legacy independent sampler misses a known rare rejected event; its disagreement curve is preserved, not treated as truth. "
            "The repaired coordinates share a loss identity, not two independent physical models. "
            "This is a scalar reference repair, not a new shader, all-domain proof or faster renderer.", step=0)
        for index, row in enumerate(data["rows"]):
            for method, values in row["methods"].items():
                dash.scalar("residual/magnified/"+method, values["fine"]*1e12, step=index)
                dash.scalar("residual/refinement/"+method, values["relative_change"]*100, step=index)
            dash.scalar("residual/disagreement/percent", row["independent_disagreement"]*100, step=index)
            if 'loss_coordinate_disagreement' in row:
                dash.scalar("residual/disagreement/loss-coordinates",row['loss_coordinate_disagreement']*100,step=index)
        dash.record("residual/measurements", data, step=0)
    send("demo", page="shading/residual-demo", title="Measure the lost light: residual reference repair",
        artifacts=[dict(kind=kind,path="residual/"+path,label=label) for kind,path,label in (
            ("text","read-me","What the magnified numbers mean"),
            ("plot","magnified","Remaining light, multiplied by one trillion"),
            ("plot","refinement","Change on refinement, percent; target 0.1%"),
            ("plot","disagreement","Old-method disagreement and repaired coordinate agreement, percent"),
            ("record","measurements","Exact values and unresolved scope"))])
    send("finish",exit_code=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--address", required=True)
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    publish(args.address, args.run)
