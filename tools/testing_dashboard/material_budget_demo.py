"""Finite energy-accounting curves; not a full material rendering demonstration."""
import argparse
import hashlib
from pathlib import Path
import subprocess

import torch
from gffx.dashboard import connect
from tools import material_budget_reference as budget
from tools import shading_energy_probe as oracle
from .reporter import Reporter


def examples():
    rows = []
    for index in range(33):
        c = torch.tensor(.1 + .8*index/32, dtype=torch.float64)
        r = torch.tensor(.6, dtype=torch.float64, requires_grad=True)
        f = torch.tensor(.04, dtype=torch.float64)
        energy = budget.directional(c, r, f, resolution=128)
        derivative = torch.autograd.grad(energy, r)[0]
        rows.append(dict(cosine=c.item(), specular=energy.item(),
                         matte_white=1-energy.item(), dE_dr=derivative.item()))
    return dict(rows=rows, roughness=.6, f0=.04,
                scope='Reference accounting illustration; not an independent furnace, public material or frame-rate claim.')


def publish(address, run):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        data = examples()
    finally:
        torch.set_num_threads(previous)
    reporter = Reporter(address, run)
    source = b''.join(Path(p).read_bytes() for p in (__file__, budget.__file__, oracle.__file__))
    provenance = dict(revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                      dirty=bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True)),
                      artifact_sha256=hashlib.sha256(source).hexdigest())
    def send(event, **kwargs):
        if not reporter.send(event, **kwargs):
            raise RuntimeError(reporter.error)
    send('start', kind='demo', provenance=provenance)
    with connect(address, run=run, wait_acks=True) as dash:
        dash.text('budget/read-me', 'Roughness 0.6, F0=0.04. Each step moves view cosine from 0.1 '
                  '(near grazing) to 0.9 (near head-on); exact cosines are in the table. '
                  'Specular is the integrated shiny reflection. Matte-white is the remaining 1-E budget '
                  'for white albedo. Their sum is one by construction: this chart is NOT an independent '
                  'conservation test. The separate material-contract page checks that. The derivative '
                  'chart shows how roughness changes the shiny budget; the matte budget must change '
                  'in the opposite direction. Units are reflected fraction per unit perceptual roughness. '
                  'Fixed quadrature is a slow mathematical reference, not a public material implementation.', step=0)
        for index, row in enumerate(data['rows']):
            for name in ('specular', 'matte_white'):
                dash.scalar('budget/energy/'+name, row[name], step=index)
            dash.scalar('budget/gradient/dE_dr', row['dE_dr'], step=index)
        dash.record('budget/samples', data, step=0)
    send('demo', page='shading/budget-demo', title='Where the light budget goes', artifacts=[
        dict(kind=kind, path='budget/'+path, label=label) for kind, path, label in (
            ('text', 'read-me', 'How to read the budget'),
            ('plot', 'energy', 'Shiny reflection and remaining white-matte budget'),
            ('plot', 'gradient', 'What changing roughness does'),
            ('record', 'samples', 'All 33 samples and their exact view cosines'))])
    send('finish', exit_code=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--address', required=True)
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    publish(args.address, args.run)
