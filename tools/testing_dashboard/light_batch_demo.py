"""Finite measured large-light-batch evidence; never reruns the integrator."""
import argparse
import hashlib
import json
from pathlib import Path

HASHES={'costs':'e50ed04cd0dee4e2e040526d1e4000a0d42c6d67891457363608f654762c2e41',
        'environment':'8246f0a53d0a714ea26cf43ce338478a0c9453216c5827e7b8458b9303a199ab'}


def read(root,name):
    with (Path(root)/(name+'.json')).open('rb') as handle:blob=handle.read(262145)
    if len(blob)>262144 or hashlib.sha256(blob).hexdigest()!=HASHES[name]:raise ValueError('batch report identity mismatch')
    row=json.loads(blob);s=row['supervisor']
    if (not row['status'].startswith('passed') or not row['large'] or s['failure'] is not None or s['exit_code']!=0
        or s['monitored_pid']!=row['worker_pid'] or s['sampled_peak_working_set_bytes']>1024**3):raise ValueError('unqualified batch report')
    return row


def publish(root,address,run):
    from gffx.dashboard import connect
    from .reporter import Reporter
    costs,env=read(root,'costs'),read(root,'environment');reporter=Reporter(address,run)
    def send(event,**values):
        if not reporter.send(event,**values):raise RuntimeError(reporter.error)
    send('start',kind='demo',provenance={'artifact_sha256':env['producer']['wheel']['sha256']})
    with connect(address,run=run,wait_acks=True) as dash:
        dash.text('light-batches/scope','Same 65536 light directions, 64 calls instead of6554. '
            'Sampling0.596s versus37.878s; fixed independent tolerance passes. This is one difficult '
            'point, not a full image. Standalone queries remain32 entries; material workspace stays64MiB. '
            'Recorded finite CPU measurements, not a live renderer or sustained-memory guarantee.',step=0)
        for i,row in enumerate(costs['costs']):
            dash.scalar('light-batches/time/ms',1000*row['median_seconds'],step=i)
            dash.scalar('light-batches/storage/saved-bytes',row['saved_unique_storage_bytes'],step=i)
        dash.record('light-batches/evidence',dict(hashes=HASHES,costs=costs['costs'],
            environment=env['result'],errors=env['errors'],limits=env['limits'],
            workers=[costs['supervisor'],env['supervisor']]),step=0)
    send('demo',page='shading/material-light-batches-demo',title='Larger light batches: fixed accuracy and memory ceilings',artifacts=[
        dict(kind='text',path='light-batches/scope',label='Scope: one point, not a frame'),
        dict(kind='plot',path='light-batches/time',label='Milliseconds: 1x1024 forward, 8x128 forward, 1x1024 backward, 4x256 backward'),
        dict(kind='plot',path='light-batches/storage',label='Saved autograd storage, not peak scratch'),
        dict(kind='record',path='light-batches/evidence',label='Exact cost, memory and independent-error evidence')])
    send('finish',exit_code=0)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--address',required=True);p.add_argument('--run',required=True)
    a=p.parse_args();publish(a.root,a.address,a.run)


if __name__=='__main__':main()
