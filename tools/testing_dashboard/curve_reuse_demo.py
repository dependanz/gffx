"""Replay finite CPU costs with exact report identities; no workload rerun."""
import argparse
import hashlib
import json
from pathlib import Path

HASHES={
    'baseline-costs':'144b2e9911c5aa57f7dab8f6b6161844b098224ce32dc070e29df1553318354b',
    'optimized-costs':'1360f55a87d0d08cd4149d5c8dcca77d152571f429f69c0335894b718ef8ea2b',
    'optimized-environment':'a5620b6fa9f1dc10a2cf6893496648594c16d0ec3d4b5865c09e27f24832c5d1',
}


def read(root,name):
    with (Path(root)/(name+'.json')).open('rb') as handle:blob=handle.read(262145)
    if len(blob)>262144 or hashlib.sha256(blob).hexdigest()!=HASHES[name]:raise ValueError('cost report identity mismatch')
    row=json.loads(blob)
    if (not row['status'].startswith('passed') or row['worker_pid']!=row['supervisor']['monitored_pid']
        or row['supervisor']['exit_code']!=0 or row['supervisor']['failure'] is not None
        or row['supervisor']['sampled_peak_working_set_bytes']>1024**3):raise ValueError('unqualified cost report')
    return row


def publish(root,address,run):
    from gffx.dashboard import connect
    from .reporter import Reporter
    rows={name:read(root,name) for name in HASHES}
    reporter=Reporter(address,run)
    def send(event,**values):
        if not reporter.send(event,**values):raise RuntimeError(reporter.error)
    send('start',kind='demo',provenance={'artifact_sha256':rows['optimized-costs']['producer']['wheel']['sha256']})
    with connect(address,run=run,wait_acks=True) as dash:
        dash.text('cpu-reuse/scope','One current graph per point and call. No stale cache or raised limits. '
                  'Exact difficult-point integration: 110.205s before, 37.878s after, identical recorded RGB. '
                  'Full-image gate remains open. These are recorded finite CPU measurements, not live rendering or coverage.',step=0)
        for step,(before,after) in enumerate(zip(rows['baseline-costs']['costs'],rows['optimized-costs']['costs'])):
            for label,row in (('before',before),('after',after)):
                dash.scalar('cpu-reuse/milliseconds/'+label,1000*row['median_seconds'],step=step)
                dash.scalar('cpu-reuse/saved-bytes/'+label,row['saved_unique_storage_bytes'],step=step)
        dash.record('cpu-reuse/details',{name:dict(sha256=HASHES[name],costs=row.get('costs'),result=row.get('result'),
                    errors=row.get('errors'),limits=row.get('limits'),supervisor=row['supervisor']) for name,row in rows.items()},step=0)
    send('demo',page='shading/cpu-curve-reuse-demo',title='CPU reuse: same light, less repeated work',artifacts=[
        dict(kind='text',path='cpu-reuse/scope',label='Scope and remaining gate'),
        dict(kind='plot',path='cpu-reuse/milliseconds',label='Milliseconds: 0 = 1 point/10 lights; 1 = 8-point forward; 2 = 4-point backward'),
        dict(kind='plot',path='cpu-reuse/saved-bytes',label='Saved autograd bytes; not peak scratch'),
        dict(kind='record',path='cpu-reuse/details',label='Measured costs, memory and reference errors')])
    send('finish',exit_code=0)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--address',required=True);p.add_argument('--run',required=True)
    a=p.parse_args();publish(a.root,a.address,a.run)


if __name__=='__main__':main()
