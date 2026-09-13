"""Finite saved timing comparison, including candidates that are not faster."""
import argparse
from pathlib import Path
from tools.albedo_sampling import replay
from tools.environment_full_images import WHEEL_HASH

def publish(result,address,run):
    from gffx.dashboard import connect
    from .reporter import Reporter
    reporter=Reporter(address,run)
    def send(event,**values):
        if not reporter.send(event,**values):raise RuntimeError(reporter.error)
    send('start',kind='demo',provenance=dict(artifact_sha256=WHEEL_HASH));artifacts=[]
    with connect(address,run=run,wait_acks=True) as dash:
        lines=['LOCAL TIMING SCREEN, NOT A GENERAL SPEED GUARANTEE. Both paths passed accuracy checks on the same 72 saved pixel/profile observations. One warmup is excluded; each displayed median has three measurements. Execution order alternates. Times include validation in both methods. A call still has at most 1,024 point/light pairs, whether it serves one point or several.']
        for w in result['workers']:
            name=w['model']+'-'+w['environment'];m=w['medians']
            lines.append(f"{name}: single {m['single']:.4f} s; batched {m['batched']:.4f} s. "+('Batched median is lower.' if w['prefer_batched'] else 'Keep the single-point path: batching was not faster.'))
            for p in w['passes']:
                if p['repetition']==0:continue
                dash.scalar('sampling/'+name+'/'+p['method'],p['duration_seconds'],step=p['repetition'])
            artifacts.append(dict(kind='plot',path='sampling/'+name,label=name+' — three measured durations in seconds'))
        dash.text('sampling/summary','\n\n'.join(lines),step=0)
        artifacts.insert(0,dict(kind='text',path='sampling/summary',label='What improved, what did not, and what the timings mean'))
        details=[dict(model=w['model'],environment=w['environment'],medians=w['medians'],prefer_batched=w['prefer_batched'],
            supervisor=w['supervisor'],passes=[dict(method=p['method'],repetition=p['repetition'],seconds=p['duration_seconds'],
            calls=p['result']['calls'],directions=p['result']['directions'],max_pairs=p['result']['max_pairs'],max_points=p['result']['max_points']) for p in w['passes']]) for w in result['workers']]
        dash.record('sampling/evidence',dict(sha256=result['sha256'],profile=result['record']['profile'],workers=details),step=0)
        artifacts.append(dict(kind='record',path='sampling/evidence',label='Raw timing repetitions, work counts and worker limits'))
    send('demo',page='shading/albedo-sampling-demo',title='Measured bounded sampling comparison',artifacts=artifacts)
    send('finish',exit_code=0)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','parent','corpus'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('sha256','address','run'):p.add_argument('--'+name,required=True)
    a=p.parse_args();publish(replay(a.root,a.sha256,a.parent,a.corpus),a.address,a.run)

if __name__=='__main__':main()
