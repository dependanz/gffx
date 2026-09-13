"""Finite, measured convergence lesson; no services or producers start on import."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from tools import shading_environment_reference as reference
from tools import shading_independent_evaluator as evaluator


def study():
    """Three small scalar experiments, including an intentionally unresolved one."""
    coefficients=[[math.sqrt(4*math.pi)]*3]+[[0.,0.,0.] for _ in range(8)]
    n=(0.,0.,1.)
    def aliased(n,v,l):
        return [(1+.5*math.cos(64*math.atan2(l[1],l[0])))/math.pi]*3
    controls=(
        ('diffuse',[.2,.4,.6],lambda n,v,l:evaluator.classic(n,v,l,albedo=[.2,.4,.6],specular=0.,exponent=2.)),
        ('classic_overreflection',[25/24]*3,lambda n,v,l:evaluator.classic(n,v,l,albedo=0.,specular=1.,exponent=2.)),
        ('aliased_sampling',[1.]*3,aliased))
    cases=[]
    for identifier,expected,brdf in controls:
        try: report=reference.environment_reference(n,n,coefficients,brdf=brdf,max_levels=3)
        except reference.ReferenceConvergenceError as error:report=error.report
        cases.append(dict(id=identifier,expected=expected,report=report))
    return dict(profile=reference.PROFILE,cases=cases,
                source_hashes={Path(p).name:hashlib.sha256(Path(p).read_bytes()).hexdigest()
                               for p in (__file__,reference.__file__,evaluator.__file__)},
                limits='Scalar diagnostic, not a rendered scene or public shader acceptance. '
                       'Classic overreflection is intentional; alias failure is retained. '
                       'Two sampling grids can still miss the same feature; agreement is not proof.')


def publish(address,run,data):
    from gffx.dashboard import connect
    from .reporter import Reporter
    reporter=Reporter(address,run)
    def send(event,**values):
        if not reporter.send(event,**values):raise RuntimeError(reporter.error)
    send('start',kind='demo',provenance=dict(source_hashes=data['source_hashes']))
    with connect(address,run=run,wait_acks=True) as dash:
        dash.text('environment-reference/read-me',
            'A measuring-stick demonstration: the plots use actual computed values, not an animation. '
            'Each step doubles the number of rings and directions per ring, and repeats with the '
            'directions shifted half a cell. Diffuse should return its albedo; the empirical classic '
            'highlight returns 25/24, deliberately more than one. The oscillating negative control '
            'fools early grids but the shifted third grid detects it, withholding a reference. '
            'The red channel is plotted; the record keeps all channels and limits. '+data['limits'],step=0)
        for case in data['cases']:
            for level,row in enumerate(case['report']['history']):
                prefix='environment-reference/'+case['id']
                dash.scalar(prefix+'/value/estimate',row['radiance'][0],step=level)
                dash.scalar(prefix+'/value/known_answer',case['expected'][0],step=level)
                dash.scalar(prefix+'/difference/offset_pair',row['offset_change'][0],step=level)
                dash.scalar(prefix+'/difference/threshold',row['threshold'][0],step=level)
                if row['change'] is not None:
                    dash.scalar(prefix+'/difference/refinement',row['change'][0],step=level)
        dash.record('environment-reference/measurements',data,step=0)
    artifacts=[dict(kind='text',path='environment-reference/read-me',label='How to read these measurements')]
    for case in data['cases']:
        for panel in ('value','difference'):
            artifacts.append(dict(kind='plot',path='environment-reference/'+case['id']+'/'+panel,
                                  label=case['id'].replace('_',' ')+' — '+panel))
    artifacts.append(dict(kind='record',path='environment-reference/measurements',label='Exact histories and provenance'))
    send('demo',page='shading/environment-reference-demo',title='When does a lighting answer settle down?',artifacts=artifacts)
    send('finish',exit_code=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path)
    parser.add_argument('--address'); parser.add_argument('--run')
    args=parser.parse_args()
    if not args.report and not args.address:parser.error('select --report or --address')
    if bool(args.address)!=bool(args.run):parser.error('--address and --run are required together')
    data=study()
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        with args.report.open('x',encoding='utf-8') as handle:
            json.dump(data,handle,indent=2,allow_nan=False)
    if args.address:publish(args.address,args.run,data)
    print(json.dumps({c['id']:c['report']['status'] for c in data['cases']}))


if __name__=='__main__':main()
