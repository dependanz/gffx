"""Finite verified studio image transport, never a renderer or a server launcher.

Inspect the saved images before separately invoking publication. A32pixel image
is a diagnostic smoke result, not a high-resolution beauty demonstration.
"""
import argparse
import hashlib
import json
from pathlib import Path

LABELS=('orbit-left','orbit-centre','orbit-right','normals','depth','coverage')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def publication_provenance(report_sha256):
    from gffx.dashboard._server import check_provenance
    return check_provenance({'artifact_sha256':report_sha256})


def load_cached(report,wheel,expected,source_root,report_sha256):
    from gffx.dashboard import ArrayView
    report=Path(report).resolve();root=Path(source_root).resolve()
    if report.stat().st_size>1024*1024 or digest(report)!=report_sha256:
        raise ValueError('report identity differs')
    row=json.loads(report.read_text(encoding='utf-8'))
    if row.get('status')!='passed' or row.get('mode')!='render':
        raise ValueError('report is not a passed render')
    supervisor=row.get('supervisor',{})
    if supervisor.get('exit_code')!=0 or supervisor.get('failure'):
        raise ValueError('render supervision failed')
    if len(expected)!=64 or row.get('wheel',{}).get('sha256')!=expected or digest(wheel)!=expected:
        raise ValueError('wheel identity differs')
    if not row.get('sources'):raise ValueError('source identities missing')
    for name,value in row['sources'].items():
        path=(root/name).resolve()
        if not path.is_relative_to(root) or digest(path)!=value:
            raise ValueError('source identity differs')
    entries=row.get('images',[]);size=row.get('size')
    if type(size)!=int or size not in (32,128):raise ValueError('unreviewed image dimensions')
    if [entry.get('label') for entry in entries]!=list(LABELS):
        raise ValueError('expected six ordered studio images')
    images=[];total=0;image_root=(report.parent/'images').resolve()
    for entry in entries:
        shape=entry.get('shape')
        if shape!=[size,size,3]:raise ValueError('image shape differs')
        sha=entry.get('raw_sha256','')
        if len(sha)!=64 or any(c not in '0123456789abcdef' for c in sha):
            raise ValueError('image identity malformed')
        raw=Path(entry['raw_path']);safe=(image_root/(sha+'.rgb')).resolve()
        if raw.is_symlink() or not safe.is_relative_to(image_root) or raw.resolve()!=safe:
            raise ValueError('image identity path differs')
        length=size*size*3
        if raw.stat().st_size!=length or digest(raw)!=sha:raise ValueError('image identity differs')
        images.append(ArrayView('uint8',tuple(shape),raw.read_bytes()));total+=length
    if total!=row.get('logical_output_bytes'):raise ValueError('logical image size differs')
    return row,images


def send_images(dash,images,evidence):
    if len(images)!=6:raise ValueError('expected six studio images')
    artifacts=[]
    for label,pixels in zip(LABELS,images):
        path='studio/'+label;dash.image(path,pixels,step=0)
        artifacts.append(dict(kind='image',path=path,label=label.replace('-',' ').title()))
    guide=('Original metallic orb and matte ribbon, rendered through public gffx deferred '
           'materials and two directional samples. Three camera views plus centre-view '
           'normal, depth and coverage diagnostics. No shadows, indirect lighting, path tracing '
           'or Cycles equivalence. RGB display clips linear values and converts to sRGB; '
           'normal/depth/coverage panels are labelled diagnostic mappings, not beauty colours. '
           '32-pixel images are smoke diagnostics, not visual quality acceptance. '
           'Cached evidence: '+json.dumps(evidence,sort_keys=True))
    dash.text('studio/guide',guide,step=0)
    artifacts.append(dict(kind='text',path='studio/guide',label='What these pixels establish'))
    return artifacts


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('report','wheel','source-root','output'):parser.add_argument('--'+key,type=Path,required=True)
    for key in ('sha256','report-sha256','address','run'):parser.add_argument('--'+key,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('preserve prior publication outcome')
    row,images=load_cached(args.report,args.wheel,args.sha256,args.source_root,args.report_sha256)
    provenance=publication_provenance(args.report_sha256)
    from gffx.dashboard import connect
    from tools.testing_dashboard.reporter import Reporter
    reporter=Reporter(args.address,args.run)
    outcome=dict(status='started',report_sha256=args.report_sha256,wheel_sha256=args.sha256,
                 image_count=6,logical_output_bytes=row['logical_output_bytes'])
    with args.output.open('x',encoding='utf-8') as stream:json.dump(outcome,stream,indent=2)
    def send(event,**kwargs):
        if not reporter.send(event,**kwargs):raise RuntimeError(reporter.error)
    try:
        send('start',kind='demo',provenance=provenance)
        with connect(args.address,run=args.run,wait_acks=True,provenance=provenance) as dash:
            artifacts=send_images(dash,images,dict(report_sha256=args.report_sha256,
                wheel_sha256=args.sha256,sources=row['sources'],size=row['size'],cached_images=True))
        send('demo',page='render/deferred-demo',title='Studio / public deferred renderer',artifacts=artifacts)
        send('finish',exit_code=0);outcome['status']='passed'
    except Exception as error:
        outcome.update(status='failed',error=f'{type(error).__name__}: {error}');raise
    finally:args.output.write_text(json.dumps(outcome,indent=2),encoding='utf-8')
    print(json.dumps(outcome));return 0


if __name__=='__main__':raise SystemExit(main())
