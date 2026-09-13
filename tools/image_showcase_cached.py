"""Verify and publish the three saved CHROMA images, without importing a renderer.

This finite transport imports only stdlib and dependency-free dashboard components.
The caller must have visually accepted the images before invoking publication.
"""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            value.update(block)
    return value.hexdigest()


def load_cached(report,wheel,expected,source_root,report_sha256):
    """Validate every identity and size before returning bounded dashboard arrays."""
    from gffx.dashboard import ArrayView
    report=Path(report).resolve(); root=Path(source_root).resolve()
    if digest(report)!=report_sha256:
        raise ValueError('report identity differs')
    row=json.loads(report.read_text(encoding='utf-8'))
    if row.get('status')!='passed' or row.get('mode')!='render':
        raise ValueError('report is not a passed full render')
    supervision=row.get('supervisor',{})
    if supervision.get('exit_code')!=0 or supervision.get('failure'):
        raise ValueError('render supervision did not pass')
    if len(expected)!=64 or row.get('wheel',{}).get('sha256')!=expected or digest(wheel)!=expected:
        raise ValueError('wheel identity differs')
    if not row.get('sources'):
        raise ValueError('source identity missing')
    for name,expected_source in row['sources'].items():
        source=(root/name).resolve()
        if not source.is_relative_to(root) or digest(source)!=expected_source:
            raise ValueError('source identity differs')
    entries=row.get('images',[])
    if [entry.get('label') for entry in entries]!=['hero','transparency','sampling']:
        raise ValueError('expected exactly three ordered images')
    images=[]; total=0
    for entry in entries:
        shape=entry.get('shape',[])
        if len(shape)!=3 or any(type(x) is not int or x<=0 for x in shape) or shape[-1]!=3:
            raise ValueError('invalid RGB shape')
        length=shape[0]*shape[1]*3
        total+=length
        if total>16*1024**2:
            raise ValueError('cached image payload exceeds finite transport bound')
        raw=Path(entry['raw_path'])
        expected_raw=entry['raw_sha256']
        safe=(report.parent/'images'/(expected_raw+'.rgb')).resolve()
        if raw.is_symlink() or raw.resolve()!=safe or not safe.is_relative_to(report.parent/'images'):
            raise ValueError('raw image path identity differs')
        if raw.stat().st_size!=length or digest(raw)!=expected_raw:
            raise ValueError('raw image identity differs')
        images.append(ArrayView('uint8',tuple(shape),raw.read_bytes()))
    if total!=row.get('logical_output_bytes'):
        raise ValueError('logical image size differs')
    return row,images


def send_images(dash,images,evidence=None):
    if len(images)!=3:
        raise ValueError('expected exactly three images')
    artifacts=[]
    for path,label,pixels in zip(('chroma/hero','chroma/transparency','chroma/sampling'),
                                ('CHROMA / public image operations','Transparent edges / wrong versus right',
                                 'Sampling / inspect the aliasing'),images):
        dash.image(path,pixels,step=0)
        artifacts.append(dict(kind='image',path=path,label=label))
    guide=('CHROMA is authored 2D graphic design using public gffx masks, opacity, exposure, affine sampling, '
           'Gaussian blur and compositing. Gradient rasters, bitmap lettering and flat graphic shadows are '
           'authored demonstration assets, not physical lighting, a font engine or new geometry. '
           'Three finite cached images are not test acceptance, a Compositron integration proof or a realtime claim. '
           'The straight-RGB interpolation counterexample is labelled WRONG; sampling panels magnify actual outputs. '
           'Linear-to-sRGB conversion and clipping happen only at the final presentation boundary.')
    if evidence is not None:
        guide+=' Cached evidence: '+json.dumps(evidence,sort_keys=True)
    dash.text('chroma/guide',guide,step=0)
    artifacts.append(dict(kind='text',path='chroma/guide',label='What this demonstration establishes'))
    return artifacts


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('report','wheel','source-root','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    for name in ('sha256','report-sha256','address','run'):
        parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise ValueError('preserve prior publication outcome')
    row,images=load_cached(args.report,args.wheel,args.sha256,args.source_root,args.report_sha256)
    from gffx.dashboard import connect
    from gffx.dashboard._server import check_provenance
    from tools.testing_dashboard.reporter import Reporter
    reporter=Reporter(args.address,args.run)
    outcome=dict(status='started',run=args.run,address=args.address,report_sha256=args.report_sha256,
                 wheel_sha256=args.sha256,logical_output_bytes=row['logical_output_bytes'],image_count=3)
    provenance=check_provenance({'artifact_sha256':args.report_sha256})
    def send(event,**kwargs):
        if not reporter.send(event,**kwargs):
            raise RuntimeError(reporter.error)
    # Exclusive creation prevents accidentally overwriting a previous transaction.
    with args.output.open('x',encoding='utf-8') as stream:
        json.dump(outcome,stream,indent=2)
    try:
        send('start',kind='demo',provenance=provenance)
        with connect(args.address,run=args.run,wait_acks=True,provenance=provenance) as dash:
            artifacts=send_images(dash,images,{'wheel_sha256':args.sha256,'render_report_sha256':args.report_sha256,
                                              'sources':row['sources'],'cached_images':True})
        send('demo',page='image/chroma-demo',title='CHROMA / linear light in layers',artifacts=artifacts)
        send('finish',exit_code=0)
        outcome['status']='passed'
    except Exception as error:
        outcome.update(status='failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        args.output.write_text(json.dumps(outcome,indent=2),encoding='utf-8')
    print(json.dumps(outcome))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
