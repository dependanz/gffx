"""Publish only verified saved PNGs; no renderer, solver, raw image log or quota change."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
from tools.testing_dashboard.presentation import AssetWriter,MAX_FILE


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path,expected):
    path=Path(path)
    if path.stat().st_size>65536 or digest(path)!=expected:raise ValueError('gallery manifest identity')
    row=json.loads(path.read_text(encoding='utf-8'))
    if row.get('version')!=1 or not isinstance(row.get('title'),str) or not 0<len(row['title'])<=200 or not 1<=len(row.get('images',[]))<=16:raise ValueError('gallery shape')
    if not isinstance(row.get('scope'),str) or not 0<len(row['scope'])<=4000:raise ValueError('gallery scope explanation required')
    report=Path(row['provenance']['producer_report'])
    if digest(report)!=row['provenance']['producer_report_sha256']:raise ValueError('producer report identity')
    images=[]
    for entry in row['images']:
        file=Path(entry['path']);a=entry['asset']
        if file.is_symlink() or file.stat().st_size!=a['bytes'] or not 0<a['bytes']<=MAX_FILE:raise ValueError('image size')
        blob=file.read_bytes()
        if hashlib.sha256(blob).hexdigest()!=a['sha256'] or a['url']!='/testing/media/'+a['sha256']+'.png':raise ValueError('image identity')
        if blob[:8]!=b'\x89PNG\r\n\x1a\n' or len(blob)<24:raise ValueError('image format')
        w,h=struct.unpack('!II',blob[16:24])
        if not 0<w<=1920 or not 0<h<=1080:raise ValueError('image dimensions')
        if not isinstance(entry.get('label'),str) or not 0<len(entry['label'])<=200 or not isinstance(entry.get('caption'),str) or not 0<len(entry['caption'])<=4000:raise ValueError('image explanation')
        images.append(blob)
    return row,images


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','root','output'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('sha256','address','run'):p.add_argument('--'+name,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('preserve publication report')
    row,images=load(a.manifest,a.sha256)
    from tools.testing_dashboard.catalog import definition
    if definition(row['page'])['kind']!='demo':raise ValueError('known demonstration page required')
    writer=AssetWriter(a.root/'presentation');gallery={'version':1,'title':row['title'],'images':[]}
    for item,blob in zip(row['images'],images):
        asset=writer.put(blob);assert asset==item['asset']
        gallery['images'].append(dict(label=item['label'],caption=item['caption'],asset=asset))
    from tools.testing_dashboard.reporter import Reporter
    from gffx.dashboard import connect
    reporter=Reporter(a.address,a.run);outcome=dict(status='working',manifest_sha256=a.sha256,page=row['page'],run=a.run,images=gallery['images'])
    def send(event,**values):
        if not reporter.send(event,**values):raise RuntimeError(reporter.error)
    try:
        # Only this small manifest/explanation enters the existing saved log.
        # PNG bytes stay in the existing separate preserve-and-stop asset store.
        send('start',kind='demo',provenance={'artifact_sha256':a.sha256})
        with connect(a.address,run=a.run,wait_acks=True,provenance={'artifact_sha256':a.sha256}) as dash:
            dash.record('presentation/gallery',gallery,step=0)
            dash.text('presentation/explanation',row['scope']+' Producer wheel '+row['provenance']['wheel_sha256']+'. Producer report SHA256 '+row['provenance']['producer_report_sha256']+'.',step=0)
        send('demo',page=row['page'],title=row['title'],artifacts=[dict(kind='record',path='presentation/gallery',label='Saved output gallery'),dict(kind='text',path='presentation/explanation',label='Method and evidence')])
        send('finish',exit_code=0);outcome['status']='passed'
    except Exception as error:
        outcome.update(status='failed',error=str(error))
        try:send('finish',exit_code=1)
        except Exception:pass
        raise
    finally:
        with a.output.open('x',encoding='utf-8') as f:json.dump(outcome,f,indent=2)
    return 0

if __name__=='__main__':raise SystemExit(main())
