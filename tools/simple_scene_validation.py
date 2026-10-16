"""Validate the installed simple scene API and run its shipped example. Never publishes."""
from __future__ import annotations
import argparse,hashlib,json,os,runpy,sys,time,zipfile
from datetime import datetime,timezone
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheel',required=True,type=Path)
    parser.add_argument('--device',required=True,choices=['cpu','cuda'])
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1]
    import gffx,torch,pytest
    assert gffx.__version__=='0.2.1'
    assert Path(gffx.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    package_root=Path(gffx.__file__).resolve().parent.parent
    with zipfile.ZipFile(args.wheel) as archive:
        members=[n for n in archive.namelist() if n.startswith(('gffx/','gffx.libs/')) and not n.endswith('/')]
        for member in members:assert (package_root/member).read_bytes()==archive.read(member),member
    torch.set_num_threads(1)
    if args.device=='cuda':
        assert torch.cuda.is_available() and gffx.cuda.capabilities()['status']=='loaded'
        torch.cuda.reset_peak_memory_stats()
    sys.path[:0]=[str(root/'tests/python'),str(root/'tests')]
    calls=[]
    class Record:
        def pytest_runtest_logreport(self,report):
            if report.when=='call' or report.skipped or (report.when=='setup' and report.failed):
                calls.append(dict(nodeid=report.nodeid,outcome=report.outcome))
    command=['-q','--import-mode=importlib','--basetemp='+str(out/'temp'),
             '--junitxml='+str(out/'tests.xml'),str(root/'tests/python/test_simple_scene_api.py')]
    if args.device=='cpu':command+=['-k','not cuda']
    code=int(pytest.main(command,plugins=[Record()]))
    counts={name:sum(row['outcome']==name for row in calls) for name in ['passed','failed','skipped']}
    if code or counts['failed'] or counts['skipped'] or not counts['passed']:
        (out/'failed-tests.json').write_text(json.dumps(dict(code=code,counts=counts,calls=calls),indent=2)+'\n',encoding='utf-8')
        raise RuntimeError('Selected scene tests must all pass without skips')
    # Explicitly test the portable example's CPU selection even with a CUDA framework.
    previous=torch.cuda.is_available
    if args.device=='cpu':torch.cuda.is_available=lambda:False
    cwd=Path.cwd();started=time.monotonic()
    try:
        os.chdir(out)
        result=runpy.run_module('gffx.examples.cornell_box',run_name='__main__')
    finally:
        os.chdir(cwd);torch.cuda.is_available=previous
    if args.device=='cuda':torch.cuda.synchronize()
    image=result['image']['colour']
    assert image.device.type==args.device and image.shape==(1,512,512,3)
    assert bool(torch.isfinite(image).all())
    png=out/'cornell_box.png'
    report=dict(version=gffx.__version__,device=args.device,python=sys.version.split()[0],torch=torch.__version__,
                wheel_sha256=hashlib.sha256(args.wheel.read_bytes()).hexdigest(),payload_members=len(members),
                exit_code=0,**counts,selected_nodeids_sha256=hashlib.sha256('\n'.join(sorted(x['nodeid'] for x in calls)).encode()).hexdigest(),
                example_seconds=round(time.monotonic()-started,3),example_shape=list(image.shape),
                png_sha256=hashlib.sha256(png.read_bytes()).hexdigest(),png_bytes=png.stat().st_size,
                completed_at=datetime.now(timezone.utc).isoformat())
    if args.device=='cuda':
        report.update(torch_allocated_peak=torch.cuda.max_memory_allocated(),torch_reserved_peak=torch.cuda.max_memory_reserved())
        assert report['torch_reserved_peak']<=2*2**30
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
