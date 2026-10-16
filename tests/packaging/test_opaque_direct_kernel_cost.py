"""Bounded old/new kernel comparison on three saved fourth-piece regions."""
import hashlib,json,os,time,importlib.util,importlib.machinery
from pathlib import Path
from fractions import Fraction as Q
import pytest

def test_direct_kernel_same_regions_bounds_and_timing(monkeypatch,record_property):
    from tools import opaque_full_material_reference as m,opaque_peano as p
    from tools.opaque_shading_intervals import I
    source=Path(os.environ['GFFX_FULL_INPUT']);base=source.parent.parent;o=base/'opaque96-20260923'
    failure=base/'opaque95-20260923/original-campaign1/piece-03-result.json'
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    assert sha(failure)=='0a9248ea39b791f5474fafe7d4d243fcd7351961779bc44559c5701d2cb12176'
    assert sha(source)=='a15bea40cb5cdead7994d828b57f5dcfac9f44fab0292a06e7d0a7b08719f73a'
    rows=json.loads(failure.read_bytes())['details']['cause']['last_complete_enclosure']['error_accounting']['largest_leaves'][:3]
    boxes=[(I(I(Q(x['box'][0])).lo,I(Q(x['box'][1])).hi),I(I(Q(x['box'][2])).lo,I(Q(x['box'][3])).hi)) for x in rows]
    oldpath=o/'opaque_peano.py.before';loader=importlib.machinery.SourceFileLoader('opaque96_prior',str(oldpath));spec=importlib.util.spec_from_loader(loader.name,loader);old=importlib.util.module_from_spec(spec);loader.exec_module(old)
    new=p._kernel_local_ranges_uncached;report=dict(status='running',max_inputs=2048,max_seconds=20,inputs=0,runs=[],input_sha256=sha(source),failure_sha256=sha(failure),old_kernel_sha256=sha(oldpath),scope='Same saved boxes, full selected correction enclosures. No complete-piece admission. All preparation inputs charged.')
    dest=Path(os.environ['GFFX_FULL_REFERENCE']);assert not dest.exists();started=time.perf_counter()
    class Stop(Exception):pass
    def check():assert time.perf_counter()-started<20,'diagnostic deadline20';assert report['inputs']<=2048
    def capture(field,**kw):
        c=kw['correction_integrator'];original=c.inputs;before=c.checkpoint
        def inputs(*args):
            check();assert report['inputs']<2048;report['inputs']+=1;return original(*args)
        def checkpoint():check();before()
        prepare=kw['prepare'];cell=prepare.__closure__[prepare.__code__.co_freevars.index('inputs')]
        assert cell.cell_contents is original
        cell.cell_contents=inputs;c.inputs=inputs;c.checkpoint=checkpoint
        try:
            for mode in ['old','new','new','old']:
                monkeypatch.setattr(p,'_kernel_local_ranges_uncached',old._kernel_local_ranges_uncached if mode=='old' else new)
                for owner in (p,old):owner._kernel_local_ranges_cached.cache_clear();owner.kernel_bounds.cache_clear();owner.gauss5.cache_clear()
                t=time.perf_counter();enclosures=[]
                for box in boxes:
                    check();prepare(0,0,box);v=c(box)[1];enclosures.append(dict(bounds=[str(v.lo),str(v.hi)],width=str(v.width),axis=c.last_integrated_axis))
                report['runs'].append(dict(mode=mode,seconds=time.perf_counter()-t,enclosures=enclosures))
            for i in range(3):
                a=I(*report['runs'][0]['enclosures'][i]['bounds']);b=I(*report['runs'][1]['enclosures'][i]['bounds'])
                assert max(a.lo,b.lo)<=min(a.hi,b.hi)
                assert b.width<=(I(a.width)+I('1e-45')).hi
            assert report['runs'][0]['enclosures']==report['runs'][3]['enclosures']
            assert report['runs'][1]['enclosures']==report['runs'][2]['enclosures']
            report['status']='passed'
        finally:cell.cell_contents=original;c.inputs=original;c.checkpoint=before;monkeypatch.setattr(p,'_kernel_local_ranges_uncached',new)
        raise Stop
    monkeypatch.setattr(m,'integrate_square',capture)
    try:
        with pytest.raises(Stop):m.full_pixel(json.loads(source.read_bytes()),selected_piece=3)
    finally:
        report['seconds']=time.perf_counter()-started;dest.write_text(json.dumps(report,indent=2));record_property('direct_kernel_comparison',json.dumps(report))

def test_saved_direct_kernel_comparison_uses_outward_margin():
    from tools.opaque_shading_intervals import I
    p=Path(os.environ['GFFX_FULL_INPUT']).parent.parent/'opaque96-20260923/kernel-cost1-reference.json'
    assert hashlib.sha256(p.read_bytes()).hexdigest()=='93876df260287f3e89a215ac593eaf1e8609a6a25367b59ba74f5eeb06c97c6f'
    r=json.loads(p.read_bytes());assert r['inputs']==1560 and r['seconds']<20
    assert [x['mode'] for x in r['runs']]==['old','new','new','old']
    assert r['runs'][0]['enclosures']==r['runs'][3]['enclosures']
    assert r['runs'][1]['enclosures']==r['runs'][2]['enclosures']
    for a,b in zip(r['runs'][0]['enclosures'],r['runs'][1]['enclosures']):
        x=I(*a['bounds']);y=I(*b['bounds'])
        assert max(x.lo,y.lo)<=min(x.hi,y.hi)
        assert y.width<=(I(x.width)+I('1e-45')).hi


def test_mass_same_regions_with_remaining_diagnostic_allowance(monkeypatch,record_property):
    from tools import opaque_full_material_reference as m,opaque_peano as p
    from tools.opaque_shading_intervals import I
    source=Path(os.environ['GFFX_FULL_INPUT']);base=source.parent.parent;o=base/'opaque96-20260923'
    prior=json.loads((o/'kernel-cost1-reference.json').read_bytes());assert prior['inputs']==1560 and len(prior['runs'])==4
    failure=json.loads((base/'opaque95-20260923/original-campaign1/piece-03-result.json').read_bytes())
    rows=failure['details']['cause']['last_complete_enclosure']['error_accounting']['largest_leaves'][:3]
    boxes=[(I(I(Q(x['box'][0])).lo,I(Q(x['box'][1])).hi),I(I(Q(x['box'][2])).lo,I(Q(x['box'][3])).hi)) for x in rows]
    report=dict(status='running',inputs=0,prior_inputs=prior['inputs'],prior_seconds=prior['seconds'],rows=[]);started=time.perf_counter();dest=Path(os.environ['GFFX_FULL_REFERENCE']);assert not dest.exists()
    class Stop(Exception):pass
    def check():assert prior['seconds']+time.perf_counter()-started<20;assert prior['inputs']+report['inputs']<=2048
    def capture(field,**kw):
        c=kw['correction_integrator'];original=c.inputs;before=c.checkpoint;prepare=kw['prepare'];cell=prepare.__closure__[prepare.__code__.co_freevars.index('inputs')]
        def inputs(*args):
            check();assert prior['inputs']+report['inputs']<2048;report['inputs']+=1;return original(*args)
        def checkpoint():check();before()
        cell.cell_contents=inputs;c.inputs=inputs;c.checkpoint=checkpoint
        p._kernel_local_mass_cached.cache_clear();p._kernel_local_ranges_cached.cache_clear()
        try:
            for i,box in enumerate(boxes):
                check();t=time.perf_counter();prepare(0,0,box);v=c(box)[1];old=I(*prior['runs'][1]['enclosures'][i]['bounds'])
                report['rows'].append(dict(box=rows[i]['box'],bounds=[str(v.lo),str(v.hi)],seconds=time.perf_counter()-t,width_ratio=float(v.width/old.width),axis=c.last_integrated_axis))
                assert max(old.lo,v.lo)<=min(old.hi,v.hi)
            report['status']='passed'
        finally:cell.cell_contents=original;c.inputs=original;c.checkpoint=before
        raise Stop
    monkeypatch.setattr(m,'integrate_square',capture)
    try:
        with pytest.raises(Stop):m.full_pixel(json.loads(source.read_bytes()),selected_piece=3)
    finally:
        report.update(seconds=time.perf_counter()-started,aggregate_inputs=prior['inputs']+report['inputs'],aggregate_seconds=prior['seconds']+time.perf_counter()-started);dest.write_text(json.dumps(report,indent=2));record_property('kernel_mass_comparison',json.dumps(report))
