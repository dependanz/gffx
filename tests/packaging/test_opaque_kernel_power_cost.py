"""Three saved regions; bounded root-contraction comparison only."""
import json,hashlib,os,time
from pathlib import Path
from fractions import Fraction as Q
import pytest
def test_saved_kernel_constants_enclosures_and_cost(monkeypatch,record_property):
    from tools import opaque_full_material_reference as m,opaque_peano as p
    from tools.opaque_shading_intervals import I
    source=Path(os.environ['GFFX_FULL_INPUT']);b=source.parent.parent;failure=b/'opaque98-20260923/original-campaign1/piece-03-result.json'
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    assert sha(failure)=='4773c7334ecaefd6f424ff54e1a265b877b01f92d61704c44e6dd433c8f828f6'
    assert sha(source)=='a15bea40cb5cdead7994d828b57f5dcfac9f44fab0292a06e7d0a7b08719f73a'
    cover=json.loads(failure.read_bytes())['details']['cause']['last_complete_enclosure'];assert cover['cells']==163 and cover['leaves']==82
    rows=cover['error_accounting']['largest_leaves'][:3];boxes=[(I(I(Q(x['box'][0])).lo,I(Q(x['box'][1])).hi),I(I(Q(x['box'][2])).lo,I(Q(x['box'][3])).hi)) for x in rows]
    import importlib.util
    legacy_path=b/'opaque98-20260923/source-final/tools/opaque_peano.py';spec=importlib.util.spec_from_file_location('legacy_kernel',legacy_path);legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy);current=p._kernel_segment_bernstein
    report=dict(legacy_sha256=sha(legacy_path),status='running',max_inputs=2048,max_seconds=20,inputs=0,runs=[],failure_sha256=sha(failure),input_sha256=sha(source));started=time.perf_counter();dest=Path(os.environ['GFFX_FULL_REFERENCE']);assert not dest.exists()
    class Stop(Exception):pass
    def check():assert time.perf_counter()-started<20;assert report['inputs']<=2048
    def capture(field,**kw):
        c=kw['correction_integrator'];original=c.inputs;before=c.checkpoint;prepare=kw['prepare'];cell=prepare.__closure__[prepare.__code__.co_freevars.index('inputs')]
        def inputs(*args):
            check();assert report['inputs']<2048;report['inputs']+=1;return original(*args)
        def checkpoint():check();before()
        cell.cell_contents=inputs;c.inputs=inputs;c.checkpoint=checkpoint
        try:
            for refine in [False,True,True,False]:
                p._kernel_segment_bernstein=current if refine else legacy._kernel_segment_bernstein;p._kernel_power_cached.cache_clear();p._kernel_local_mass_cached.cache_clear();p._kernel_local_ranges_cached.cache_clear();t=time.perf_counter();enclosures=[]
                for box in boxes:
                    check();prepare(0,0,box);v=c(box)[1];enclosures.append(dict(bounds=[str(v.lo),str(v.hi)],width=str(v.width),axis=c.last_integrated_axis))
                report['runs'].append(dict(refined=refine,seconds=time.perf_counter()-t,enclosures=enclosures))
            assert report['runs'][0]['enclosures']==report['runs'][3]['enclosures']
            assert report['runs'][1]['enclosures']==report['runs'][2]['enclosures']
            report['width_ratios']=[]
            for a,b in zip(report['runs'][0]['enclosures'],report['runs'][1]['enclosures']):
                a,b=I(*a['bounds']),I(*b['bounds']);assert max(a.lo,b.lo)<=min(a.hi,b.hi);report['width_ratios'].append(float(b.width/a.width))
            report['status']='passed'
        finally:cell.cell_contents=original;c.inputs=original;c.checkpoint=before;p._kernel_segment_bernstein=current
        raise Stop
    monkeypatch.setattr(m,'integrate_square',capture)
    try:
        with pytest.raises(Stop):m.full_pixel(json.loads(source.read_bytes()),selected_piece=3)
    finally:
        report['seconds']=time.perf_counter()-started;dest.write_text(json.dumps(report,indent=2));record_property('kernel_constants_comparison',json.dumps(report))
