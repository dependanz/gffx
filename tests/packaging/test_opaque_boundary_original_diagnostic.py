"""Three frozen failed-piece boxes, both orientations; no full-piece quadrature."""
import hashlib,json,os
from fractions import Fraction as Q
from pathlib import Path
import pytest


def test_saved_original_boundary_boxes(monkeypatch,record_property):
    from tools import opaque_full_material_reference as module
    from tools.opaque_boundary_diagnostic import diagnose,DiagnosticStopped
    from tools.opaque_shading_intervals import I
    source=Path(os.environ['GFFX_FULL_INPUT']);base=source.parent.parent
    failure=base/'opaque72-20260920/original-campaign1/piece-01-result.json'
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    assert digest(source)=='a15bea40cb5cdead7994d828b57f5dcfac9f44fab0292a06e7d0a7b08719f73a'
    assert digest(failure)=='616ae232534dfbea5c651f17e37f23a876d671af9ae30149ddd71dabdd76df1c'
    trace=json.loads(failure.read_bytes())['details']['cause']['trace']
    chosen=[r for r in trace if r['cell'] in (30,31,32)]
    assert len(chosen)==3
    boxes=[(I(I(Q(r['box'][0])).lo,I(Q(r['box'][1])).hi),I(I(Q(r['box'][2])).lo,I(Q(r['box'][3])).hi)) for r in chosen]
    dest=Path(os.environ['GFFX_FULL_REFERENCE']);assert not dest.exists()
    def encode(v):
        if isinstance(v,I):return [str(v.lo),str(v.hi)]
        raise TypeError(type(v))
    class Captured(Exception):pass
    def capture(field,**kwargs):
        report=None
        try:report=diagnose(kwargs['correction_integrator'],boxes,prepare=lambda b:kwargs['prepare'](0,0,b),max_inputs=2048,max_seconds=20)
        except DiagnosticStopped as e:report=e.report;raise
        finally:
            if report is not None:
                report.update(input_sha256=digest(source),failure_sha256=digest(failure),piece=1,trace_cells=[30,31,32],scope='Fixed three overlapping trace boxes only,not a complete-piece decomposition or certificate')
                dest.write_text(json.dumps(report,indent=2,default=encode),encoding='utf8')
                record_property('boundary_diagnostic',json.dumps(dict(status=report['status'],input_calls=report['input_calls'],witnesses=report['witnesses'],seconds=report.get('seconds'),path=str(dest))))
        assert report['status']=='diagnostic_only' and len(report['boxes'])==3
        raise Captured
    monkeypatch.setattr(module,'integrate_square',capture)
    with pytest.raises(Captured):module.full_pixel(json.loads(source.read_bytes()),selected_piece=1)


def test_saved_original_smooth_components(monkeypatch,record_property):
    from tools import opaque_full_material_reference as module
    from tools.opaque_boundary_diagnostic import diagnose,DiagnosticStopped
    from tools.opaque_shading_intervals import I
    source=Path(os.environ['GFFX_FULL_INPUT']);base=source.parent.parent
    failure=base/'opaque74-20260920/original-campaign1/piece-01-result.json'
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    assert digest(source)=='a15bea40cb5cdead7994d828b57f5dcfac9f44fab0292a06e7d0a7b08719f73a'
    assert digest(failure)=='9c20b203bb2aeffe2a8a40976428af01aea3094418d3cf3b9f4d8e631ee4ccfd'
    trace=json.loads(failure.read_bytes())['details']['cause']['trace']
    chosen=[r for r in trace if r['cell'] in (30,31,32)]
    assert len(chosen)==3 and all(r['okay'] for r in chosen)
    boxes=[(I(I(Q(r['box'][0])).lo,I(Q(r['box'][1])).hi),I(I(Q(r['box'][2])).lo,I(Q(r['box'][3])).hi)) for r in chosen]
    dest=Path(os.environ['GFFX_FULL_REFERENCE']);assert not dest.exists()
    def encode(v):
        if isinstance(v,I):return [str(v.lo),str(v.hi)]
        raise TypeError(type(v))
    class Captured(Exception):pass
    def capture(field,**kwargs):
        report=None
        try:report=diagnose(kwargs['correction_integrator'],boxes,prepare=lambda b:kwargs['prepare'](0,0,b),max_inputs=2048,max_seconds=20,partition_smooth=False)
        except DiagnosticStopped as e:report=e.report;raise
        finally:
            if report is not None:
                report.update(input_sha256=digest(source),failure_sha256=digest(failure),piece=1,trace_cells=[30,31,32],scope='Fixed three overlapping trace boxes only,not a complete-piece decomposition or certificate')
                dest.write_text(json.dumps(report,indent=2,default=encode),encoding='utf8')
                record_property('boundary_diagnostic',json.dumps(dict(status=report['status'],input_calls=report['input_calls'],witnesses=report['witnesses'],seconds=report.get('seconds'),path=str(dest))))
        assert kwargs['correction_integrator'].smooth_partition is True
        assert report['status']=='diagnostic_only' and len(report['boxes'])==3
        assert all('ordinary_comparison' in a for row in report['boxes'] for a in row['orientations'])
        raise Captured
    monkeypatch.setattr(module,'integrate_square',capture)
    with pytest.raises(Captured):module.full_pixel(json.loads(source.read_bytes()),selected_piece=1)


def test_saved_largest_leaf_terms_and_directions(monkeypatch,record_property):
    from tools import opaque_full_material_reference as module
    from tools.opaque_boundary_diagnostic import diagnose,DiagnosticStopped
    from tools.opaque_shading_intervals import I
    source=Path(os.environ['GFFX_FULL_INPUT']);base=source.parent.parent
    failure=base/'opaque76-20260920/original-campaign1/piece-01-result.json'
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    assert digest(source)=='a15bea40cb5cdead7994d828b57f5dcfac9f44fab0292a06e7d0a7b08719f73a'
    assert digest(failure)=='ad336617224047510cb4ce2be1bf944fe0cbbdd07423843a59d29bdfcb223853'
    cover=json.loads(failure.read_bytes())['details']['cause']['last_complete_enclosure']
    assert cover['complete_cover'] and cover['leaves']==101
    chosen=cover['error_accounting']['largest_leaves'][:3]
    assert len(chosen)==3 and [r['depth'] for r in chosen]==[6,7,6]
    boxes=[(I(I(Q(r['box'][0])).lo,I(Q(r['box'][1])).hi),I(I(Q(r['box'][2])).lo,I(Q(r['box'][3])).hi)) for r in chosen]
    dest=Path(os.environ['GFFX_FULL_REFERENCE']);assert not dest.exists()
    def encode(v):
        if isinstance(v,I):return [str(v.lo),str(v.hi)]
        raise TypeError(type(v))
    class Captured(Exception):pass
    def capture(field,**kwargs):
        report=None
        try:report=diagnose(kwargs['correction_integrator'],boxes,prepare=lambda b:kwargs['prepare'](0,0,b),max_inputs=2048,max_seconds=20,partition_smooth=False)
        except DiagnosticStopped as e:report=e.report;raise
        finally:
            if report is not None:
                report.update(input_sha256=digest(source),failure_sha256=digest(failure),piece=1,largest_leaf_ranks=[1,2,3],scope='Three disjoint largest saved leaves only; both orientations; not a complete-piece certificate')
                dest.write_text(json.dumps(report,indent=2,default=encode),encoding='utf8')
                record_property('boundary_diagnostic',json.dumps(dict(status=report['status'],input_calls=report['input_calls'],witnesses=report['witnesses'],seconds=report.get('seconds'),path=str(dest))))
        assert kwargs['correction_integrator'].smooth_partition is True
        assert report['status']=='diagnostic_only' and len(report['boxes'])==3
        assert all('ordinary_comparison' in a for row in report['boxes'] for a in row['orientations'])
        raise Captured
    monkeypatch.setattr(module,'integrate_square',capture)
    with pytest.raises(Captured):module.full_pixel(json.loads(source.read_bytes()),selected_piece=1)


def test_saved_boundary_uncertainty_and_derivative_cost(monkeypatch,record_property):
    from tools import opaque_full_material_reference as module
    from tools.opaque_boundary_diagnostic import diagnose,DiagnosticStopped
    from tools.opaque_shading_intervals import I
    source=Path(os.environ['GFFX_FULL_INPUT']);base=source.parent.parent
    failure=base/'opaque78-20260920/original-campaign1/piece-01-result.json'
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    assert digest(source)=='a15bea40cb5cdead7994d828b57f5dcfac9f44fab0292a06e7d0a7b08719f73a'
    assert digest(failure)=='949fd3b74457e7ee514389cc4f311d37b15dc685894384e786025afa9f05e6d6'
    cover=json.loads(failure.read_bytes())['details']['cause']['last_complete_enclosure']
    assert cover['complete_cover'] and cover['leaves']==102
    chosen=cover['error_accounting']['largest_leaves'][:3]
    assert len(chosen)==3
    boxes=[(I(I(Q(r['box'][0])).lo,I(Q(r['box'][1])).hi),I(I(Q(r['box'][2])).lo,I(Q(r['box'][3])).hi)) for r in chosen]
    dest=Path(os.environ['GFFX_FULL_REFERENCE']);assert not dest.exists()
    def encode(v):
        if isinstance(v,I):return [str(v.lo),str(v.hi)]
        raise TypeError(type(v))
    class Captured(Exception):pass
    def capture(field,**kwargs):
        report=None
        import cProfile,pstats,time
        correction=kwargs['correction_integrator'];original=correction.inputs;input_cost={}
        def timed(*args):
            order=max(v.order for v in args);started=time.perf_counter()
            try:return original(*args)
            finally:
                row=input_cost.setdefault(order,dict(calls=0,seconds=0));row['calls']+=1;row['seconds']+=time.perf_counter()-started
        correction.inputs=timed;prof=cProfile.Profile();prof.enable()
        try:report=diagnose(kwargs['correction_integrator'],boxes,prepare=lambda b:kwargs['prepare'](0,0,b),max_inputs=2048,max_seconds=20,partition_smooth=False)
        except DiagnosticStopped as e:report=e.report;raise
        finally:
            prof.disable();correction.inputs=original
            if report is not None:
                stats=pstats.Stats(prof);costs=[]
                for (file,line,function),(primitive,calls,self_seconds,total_seconds,callers) in stats.stats.items():
                    if 'opaque_' in file:
                        costs.append(dict(file=file,line=line,function=function,calls=calls,self_seconds=self_seconds,total_seconds=total_seconds))
                report['profile']=dict(input_cost=input_cost,functions=sorted(costs,key=lambda r:r['total_seconds'],reverse=True),scope='cProfile instrumented diagnostic;inclusive function times overlap;not a production runtime benchmark')
                report.update(input_sha256=digest(source),failure_sha256=digest(failure),piece=1,largest_leaf_ranks=[1,2,3],scope='Three disjoint largest saved leaves only; both orientations; not a complete-piece certificate')
                dest.write_text(json.dumps(report,indent=2,default=encode),encoding='utf8')
                record_property('boundary_diagnostic',json.dumps(dict(status=report['status'],input_calls=report['input_calls'],witnesses=report['witnesses'],seconds=report.get('seconds'),path=str(dest))))
        assert kwargs['correction_integrator'].smooth_partition is True
        assert report['status']=='diagnostic_only' and len(report['boxes'])==3
        assert all('ordinary_comparison' in a for row in report['boxes'] for a in row['orientations'])
        raise Captured
    monkeypatch.setattr(module,'integrate_square',capture)
    with pytest.raises(Captured):module.full_pixel(json.loads(source.read_bytes()),selected_piece=1)
