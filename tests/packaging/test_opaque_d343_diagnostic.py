"""Pinned third-piece diagnostics; no complete-piece quadrature or admission."""
import hashlib,json,os
from fractions import Fraction as Q
from pathlib import Path
import pytest


def test_saved_d343_third_piece_terms_directions_and_cost(monkeypatch,record_property):
    from tools import opaque_full_material_reference as module
    from tools.opaque_boundary_diagnostic import diagnose,DiagnosticStopped
    from tools.opaque_shading_intervals import I
    source=Path(os.environ['GFFX_FULL_INPUT']);base=source.parent.parent
    failure=base/'opaque90-20260923/original-campaign1/piece-02-result.json'
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    assert digest(source)=='a15bea40cb5cdead7994d828b57f5dcfac9f44fab0292a06e7d0a7b08719f73a'
    assert digest(failure)=='cbaa2049f920b46e32b1db598fc3173f9447af3820e91619a2e93be761ecc19d'
    failed=json.loads(failure.read_bytes());assert failed['piece']==2 and failed['details']['face']==5740
    cover=failed['details']['cause']['last_complete_enclosure']
    assert cover['complete_cover'] and cover['leaves']==79
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
        prior_path=os.environ.get('GFFX_ARITHMETIC_PRIOR');prior=json.loads(Path(prior_path).read_bytes()) if prior_path else None
        aggregate_path=os.environ.get('GFFX_DIAGNOSTIC_AGGREGATE');aggregate=json.loads(Path(aggregate_path).read_bytes()) if aggregate_path else None
        charged_seconds=aggregate['aggregate_seconds'] if aggregate else (prior['seconds'] if prior else 0)
        charged_inputs=aggregate['aggregate_calls'] if aggregate else (prior['input_calls'] if prior else 0)
        import cProfile,pstats,time
        correction=kwargs['correction_integrator'];original=correction.inputs;input_cost={}
        def timed(*args):
            order=max(v.order for v in args);started=time.perf_counter()
            try:return original(*args)
            finally:
                row=input_cost.setdefault(order,dict(calls=0,seconds=0));row['calls']+=1;row['seconds']+=time.perf_counter()-started
        correction.inputs=timed;prof=cProfile.Profile();prof.enable()
        try:report=diagnose(kwargs['correction_integrator'],boxes,prepare=lambda b:kwargs['prepare'](0,0,b),max_inputs=2048-charged_inputs,max_seconds=20-charged_seconds,partition_smooth=False)
        except DiagnosticStopped as e:report=e.report;raise
        finally:
            prof.disable();correction.inputs=original
            if report is not None:
                stats=pstats.Stats(prof);costs=[]
                for (file,line,function),(primitive,calls,self_seconds,total_seconds,callers) in stats.stats.items():
                    if 'opaque_' in file:
                        costs.append(dict(file=file,line=line,function=function,calls=calls,self_seconds=self_seconds,total_seconds=total_seconds))
                report['profile']=dict(input_cost=input_cost,functions=sorted(costs,key=lambda r:r['total_seconds'],reverse=True),scope='cProfile instrumented diagnostic;inclusive function times overlap;not a production runtime benchmark')
                if prior:
                    def numeric(v):
                        if isinstance(v,I):return [str(v.lo),str(v.hi)]
                        if isinstance(v,dict):return {k:numeric(x) for k,x in v.items() if 'seconds' not in k}
                        if isinstance(v,list):return [numeric(x) for x in v]
                        if isinstance(v,tuple):return [numeric(x) for x in v]
                        return v
                    report['numerical_fields_exact']=numeric(report['boxes'])==numeric(prior['boxes'])
                    report['prior_charge']=dict(seconds=charged_seconds,inputs=charged_inputs,sha256=digest(Path(prior_path)),aggregate_path=aggregate_path)
                    report['aggregate_seconds']=charged_seconds+report['seconds'];report['aggregate_calls']=charged_inputs+report['input_calls']
                report['saved_cover']=dict(cells=cover['cells'],leaves=cover['leaves'],widths=cover['widths'],accounting=cover['error_accounting'])
                report.update(input_sha256=digest(source),failure_sha256=digest(failure),piece=2,largest_leaf_ranks=[1,2,3],scope='Three disjoint largest saved leaves only; both orientations; not a complete-piece certificate')
                dest.write_text(json.dumps(report,indent=2,default=encode),encoding='utf8')
                record_property('boundary_diagnostic',json.dumps(dict(status=report['status'],input_calls=report['input_calls'],witnesses=report['witnesses'],seconds=report.get('seconds'),path=str(dest))))
        if prior:assert report['numerical_fields_exact']
        assert kwargs['correction_integrator'].smooth_partition is True
        assert report['status']=='diagnostic_only' and len(report['boxes'])==3
        assert all('ordinary_comparison' in a for row in report['boxes'] for a in row['orientations'])
        raise Captured
    monkeypatch.setattr(module,'integrate_square',capture)
    with pytest.raises(Captured):module.full_pixel(json.loads(source.read_bytes()),selected_piece=2)
