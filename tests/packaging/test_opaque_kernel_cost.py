"""Controlled cold-cache timing; local evidence, never whole-piece admission."""
import json,os,hashlib,time,collections
from pathlib import Path
from fractions import Fraction as Q
import pytest

def test_cold_cache_input_reuse_and_kernel_cost(monkeypatch,record_property):
    from tools import opaque_full_material_reference as module
    from tools import opaque_peano as peano
    from tools import opaque_peano_material as material
    from tools import opaque_transverse_peano as transverse
    from tools import opaque_segmented_remainder as segment
    from tools.opaque_smooth_branch import SmoothBranches
    from tools.opaque_boundary_diagnostic import diagnose
    from tools.opaque_shading_intervals import I
    source=Path(os.environ['GFFX_FULL_INPUT']);failure=source.parent.parent/'opaque84-20260921/original-campaign1/piece-01-result.json'
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    assert sha(source)=='a15bea40cb5cdead7994d828b57f5dcfac9f44fab0292a06e7d0a7b08719f73a'
    assert sha(failure)=='23912aacbbb436d8cc229125738227a69e0b8953eac0707a4ccf7007863a4449'
    saved=json.loads(failure.read_bytes());assert saved['piece']==1
    cover=saved['details']['cause']['last_complete_enclosure'];assert cover['complete_cover'] and cover['leaves']==79
    chosen=cover['error_accounting']['largest_leaves'][:3]
    boxes=[(I(I(Q(r['box'][0])).lo,I(Q(r['box'][1])).hi),I(I(Q(r['box'][2])).lo,I(Q(r['box'][3])).hi)) for r in chosen]
    dest=Path(os.environ['GFFX_FULL_REFERENCE']);assert not dest.exists()
    def encode(v):
        if isinstance(v,I):return [str(v.lo),str(v.hi)]
        raise TypeError(type(v))
    def numeric(v):
        if isinstance(v,dict):return {k:numeric(x) for k,x in v.items() if 'seconds' not in k}
        if isinstance(v,list):return [numeric(x) for x in v]
        return v
    class Captured(Exception):pass
    def capture(field,**kwargs):
        correction=kwargs['correction_integrator'];assert correction.segmented_conditioned
        report=dict(status='working',full_piece_certificate=False,input_sha256=sha(source),failure_sha256=sha(failure),max_seconds=20,max_inputs=2048,runs=[],input_calls=0,order=['uncached','reuse','reuse','uncached'],timing_scope='Same clock instrumentation, no cProfile; caches cleared before each run; all preparation charged. Two observations per path, no general throughput claim.')
        original=correction.inputs;factory=material.InputReuse;kernel=transverse.kernel_integral;weighted=transverse.weighted_kernel_parts;local=peano._kernel_local_ranges_uncached
        started=time.perf_counter();current={};context=[];seen={}
        def check():
            if time.perf_counter()-started>=20:raise RuntimeError('aggregate diagnostic deadline20')
        def timed(name,fn,*args):
            check();t=time.perf_counter()
            try:return fn(*args)
            finally:
                row=current.setdefault('costs',{}).setdefault(name,dict(calls=0,seconds=0));row['calls']+=1;row['seconds']+=time.perf_counter()-t
        def inputs(*args):
            check();assert report['input_calls']<2048;report['input_calls']+=1
            return timed('input_order_'+str(max(v.order for v in args)),original,*args)
        def observed_kernel(a,b,order):return timed('kernel_integral_order_'+str(order),kernel,a,b,order)
        def observed_local(position,order=None):
            role='weighted_'+str(context[-1]) if context else 'other'
            current.setdefault('local_miss_context',{}).setdefault(role,0);current['local_miss_context'][role]+=1
            return timed('kernel_local_miss',local,position) if order is None else timed('kernel_local_miss',local,position,order)
        def observed_weighted(coefficient,a,b,order):
            key=(str(a.lo),str(a.hi),str(b.lo),str(b.hi),order);row=current.setdefault('weighted_geometry',dict(calls=0,duplicates=0));row['calls']+=1;row['duplicates']+=key in seen;seen[key]=True
            context.append(order)
            try:return timed('weighted_kernel',weighted,coefficient,a,b,order)
            finally:context.pop()
        correction.inputs=inputs
        monkeypatch.setattr(transverse,'kernel_integral',observed_kernel);monkeypatch.setattr(transverse,'weighted_kernel_parts',observed_weighted);monkeypatch.setattr(segment,'weighted_kernel_parts',observed_weighted);monkeypatch.setattr(peano,'_kernel_local_ranges_uncached',observed_local)
        try:
            for mode in report['order']:
                check();peano._kernel_local_ranges_cached.cache_clear();peano.kernel_bounds.cache_clear();peano.gauss5.cache_clear();correction.extension.coefficient_cache.clear();correction.smooth=SmoothBranches(correction.extension)
                current=dict(mode=mode,caches_before=dict(local=peano._kernel_local_ranges_cached.cache_info()._asdict(),global_bounds=peano.kernel_bounds.cache_info()._asdict(),nodes=peano.gauss5.cache_info()._asdict()));seen={}
                assert all(x['currsize']==0 for x in current['caches_before'].values())
                monkeypatch.setattr(material,'InputReuse',factory if mode=='reuse' else lambda inputs,checkpoint:inputs)
                d=diagnose(correction,boxes,prepare=lambda b:kwargs['prepare'](0,0,b),max_inputs=2048-report['input_calls'],max_seconds=20-(time.perf_counter()-started),checkpoint=check)
                current.update(diagnostic=d,caches_after=dict(local=peano._kernel_local_ranges_cached.cache_info()._asdict(),global_bounds=peano.kernel_bounds.cache_info()._asdict(),nodes=peano.gauss5.cache_info()._asdict()));report['runs'].append(current)
            reference=json.loads(json.dumps(numeric(report['runs'][0]['diagnostic']['boxes']),default=encode))
            assert all(json.loads(json.dumps(numeric(r['diagnostic']['boxes']),default=encode))==reference for r in report['runs'])
            report['numerical_fields_exact']=True;report['status']='diagnostic_only';check()
        finally:
            correction.inputs=original;monkeypatch.setattr(material,'InputReuse',factory)
            report['seconds']=time.perf_counter()-started;dest.write_text(json.dumps(report,indent=2,default=encode),encoding='utf8')
            record_property('cold_cache_timing',json.dumps(dict(status=report['status'],seconds=report['seconds'],input_calls=report['input_calls'],path=str(dest))))
        raise Captured
    monkeypatch.setattr(module,'integrate_square',capture)
    with pytest.raises(Captured):module.full_pixel(json.loads(source.read_bytes()),selected_piece=1)
