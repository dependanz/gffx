"""Bounded matched-box diagnostic of D335 piece1; no whole-piece admission."""
import hashlib,json,os,time,cProfile,pstats
from pathlib import Path
from fractions import Fraction as Q
import pytest

def test_saved_piece1_matched_segment_cost(monkeypatch,record_property):
    from tools import opaque_full_material_reference as module
    from tools import opaque_segmented_remainder as segment
    from tools.opaque_boundary_diagnostic import diagnose,DiagnosticStopped
    from tools.opaque_shading_intervals import I
    source=Path(os.environ['GFFX_FULL_INPUT']);base=source.parent.parent
    failure=base/'opaque84-20260921/original-campaign1/piece-01-result.json'
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
    class Captured(Exception):pass
    def capture(field,**kwargs):
        correction=kwargs['correction_integrator'];assert correction.segmented_conditioned
        report=dict(status='working',piece=1,input_sha256=sha(source),failure_sha256=sha(failure),full_piece_certificate=False,max_seconds=20,max_inputs=2048,runs=[],slabs=[],lines=[],input_calls=0)
        prior_path=os.environ.get('GFFX_COST_PRIOR');prior=json.loads(Path(prior_path).read_bytes()) if prior_path else None
        prior_seconds=prior['seconds'] if prior else 0;prior_calls=prior['input_calls'] if prior else 0
        report['prior']=dict(path=prior_path,seconds=prior_seconds,input_calls=prior_calls,sha256=sha(Path(prior_path)) if prior else None)
        started=time.perf_counter();current={};seen={};prof=cProfile.Profile()
        original_inputs=correction.inputs;original_axis=correction.axis_bounds;original_segment=segment.segmented_remainder
        def check():
            if time.perf_counter()-started+prior_seconds>=20:raise RuntimeError('aggregate diagnostic deadline20')
        def inputs(*args):
            check();assert report['input_calls']+prior_calls<2048,'aggregate input cap2048'
            report['input_calls']+=1
            key=tuple((v.order,tuple((i,j,str(q.lo),str(q.hi)) for (i,j),q in sorted(v.c.items()))) for v in args)
            order=max(v.order for v in args);row=current.setdefault('inputs',{}).setdefault(order,dict(calls=0,duplicates=0,seconds=0))
            row['calls']+=1;row['duplicates']+=key in seen;seen[key]=True;t=time.perf_counter()
            try:return original_inputs(*args)
            finally:row['seconds']+=time.perf_counter()-t
        def observed_segment(third,domain,roots,**kw):
            details=[];t=time.perf_counter()
            def observed_third(slab):
                at=time.perf_counter();value=third(slab)
                details.append(dict(domain=slab,third=value,seconds=time.perf_counter()-at,uncertainty_zone=any(r.lo<=slab.lo and r.hi>=slab.hi for r in roots)))
                return value
            value=original_segment(observed_third,domain,roots,**kw)
            report['slabs'].append(dict(mode=current['mode'],seconds=time.perf_counter()-t,slabs=details,evaluations=None if value is None else value['evaluations']))
            return value
        def axis(box,active):
            rows=original_axis(box,active);w=rows[1]['localized_jumps']
            report['lines'].append(dict(mode=current['mode'],axis=active,box=box,witnesses=[dict(required=x.get('required'),orientation=x.get('orientation'),role=x['role'],position=x['position']) for x in w],signed='ordinary_signed' in rows[1]))
            return rows
        correction.inputs=inputs;correction.axis_bounds=axis;monkeypatch.setattr(segment,'segmented_remainder',observed_segment)
        prof.enable()
        try:
            for enabled in ((True,) if prior else (False,True)):
                current={'mode':'segmented' if enabled else 'conservative'};seen={};correction.segmented_conditioned=enabled
                r=diagnose(correction,boxes,prepare=lambda b:kwargs['prepare'](0,0,b),max_inputs=2048-prior_calls-report['input_calls'],max_seconds=20-prior_seconds-(time.perf_counter()-started),checkpoint=check)
                report['runs'].append(dict(**current,diagnostic=r))
            check();assert all(len(r['diagnostic']['boxes'])==3 for r in report['runs']);report['status']='diagnostic_only'
        finally:
            prof.disable();correction.inputs=original_inputs;correction.axis_bounds=original_axis;correction.segmented_conditioned=True
            report['seconds']=time.perf_counter()-started
            report['profile']=[dict(file=f,line=l,function=n,calls=c,self_seconds=s,total_seconds=t) for (f,l,n),(p,c,s,t,callers) in pstats.Stats(prof).stats.items() if 'opaque_' in f]
            report['profile'].sort(key=lambda r:r['total_seconds'],reverse=True)
            report['profile_scope']='Instrumented inclusive costs overlap; not a production timing benchmark. Preparation charged and separately timed. Duplicate input keys include every jet coefficient.'
            dest.write_text(json.dumps(report,indent=2,default=encode),encoding='utf8')
            record_property('matched_cost',json.dumps(dict(status=report['status'],seconds=report['seconds'],input_calls=report['input_calls'],path=str(dest))))
        raise Captured
    monkeypatch.setattr(module,'integrate_square',capture)
    with pytest.raises(Captured):module.full_pixel(json.loads(source.read_bytes()),selected_piece=1)
