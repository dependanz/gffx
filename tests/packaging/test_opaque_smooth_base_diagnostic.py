"""Compare smooth-base representations on the same three frozen D338 boxes."""
import os,json,hashlib,time
from pathlib import Path
from fractions import Fraction as Q
import pytest
from tools.opaque_shading_intervals import I

def test_saved_d338_smooth_base_bounds_and_cost(monkeypatch,record_property):
    from tools import opaque_full_material_reference as m
    from tools.opaque_square_reference import integrate_square
    source=Path(os.environ['GFFX_FULL_INPUT']);dest=Path(os.environ['GFFX_FULL_REFERENCE']);o=dest.parent
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();assert sha(source)=='a15bea40cb5cdead7994d828b57f5dcfac9f44fab0292a06e7d0a7b08719f73a'
    prior=[json.loads((o/(name+'-reference.json')).read_bytes()) for name in ['diagnostic1','diagnostic2']]
    prior_seconds=sum(p['seconds'] for p in prior);prior_calls=sum(p['input_calls'] for p in prior);assert prior_calls==856
    failed_prior=json.loads(Path(os.environ['GFFX_BASE_PRIOR']).read_bytes()) if os.environ.get('GFFX_BASE_PRIOR') else None
    if failed_prior:prior_seconds+=failed_prior['seconds'];prior_calls+=failed_prior['base_field_calls']
    failure=source.parent.parent/'opaque88-20260923/original-campaign1/piece-02-result.json';assert sha(failure)=='35aacde85f58bd8a9df763ac28f1a87ef9c58257bfef1227ca6ab72bbeaf99e7'
    saved=json.loads(failure.read_bytes())['details']['cause']['last_complete_enclosure']['error_accounting']['largest_leaves'][:3]
    class Stop(Exception):pass
    def capture(field,**kwargs):
        started=time.perf_counter();calls=0;rows=[];complete=False;corr=kwargs['correction_integrator'];new=corr.smooth.at
        def check():
            assert time.perf_counter()-started+prior_seconds<20,'aggregate diagnostic deadline'
            assert calls+prior_calls<=2048,'aggregate diagnostic input allowance'
        def encode(v):
            if isinstance(v,I):return [str(v.lo),str(v.hi)]
            raise TypeError(type(v))
        try:
            for leaf in saved:
                check();a,b,c,d=map(Q,leaf['box']);box=(I(I(a).lo,I(b).hi),I(I(c).lo,I(d).hi));hs=I(b-a);ht=I(d-c);kwargs['prepare'](0,0,box);row=dict(box=leaf['box'],variants=[]);rows.append(row)
                for mode in ['bernstein','centered']:
                    corr.smooth.at=(lambda q,index:corr.extension.at(q,index)) if mode=='bernstein' else new
                    def mapped(u,v):
                        nonlocal calls
                        check();calls+=1;return [x*hs*ht for x in field(u*hs+I(a),v*ht+I(c))]
                    t=time.perf_counter();got=integrate_square(mapped,tolerance='1e-6',max_cells=1,checkpoint=check)
                    row['variants'].append(dict(mode=mode,seconds=time.perf_counter()-t,value=got['value'],cells=got['cells'],widths=[str(x.width) for x in got['value']]))
                old,fresh=[v['value'] for v in row['variants']]
                assert all(max(x.lo,y.lo)<=min(x.hi,y.hi) for x,y in zip(old,fresh))
            complete=True
        finally:
            corr.smooth.at=new;elapsed=time.perf_counter()-started
            result=dict(status='local_base_diagnostic_only' if complete else 'stopped_partial_diagnostic',prior_seconds=prior_seconds,prior_correction_inputs=prior_calls,seconds=elapsed,base_field_calls=calls,rows=rows,scope='Three saved boxes; base enclosure only, all knot corrections still required. One observation per representation, no whole-piece prediction.')
            dest.write_text(json.dumps(result,indent=2,default=encode));record_property('smooth_base_diagnostic',json.dumps(dict(seconds=elapsed,base_field_calls=calls,path=str(dest))))
        check();assert len(rows)==3;raise Stop
    monkeypatch.setattr(m,'integrate_square',capture)
    with pytest.raises(Stop):m.full_pixel(json.loads(source.read_bytes()),selected_piece=2)
