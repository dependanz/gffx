"""Bounded split-response diagnostic on three saved original-material leaves."""
import os,json,time,hashlib
from pathlib import Path
from fractions import Fraction as Q
import pytest
from tools.opaque_shading_intervals import I

def test_saved_d340_split_response(monkeypatch,record_property):
    from tools import opaque_full_material_reference as m
    source=Path(os.environ['GFFX_FULL_INPUT']);dest=Path(os.environ['GFFX_FULL_REFERENCE'])
    failure=source.parent.parent/'opaque89-20260923/original-campaign1/piece-01-result.json'
    assert hashlib.sha256(failure.read_bytes()).hexdigest()=='85623f00a48a00664e836da88662750af9e1d25c2e511a48057ea4146833a67a'
    saved=json.loads(failure.read_bytes())['details']['cause']['last_complete_enclosure']['error_accounting']['largest_leaves'][:3]
    class Stop(Exception):pass
    def capture(field,**kwargs):
        prior=json.loads((dest.parent/'split-response1-reference.json').read_bytes());start=time.perf_counter();calls=prior['input_calls'];base_calls=0;rows=[];complete=False;corr=kwargs['correction_integrator'];original=corr.inputs;prior_check=corr.checkpoint
        def check():
            prior_check()
            assert time.perf_counter()-start+prior['seconds']<20,'diagnostic deadline'
            assert calls+base_calls<=2048,'diagnostic inputs'
        def inputs(*args):
            nonlocal calls
            check();assert calls+base_calls<2048;calls+=1;return original(*args)
        corr.inputs=inputs;corr.checkpoint=check
        def evaluate(coords):
            nonlocal base_calls
            from tools.opaque_square_reference import integrate_square
            a,b,c,d=coords;box=(I(I(a).lo,I(b).hi),I(I(c).lo,I(d).hi));check();kwargs['prepare'](0,0,box)
            t=time.perf_counter();value=corr(box)
            hint=corr.preferred_split_axis;hs=I(b-a);ht=I(d-c)
            def mapped(u,v):
                nonlocal base_calls
                check();assert calls+base_calls<2048;base_calls+=1
                return [x*hs*ht for x in field(u*hs+I(a),v*ht+I(c))]
            smooth=integrate_square(mapped,tolerance='1e-6',max_cells=1,checkpoint=check)['value']
            value=[x+y for x,y in zip(value,smooth)]
            return dict(box=list(map(str,coords)),value=value,widths=[str(v.width) for v in value],preferred_axis=hint,seconds=time.perf_counter()-t)
        try:
            for leaf in saved:
                a,b,c,d=map(Q,leaf['box']);row=dict(parent=evaluate((a,b,c,d)),splits=[]);rows.append(row)
                for axis in (0,1):
                    mid=(a+b)/2 if axis==0 else (c+d)/2
                    boxes=[(a,mid,c,d),(mid,b,c,d)] if axis==0 else [(a,b,c,mid),(a,b,mid,d)]
                    children=[evaluate(v) for v in boxes];total=[sum((v['value'][k] for v in children),I(0)) for k in range(2)]
                    assert all(max(v.lo,w.lo)<=min(v.hi,w.hi) for v,w in zip(total,row['parent']['value']))
                    row['splits'].append(dict(axis=axis,children=children,total=total,widths=[str(v.width) for v in total]))
            complete=True
        finally:
            corr.inputs=original;corr.checkpoint=prior_check
            report=dict(status='diagnostic_only' if complete else 'stopped_partial',seconds=time.perf_counter()-start,input_calls=calls,base_field_calls=base_calls,prior_seconds=prior['seconds'],aggregate_seconds=time.perf_counter()-start+prior['seconds'],aggregate_calls=calls+base_calls,max_seconds=20,max_inputs=2048,rows=rows,scope='Three saved parent regions and both midpoint subdivisions; no full-piece certificate, no production scheduling change.')
            dest.write_text(json.dumps(report,indent=2,default=lambda v:[str(v.lo),str(v.hi)] if isinstance(v,I) else str(v)))
            record_property('split_diagnostic',json.dumps(dict(seconds=report['seconds'],inputs=calls,path=str(dest))))
        check();raise Stop
    monkeypatch.setattr(m,'integrate_square',capture)
    with pytest.raises(Stop):m.full_pixel(json.loads(source.read_bytes()),selected_piece=1)
