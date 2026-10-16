"""Recheck the same three D340 regions within the existing diagnostic budget."""
import os,json,time
from pathlib import Path
from fractions import Fraction as Q
import pytest
from tools.opaque_shading_intervals import I

def test_signed_partition_same_saved_regions(monkeypatch,record_property):
    from tools import opaque_full_material_reference as m
    source=Path(os.environ['GFFX_FULL_INPUT']);dest=Path(os.environ['GFFX_FULL_REFERENCE']);prior=json.loads((dest.parent/'split-response2-reference.json').read_bytes())
    class Stop(Exception):pass
    def capture(field,**kwargs):
        start=time.perf_counter();calls=prior['aggregate_calls'];rows=[];complete=False;corr=kwargs['correction_integrator'];original=corr.inputs;oldcheck=corr.checkpoint
        def check():
            oldcheck();assert time.perf_counter()-start+prior['aggregate_seconds']<20;assert calls<=2048
        def inputs(*args):
            nonlocal calls
            check();assert calls<2048;calls+=1;return original(*args)
        corr.inputs=inputs;corr.checkpoint=check
        try:
            for saved in prior['rows']:
                a,b,c,d=map(Q,saved['parent']['box']);box=(I(I(a).lo,I(b).hi),I(I(c).lo,I(d).hi));check();kwargs['prepare'](0,0,box)
                value=corr(box)
                rows.append(dict(box=saved['parent']['box'],previous_total_width=saved['parent']['widths'][1],correction_width=str(value[1].width),correction=value,parts=corr.error_breakdown))
            complete=True
        finally:
            corr.inputs=original;corr.checkpoint=oldcheck
            result=dict(status='diagnostic_only' if complete else 'stopped_partial',seconds=time.perf_counter()-start,aggregate_seconds=time.perf_counter()-start+prior['aggregate_seconds'],input_calls=calls-prior['aggregate_calls'],aggregate_calls=calls,max_seconds=20,max_inputs=2048,rows=rows,scope='Same three parent regions; new correction width versus previous complete width, whose base is negligible. No complete-piece timing prediction or certificate.')
            dest.write_text(json.dumps(result,indent=2,default=lambda v:[str(v.lo),str(v.hi)] if isinstance(v,I) else str(v)))
            record_property('signed_partition_diagnostic',json.dumps(dict(path=str(dest),seconds=result['aggregate_seconds'],calls=calls)))
        check();raise Stop
    monkeypatch.setattr(m,'integrate_square',capture)
    with pytest.raises(Stop):m.full_pixel(json.loads(source.read_bytes()),selected_piece=1)
