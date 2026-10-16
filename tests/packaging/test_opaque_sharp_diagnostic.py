"""Bounded same-region endpoint and timing comparison; no new piece trial."""
import os,json,time,hashlib
from pathlib import Path
from fractions import Fraction as Q
import pytest

def test_saved_regions_compare_grouped_and_individual_segment_bounds(monkeypatch,record_property):
    from tools import opaque_full_material_reference as m
    from tools.opaque_residual_range import ResidualRangeIndex
    from tools.opaque_sharp_range import SharpRangeIndex
    from tools.opaque_shading_intervals import I
    source=Path(os.environ['GFFX_FULL_INPUT']);failure=source.parent.parent/'opaque103-20260923/ninth-preflight1-reference.json'
    assert hashlib.sha256(failure.read_bytes()).hexdigest()=='57eb00d29b532467fdc8952cb241ed70e782aaf748bb466ef67507f2ac998aae'
    leaves=json.loads(failure.read_bytes())['details']['cause']['last_complete_enclosure']['error_accounting']['largest_leaves'];rows=[leaves[i] for i in [0,1,3]]
    dest=Path(os.environ['GFFX_FULL_REFERENCE']);assert not dest.exists();started=time.perf_counter();report=dict(status='running',max_seconds=20,max_inputs=2048,inputs=0,input_seconds=0.,regions=[])
    class Stop(Exception):pass
    def check():assert time.perf_counter()-started<20;assert report['inputs']<=2048
    def capture(field,**kw):
        c=kw['correction_integrator'];prepare=kw['prepare'];cell=prepare.__closure__[prepare.__code__.co_freevars.index('inputs')];original=cell.cell_contents;before=c.checkpoint;saved=c.inputs
        def inputs(*a,**k):
            check();assert report['inputs']<2048;report['inputs']+=1;t=time.perf_counter()
            try:return original(*a,**k)
            finally:report['input_seconds']+=time.perf_counter()-t
        def checkpoint():check();before()
        cell.cell_contents=inputs;c.checkpoint=checkpoint
        try:
            # Shared immutable range index is setup charged to the diagnostic, not timed twice.
            indices=dict(grouped=ResidualRangeIndex(c.extension,checkpoint=checkpoint),individual=SharpRangeIndex(c.extension,checkpoint=checkpoint))
            for number,row in enumerate(rows):
                coords=list(map(Q,row['box']));box=(I(I(coords[0]).lo,I(coords[1]).hi),I(I(coords[2]).lo,I(coords[3]).hi));item=dict(box=row['box'],profiles=[]);report['regions'].append(item)
                for mode in (['grouped','individual'] if number%2==0 else ['individual','grouped']):
                    c.range_index=indices[mode];c.inputs=lambda *a:inputs(*a,material_only=True);prepare(0,0,box);n=report['inputs'];ti=report['input_seconds'];t=time.perf_counter();v=c(box)
                    item['profiles'].append(dict(mode=mode,bounds=[[str(q.lo),str(q.hi)] for q in v],seconds=time.perf_counter()-t,inputs=report['inputs']-n,input_seconds=report['input_seconds']-ti))
                group={p['mode']:p for p in item['profiles']}
                for a,b in zip(group['grouped']['bounds'],group['individual']['bounds']):
                    assert I(a[0]).lo<=I(b[0]).lo<=I(b[1]).hi<=I(a[1]).hi
            report['status']='passed'
        finally:cell.cell_contents=original;c.inputs=saved;c.checkpoint=before
        raise Stop
    monkeypatch.setattr(m,'integrate_square',capture)
    try:
        with pytest.raises(Stop):m.full_pixel(json.loads(source.read_bytes()),selected_piece=8,material_only_inputs=True)
    finally:
        report['seconds']=time.perf_counter()-started;dest.write_text(json.dumps(report,default=str,indent=2),encoding='utf8');record_property('sharp_range_cost',json.dumps(report,default=str))
