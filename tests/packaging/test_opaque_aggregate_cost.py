"""Three largest saved complete-cover regions; all proof preparation charged."""
import os,json,hashlib,time
from pathlib import Path
from fractions import Fraction as Q
import pytest

def test_saved_complete_regions_aggregate_bounds_and_cost(monkeypatch,record_property):
    from tools import opaque_full_material_reference as m
    from tools.opaque_aggregate_peano import AggregateMaterial
    from tools.opaque_shading_intervals import I
    source=Path(os.environ['GFFX_FULL_INPUT']);failure=source.parent.parent/'opaque102-20260923/ninth-preflight4-reference.json'
    assert hashlib.sha256(failure.read_bytes()).hexdigest()=='673a7edc5e2e8e9cc09ff7624e3651a8f8424959d17e3868391a7daa1063bde8'
    assert hashlib.sha256(source.read_bytes()).hexdigest()=='a15bea40cb5cdead7994d828b57f5dcfac9f44fab0292a06e7d0a7b08719f73a'
    cover=json.loads(failure.read_bytes())['details']['cause']['last_complete_enclosure'];rows=cover['error_accounting']['largest_leaves'][:3]
    boxes=[(I(I(Q(v['box'][0])).lo,I(Q(v['box'][1])).hi),I(I(Q(v['box'][2])).lo,I(Q(v['box'][3])).hi)) for v in rows]
    dest=Path(os.environ['GFFX_FULL_REFERENCE']);assert not dest.exists();started=time.perf_counter();report=dict(status='running',max_inputs=2048,max_seconds=20,inputs=0,regions=[])
    class Stop(Exception):pass
    def check():assert time.perf_counter()-started<20;assert report['inputs']<=2048
    def capture(field,**kw):
        c=kw['correction_integrator'];prepare=kw['prepare'];cell=prepare.__closure__[prepare.__code__.co_freevars.index('inputs')];original=cell.cell_contents;before=c.checkpoint
        def inputs(*args):check();assert report['inputs']<2048;report['inputs']+=1;return original(*args)
        def checkpoint():check();before()
        cell.cell_contents=inputs;c.inputs=inputs;c.checkpoint=checkpoint;c._certified_integral=lambda box:(_ for _ in ()).throw(ValueError('bounded angular segment count'))
        try:
            proof=AggregateMaterial(c)
            for row,box in zip(rows,boxes):
                check();prepare(0,0,box);item=dict(box=row['box']);report['regions'].append(item);old=c(box)[1];item['old']=[str(old.lo),str(old.hi)]
                try:
                    new=proof(box)[1];item['new']=[str(new.lo),str(new.hi)];item['width_ratio']=float(new.width/old.width);assert max(new.lo,old.lo)<=min(new.hi,old.hi)
                except ValueError as e:item['domain']=str(e)
            report['status']='passed'
        finally:cell.cell_contents=original;c.inputs=original;c.checkpoint=before
        raise Stop
    monkeypatch.setattr(m,'integrate_square',capture)
    try:
        with pytest.raises(Stop):m.full_pixel(json.loads(source.read_bytes()),selected_piece=8)
    finally:
        report['seconds']=time.perf_counter()-started;dest.write_text(json.dumps(report,indent=2),encoding='utf8');record_property('aggregate_cost',json.dumps(report))
