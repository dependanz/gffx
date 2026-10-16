"""Bounded same-region endpoint and timing comparison; no new piece trial."""
import os,json,time,hashlib
from pathlib import Path
from fractions import Fraction as Q
import pytest

def test_saved_regions_signed_jump_prefix_feasibility(monkeypatch,record_property):
    from tools import opaque_full_material_reference as m
    from tools.opaque_residual_range import ResidualRangeIndex
    from tools.opaque_shading_intervals import I,Jet
    from bisect import bisect_left,bisect_right
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
            c.range_index=ResidualRangeIndex(c.extension,checkpoint=checkpoint)
            for number,row in enumerate(rows):
                coords=list(map(Q,row['box']));box=(I(I(coords[0]).lo,I(coords[1]).hi),I(I(coords[2]).lo,I(coords[3]).hi));item=dict(box=row['box'],profiles=[]);report['regions'].append(item)
                prepare(0,0,box);values=inputs(Jet(box[0]),Jet(box[1]))[:2];index=c.range_index;item['roles']=[]
                for role,value in enumerate(values):
                    q=value[0,0];index.jump_sums(q);first=bisect_left(index.upper[:-1],q.lo);last=bisect_right(index.lower[:-1],q.hi);terms=[c.mix([index.jump_values[k][ch][2] for ch in range(2)]) for k in range(first,last)]
                    total=I(0);absolute=I(0);largest=I(0);positive=negative=uncertain=0
                    for v in terms:
                        check();total=total+v;absolute=absolute+I(v.bound);largest=I(max(largest.hi,total.bound))
                        if v.lo>0:positive+=1
                        elif v.hi<0:negative+=1
                        else:uncertain+=1
                    item['roles'].append(dict(role=role,knots=[first,last],count=len(terms),positive=positive,negative=negative,uncertain=uncertain,signed_sum=[str(total.lo),str(total.hi)],absolute_sum=str(absolute.hi),largest_prefix=str(largest.hi),prefix_to_absolute=str((largest/absolute).hi) if absolute.lo>0 else None))
            report['status']='passed'
        finally:cell.cell_contents=original;c.inputs=saved;c.checkpoint=before
        raise Stop
    monkeypatch.setattr(m,'integrate_square',capture)
    try:
        with pytest.raises(Stop):m.full_pixel(json.loads(source.read_bytes()),selected_piece=8,material_only_inputs=True)
    finally:
        report['seconds']=time.perf_counter()-started;dest.write_text(json.dumps(report,default=str,indent=2),encoding='utf8');record_property('signed_prefix_feasibility',json.dumps(report,default=str))
