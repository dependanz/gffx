"""Matched saved-region comparison, all preparation and attempted work charged."""
import os,json,time,hashlib
from pathlib import Path
from fractions import Fraction as Q
import pytest

def test_saved_regions_direct_complete_aggregate_cost(monkeypatch,record_property):
    from tools import opaque_full_material_reference as m
    from tools.opaque_shading_intervals import I
    source=Path(os.environ['GFFX_FULL_INPUT']);failure=source.parent.parent/'opaque103-20260923/ninth-preflight1-reference.json'
    assert hashlib.sha256(failure.read_bytes()).hexdigest()=='57eb00d29b532467fdc8952cb241ed70e782aaf748bb466ef67507f2ac998aae'
    leaves=json.loads(failure.read_bytes())['details']['cause']['last_complete_enclosure']['error_accounting']['largest_leaves'];rows=[leaves[i] for i in (0,1,3)]
    dest=Path(os.environ['GFFX_FULL_REFERENCE']);assert not dest.exists();start=time.perf_counter();report=dict(status='running',inputs=0,max_inputs=2048,max_seconds=20,regions=[])
    class Stop(Exception):pass
    def check():assert time.perf_counter()-start<20;assert report['inputs']<=2048
    def capture(field,**kw):
        c=kw['correction_integrator'];prepare=kw['prepare'];cell=prepare.__closure__[prepare.__code__.co_freevars.index('inputs')];original=cell.cell_contents;before=c.checkpoint
        def inputs(*args):check();assert report['inputs']<2048;report['inputs']+=1;return original(*args)
        def checkpoint():check();before()
        cell.cell_contents=inputs;c.inputs=inputs;c.checkpoint=checkpoint
        try:
            for i,row in enumerate(rows):
                q=list(map(Q,row['box']));box=(I(I(q[0]).lo,I(q[1]).hi),I(I(q[2]).lo,I(q[3]).hi));item=dict(box=row['box']);report['regions'].append(item)
                for label in (('old','direct') if i%2==0 else ('direct','old')):
                    prepare(0,0,box);t=time.perf_counter();n=report['inputs'];v=(c(box) if label=='old' else c.envelope_integral(box))[1];item[label]=dict(bounds=[str(v.lo),str(v.hi)],width=str(v.width),seconds=time.perf_counter()-t,inputs=report['inputs']-n)
                assert item['old']['bounds']==item['direct']['bounds']
            report['status']='passed'
        finally:cell.cell_contents=original;c.inputs=original;c.checkpoint=before
        raise Stop
    monkeypatch.setattr(m,'integrate_square',capture)
    try:
        with pytest.raises(Stop):m.full_pixel(json.loads(source.read_bytes()),selected_piece=8)
    finally:
        report['seconds']=time.perf_counter()-start;dest.write_text(json.dumps(report,indent=2),encoding='utf8');record_property('direct_cost',json.dumps(report))
