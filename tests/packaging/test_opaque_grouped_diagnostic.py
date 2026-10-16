"""Three saved leaves, both smooth directions and independently valid corrections."""
import os,json,time,hashlib,math
from pathlib import Path
from fractions import Fraction as Q
import pytest

def test_saved_ninth_grouped_terms_and_directions(monkeypatch,record_property):
    from tools import opaque_full_material_reference as m
    from tools.opaque_aggregate_peano import AggregateMaterial
    from tools.opaque_grouped_peano import GroupedMaterial
    from tools.opaque_shading_intervals import I,Jet
    source=Path(os.environ['GFFX_FULL_INPUT']);failure=source.parent.parent/'opaque103-20260923/ninth-preflight1-reference.json'
    assert hashlib.sha256(failure.read_bytes()).hexdigest()=='57eb00d29b532467fdc8952cb241ed70e782aaf748bb466ef67507f2ac998aae'
    leaves=json.loads(failure.read_bytes())['details']['cause']['last_complete_enclosure']['error_accounting']['largest_leaves'];rows=[leaves[i] for i in [0,1,3]]
    dest=Path(os.environ['GFFX_FULL_REFERENCE']);assert not dest.exists();started=time.perf_counter();report=dict(status='running',max_seconds=20,max_inputs=2048,inputs=0,regions=[])
    class Stop(Exception):pass
    def check():assert time.perf_counter()-started<20;assert report['inputs']<=2048
    def capture(field,**kw):
        c=kw['correction_integrator'];prepare=kw['prepare'];cell=prepare.__closure__[prepare.__code__.co_freevars.index('inputs')];original=cell.cell_contents;before=c.checkpoint
        def inputs(*a):check();assert report['inputs']<2048;report['inputs']+=1;return original(*a)
        def checkpoint():check();before()
        cell.cell_contents=inputs;c.inputs=inputs;c.checkpoint=checkpoint
        try:
            proof=AggregateMaterial(c);grouped=GroupedMaterial(c);factor=I(Q(math.factorial(5)**4,11*math.factorial(10)**2))
            for row in rows:
                coords=list(map(Q,row['box']));box=(I(I(coords[0]).lo,I(coords[1]).hi),I(I(coords[2]).lo,I(coords[3]).hi));prepare(0,0,box);h=[I(v.hi)-I(v.lo) for v in box];item=dict(box=row['box'],smooth=[],corrections={});report['regions'].append(item)
                for axis in [0,1]:
                    args=[Jet(v,order=10) for v in box];args[axis]=Jet.variable(box[axis],order=10);values=field(*args);item['smooth'].append([str((h[1-axis]*h[axis]**11*I(q[10,0].bound)*factor).hi) for q in values])
                for name,fn in [('current',c),('aggregate',proof),('conditioned0',lambda b:proof(b,integrated_axis=0)),('conditioned1',lambda b:proof(b,integrated_axis=1)),('grouped0',lambda b:grouped(b,integrated_axis=0)),('grouped1',lambda b:grouped(b,integrated_axis=1))]:
                    t=time.perf_counter()
                    try:
                        v=fn(box)[1];item['corrections'][name]=dict(bounds=[str(v.lo),str(v.hi)],width=str(v.width),seconds=time.perf_counter()-t,axis=c.preferred_split_axis,parts=c.error_breakdown)
                    except ValueError as e:item['corrections'][name]=dict(domain=str(e),seconds=time.perf_counter()-t)
                item['aggregate_directions']=[]
                for axis in [0,1]:
                    try:item['aggregate_directions'].append(proof.axis_bounds(box,axis)[1])
                    except ValueError as e:item['aggregate_directions'].append(dict(domain=str(e)))
            report['status']='passed'
        finally:cell.cell_contents=original;c.inputs=original;c.checkpoint=before
        raise Stop
    monkeypatch.setattr(m,'integrate_square',capture)
    try:
        with pytest.raises(Stop):m.full_pixel(json.loads(source.read_bytes()),selected_piece=8)
    finally:
        report['seconds']=time.perf_counter()-started;dest.write_text(json.dumps(report,default=str,indent=2),encoding='utf8');record_property('saved_directions',json.dumps(report,default=str))
