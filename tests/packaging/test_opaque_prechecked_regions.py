"""Three saved leaves, both smooth directions and independently valid corrections."""
import os,json,time,hashlib,math
from pathlib import Path
from fractions import Fraction as Q
import pytest

@pytest.mark.parametrize('index',[11])
def test_prechecked_direction_on_saved_remaining_regions(monkeypatch,record_property,index):
    from tools import opaque_full_material_reference as m
    from tools.opaque_decimal_convolution import decimal_convolution
    from tools.opaque_high_transport import TransportMaterial
    from tools.opaque_direction_precheck import DirectionBoundary
    from tools.opaque_shading_intervals import I,Jet
    source=Path(os.environ['GFFX_FULL_INPUT']);failure=source.parent.parent/'opaque118-20260924'/('twelfth-preflight1-reference.json' if index==11 else 'thirteenth-preflight1-reference.json')
    assert hashlib.sha256(failure.read_bytes()).hexdigest()=='532334510191e1feb41f54f083233d287651ea44ac92444791b732f3c3109606'
    leaves=json.loads(failure.read_bytes())['details']['cause']['last_complete_enclosure']['error_accounting']['largest_leaves'];rows=leaves[:3]
    assert hashlib.sha256(source.read_bytes()).hexdigest()=='a15bea40cb5cdead7994d828b57f5dcfac9f44fab0292a06e7d0a7b08719f73a'
    dest=Path(os.environ['GFFX_FULL_REFERENCE']);dest=dest.with_name(dest.stem+f'-piece{index}.json');assert not dest.exists();started=time.perf_counter();report=dict(piece=index,status='running',input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),failure_sha256=hashlib.sha256(failure.read_bytes()).hexdigest(),max_seconds=20,max_inputs=2048,inputs=0,regions=[],input_types={})
    class Stop(Exception):pass
    def check():assert time.perf_counter()-started<20;assert report['inputs']<=2048
    def capture(field,**kw):
        c=kw['correction_integrator'];report['curved']=c.curved;prepare=kw['prepare'];cell=prepare.__closure__[prepare.__code__.co_freevars.index('inputs')];original=cell.cell_contents;before=c.checkpoint
        def wrap(fn,kind):
            def counted(*a,**k):
                check();assert report['inputs']<2048;report['inputs']+=1;report['input_types'][kind]=report['input_types'].get(kind,0)+1;return fn(*a,**k)
            return counted
        inputs=wrap(original,'full');old_role=c.role_inputs;old_implicit=c.implicit_inputs
        c.role_inputs=wrap(old_role,'role');c.implicit_inputs=wrap(old_implicit,'implicit')
        def checkpoint():check();before()
        cell.cell_contents=inputs;c.inputs=wrap(lambda *a:original(*a,material_only=True),'material');c.checkpoint=checkpoint
        try:
            proof=TransportMaterial(c,order=6);boundary=DirectionBoundary(c,tolerance=kw['tolerance'],piece=11);factor=I(Q(math.factorial(5)**4,11*math.factorial(10)**2))
            for row in rows:
                coords=list(map(Q,row['box']));box=(I(I(coords[0]).lo,I(coords[1]).hi),I(I(coords[2]).lo,I(coords[3]).hi));prepare(0,0,box);h=[I(v.hi)-I(v.lo) for v in box];item=dict(box=row['box'],smooth=[],corrections={});report['regions'].append(item)
                for name,fn in [('selected_complete_proof',lambda b:boundary(b))]:
                    t=time.perf_counter()
                    try:
                        v=fn(box)[1];item['corrections'][name]=dict(bounds=[str(v.lo),str(v.hi)],width=str(v.width),seconds=time.perf_counter()-t,axis=c.preferred_split_axis,parts=c.error_breakdown,boundary=getattr(boundary,'last_boundary_details',None))
                    except ValueError as e:item['corrections'][name]=dict(domain=str(e),seconds=time.perf_counter()-t)
            report['status']='passed';report['prechecks']=boundary.prechecks;report['events']=boundary.events
        finally:cell.cell_contents=original;c.inputs=original;c.checkpoint=before;c.role_inputs=old_role;c.implicit_inputs=old_implicit
        raise Stop
    monkeypatch.setattr(m,'integrate_square',capture)
    try:
        with decimal_convolution() as stats:
            with pytest.raises(Stop):m.full_pixel(json.loads(source.read_bytes()),selected_piece=index)
        report['native_stats']=stats
    finally:
        report['seconds']=time.perf_counter()-started;dest.write_text(json.dumps(report,default=str,indent=2),encoding='utf8');record_property('saved_directions',json.dumps(report,default=str))
