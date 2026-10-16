"""Bounded same-region endpoint and timing comparison; no new piece trial."""
import os,json,time,hashlib
from pathlib import Path
from fractions import Fraction as Q
import pytest

def test_saved_regions_binary_high_derivative_cost(monkeypatch,record_property):
    from tools import opaque_full_material_reference as m
    from tools.opaque_residual_range import ResidualRangeIndex
    from tools.opaque_shading_intervals import I,Jet
    from tools.opaque_binary_derivatives import binary_derivatives
    from contextlib import nullcontext
    import math
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
                for mode in (['decimal','binary','binary','decimal'] if number%2==0 else ['binary','decimal','decimal','binary']):
                    c.inputs=lambda *a:inputs(*a,material_only=True)
                    n=report['inputs'];ti=report['input_seconds'];t=time.perf_counter()
                    with binary_derivatives() if mode=='binary' else nullcontext({}) as stats:
                        prepare(0,0,box);smooth=[];low=[];h=[I(v.hi)-I(v.lo) for v in box];factor=I(Q(math.factorial(5)**4,11*math.factorial(10)**2))
                        for axis in (0,1):
                            args=[Jet(v,order=10) for v in box];args[axis]=Jet.variable(box[axis],order=10);values=field(*args)
                            smooth.append([str((h[1-axis]*h[axis]**11*I(q[10,0].bound)*factor).hi) for q in values])
                            low.append([[[str(q[i,j].lo),str(q[i,j].hi)] for i in range(3) for j in range(2)] for q in values])
                        v=c(box)
                    item['profiles'].append(dict(mode=mode,smooth=smooth,low=low,stats=stats,bounds=[[str(q.lo),str(q.hi)] for q in v],seconds=time.perf_counter()-t,inputs=report['inputs']-n,input_seconds=report['input_seconds']-ti))
                assert all(p['low']==item['profiles'][0]['low'] for p in item['profiles'])
            report['status']='passed'
        finally:cell.cell_contents=original;c.inputs=saved;c.checkpoint=before
        raise Stop
    monkeypatch.setattr(m,'integrate_square',capture)
    try:
        with pytest.raises(Stop):m.full_pixel(json.loads(source.read_bytes()),selected_piece=8,material_only_inputs=True)
    finally:
        report['seconds']=time.perf_counter()-started;dest.write_text(json.dumps(report,default=str,indent=2),encoding='utf8');record_property('binary_derivative_diagnostic',json.dumps(report,default=str))
