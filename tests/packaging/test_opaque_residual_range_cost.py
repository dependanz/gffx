"""Bounded diagnosis of the saved ninth-piece angular domain refusal."""
import json,hashlib,os,time
from pathlib import Path
from fractions import Fraction as Q
import pytest

def test_saved_ninth_whole_residual_envelopes(monkeypatch,record_property):
    from tools import opaque_full_material_reference as m
    from tools.opaque_shading_intervals import I,Jet
    source=Path(os.environ['GFFX_FULL_INPUT']);failure=source.parent.parent/'opaque102-20260923/ninth-preflight1-reference.json'
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    assert sha(source)=='a15bea40cb5cdead7994d828b57f5dcfac9f44fab0292a06e7d0a7b08719f73a'
    assert sha(failure)=='4b9dd54690afbb536a31bd2dd87c8844b096cc76b5ce58b8296832980a2ba3b7'
    saved=json.loads(failure.read_bytes())['details']['cause']['last'];assert saved['box']==['73/256','37/128','1/4','1/2']
    dest=Path(os.environ['GFFX_FULL_REFERENCE']);assert not dest.exists();started=time.perf_counter()
    report=dict(status='running',input_sha256=sha(source),failure_sha256=sha(failure),max_inputs=2048,max_seconds=20,inputs=0,boxes=[])
    class Stop(Exception):pass
    def check():assert time.perf_counter()-started<20;assert report['inputs']<=2048
    def capture(field,**kw):
        c=kw['correction_integrator'];prepare=kw['prepare'];cell=prepare.__closure__[prepare.__code__.co_freevars.index('inputs')];original=cell.cell_contents;before=c.checkpoint
        def inputs(*args):
            check();assert report['inputs']<2048;report['inputs']+=1;return original(*args)
        def checkpoint():check();before()
        cell.cell_contents=inputs;c.inputs=inputs;c.checkpoint=checkpoint;report['curved']=c.curved
        c.envelope_fallback=True
        c._certified_integral=lambda box:(_ for _ in ()).throw(ValueError('bounded angular segment count'))
        try:
            for label,a,b,c0,d in [('saved',Q(73,256),Q(37,128),Q(1,4),Q(1,2)),('radial_half',Q(73,256),Q(147,512),Q(1,4),Q(1,2)),('transverse_half',Q(73,256),Q(37,128),Q(1,4),Q(3,8))]:
                box=(I(I(a).lo,I(b).hi),I(I(c0).lo,I(d).hi));row=dict(label=label,box=list(map(str,[a,b,c0,d])),roles=[]);report['boxes'].append(row)
                check();prepare(0,0,box);ext=c.extension if hasattr(c,'extension') else c.e
                row['hint']=c.domain_split_axis;row['indices']=ext.indices;row['bounds']=[[str(v.lo),str(v.hi)] for v in ext.bounds]
                dx=inputs(Jet.variable(box[0],order=1),Jet(box[1],order=1));dy=inputs(Jet(box[0],order=1),Jet.variable(box[1],order=1))
                for role in range(2):
                    v=ext.bounds[role];chosen=([-1] if v.lo<=ext.cn[0].hi else [])+[i for i in range(len(ext.cn)-1) if v.hi>=ext.cn[i].lo and v.lo<=ext.cn[i+1].hi]
                    row['roles'].append(dict(role=role,segments=len(chosen),chosen=chosen,dx=[str(dx[role][1,0].lo),str(dx[role][1,0].hi)],dy=[str(dy[role][1,0].lo),str(dy[role][1,0].hi)]))
                try:
                    values=c(box);row['correction']=[[str(v.lo),str(v.hi)] for v in values];row['width']=str(values[1].width);row['blocks']=sum(len(level) for level in c.range_index.blocks);row['query_blocks']=c.range_index.last_query_blocks
                    assert row['query_blocks']<=3
                except ValueError as e:row['domain']=str(e)
            report['status']='passed'
        finally:cell.cell_contents=original;c.inputs=original;c.checkpoint=before
        raise Stop
    monkeypatch.setattr(m,'integrate_square',capture)
    try:
        with pytest.raises(Stop):m.full_pixel(json.loads(source.read_bytes()),selected_piece=8)
    finally:
        report['seconds']=time.perf_counter()-started;dest.write_text(json.dumps(report,indent=2),encoding='utf8');record_property('domain_directions',json.dumps(report))
