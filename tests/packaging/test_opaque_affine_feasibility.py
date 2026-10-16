"""Bounded feasibility budget only; this diagnostic never admits an integral."""
import os,json,time,hashlib
from pathlib import Path
from fractions import Fraction as Q
import pytest

def test_saved_regions_affine_control_error_budget(monkeypatch,record_property):
    from tools import opaque_full_material_reference as m
    from tools.opaque_residual_range import ResidualRangeIndex
    from tools.opaque_shading_intervals import I,Jet
    source=Path(os.environ['GFFX_FULL_INPUT']);failure=source.parent.parent/'opaque103-20260923/ninth-preflight1-reference.json'
    assert hashlib.sha256(failure.read_bytes()).hexdigest()=='57eb00d29b532467fdc8952cb241ed70e782aaf748bb466ef67507f2ac998aae'
    leaves=json.loads(failure.read_bytes())['details']['cause']['last_complete_enclosure']['error_accounting']['largest_leaves'];rows=[leaves[i] for i in [0,1,3]]
    dest=Path(os.environ['GFFX_FULL_REFERENCE']);assert not dest.exists();started=time.perf_counter();report=dict(status='running',max_seconds=20,max_inputs=2048,inputs=0,regions=[],scope='Feasibility bounds only, not a numerical integral certificate')
    class Stop(Exception):pass
    def check():assert time.perf_counter()-started<20;assert report['inputs']<=2048
    def capture(field,**kw):
        c=kw['correction_integrator'];prepare=kw['prepare'];cell=prepare.__closure__[prepare.__code__.co_freevars.index('inputs')];original=cell.cell_contents;before=c.checkpoint;saved=c.inputs
        def inputs(*a,**k):check();assert report['inputs']<2048;report['inputs']+=1;return original(*a,**k)
        def checkpoint():check();before()
        cell.cell_contents=inputs;c.inputs=lambda *a:inputs(*a,material_only=True);c.checkpoint=checkpoint
        try:
            index=ResidualRangeIndex(c.extension,checkpoint=checkpoint)
            for row in rows:
                coords=list(map(Q,row['box']));box=(I(I(coords[0]).lo,I(coords[1]).hi),I(I(coords[2]).lo,I(coords[3]).hi));prepare(0,0,box);item=dict(box=row['box'],roles=[]);report['regions'].append(item)
                mid=[I((Q(v.lo)+Q(v.hi))/2) for v in box];dx=[v-a for v,a in zip(box,mid)];radius=[I(v.bound) for v in dx];area=(I(box[0].hi)-I(box[0].lo))*(I(box[1].hi)-I(box[1].lo))
                def xjet(u,v):return c.inputs(Jet.variable(u,order=2),Jet(v,order=2,tangent=1))
                center=xjet(*mid);xx=xjet(*box);yy=c.inputs(Jet(box[0],order=2),Jet.variable(box[1],order=2))
                e=c.extension
                def weight(v,role):return v[1]*v[3]*c.mix([index.base_jet(v[1-role],e.indices[1-role],ch) for ch in range(2)])/(v[0].lift(e.pi)*v[0].lift(c.mean))
                for role in [0,1]:
                    q=center[role];qa=q[0,0]+q[1,0]*dx[0]+q[0,1]*dx[1];qr=I(min(qa.lo,xx[role][0,0].lo),max(qa.hi,xx[role][0,0].hi))
                    if qr.lo<0 or qr.hi>1:item['roles'].append(dict(role=role,refused='affine angular range leaves unit domain'));continue
                    dq=I(xx[role][2,0].bound)*radius[0]**2+I(xx[role][1,1].bound)*radius[0]*radius[1]+I(yy[role][2,0].bound)*radius[1]**2
                    gx,gy=weight(xx,role),weight(yy,role);dg=I(gx[2,0].bound)*radius[0]**2+I(gx[1,1].bound)*radius[0]*radius[1]+I(gy[2,0].bound)*radius[1]**2
                    rp=I(c.mix(index.high_delta_jet(Jet.variable(qr,order=1),e.indices[role]))[1,0].bound);jumps=c.mix([v[0] for v in index.jump_sums(qr)]);r=I(c.mix(index.delta(qa,e.indices[role])).bound)
                    angular=I(gx[0,0].bound)*(rp*dq+jumps)*area;weighted=r*dg*area
                    item['roles'].append(dict(role=role,affine_cosine=qa,cosine_error=dq,residual_slope=rp,value_jumps=jumps,weight_error=dg,angular_error=angular,weight_remainder=weighted,total_error=angular+weighted))
            report['status']='passed'
        finally:cell.cell_contents=original;c.inputs=saved;c.checkpoint=before
        raise Stop
    monkeypatch.setattr(m,'integrate_square',capture)
    try:
        with pytest.raises(Stop):m.full_pixel(json.loads(source.read_bytes()),selected_piece=8,material_only_inputs=True)
    finally:
        report['seconds']=time.perf_counter()-started;dest.write_text(json.dumps(report,default=str,indent=2),encoding='utf8');record_property('affine_feasibility',json.dumps(report,default=str))
