"""Three saved leaves, both smooth directions and independently valid corrections."""
import os,json,time,hashlib,math
from pathlib import Path
from fractions import Fraction as Q
import pytest

@pytest.mark.parametrize('index',[11])
def test_value_centering_on_saved_expensive_regions(monkeypatch,record_property,index):
    from tools import opaque_full_material_reference as m
    from tools.opaque_decimal_convolution import decimal_convolution
    from tools.opaque_high_transport import TransportMaterial
    from tools.opaque_centered_gradient import ValueCenteredBoundary as BoundaryMaterial
    from tools.opaque_shading_intervals import I,Jet
    source=Path(os.environ['GFFX_FULL_INPUT']);failure=source.parent.parent/'opaque117-20260924'/('twelfth-preflight1-reference.json' if index==11 else 'thirteenth-preflight1-reference.json')
    assert hashlib.sha256(failure.read_bytes()).hexdigest()=='ee2962b8b6955934adfe1b287f63a6f2bba4518c13d6ff6e55bfaf1c703d8715'
    events=json.loads(failure.read_bytes())['boundary_events'];chosen=sorted([z for z in events if z['axis']==1 and z['status']=='intersected'],key=lambda z:z['seconds'],reverse=True)[:3];rows=[dict(box=[v for q in z['box'] for v in q]) for z in chosen]
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
            proof=TransportMaterial(c,order=6);boundary=BoundaryMaterial(c,order=6,parts=2,localize_higher=True,algebraic=True);factor=I(Q(math.factorial(5)**4,11*math.factorial(10)**2))
            for row in rows:
                coords=list(map(Q,row['box']));box=(I(I(coords[0]).lo,I(coords[1]).hi),I(I(coords[2]).lo,I(coords[3]).hi));prepare(0,0,box);h=[I(v.hi)-I(v.lo) for v in box];item=dict(box=row['box'],smooth=[],corrections={});report['regions'].append(item)
                item['gradient_bounds']=[]
                mid=[I((Q(v.lo)+Q(v.hi))/2) for v in box]
                for axis in (0,1):
                    other=1-axis;args=[Jet(v,order=2) for v in box];args[axis]=Jet.variable(box[axis],order=2);args[other]=Jet.variable(box[other],axis=1,order=2)
                    whole=inputs(*args)[:2];point=[Jet(v,order=2) for v in mid];point[axis]=Jet.variable(mid[axis],order=2);center=inputs(*point)[:2];deltas=[v-m for v,m in zip(box,mid)];gradient=[]
                    for role,(a,z) in enumerate(zip(whole,center)):
                        centered=z[1,0]+2*a[2,0]*deltas[axis]+a[1,1]*deltas[other]
                        intersection=I(max(a[1,0].lo,centered.lo),min(a[1,0].hi,centered.hi))
                        value_centered=z[0,0]+a[1,0]*deltas[axis]+a[0,1]*deltas[other]
                        gradient.append(dict(role=role,value=str(a[0,0]),value_centered=str(value_centered),second=str(2*a[2,0]),natural=str(a[1,0]),center=str(z[1,0]),centered=str(centered),intersection=str(intersection)))
                    item['gradient_bounds'].append(dict(axis=axis,roles=gradient))
                for axis in [0,1]:
                    args=[Jet(v,order=10) for v in box];args[axis]=Jet.variable(box[axis],order=10);values=field(*args);item['smooth'].append([str((h[1-axis]*h[axis]**11*I(q[10,0].bound)*factor).hi) for q in values])
                from tools.opaque_peano import gauss5
                item['inner_lines']=[]
                for axis in (0,1):
                    for node in gauss5()[0]:
                        args=[Jet(v,order=2) for v in box];args[axis]=Jet.variable(box[axis],order=2);args[1-axis]=Jet(I(box[1-axis].lo)+h[1-axis]*node,order=2)
                        vals=inputs(*args);center=list(args);center[axis]=Jet(mid[axis],order=2);at=inputs(*center)
                        item['inner_lines'].append(dict(axis=axis,roles=[dict(value=str(v[0,0]),centered_value=str(z[0,0]+v[1,0]*(box[axis]-mid[axis])),derivative=str(v[1,0]),second=str(2*v[2,0])) for v,z in zip(vals[:2],at[:2])]))
                for name,fn in [('direct',c.envelope_integral),('boundary6axis0',lambda b:boundary(b,integrated_axis=0)),('boundary6axis1',lambda b:boundary(b,integrated_axis=1))]:
                    t=time.perf_counter()
                    try:
                        v=fn(box)[1];item['corrections'][name]=dict(bounds=[str(v.lo),str(v.hi)],width=str(v.width),seconds=time.perf_counter()-t,axis=c.preferred_split_axis,parts=c.error_breakdown,boundary=getattr(boundary,'last_boundary_details',None))
                    except ValueError as e:item['corrections'][name]=dict(domain=str(e),seconds=time.perf_counter()-t)
            report['status']='passed';report['refinements']=boundary.refinements;report['recovered_signs']=boundary.recovered_signs;report['algebraic_uses']=boundary.algebraic_uses;report['algebraic_fallbacks']=boundary.algebraic_fallbacks
        finally:cell.cell_contents=original;c.inputs=original;c.checkpoint=before;c.role_inputs=old_role;c.implicit_inputs=old_implicit
        raise Stop
    monkeypatch.setattr(m,'integrate_square',capture)
    try:
        with decimal_convolution() as stats:
            with pytest.raises(Stop):m.full_pixel(json.loads(source.read_bytes()),selected_piece=index)
        report['native_stats']=stats
    finally:
        report['seconds']=time.perf_counter()-started;dest.write_text(json.dumps(report,default=str,indent=2),encoding='utf8');record_property('saved_directions',json.dumps(report,default=str))
