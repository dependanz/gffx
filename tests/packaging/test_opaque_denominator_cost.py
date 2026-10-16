"""Controlled old/new arithmetic on complete base and guidance fields."""
import os,json,hashlib,ast,time
from pathlib import Path
from fractions import Fraction as Q
import pytest
from tools.opaque_shading_intervals import I,Jet

def test_saved_d343_base_guidance_arithmetic_cost(monkeypatch,record_property):
    from tools import opaque_full_material_reference as m
    from tools import opaque_shading_intervals as arithmetic
    source=Path(os.environ['GFFX_FULL_INPUT']);dest=Path(os.environ['GFFX_FULL_REFERENCE']);out=dest.parent
    prior=[json.loads((out/(n+'-reference.json')).read_bytes()) for n in ('diagnostic1','diagnostic2')]
    seconds=sum(x['seconds'] for x in prior);inputs=sum(x['input_calls'] for x in prior)
    aggregate_path=os.environ.get('GFFX_COST_AGGREGATE')
    if aggregate_path:
        aggregate=json.loads(Path(aggregate_path).read_bytes());seconds=aggregate['aggregate_seconds'];inputs=aggregate['aggregate_calls']
    boxes=prior[0]['boxes'];assert len(boxes)==3
    old=source.parent.parent/'opaque90-20260923/source-final/tools/opaque_shading_intervals.py'
    frozen=json.loads((old.parents[2]/'final-source1.json').read_bytes());assert hashlib.sha256(old.read_bytes()).hexdigest()==next(x['sha256'] for x in frozen['source'] if x['path']=='tools/opaque_shading_intervals.py')
    tree=ast.parse(old.read_text());methods={}
    for cls,name in [('Interval','__truediv__'),('Interval','__add__'),('Interval','__mul__'),('Jet','reciprocal'),('Jet','sqrt')]:
        c=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name==cls);node=next(x for x in c.body if isinstance(x,ast.FunctionDef) and x.name==name)
        namespace=dict(arithmetic.__dict__);exec(compile(ast.Module(body=[node],type_ignores=[]),str(old),'exec'),namespace);methods[cls,name]=namespace[name]
        if cls=='Interval' and name in ('__add__','__mul__'):methods[cls,name.replace('__','__r',1)]=namespace[name]
    class Stop(Exception):pass
    def capture(field,**kwargs):
        start=time.perf_counter();calls=0;passes=[];complete=False
        def check():
            kwargs['checkpoint']();assert time.perf_counter()-start+seconds<20;assert calls+inputs<=2048
        try:
            for mode in ('old','new','new','old'):
                with monkeypatch.context() as patch:
                    if mode=='old':
                        for (cls,name),fn in methods.items():patch.setattr(getattr(arithmetic,cls),name,fn)
                    t=time.perf_counter();values=[]
                    for row in boxes:
                        box=tuple(I(*v) for v in row['box']);check();kwargs['correction_integrator'].smooth._cache.clear();kwargs['prepare'](0,0,box)
                        for axis in (0,1):
                            args=[Jet(v,order=10) for v in box];args[axis]=Jet.variable(box[axis],order=10)
                            for fn in (field,kwargs['refinement_field']):
                                check();calls+=1;got=fn(*args)
                                values.append([{str(k):[str(v.lo),str(v.hi)] for k,v in q.c.items()} for q in got])
                    passes.append(dict(mode=mode,seconds=time.perf_counter()-t,values=values))
            assert all(x['values']==passes[0]['values'] for x in passes)
            complete=True
        finally:
            report=dict(status='diagnostic_only' if complete else 'stopped_partial',seconds=time.perf_counter()-start,prior_seconds=seconds,aggregate_seconds=time.perf_counter()-start+seconds,field_calls=calls,aggregate_calls=inputs+calls,max_seconds=20,max_inputs=2048,passes=passes,numerical_fields_exact=complete,scope='Same three saved boxes; old-new-new-old high-order base+guidance with branch cache cleared and preparation timed. Source of old arithmetic pinned; no full-piece admission.')
            dest.write_text(json.dumps(report,indent=2));record_property('arithmetic_cost',json.dumps({k:v for k,v in report.items() if k!='passes'}))
        check();raise Stop
    monkeypatch.setattr(m,'integrate_square',capture)
    with pytest.raises(Stop):m.full_pixel(json.loads(source.read_bytes()),selected_piece=2)
