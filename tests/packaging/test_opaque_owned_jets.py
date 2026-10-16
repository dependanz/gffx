"""Internal fresh arithmetic results may transfer ownership, never input aliases."""
from decimal import Decimal as D,localcontext,ROUND_FLOOR,ROUND_CEILING
from fractions import Fraction as Q
import pytest
from tools.opaque_shading_intervals import I,Jet

@pytest.mark.parametrize('operation',['add','mul','neg','reciprocal','sqrt'])
@pytest.mark.parametrize('tangent',[0,'1e-80'])
def test_arithmetic_does_not_recopy_owned_coefficient_dictionary(operation,tangent,monkeypatch):
    a=Jet.variable(I('2','2.001'),order=3)
    if tangent:a.c[0,1]=I(tangent)
    b=Jet(I(3),order=3);original=Jet.__init__;copies=[]
    def observed(self,*args,**kwargs):
        if kwargs.get('coeff') is not None:copies.append(1)
        original(self,*args,**kwargs)
    monkeypatch.setattr(Jet,'__init__',observed)
    result=a+b if operation=='add' else a*b if operation=='mul' else -a if operation=='neg' else getattr(a,operation)()
    assert result.c and not copies

@pytest.mark.parametrize('operation',['add','mul','neg','reciprocal','sqrt'])
@pytest.mark.parametrize('pattern',['left_only','right_only','shared','tiny_tangent'])
def test_result_mutation_cannot_change_inputs(operation,pattern):
    a=Jet.variable(I('2','2.01'),order=3);b=Jet.variable(I('3','3.01'),order=3)
    if pattern=='left_only':b=Jet(3,order=3)
    if pattern=='right_only':a=Jet(2,order=3)
    if pattern=='tiny_tangent':a.c[0,1]=I('1e-80')
    original=[{k:(v.lo,v.hi) for k,v in q.c.items()} for q in (a,b)]
    r=a+b if operation=='add' else a*b if operation=='mul' else -a if operation=='neg' else getattr(a,operation)()
    for v in r.c.values():v.lo=D(-1000);v.hi=D(1000)
    assert original==[{k:(v.lo,v.hi) for k,v in q.c.items()} for q in (a,b)]


def test_external_coefficient_dictionary_still_copies():
    v=I(2);d={(0,0):v};q=Jet(order=3,coeff=d);v.lo=D(-9);d.clear()
    assert q[0,0].lo==2

@pytest.mark.parametrize('a,b',[(I(1,2),I(3,4)),(I('-1e-80','2e70'),I('-3e70','1e-80'))])
def test_addition_uses_one_context_and_directed_endpoints(a,b,monkeypatch):
    import tools.opaque_shading_intervals as m
    with localcontext() as c:
        c.prec=60;c.rounding=ROUND_FLOOR;lo=a.lo+b.lo
        c.rounding=ROUND_CEILING;hi=a.hi+b.hi
    real=m.localcontext;calls=[]
    def observed(*a,**kw):calls.append(1);return real(*a,**kw)
    monkeypatch.setattr(m,'localcontext',observed)
    with localcontext() as c:c.prec=6;q=a+b
    assert (q.lo,q.hi)==(lo,hi) and len(calls)==1

@pytest.mark.parametrize('lo,hi',[(D('-0'),D(0)),(D('1e-80'),D('1e80'))])
def test_decimal_endpoint_construction_avoids_conversion(lo,hi,monkeypatch):
    import tools.opaque_shading_intervals as m
    original=m.dec;calls=[]
    def observed(x):calls.append(x);return original(x)
    monkeypatch.setattr(m,'dec',observed);v=I(lo,hi)
    assert v.lo.as_tuple()==lo.as_tuple() and v.hi.as_tuple()==hi.as_tuple() and not calls

@pytest.mark.parametrize('lo,hi',[(D('NaN'),D(1)),(D(0),D('Infinity')),(D(2),D(1))])
def test_decimal_fast_path_retains_invalid_endpoint_refusal(lo,hi):
    with pytest.raises(ValueError):I(lo,hi)


def test_saved_regions_retain_exact_bounds_with_owned_arithmetic(monkeypatch):
    import os,json,time
    from pathlib import Path
    from tools import opaque_full_material_reference as m
    from tools.opaque_square_reference import integrate_square
    source=Path(os.environ['GFFX_FULL_INPUT']);dest=Path(os.environ['GFFX_FULL_REFERENCE']);prior=json.loads((dest.parent/'split1-reference.json').read_bytes())
    assert not dest.exists();aggregate_path=os.environ.get('GFFX_OWNED_AGGREGATE');charge=json.loads(Path(aggregate_path).read_bytes()) if aggregate_path else prior
    rows=[];calls=0;start=time.perf_counter();report=dict(status='working',rows=rows,prior_aggregate_path=aggregate_path)
    class Stop(Exception):pass
    def capture(field,**kwargs):
        nonlocal calls
        corr=kwargs['correction_integrator'];original=corr.inputs;prior_check=corr.checkpoint
        def check():
            prior_check()
            assert charge['aggregate_seconds']+time.perf_counter()-start<20,'diagnostic20seconds'
            assert charge['aggregate_calls']+calls<=2048,'diagnostic2048inputs'
        def inputs(*args):
            nonlocal calls
            check();assert charge['aggregate_calls']+calls<2048;calls+=1;return original(*args)
        corr.inputs=inputs;corr.checkpoint=check
        for row in (prior['rows'][:1] if aggregate_path else prior['rows']):
            saved=row['parent'];a,b,c,d=map(Q,saved['box']);box=(I(I(a).lo,I(b).hi),I(I(c).lo,I(d).hi));kwargs['prepare'](0,0,box);t=time.perf_counter();value=corr(box);hs=I(b-a);ht=I(d-c)
            def mapped(u,v):
                nonlocal calls
                check();assert charge['aggregate_calls']+calls<2048;calls+=1
                return [x*hs*ht for x in field(u*hs+I(a),v*ht+I(c))]
            smooth=integrate_square(mapped,tolerance='1e-6',max_cells=1,checkpoint=check)['value'];value=[x+y for x,y in zip(value,smooth)];elapsed=time.perf_counter()-t
            encoded=[[str(v.lo),str(v.hi)] for v in value];assert encoded==saved['value']
            rows.append(dict(box=saved['box'],value=encoded,exact=True,old_seconds=saved['seconds'],new_seconds=elapsed))
        report['status']='passed';raise Stop
    monkeypatch.setattr(m,'integrate_square',capture)
    try:
        with pytest.raises(Stop):m.full_pixel(json.loads(source.read_bytes()),selected_piece=3)
    finally:
        report.update(seconds=time.perf_counter()-start,calls=calls,aggregate_seconds=charge['aggregate_seconds']+time.perf_counter()-start,aggregate_calls=charge['aggregate_calls']+calls)
        dest.write_text(json.dumps(report,indent=2))
