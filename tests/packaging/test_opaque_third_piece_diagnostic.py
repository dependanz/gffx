"""Pinned third-piece diagnostics; no complete-piece quadrature or admission."""
import hashlib,json,os
from fractions import Fraction as Q
from pathlib import Path
import pytest


def test_saved_third_piece_terms_directions_and_cost(monkeypatch,record_property):
    from tools import opaque_full_material_reference as module
    from tools.opaque_boundary_diagnostic import diagnose,DiagnosticStopped
    from tools.opaque_shading_intervals import I
    source=Path(os.environ['GFFX_FULL_INPUT']);base=source.parent.parent
    failure=base/'opaque81-20260921/original-campaign1/piece-02-result.json'
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    assert digest(source)=='a15bea40cb5cdead7994d828b57f5dcfac9f44fab0292a06e7d0a7b08719f73a'
    assert digest(failure)=='b236f9c4eb4279e8d340b3a9a11e6cea76d79407e985397116561ffc46064c7d'
    failed=json.loads(failure.read_bytes());assert failed['piece']==2 and failed['details']['face']==5740
    cover=failed['details']['cause']['last_complete_enclosure']
    assert cover['complete_cover'] and cover['leaves']==111
    chosen=cover['error_accounting']['largest_leaves'][:3]
    assert len(chosen)==3
    boxes=[(I(I(Q(r['box'][0])).lo,I(Q(r['box'][1])).hi),I(I(Q(r['box'][2])).lo,I(Q(r['box'][3])).hi)) for r in chosen]
    dest=Path(os.environ['GFFX_FULL_REFERENCE']);assert not dest.exists()
    def encode(v):
        if isinstance(v,I):return [str(v.lo),str(v.hi)]
        raise TypeError(type(v))
    class Captured(Exception):pass
    def capture(field,**kwargs):
        report=None
        import cProfile,pstats,time
        correction=kwargs['correction_integrator'];original=correction.inputs;input_cost={}
        def timed(*args):
            order=max(v.order for v in args);started=time.perf_counter()
            try:return original(*args)
            finally:
                row=input_cost.setdefault(order,dict(calls=0,seconds=0));row['calls']+=1;row['seconds']+=time.perf_counter()-started
        correction.inputs=timed;prof=cProfile.Profile();prof.enable()
        try:report=diagnose(kwargs['correction_integrator'],boxes,prepare=lambda b:kwargs['prepare'](0,0,b),max_inputs=1024,max_seconds=8,partition_smooth=False)
        except DiagnosticStopped as e:report=e.report;raise
        finally:
            prof.disable();correction.inputs=original
            if report is not None:
                stats=pstats.Stats(prof);costs=[]
                for (file,line,function),(primitive,calls,self_seconds,total_seconds,callers) in stats.stats.items():
                    if 'opaque_' in file:
                        costs.append(dict(file=file,line=line,function=function,calls=calls,self_seconds=self_seconds,total_seconds=total_seconds))
                report['profile']=dict(input_cost=input_cost,functions=sorted(costs,key=lambda r:r['total_seconds'],reverse=True),scope='cProfile instrumented diagnostic;inclusive function times overlap;not a production runtime benchmark')
                report['saved_cover']=dict(cells=cover['cells'],leaves=cover['leaves'],widths=cover['widths'],accounting=cover['error_accounting'])
                report.update(input_sha256=digest(source),failure_sha256=digest(failure),piece=2,largest_leaf_ranks=[1,2,3],scope='Three disjoint largest saved leaves only; both orientations; not a complete-piece certificate')
                dest.write_text(json.dumps(report,indent=2,default=encode),encoding='utf8')
                record_property('boundary_diagnostic',json.dumps(dict(status=report['status'],input_calls=report['input_calls'],witnesses=report['witnesses'],seconds=report.get('seconds'),path=str(dest))))
        assert kwargs['correction_integrator'].smooth_partition is True
        assert report['status']=='diagnostic_only' and len(report['boxes'])==3
        assert all('ordinary_comparison' in a for row in report['boxes'] for a in row['orientations'])
        raise Captured
    monkeypatch.setattr(module,'integrate_square',capture)
    with pytest.raises(Captured):module.full_pixel(json.loads(source.read_bytes()),selected_piece=2)


def test_saved_third_piece_conditioned_dependence(monkeypatch,record_property):
    from tools import opaque_full_material_reference as module
    from tools.opaque_boundary_diagnostic import diagnose,DiagnosticStopped
    from tools.opaque_shading_intervals import I
    source=Path(os.environ['GFFX_FULL_INPUT']);base=source.parent.parent
    failure=base/'opaque81-20260921/original-campaign1/piece-02-result.json'
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    assert digest(source)=='a15bea40cb5cdead7994d828b57f5dcfac9f44fab0292a06e7d0a7b08719f73a'
    assert digest(failure)=='b236f9c4eb4279e8d340b3a9a11e6cea76d79407e985397116561ffc46064c7d'
    failed=json.loads(failure.read_bytes());assert failed['piece']==2 and failed['details']['face']==5740
    cover=failed['details']['cause']['last_complete_enclosure']
    assert cover['complete_cover'] and cover['leaves']==111
    chosen=cover['error_accounting']['largest_leaves'][:3]
    assert len(chosen)==3
    boxes=[(I(I(Q(r['box'][0])).lo,I(Q(r['box'][1])).hi),I(I(Q(r['box'][2])).lo,I(Q(r['box'][3])).hi)) for r in chosen]
    dest=Path(os.environ['GFFX_FULL_REFERENCE']);assert not dest.exists()
    def encode(v):
        if isinstance(v,I):return [str(v.lo),str(v.hi)]
        raise TypeError(type(v))
    class Captured(Exception):pass
    def capture(field,**kwargs):
        report=None
        import cProfile,pstats,time
        correction=kwargs['correction_integrator'];original=correction.inputs;input_cost={}
        def timed(*args):
            order=max(v.order for v in args);started=time.perf_counter()
            try:return original(*args)
            finally:
                row=input_cost.setdefault(order,dict(calls=0,seconds=0));row['calls']+=1;row['seconds']+=time.perf_counter()-started
        correction.inputs=timed
        from tools.opaque_shading_intervals import Jet
        from tools.opaque_peano import gauss5,kernel_bounds
        from tools.opaque_smooth_peano import ordinary_bounds
        from tools.opaque_peano_material import segmented
        nodes,weights=gauss5();lines={};details=[];original_axis=correction.axis_bounds
        key=lambda box,axis:(axis,tuple((v.lo,v.hi) for v in box))
        for region,box in enumerate(boxes):
            for integrated in (0,1):
                width=I(box[integrated].hi)-I(box[integrated].lo)
                for node,weight in zip(nodes,weights):
                    line=list(box);line[integrated]=I(box[integrated].lo)+width*node
                    lines[key(line,1-integrated)]=(region,integrated,width*weight)
        def observed_axis(box,axis):
            rows=original_axis(box,axis)
            identity=lines.get(key(box,axis))
            if identity is None:return rows
            correction.checkpoint();region,integrated,mass=identity;domain=box[axis]
            midpoint=I((Q(domain.lo)+Q(domain.hi))/2);halves=[I(domain.lo,midpoint.hi),I(midpoint.lo,domain.hi)]
            ranges=[];samples=[]
            for half in halves:
                correction.checkpoint();args=[Jet(v,order=3) for v in box];args[axis]=Jet.variable(half,order=3)
                values=correction.inputs(*args);nv,nl,_,jac=values;e=correction.extension
                regular=correction._ordinary_inputs(args[0],values)[1];ranges.append(regular[3,0]*6)
                # Separate terms only diagnose their relative scale. They do
                # not replace the production grouping or establish a new bound.
                av=correction.mix(correction.smooth.at(nv,e.indices[0]));bl=correction.mix(correction.smooth.at(nl,e.indices[1]))
                dv=correction.mix(segmented(e,nv,e.indices[0]));dl=correction.mix(segmented(e,nl,e.indices[1]))
                terms=[nl*term*jac/(e.pi*correction.mean) for term in (dv*bl,av*dl,dv*dl)]
                def angle(q,base):
                    value=q[0,0];chosen=([-1] if value.lo<=e.cn[0].hi else [])+[i for i in range(len(e.cn)-1) if value.hi>=e.cn[i].lo and value.lo<=e.cn[i+1].hi]
                    return dict(value=value,third=q[3,0]*6,base=base,possible_segments=chosen)
                samples.append(dict(domain=half,third=regular[3,0]*6,term_thirds=[v[3,0]*6 for v in terms],view=angle(nv,e.indices[0]),light=angle(nl,e.indices[1])))
            hull=I(min(v.lo for v in ranges),max(v.hi for v in ranges));anchor=I(I((Q(hull.lo)+Q(hull.hi))/2).lo)
            factor=(I(domain.hi)-I(domain.lo))**4*kernel_bounds()[1][2]*mass/2
            candidate=min(sum((I(v.bound)*factor for v in ranges),I(0)).hi,sum((I((v-anchor).bound)*factor for v in ranges),I(0)).hi)
            before=ordinary_bounds(rows[1],domain,mass)['effective']
            details.append(dict(region=region,integrated_axis=integrated,active_axis=axis,box=box,mass=mass,original_third=rows[1]['third_range'],original_radius=before,candidate_radius=I(0,candidate),shared_anchor=anchor,halves=samples,scope='Prospective two-half ordinary enclosure only;all boundary/base terms remain separately required;not adopted or whole-piece qualified'))
            return rows  # Exact original rows; observer cannot affect the bound.
        correction.axis_bounds=observed_axis
        prof=cProfile.Profile();prof.enable()
        try:report=diagnose(kwargs['correction_integrator'],boxes,prepare=lambda b:kwargs['prepare'](0,0,b),max_inputs=1024,max_seconds=12,partition_smooth=False)
        except DiagnosticStopped as e:report=e.report;raise
        finally:
            prof.disable();correction.inputs=original;correction.axis_bounds=original_axis
            if report is not None:
                stats=pstats.Stats(prof);costs=[]
                for (file,line,function),(primitive,calls,self_seconds,total_seconds,callers) in stats.stats.items():
                    if 'opaque_' in file:
                        costs.append(dict(file=file,line=line,function=function,calls=calls,self_seconds=self_seconds,total_seconds=total_seconds))
                report['conditioned_inspection']=details
                report['profile']=dict(input_cost=input_cost,functions=sorted(costs,key=lambda r:r['total_seconds'],reverse=True),scope='cProfile instrumented diagnostic;inclusive function times overlap;not a production runtime benchmark')
                report['saved_cover']=dict(cells=cover['cells'],leaves=cover['leaves'],widths=cover['widths'],accounting=cover['error_accounting'])
                report.update(input_sha256=digest(source),failure_sha256=digest(failure),piece=2,largest_leaf_ranks=[1,2,3],scope='Three disjoint largest saved leaves only; both orientations; not a complete-piece certificate')
                dest.write_text(json.dumps(report,indent=2,default=encode),encoding='utf8')
                record_property('boundary_diagnostic',json.dumps(dict(status=report['status'],input_calls=report['input_calls'],witnesses=report['witnesses'],seconds=report.get('seconds'),path=str(dest))))
        assert len(details)==30
        assert kwargs['correction_integrator'].smooth_partition is True
        assert report['status']=='diagnostic_only' and len(report['boxes'])==3
        assert all('ordinary_comparison' in a for row in report['boxes'] for a in row['orientations'])
        raise Captured
    monkeypatch.setattr(module,'integrate_square',capture)
    with pytest.raises(Captured):module.full_pixel(json.loads(source.read_bytes()),selected_piece=2)


def test_saved_third_piece_segmented_remainder(monkeypatch,record_property):
    from tools import opaque_full_material_reference as module
    from tools.opaque_boundary_diagnostic import diagnose,DiagnosticStopped
    from tools.opaque_shading_intervals import I
    source=Path(os.environ['GFFX_FULL_INPUT']);base=source.parent.parent
    failure=base/'opaque81-20260921/original-campaign1/piece-02-result.json'
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    assert digest(source)=='a15bea40cb5cdead7994d828b57f5dcfac9f44fab0292a06e7d0a7b08719f73a'
    assert digest(failure)=='b236f9c4eb4279e8d340b3a9a11e6cea76d79407e985397116561ffc46064c7d'
    failed=json.loads(failure.read_bytes());assert failed['piece']==2 and failed['details']['face']==5740
    cover=failed['details']['cause']['last_complete_enclosure']
    assert cover['complete_cover'] and cover['leaves']==111
    chosen=cover['error_accounting']['largest_leaves'][:3]
    assert len(chosen)==3
    boxes=[(I(I(Q(r['box'][0])).lo,I(Q(r['box'][1])).hi),I(I(Q(r['box'][2])).lo,I(Q(r['box'][3])).hi)) for r in chosen]
    dest=Path(os.environ['GFFX_FULL_REFERENCE']);assert not dest.exists()
    def encode(v):
        if isinstance(v,I):return [str(v.lo),str(v.hi)]
        raise TypeError(type(v))
    class Captured(Exception):pass
    def capture(field,**kwargs):
        report=None
        import cProfile,pstats,time
        correction=kwargs['correction_integrator'];original=correction.inputs;input_cost={}
        def timed(*args):
            order=max(v.order for v in args);started=time.perf_counter()
            try:return original(*args)
            finally:
                row=input_cost.setdefault(order,dict(calls=0,seconds=0));row['calls']+=1;row['seconds']+=time.perf_counter()-started
        correction.inputs=timed;assert correction.segmented_conditioned is True;prof=cProfile.Profile();prof.enable()
        try:report=diagnose(kwargs['correction_integrator'],boxes,prepare=lambda b:kwargs['prepare'](0,0,b),max_inputs=2048,max_seconds=20,partition_smooth=False)
        except DiagnosticStopped as e:report=e.report;raise
        finally:
            prof.disable();correction.inputs=original
            if report is not None:
                stats=pstats.Stats(prof);costs=[]
                for (file,line,function),(primitive,calls,self_seconds,total_seconds,callers) in stats.stats.items():
                    if 'opaque_' in file:
                        costs.append(dict(file=file,line=line,function=function,calls=calls,self_seconds=self_seconds,total_seconds=total_seconds))
                report['profile']=dict(input_cost=input_cost,functions=sorted(costs,key=lambda r:r['total_seconds'],reverse=True),scope='cProfile instrumented diagnostic;inclusive function times overlap;not a production runtime benchmark')
                report['saved_cover']=dict(cells=cover['cells'],leaves=cover['leaves'],widths=cover['widths'],accounting=cover['error_accounting'])
                report.update(input_sha256=digest(source),failure_sha256=digest(failure),piece=2,largest_leaf_ranks=[1,2,3],scope='Three disjoint largest saved leaves only; both orientations; not a complete-piece certificate')
                dest.write_text(json.dumps(report,indent=2,default=encode),encoding='utf8')
                record_property('boundary_diagnostic',json.dumps(dict(status=report['status'],input_calls=report['input_calls'],witnesses=report['witnesses'],seconds=report.get('seconds'),path=str(dest))))
        assert kwargs['correction_integrator'].smooth_partition is True
        assert report['status']=='diagnostic_only' and len(report['boxes'])==3
        assert all('ordinary_comparison' in a for row in report['boxes'] for a in row['orientations'])
        raise Captured
    monkeypatch.setattr(module,'integrate_square',capture)
    with pytest.raises(Captured):module.full_pixel(json.loads(source.read_bytes()),selected_piece=2)
