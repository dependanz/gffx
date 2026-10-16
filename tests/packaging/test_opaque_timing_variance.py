"""Bounded identical-region timing investigation; no full-piece admission."""
import json,hashlib,os,time,importlib.util,cProfile,pstats
from pathlib import Path
from fractions import Fraction as Q
import pytest

def test_second_piece_same_regions_old_new_timing(monkeypatch,record_property):
 from tools import opaque_full_material_reference as m,opaque_peano as p
 from tools.opaque_shading_intervals import I,Jet
 src=Path(os.environ['GFFX_FULL_INPUT']);base=src.parent.parent;old=base/'opaque96-20260923/source-final/tools/opaque_peano_material.py';failed=base/'opaque97-20260923/original-campaign1/piece-01-result.json'
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 assert sha(failed)=='bb9daf2cd73d8dd4e401566cbf44181b579b112c8ee1af202690cfc88018f09f'
 assert sha(src)=='a15bea40cb5cdead7994d828b57f5dcfac9f44fab0292a06e7d0a7b08719f73a'
 manifest=json.loads((base/'opaque96-20260923/final-source1.json').read_bytes());assert sha(old)==next(r['sha256'] for r in manifest['source'] if r['path']=='tools/opaque_peano_material.py')
 spec=importlib.util.spec_from_file_location('old_material',old);legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
 saved=json.loads(failed.read_bytes());cover=saved['details']['cause']['last_complete_enclosure'];assert cover['cells']==95
 boxes=[(I(I(Q(r['box'][0])).lo,I(Q(r['box'][1])).hi),I(I(Q(r['box'][2])).lo,I(Q(r['box'][3])).hi)) for r in cover['error_accounting']['largest_leaves'][:3]]
 dest=Path(os.environ['GFFX_FULL_REFERENCE']);assert not dest.exists();start=time.perf_counter();cpu=time.process_time();report=dict(status='running',max_inputs=2048,max_seconds=20,inputs=0,runs=[],failure_sha256=sha(failed),legacy_sha256=sha(old),piece=1)
 class Stop(Exception):pass
 def check():assert time.perf_counter()-start<20;assert report['inputs']<=2048
 def capture(field,**kw):
  c=kw['correction_integrator'];orig=c.inputs;prepare=kw['prepare'];cell=prepare.__closure__[prepare.__code__.co_freevars.index('inputs')];axis=c._axis_bounds;nodes,weights=p.gauss5();prof=cProfile.Profile()
  def inputs(*args):
   check();assert report['inputs']<2048;report['inputs']+=1;return orig(*args)
  cell.cell_contents=inputs;c.inputs=inputs
  def encode(v):
   if isinstance(v,I):return [str(v.lo),str(v.hi)]
   if isinstance(v,Jet):return {str(k):encode(q) for k,q in sorted(v.c.items())}
   if isinstance(v,(list,tuple)):return [encode(q) for q in v]
   return v
  try:
   for mode in ['old','new','new','old']:
    c._axis_bounds=legacy.MaterialCorrection._axis_bounds.__get__(c) if mode=='old' else axis;p._kernel_local_mass_cached.cache_clear();p._kernel_local_ranges_cached.cache_clear();rows=[];t=time.perf_counter();cp=time.process_time()
    for box in boxes:
     row={};results=[]
     def timed(label,fn):
      check();at=time.perf_counter();ct=time.process_time();v=fn();row[label]=dict(wall=time.perf_counter()-at,cpu=time.process_time()-ct);results.append(encode(v));return v
     timed('prepare',lambda:prepare(0,0,box))
     timed('smooth_nodes',lambda:[field(Jet(I(box[0].lo)+(I(box[0].hi)-I(box[0].lo))*x),Jet(I(box[1].lo)+(I(box[1].hi)-I(box[1].lo))*y)) for x in nodes for y in nodes])
     timed('smooth_derivatives',lambda:[field(Jet.variable(box[0],order=10),Jet(box[1],order=10)),field(Jet(box[0],order=10),Jet.variable(box[1],order=10))])
     if mode=='new' and len(report['runs'])==1:prof.enable()
     timed('correction',lambda:c(box))
     prof.disable()
     timed('guidance_derivatives',lambda:[kw['refinement_field'](Jet.variable(box[0],order=10),Jet(box[1],order=10)),kw['refinement_field'](Jet(box[0],order=10),Jet.variable(box[1],order=10))])
     row['results_sha256']=hashlib.sha256(json.dumps(results,sort_keys=True).encode()).hexdigest();rows.append(row)
    report['runs'].append(dict(mode=mode,wall=time.perf_counter()-t,cpu=time.process_time()-cp,regions=rows,profiled=mode=='new' and len(report['runs'])==1))
   assert all([x['results_sha256'] for x in r['regions']]==[x['results_sha256'] for x in report['runs'][0]['regions']] for r in report['runs'])
   report['profile']=[dict(file=f,line=l,function=n,calls=c,self_seconds=s,total_seconds=t) for (f,l,n),(pc,c,s,t,callers) in pstats.Stats(prof).stats.items() if 'opaque_' in f];report['profile'].sort(key=lambda x:x['total_seconds'],reverse=True);report['status']='passed'
  finally:prof.disable();cell.cell_contents=orig;c.inputs=orig;c._axis_bounds=axis
  raise Stop
 monkeypatch.setattr(m,'integrate_square',capture)
 try:
  with pytest.raises(Stop):m.full_pixel(json.loads(src.read_bytes()),selected_piece=1)
 finally:
  report.update(seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu);dest.write_text(json.dumps(report,indent=2));record_property('matched_timing',json.dumps(report))


def test_lookup_same_regions_with_remaining_budget(monkeypatch,record_property):
 from tools import opaque_full_material_reference as m,opaque_peano as p
 from tools.opaque_shading_intervals import I,Jet
 src=Path(os.environ['GFFX_FULL_INPUT']);base=src.parent.parent;old=base/'opaque96-20260923/source-final/tools/opaque_peano_material.py';failed=base/'opaque97-20260923/original-campaign1/piece-01-result.json'
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 assert sha(failed)=='bb9daf2cd73d8dd4e401566cbf44181b579b112c8ee1af202690cfc88018f09f'
 assert sha(src)=='a15bea40cb5cdead7994d828b57f5dcfac9f44fab0292a06e7d0a7b08719f73a'
 manifest=json.loads((base/'opaque96-20260923/final-source1.json').read_bytes());assert sha(old)==next(r['sha256'] for r in manifest['source'] if r['path']=='tools/opaque_peano_material.py')
 spec=importlib.util.spec_from_file_location('old_material',old);legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
 saved=json.loads(failed.read_bytes());cover=saved['details']['cause']['last_complete_enclosure'];assert cover['cells']==95
 boxes=[(I(I(Q(r['box'][0])).lo,I(Q(r['box'][1])).hi),I(I(Q(r['box'][2])).lo,I(Q(r['box'][3])).hi)) for r in cover['error_accounting']['largest_leaves'][:3]]
 dest=Path(os.environ['GFFX_FULL_REFERENCE']);assert not dest.exists();start=time.perf_counter();cpu=time.process_time();prior_path=base/'opaque98-20260923/timing1-reference.json';prior=json.loads(prior_path.read_bytes());assert prior['inputs']==1244 and prior['status']=='passed';report=dict(status='running',max_inputs=2048,max_seconds=20,inputs=0,runs=[],failure_sha256=sha(failed),legacy_sha256=sha(old),piece=1,prior_sha256=sha(prior_path),prior_inputs=prior['inputs'],prior_seconds=prior['seconds'])
 class Stop(Exception):pass
 def check():assert time.perf_counter()-start+prior['seconds']<20;assert report['inputs']+prior['inputs']<=2048
 def capture(field,**kw):
  c=kw['correction_integrator'];orig=c.inputs;prepare=kw['prepare'];cell=prepare.__closure__[prepare.__code__.co_freevars.index('inputs')];axis=c._axis_bounds;nodes,weights=p.gauss5();prof=cProfile.Profile()
  def inputs(*args):
   check();assert report['inputs']+prior['inputs']<2048;report['inputs']+=1;return orig(*args)
  cell.cell_contents=inputs;c.inputs=inputs
  def encode(v):
   if isinstance(v,I):return [str(v.lo),str(v.hi)]
   if isinstance(v,Jet):return {str(k):encode(q) for k,q in sorted(v.c.items())}
   if isinstance(v,(list,tuple)):return [encode(q) for q in v]
   return v
  try:
   for mode in ['new','new']:
    c._axis_bounds=legacy.MaterialCorrection._axis_bounds.__get__(c) if mode=='old' else axis;p._kernel_local_mass_cached.cache_clear();p._kernel_local_ranges_cached.cache_clear();rows=[];t=time.perf_counter();cp=time.process_time()
    for box in boxes:
     row={};results=[]
     def timed(label,fn):
      check();at=time.perf_counter();ct=time.process_time();v=fn();row[label]=dict(wall=time.perf_counter()-at,cpu=time.process_time()-ct);results.append(encode(v));return v
     timed('prepare',lambda:prepare(0,0,box))
     timed('smooth_nodes',lambda:[field(Jet(I(box[0].lo)+(I(box[0].hi)-I(box[0].lo))*x),Jet(I(box[1].lo)+(I(box[1].hi)-I(box[1].lo))*y)) for x in nodes for y in nodes])
     timed('smooth_derivatives',lambda:[field(Jet.variable(box[0],order=10),Jet(box[1],order=10)),field(Jet(box[0],order=10),Jet.variable(box[1],order=10))])
     if False:prof.enable()
     timed('correction',lambda:c(box))
     prof.disable()
     timed('guidance_derivatives',lambda:[kw['refinement_field'](Jet.variable(box[0],order=10),Jet(box[1],order=10)),kw['refinement_field'](Jet(box[0],order=10),Jet.variable(box[1],order=10))])
     row['results_sha256']=hashlib.sha256(json.dumps(results,sort_keys=True).encode()).hexdigest();rows.append(row)
    report['runs'].append(dict(mode=mode,wall=time.perf_counter()-t,cpu=time.process_time()-cp,regions=rows,profiled=False))
   assert all([x['results_sha256'] for x in r['regions']]==[x['results_sha256'] for x in report['runs'][0]['regions']] for r in report['runs'])
   assert all([x['results_sha256'] for x in r['regions']]==[x['results_sha256'] for x in prior['runs'][0]['regions']] for r in report['runs'])
   report['profile']=[];report['status']='passed'
  finally:prof.disable();cell.cell_contents=orig;c.inputs=orig;c._axis_bounds=axis
  raise Stop
 monkeypatch.setattr(m,'integrate_square',capture)
 try:
  with pytest.raises(Stop):m.full_pixel(json.loads(src.read_bytes()),selected_piece=1)
 finally:
  report.update(seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu);dest.write_text(json.dumps(report,indent=2));record_property('matched_timing',json.dumps(report))


def test_completed_lookup_diagnostic_is_auditable_without_rerun():
 src=Path(os.environ['GFFX_FULL_INPUT']);o=src.parent.parent/'opaque98-20260923';p=o/'lookup-cost1-reference.json';assert hashlib.sha256(p.read_bytes()).hexdigest()=='eb70ee61f2e2a52b40f62cbf4431a850611e05799e472365c673d1e6415f4a56'
 r=json.loads(p.read_bytes());prior=json.loads((o/'timing1-reference.json').read_bytes());assert len(r['runs'])==2
 assert r['inputs']+prior['inputs']==1866 and r['seconds']+prior['seconds']<20
 for run in r['runs']:
  assert run['profiled'] is False
  assert [x['results_sha256'] for x in run['regions']]==[x['results_sha256'] for x in prior['runs'][0]['regions']]
 assert r['prior_sha256']==hashlib.sha256((o/'timing1-reference.json').read_bytes()).hexdigest()
