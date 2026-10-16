"""Separately measured complete fixed-image certificate audit.
No original visibility/native render or numerical quadrature is replayed.
Check actual partitions, field pullbacks, border values, certificate composition,
closed candidates and every original native colour/mask comparison independently.
"""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
from fractions import Fraction as Q
from route2_common import ROOT as P
sys.path.insert(0,'C:/dev/gffx')
from tools import material_loss_cost as cost
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def _measure_call(fn,label):
 def call(*args,**kwargs):
  from route2_phases import current
  m=current()
  if m is None:return fn(*args,**kwargs)
  with m.measure(label):return fn(*args,**kwargs)
 return call
read=_measure_call(read,'artifact_read_and_JSON_decode')
sha=_measure_call(sha,'artifact_hash_verification')

def main():
 pa=argparse.ArgumentParser();pa.add_argument('--worker',action='store_true');pa.add_argument('--output',type=Path,required=True);pa.add_argument('--launch-index',type=int,required=True);a=pa.parse_args();start=time.perf_counter();d={'status':'working','phase':'admission','worker_pid':os.getpid(),'launch_index':a.launch_index,'native_calls':0,'render_calls':0,'projection_calls':0,'visibility_calls':0,'whole_image_complete':False,'pixels_audited':0,'photo_readiness':False}
 from report_journal142 import Scope
 from route2_phases import Recorder
 with Recorder(a.output) as phases, Scope(cost,a.output):
  from route2_serializer import configure,finish_context
  try:
   cost.write_report(a.output,d);p=read(P/'pipeline-packet1.json');i=a.launch_index-p['base_launch_index'];from route2_admission import task_for_index;task=task_for_index(i,p);assert a.worker and a.output==Path(p['stage'])/task['output'] and read(str(a.output)+'.spec.json')==task['spec']
   from route2_admission import before_arrays,task_for_index
   with phases.measure('worker_prerequisite_verification'):prior=before_arrays(i)
   configure(P,a.launch_index,checkpoint=lambda:cost.check_deadline(start+task['spec']['seconds']))
   scene=prior['complete_visibility_scene']['data'];native=prior['timing_reference'];fields=prior['complete_whole_fields']['data'];chunks=prior['complete_pixel_chunks'];assert len(chunks)==31
   for ref in fields['artifacts'].values():assert sha(ref['path'])==ref['sha256']
   import numpy as np
   import duffy_integral271 as g
   from route2_serializer import dump_exact
   import scene_support289 as support
   from visibility_field_view308 import visibility_field_view
   from tools.opaque_shading_intervals import I
   from whole_runtime274.pixel_fields258 import OwnedWork
   from whole_runtime274.pixel_runtime257.visibility_independent162 import area,intersection,decode
   check=lambda:cost.check_deadline(start+task['spec']['seconds']);o=OwnedWork(9223372036854775807,9223372036854775807,check);support._owned_optional_owner(o,None)
   data=g.decode(read(fields['artifacts']['field-data.json']['path']));view_result=visibility_field_view(decode(read(scene['visibility']['path'])),data['eligible'],checkpoint=check);visibility=view_result['data'];o.hold(data);o.hold(view_result);o.hold(visibility)
   with np.load(native['frame']['path'],allow_pickle=False) as z:comparison={n:z[n].copy() for n in z.files}
   assert data['input_identity']==scene['input_identity']==native['input_identity'] and all(hashlib.sha256(comparison[n].tobytes()).hexdigest()==data['parameter_sha256'][n] for n in data['parameter_sha256'])
   # Independent closed bbox inventory, with zero-area-only contacts included.
   expected=[[] for _ in range(1024)];o.hold(expected)
   import math
   for identifier,(ids,active) in enumerate(zip(data['faces'],data['eligible'])):
    o.step()
    if not active:continue
    t=tuple(data['projected'][j] for j in ids);xmin=min(q[0] for q in t);xmax=max(q[0] for q in t);ymin=min(q[1] for q in t);ymax=max(q[1] for q in t)
    for y in range(max(0,math.ceil(ymin-1)),min(31,math.floor(ymax))+1):
     for x in range(max(0,math.ceil(xmin-1)),min(31,math.floor(xmax))+1):o.step();expected[y*32+x].append(identifier)
   assert tuple(tuple(ids) for ids in expected)==data['atlas']['buckets']
   def cross(a,b,c):return o.rational((b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]))
   def contains(poly,point):
    values=tuple(cross(a,b,point) for a,b in zip(poly,poly[1:]+poly[:1]));o.hold(values);return all(x>=0 for x in values) or all(x<=0 for x in values)
   def interpolate(xy,values,point):
    a,b,c=xy;den=cross(a,b,c)
    if not den:raise ValueError('nondegenerate exact source')
    weights=(cross(point,b,c)/den,cross(a,point,c)/den,cross(a,b,point)/den);o.hold(weights);o.step()
    return o.rational(sum(w*v for w,v in zip(weights,values)))
   all_artifacts=[ref for chunk in chunks for ref in chunk['data']['certificates']];assert sorted(ref['pixel'] for ref in all_artifacts)==list(range(496));o.hold(all_artifacts);begin,end=task['pixels'];assert type(begin) is int and type(end) is int and 0<=begin<end<=1024 and end-begin==16;selected=[];o.hold(selected)
   for ref in all_artifacts:
    o.step()
    if begin<=ref['pixel']<end:selected.append(ref)
   o.hold(tuple(selected));index=[];o.hold(index);max_rgb=max_mask=Q(0);fragments=pieces=boundary_queries=0
   d.update(phase='all-pixel-partition-field-error-native-audit',all_current_complete_and_stopped_chains_before_arrays=True);cost.write_report(a.output,d)
   for ref in sorted(selected,key=lambda q:q['pixel']):
    check();assert sha(ref['path'])==ref['sha256'];r=g.decode(read(ref['path']));o.hold(r);number=ref['pixel'];assert r['pixel']==number and r['field_data_sha256']==fields['artifacts']['field-data.json']['sha256'];y,x=divmod(number,32)
    ids,polys=visibility['input_cells'][number];groups=visibility['visible_cells'][number];assert r['original_input_ids']==ids and r['original_input_polygons']==polys and r['visible_owner_groups']==groups
    flat=tuple((fid,poly) for fid,group in groups for poly in group);assert tuple((q['owner'],q['polygon']) for q in r['fragments'])==flat
    sums=[I(0),I(0),I(0)];mask=Q(0)
    for f in r['fragments']:
     fragments+=1;poly=f['polygon'];mask+=area(poly)/2;prep=f['preparation'];entries=f['entries'];assert prep['source_double_area']==area(poly) and tuple(q['piece'] for q in entries)==prep['pieces'] and sum(q['piece']['det'] for q in entries)==area(poly)
     face=data['faces'][f['owner']];xy=tuple(data['projected'][j][:2] for j in face);normal=tuple(data['normal'][j] for j in face);uv=data['uv'][f['owner']];triangles=[]
     for entry in entries:
      piece=entry['piece'];corners=piece['corners'];pieces+=1;assert area(corners)==piece['det']>0 and all(contains(poly,point) for point in corners);triangles.append(corners)
      for j,point in enumerate(corners):
       assert tuple(interpolate(xy,tuple(n[k] for n in normal),point) for k in range(3))==piece['normal'][j]
       expected_uv=tuple(interpolate(xy,tuple(q[k]*256-Q(1,2) for q in uv),point)-piece['cell'][k] for k in range(2));assert expected_uv==piece['uv'][j]
      cell=piece['cell'];texels=tuple(data['texture'][cell[1]+dy][cell[0]+dx] if 0<=cell[0]+dx<256 and 0<=cell[1]+dy<256 else (Q(0),)*3 for dx,dy in ((0,0),(1,0),(0,1),(1,1)));assert entry['texels']==texels
      # Verify zero support and exact cell-domain hypotheses on every corner.
      if piece['zero_border']:assert any(max(q[k]+cell[k] for q in piece['uv'])<=-1 or min(q[k]+cell[k] for q in piece['uv'])>=256 for k in (0,1))
      else:assert all(0<=q[k]<=1 for q in piece['uv'] for k in (0,1))
      assert entry['geometric_active']==data['geometric_active'][f['owner']] and g.validate_certificate(entry['integral'])
      for k,pair in enumerate(entry['integral']['value']):sums[k]=sums[k]+g.interval(pair)
     for j,t in enumerate(triangles):
      for other in triangles[j+1:]:assert area(intersection(t,other,o.work))==0
    assert tuple(g.endpoints(q) for q in sums)==r['RGB'] and mask==r['exact_mask'] and 0<=mask<=1
    errors=[]
    for k,q in enumerate(sums):
     assert q.width<=I(Q(1,10**12)).hi;v=Q.from_float(float(comparison['colour'][0,y,x,k]));err=max(abs(Q(q.lo)-v),abs(Q(q.hi)-v));assert err<=Q(1,10**10);errors.append(err);max_rgb=max(max_rgb,err)
    mask_error=abs(mask-Q.from_float(float(comparison['mask'][0,y,x])));assert mask_error==r['mask_native_error']<=Q(1,10**10) and tuple(errors)==r['RGB_native_error_upper'];max_mask=max(max_mask,mask_error)
    for point in r['boundary_points']:
     boundary_queries+=1;q=point['point'];cert=point['certificate']['data'];assert cert['point']==q and cert['pixel']==number and cert['candidate_ids']==data['atlas']['buckets'][number];hits=[]
     for identifier in cert['candidate_ids']:
      t=tuple(data['projected'][j] for j in data['faces'][identifier]);xy=tuple(c[:2] for c in t);o.step()
      if contains(xy,q):hits.append((interpolate(xy,tuple(c[2] for c in t),q),identifier))
     assert tuple(hits)==cert['all_closed_support_hits'] and cert['owner']==(None if not hits else min(hits)[1])
    index.append({'pixel':number,'path':ref['path'],'sha256':ref['sha256'],'RGB':r['RGB'],'exact_mask':mask});d.update(pixels_audited=number-begin+1,fragments_audited=fragments,pieces_audited=pieces);cost.write_report(a.output,d)
   expected_fragment_count=0
   for number in range(begin,end):
    o.step()
    for _,poly in visibility['visible_cells'][number]:o.step();o.step();expected_fragment_count+=len(poly)
   assert fragments==expected_fragment_count and len(index)==16 and tuple(q['pixel'] for q in index)==tuple(range(begin,end))
   path=a.output.parent/'whole-image-certificates.json';result={'pixel_range':(begin,end),'pixels':index,'width':32,'height':32,'max_RGB_native_error_upper':max_rgb,'max_mask_native_error':max_mask,'pixel_width_limit':Q(1,10**12),'native_agreement_limit':Q(1,10**10),'input_identity':scene['input_identity'],'field_data_sha256':fields['artifacts']['field-data.json']['sha256'],'fragments':fragments,'pieces':pieces,'boundary_queries':boundary_queries,'independent_nonlinear_upstream':False}
   with path.open('x',encoding='utf-8') as out:dump_exact(result,out)
   history_path=a.output.parent/'whole-audit-history.json'
   with history_path.open('x',encoding='utf-8') as out:dump_exact(tuple(o.history),out)
   d.update(status='passed',phase='complete',audit_history={'path':str(history_path),'sha256':sha(history_path),'bytes':history_path.stat().st_size},whole_image_complete=False,pixel_group_audit_complete=True,pixel_range=(begin,end),complete16_pixels48_RGB_verified=True,complete_exact_visibility_area_and_field_partition=True,whole_rigorous_error_bounds=True,whole_texture_border_normal_lighting_verified=True,closed_boundary_all_candidates_and_global_depth_ties_verified=True,whole_native_colour_and_mask_agreement=False,native_group_colour_and_mask_agreement=True,max_RGB_native_error_upper=g.encode(max_rgb),max_mask_native_error=g.encode(max_mask),certificate_index={'path':str(path),'sha256':sha(path)},audit_work=o.work.count+view_result['work'],audit_charged_owned_bytes=o.storage.used+view_result['charged_owned_bytes'],independent_nonlinear_upstream=False,independent_derivative_accuracy=False);check()
  except Exception as ex:d.update(status='stopped',error=repr(ex));raise
  finally:d['serializer_adoption']=finish_context();d['phase_metrics']=phases.finish();d['seconds']=time.perf_counter()-start;cost.write_report(a.output,d)
if __name__=='__main__':main()
