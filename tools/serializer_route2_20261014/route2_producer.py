"""Fresh complete sixteen-pixel certificate subject; no native replay.
All construction histories are retained during this worker and saved per pixel.
All fixed-state fields come from the separately measured complete construction,
whose exact artifact and typed receipt precede arrays. No baseline perturbation.
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
 pa=argparse.ArgumentParser();pa.add_argument('--worker',action='store_true');pa.add_argument('--output',type=Path,required=True);pa.add_argument('--launch-index',type=int,required=True);a=pa.parse_args();start=time.perf_counter();d={'status':'working','phase':'admission','worker_pid':os.getpid(),'launch_index':a.launch_index,'render_calls':0,'projection_calls':0,'visibility_calls':0,'pixels_complete':0,'whole_image_complete':False,'photo_readiness':False}
 from report_journal142 import Scope
 from route2_phases import Recorder
 with Recorder(a.output) as phases, Scope(cost,a.output):
  from route2_serializer import configure,finish_context
  try:
   cost.write_report(a.output,d);p=read(P/'pipeline-packet1.json');i=a.launch_index-p['base_launch_index'];from route2_admission import task_for_index;task=task_for_index(i,p);assert a.worker and a.output==Path(p['stage'])/task['output'] and read(str(a.output)+'.spec.json')==task['spec']
   from route2_admission import before_arrays,task_for_index
   with phases.measure('worker_prerequisite_verification'):prior=before_arrays(i)
   configure(P,a.launch_index,checkpoint=lambda:cost.check_deadline(start+task['spec']['seconds']))
   scene=prior['complete_visibility_scene']['data'];native=prior['timing_reference'];fields=prior['complete_whole_fields']['data'];assert scene['input_identity']==native['input_identity']==fields['input_identity']
   for ref in [scene['visibility'],native['frame'],fields['artifacts']['field-data.json']]:assert sha(ref['path'])==ref['sha256']
   import numpy as np
   import duffy_integral271 as g
   g.integrate_piece=phases.wrap(g.integrate_piece,'rigorous_piece_integration')
   from route2_serializer import dump_exact
   import scene_support289 as support
   from visibility_field_view308 import visibility_field_view
   from scene_fields272 import ArrayOwner,LIMIT
   from tools.opaque_shading_intervals import I
   from whole_runtime274.pixel_fields258 import prepare_face,OwnedWork
   prepare_face=phases.wrap(prepare_face,'exact_field_partition')
   from whole_runtime274.pixel_runtime257.visibility_independent162 import decode,area
   check=lambda:cost.check_deadline(start+task['spec']['seconds'])
   data=g.decode(read(fields['artifacts']['field-data.json']['path']));view_result=visibility_field_view(decode(read(scene['visibility']['path'])),data['eligible'],checkpoint=check);visibility=view_result['data']
   comparison_owner=ArrayOwner(LIMIT,LIMIT,check)
   with np.load(native['frame']['path'],allow_pickle=False) as z:comparison={n:comparison_owner.hold(np.array(z[n],copy=True,order='C')) for n in z.files}
   comparison_owner.charge(sum(int(q.size) for q in comparison.values()))
   assert data['input_identity']==scene['input_identity'] and all(hashlib.sha256(comparison[n].tobytes()).hexdigest()==data['parameter_sha256'][n] for n in data['parameter_sha256'])
   texture=support.OwnedTexture(data['texture'],checkpoint=check);coord=OwnedWork(LIMIT,LIMIT,check);support._owned_optional_owner(coord,None);coord.hold(data);coord.hold(view_result);coord.hold(visibility);record_list=[];coord.hold(record_list);header=sys.getsizeof(record_list)
   begin,end=task['pixels'];assert type(begin) is int and type(end) is int and 0<=begin<end<=1024 and end-begin==16
   artifacts=[];max_rgb=max_mask=Q(0);fragment_count=piece_count=integral_cells=0;total_work=view_result['work'];total_owned=view_result['charged_owned_bytes']
   d.update(phase='complete-pixel-integrals',pixel_range=(begin,end),all_current_complete_and_stopped_chains_before_arrays=True);cost.write_report(a.output,d)
   for number in range(begin,end):
    y,x=divmod(number,32);groups=visibility['visible_cells'][number];original_ids,original_polys=visibility['input_cells'][number];assert tuple(fid for fid,_ in groups)==original_ids
    sum_rgb=[I(0),I(0),I(0)];mask=Q(0);fragments=[];points={(Q(x),Q(y)),(Q(x+1),Q(y)),(Q(x),Q(y+1)),(Q(x+1),Q(y+1))};coord.hold(('point_set_header',sys.getsizeof(points)))
    for identifier,polygons in groups:
     coord.step()
     for polygon in polygons:
      check();fragment_count+=1;mask+=area(polygon)/2;points.update(polygon)
      for aa,bb in zip(polygon,polygon[1:]+polygon[:1]):points.add(coord.hold(tuple(coord.rational((aa[k]+bb[k])/2) for k in (0,1))))
      ids=data['faces'][identifier];xy=tuple(data['projected'][j][:2] for j in ids);normal=tuple(data['normal'][j] for j in ids)
      prep=prepare_face(xy,normal,data['uv'][identifier],polygon,texture.size,checkpoint=check);coord.hold(prep);total_work+=prep['work'];total_owned+=prep['charged_owned_bytes'];entries=[]
      active=data['geometric_active'][identifier];coeff=data['lighting'] if active else ((Q(0),)*3,)*9
      for piece in prep['pieces']:
       # Total real width <= mask/(2e12); ample outward-sum rounding slack.
       tolerance=piece['det']/Q(8*10**12);texels=texture.texels(piece['cell']);coord.step()
       r=g.integrate_piece(piece,texels,coeff,tolerance=tolerance,checkpoint=check);assert g.validate_certificate(r);coord.hold(r);piece_count+=1;integral_cells+=r['cells'];total_work+=r['work'];total_owned+=r['charged_owned_bytes']
       for k,pair in enumerate(r['value']):sum_rgb[k]=sum_rgb[k]+g.interval(pair)
       entries.append(coord.hold({'piece':piece,'texels':texels,'integral':r,'geometric_active':active}))
      fragments.append(coord.hold({'owner':identifier,'polygon':polygon,'preparation':prep,'entries':tuple(entries)}))
    if not 0<=mask<=1:raise ValueError('complete exact pixel coverage')
    errors=[]
    for k,interval in enumerate(sum_rgb):
     native_value=Q.from_float(float(comparison['colour'][0,y,x,k]));error=max(abs(Q(interval.lo)-native_value),abs(Q(interval.hi)-native_value));errors.append(error);max_rgb=max(max_rgb,error)
     if interval.width>I(Q(1,10**12)).hi or error>Q(1,10**10):raise ValueError('rigorous width/native RGB disagreement',number,k,g.encode(error),str(interval.width))
    mask_error=abs(mask-Q.from_float(float(comparison['mask'][0,y,x])));max_mask=max(max_mask,mask_error)
    if mask_error>Q(1,10**10):raise ValueError('exact coverage/native mask disagreement',number,g.encode(mask_error))
    point_records=[];coord.hold(point_records);coord.hold(tuple(sorted(points)));coord.hold(('point_set_final_header',sys.getsizeof(points)))
    for point in sorted(points):
     r=support.query_boundary_atlas(data['atlas'],number,point,checkpoint=check);coord.hold(r);total_work+=r['work'];total_owned+=r['charged_owned_bytes'];point_records.append({'point':point,'certificate':r})
    record=coord.hold({'pixel':number,'xy':(x,y),'RGB':tuple(g.endpoints(q) for q in sum_rgb),'exact_mask':mask,'RGB_native_error_upper':tuple(errors),'mask_native_error':mask_error,'original_input_ids':original_ids,'original_input_polygons':original_polys,'visible_owner_groups':groups,'fragments':tuple(fragments),'boundary_points':tuple(point_records),'field_data_sha256':fields['artifacts']['field-data.json']['sha256'],'no_positive_area_or_border_piece_omission':True})
    record_list.append(record);n=sys.getsizeof(record_list);coord.storage.used+=max(0,n-header);header=max(header,n);coord.hold(('record_header',header))
    path=a.output.parent/f'pixel-{number:04}.json'
    with path.open('x',encoding='utf-8') as out:dump_exact(record,out)
    artifacts.append({'pixel':number,'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size});d.update(pixels_complete=number-begin+1,latest_complete_pixel=number,fragments=fragment_count,pieces=piece_count,integration_cells=integral_cells);cost.write_report(a.output,d)
   texture_result=texture.result();coord.hold(texture_result);path=a.output.parent/'chunk-history.json'
   # Keep coordinator/query/texture histories as well as every pixel's full trace.
   history={'coordinator_history':coord.hold(tuple(coord.history)),'texture':texture_result,'comparison_ordinary_history':tuple(comparison_owner.o.history),'comparison_array_shapes':{n:tuple(int(x) for x in v.shape) for n,v in comparison.items()},'comparison_array_sha256':{n:hashlib.sha256(v.tobytes()).hexdigest() for n,v in comparison.items()}}
   with path.open('x',encoding='utf-8') as out:dump_exact(history,out)
   d.update(status='passed',phase='complete',pixel_range=(begin,end),certificates=artifacts,chunk_history={'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size},max_RGB_native_error_upper=g.encode(max_rgb),max_mask_native_error=g.encode(max_mask),all_pixel_integral_widths_at_most_1e12_inverse=True,native_forward_agreement_at_most_1e10_inverse=True,complete_closed_boundary_candidates=True,all_histories_saved_losslessly=True,charged_new_work=total_work+coord.work.count+texture_result['work']+comparison_owner.o.work.count,charged_new_owned_bytes=total_owned+coord.storage.used+texture_result['charged_owned_bytes']+comparison_owner.o.storage.used,independent_nonlinear_upstream=False);check()
  except Exception as ex:d.update(status='stopped',error=repr(ex));raise
  finally:d['serializer_adoption']=finish_context();d['phase_metrics']=phases.finish();d['seconds']=time.perf_counter()-start;cost.write_report(a.output,d)
if __name__=='__main__':main()
