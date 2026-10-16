"""Explicit measurement-only CPU output ownership and fixed albedo first VJP.

Ordinary render() remains unchanged. This profile is one fixed scene/material,
one directional light and one constant RGB albedo; it is not a general tile API.
The caller must discard output buffers on StreamFailure. No partial success.
"""
from collections.abc import Mapping
from dataclasses import fields,is_dataclass
import copy,math,time
import torch


class StreamFailure(RuntimeError):
 def __init__(self,message,report):
  super().__init__(message);self.report=report


def _tensors(value):
 from gffx.torch.material import PrincipledBudget
 if isinstance(value,torch.Tensor):yield value
 elif isinstance(value,PrincipledBudget):yield from _tensors(value._data)
 elif isinstance(value,Mapping):
  for child in value.values():yield from _tensors(child)
 elif isinstance(value,(tuple,list)):
  for child in value:yield from _tensors(child)
 elif is_dataclass(value):
  for field in fields(value):yield from _tensors(getattr(value,field.name))


def _profile(model,camera,out,backward):
 from gffx.torch import scene,material,light
 mats=[p for p in model.prims if type(p)==scene.Material]
 lamps=[p for p in model.prims if type(p)==scene.Light]
 meshes=[p for p in model.prims if type(p)==scene.Mesh]
 if len(mats)!=1 or len(lamps)!=1 or len(meshes)!=1 or camera.batch_size!=1:
  raise ValueError('stream profile requires one mesh,material,light and camera')
 mat=mats[0]
 if mat.evaluator is not material.principled or lamps[0].evaluator is not light.directional:
  raise ValueError('stream profile requires Principled and directional illumination')
 if any(type(p) not in (scene.Mesh,scene.Material,scene.Light) for p in model.prims):raise ValueError('unsupported stream scene binding')
 albedo=mat.parameters['albedo']
 if not isinstance(albedo,torch.Tensor) or albedo.shape!=(3,) or albedo.dtype!=torch.float64:
  raise ValueError('stream profile requires constant float64 RGB albedo')
 for name in ('roughness','f0'):
  if not isinstance(mat.parameters[name],torch.Tensor) or mat.parameters[name].ndim!=0:
   raise ValueError('stream profile requires scalar roughness and f0')
 values=list(_tensors(tuple(model.prims)))+list(_tensors(vars(camera)))
 if any(v.requires_grad and v is not albedo for v in values):raise ValueError('only constant albedo backward is qualified')
 if backward and not albedo.requires_grad:raise ValueError('backward requires gradient-bearing albedo')
 h,w=camera.image_height,camera.image_width
 if set(out)!= {'colour','mask'}:raise ValueError('stream requires colour and mask destinations')
 roots={v.untyped_storage().data_ptr() for v in values if v.device.type=='cpu'}
 seen=set()
 for key,shape in (('colour',(1,h,w,3)),('mask',(1,h,w))):
  value=out[key]
  if not isinstance(value,torch.Tensor) or value.device.type!='cpu' or value.dtype!=torch.float64 or value.shape!=shape or not value.is_contiguous() or value.requires_grad:
   raise ValueError('destination must be contiguous CPU float64 with exact shape and no gradient')
  root=value.untyped_storage().data_ptr()
  if root in roots or root in seen:raise ValueError('destination aliases input or another output')
  seen.add(root)
 return albedo,[(v,v._version) for v in values]


def plan_windows(model,camera,*,soft=0.,backward=False,tile_rows=None,workspace_limit_bytes=67108864,group_point_limit=None):
 from gffx.torch._deferred import _plan_workload
 if group_point_limit is not None and (type(group_point_limit)!=int or group_point_limit!=128):raise ValueError('only the existing128-point fixed group profile is qualified')
 reserve=131072 if camera.pose.device.type=='cuda' and backward else 32768
 if type(workspace_limit_bytes)!=int or workspace_limit_bytes<=reserve:raise ValueError('window allowance too small')
 if type(soft) not in (int,float) or not math.isfinite(soft) or soft<0:raise ValueError('invalid soft radius')
 h,w=camera.image_height,camera.image_width
 if tile_rows is not None and (type(tile_rows)!=int or not 1<=tile_rows<=h):raise ValueError('invalid tile rows')
 # This shallow extent proxy is ONLY for the shape-based planner. Actual
 # projection always uses the untouched original camera and native pixel window.
 proxy=copy.copy(camera)
 def check(rows):
  proxy.image_height=rows
  with torch.set_grad_enabled(backward):plan=_plan_workload(model,proxy,1,workspace_limit_bytes-reserve,('colour','mask'),soft=soft)
  if plan['principled_point_block_limit']!=4:raise ValueError('workspace requires an unqualified stream point partition')
  if group_point_limit is not None and plan['principled_group_point_limit']!=group_point_limit:
   raise ValueError('workspace cannot retain the fixed group capacity in this window')
  return plan
 if tile_rows is not None and backward and tile_rows*w<min(4,h*w):raise ValueError("backward window must contain a complete four-point group")
 if tile_rows is None:
  low,high,best=((min(4,h*w)+w-1)//w if backward else 1),h,0
  while low<=high:
   mid=(low+high)//2
   try:check(mid)
   except ValueError as error:
    if 'workspace' not in str(error):raise
    high=mid-1
   else:best=mid;low=mid+1
  if not best:raise ValueError('no window fits the unchanged workspace allowance')
  tile_rows=best
 plan=check(tile_rows)
 return dict(tile_rows=tile_rows,window_count=(h+tile_rows-1)//tile_rows,
  window_allowance_bytes=workspace_limit_bytes,carry_reserve_bytes=reserve,estimate=plan,
  caller_output_bytes=1*h*w*4*8,backward=backward)


def _tiny(records):
 points=len(records['mask']);sample=records['mask']
 return dict(world_position=records['world_position'].reshape(1,1,points,3),
  normal=records['normal'].reshape(1,1,points,3),mask=sample.reshape(1,1,points),
  _covered=torch.ones((1,1,points),dtype=torch.bool,device=sample.device),
  material_index=torch.zeros((1,1,points),dtype=torch.int64,device=sample.device))


def _validate_cohort_inputs(n,v,packet,params,*,_native_checks=False):
 # Private bounded fast path. Every predicate remains; any failed combined
 # flag is diagnosed by the original validators in their original order.
 from gffx.torch import light as lighting
 from gffx.torch.material import PrincipledBudget
 from gffx.torch._principled_groups import _validate_original
 budget=params.get('budget');a=params.get('albedo');r=params.get('roughness');f=params.get('f0')
 eligible=(isinstance(n,torch.Tensor) and n.device.type=='cuda' and n.dtype==torch.float64
  and n.ndim==2 and n.shape[1]==3 and 1<=len(n)<=128
  and isinstance(packet,dict) and set(packet)=={'measure','direction','irradiance','valid'}
  and packet['measure']=='delta' and isinstance(budget,PrincipledBudget))
 if eligible:
  direction,incoming,valid=packet['direction'],packet['irradiance'],packet['valid']
  values=(n,v,direction,incoming,a,r,f)
  eligible=(all(isinstance(x,torch.Tensor) and x.dtype==n.dtype and x.device==n.device for x in values)
   and v.shape==n.shape and direction.shape==(len(n),1,3) and incoming.shape==direction.shape
   and a.shape==(3,) and r.ndim==0 and f.ndim==0
   and isinstance(valid,torch.Tensor) and valid.dtype==torch.bool and valid.device==n.device
   and valid.shape==(len(n),1) and budget._data['roughness_nodes'].device==n.device)
  if eligible:
   lo,hi=budget._metadata['domain']['roughness']
   # The native material and delta-light calls below enforce every finite and
   # physical-domain predicate on every cohort. Resource-specific roughness
   # bounds and all-valid packet eligibility are additional obligations here.
   if _native_checks and not bool((~valid).any() | ((r<lo)|(r>hi)).any()):
    return direction,incoming,valid,None,direction
   # <=1541 doubles; the additional96KiB planner reserve covers flattened
   # expanded fields, concatenation, finite temporaries and all predicate masks.
   finite=torch.isfinite(torch.cat([x.reshape(-1) for x in values])).all()
   domains=torch.cat((a,f.reshape(1),r.reshape(1)))
   lower=domains.new_tensor([0.,0.,0.,0.,lo]);upper=domains.new_tensor([1.,1.,1.,1.,hi])
   invalid=(~valid)[...,None]
   bad=torch.stack((~finite,((domains<lower)|(domains>upper)).any(),(incoming<0).any(),
       (((direction!=0)|(incoming!=0))&invalid).any())).any()
   if not bool(bad):
    safe=torch.where(valid[...,None],direction,direction.new_tensor([0.,0.,1.]))
    return direction,incoming,valid,None,safe
 direction,incoming,valid,omega=lighting._packet(n,packet)
 safe=torch.where(valid[...,None],direction,direction.new_tensor([0.,0.,1.]))
 keys=tuple(sorted(k for k,x in params.items() if isinstance(x,torch.Tensor)))
 static={k:x for k,x in params.items() if k not in keys}
 _validate_original(keys,static,(n,v,safe,*(params[k] for k in keys)))
 return direction,incoming,valid,omega,safe


@torch.no_grad()
def _cohort_partials(packed,camera,groups,cotangent,prepared):
 # Private constant-RGB-albedo profile only; stream_into owns admission and
 # mutation checks. Keep original <=4-point layouts and full native derivatives.
 from gffx.torch import _deferred as d,light as lighting
 from gffx.torch._principled_groups import _validate_original,_prepare_material_curve,_constant_layout
 from gffx.torch.material import _principled_layout
 from gffx.torch._principled_native import _error
 from gffx.torch._common import materialize
 from gffx.torch._reference_channels import sum3,_sum_last
 params=packed.materials[0].parameters;lamp=packed.lights[0];budget=params['budget']
 keys=tuple(sorted(k for k,v in params.items() if isinstance(v,torch.Tensor)))
 static={k:v for k,v in params.items() if k not in keys}
 sizes=[len(record['mask']) for record in groups]
 if not 1<=len(sizes)<=32 or any(not 1<=size<=4 for size in sizes):
  raise ValueError('internal groups require nonempty four-point layouts')
 n=torch.cat([record['normal'] for record in groups])
 points=torch.cat([record['world_position'] for record in groups])
 mask=torch.cat([record['mask'] for record in groups])
 v=camera.pose[0,:3,3].to(dtype=points.dtype)-points
 # This state belongs to one stream invocation whose inputs are guarded.
 # A directional packet depends on the fixed light and the point count only.
 # At most eight samples retain <512 bytes in the existing carry reserve.
 if prepared is not None and lamp.evaluator is lighting.directional and lamp.parameters['directions'].shape[-2]<=8:
  if 'light_packet' not in prepared:prepared['light_packet']=d._light_packet(lamp,packed,0,points[:1])
  base=prepared['light_packet']
  packet={key:(value.expand((len(points),)+value.shape[1:]) if isinstance(value,torch.Tensor) else value) for key,value in base.items()}
 else:packet=d._light_packet(lamp,packed,0,points)
 direction,incoming,valid,omega,safe=_validate_cohort_inputs(n,v,packet,params,_native_checks=True)
 def fail(error):
  # Re-establish the original first Python diagnostic before propagating a
  # native/layout failure. Failed calls never expose their temporary outputs.
  _validate_cohort_inputs(n,v,packet,params)
  if isinstance(error,RuntimeError):raise _error(error) from error
  raise error
 try:nn,vv,ll,a,r,f=_constant_layout(n,v,safe,params,sizes,True)
 except (RuntimeError,ValueError,TypeError) as error:fail(error)
 try:
  if prepared is None:cn,curve,partial,mean,width=_prepare_material_curve(budget,r)
  else:
   if 'curve' not in prepared:prepared['curve']=_prepare_material_curve(budget,r[:1])
   cn,curve,partial,mean,width=prepared['curve']
  # The fixed profile's original roughness is scalar. The shared native entry
  # validates the same values and computes all 16 Jacobian columns, while
  # reading this one call-owned table instead of copying it for each point.
  operation=(torch.ops.gffx.shading_principled_shared_groups if prepared is not None
             else torch.ops.gffx.shading_principled_groups)
  brdf,jacobian=operation(cn[1:],budget._data['_clog'],
   curve,partial,mean,width,nn,vv,ll,a,r,f,params.get('two_sided',False),True)
 except (RuntimeError,ValueError,TypeError) as error:fail(error)
 oriented=n
 if params.get('two_sided',False):
  sign=torch.where(sum3(n*v)<0,-torch.ones_like(n[:,0]),torch.ones_like(n[:,0]));oriented=n*sign[:,None]
 # Fixed directional packets have valid samples only. The bounded native
 # operation checks the complete lighting value as well as its BRDF-only VJP.
 # Material derivatives and all sixteen-column finite checks remain below.
 colour_cotangent=(torch.ones_like(points)*cotangent)*mask[:,None]
 # The eager reduction VJP adds zero for Q>1, whereas this historical helper
 # expands its cotangent directly. They coincide for nonzero cotangents; keep
 # the original path for any multi-sample zero, including negative zero.
 fused=(lamp.evaluator is lighting.directional and omega is None and direction.shape[1]<=8
        and (direction.shape[1]==1 or bool((colour_cotangent!=0).all())))
 if fused:
  try:
   _,gradient=torch.ops.gffx.shading_delta_brdf(*(materialize(x) for x in (oriented,brdf,direction,incoming,colour_cotangent)))
  except (RuntimeError,ValueError,TypeError) as error:fail(error)
 else:
  # Execute the original lighting forward, preserving every validation/finite
  # check. Its only differentiable input here is BRDF. Reverse its exact factors
  # in order, without a per-cohort autograd graph or cold engine setup.
  value=points*0
  for key in keys:value=value+params[key].reshape(-1)[:0].sum()
  value=value+lighting.integrate(oriented,brdf,packet)
  rgb=value*mask[:,None]
  normal=lighting._unit(oriented,'normals');unit=lighting._unit(safe,'direction')
  cosine=sum3(normal[...,None,:].expand(unit.shape)*unit)
  active=valid & (cosine>0);cosine=torch.where(active,cosine,torch.zeros_like(cosine))
  gradient=((torch.ones_like(rgb)*cotangent)*mask[:,None])[:,None,:].expand_as(brdf)
  if omega is not None:gradient=gradient*torch.where(active,omega,0.)[...,None]
  gradient=gradient*cosine[...,None]
  gradient=gradient*torch.where(active[...,None],incoming,0.)
  gradient=torch.where(active[...,None],gradient,0.)
 try:point,light=torch.ops.gffx.shading_principled_groups_backward(jacobian,materialize(gradient))
 except (RuntimeError,ValueError,TypeError) as error:fail(error)
 # Same _Expand adjacent-pair tree inside EACH original group. No cross-group
 # sum: stream_into still accumulates these in declared global reverse order.
 result=[];left=0;group=0
 while group<len(sizes):
  size=sizes[group];end=group+1
  while end<len(sizes) and sizes[end]==size:end+=1
  count=end-group
  row=point[left:left+count*size,9:12].contiguous().reshape(count,size,3)
  # Each independent row retains the same adjacent-pair tree. No padding,
  # shared-axis sum, or changed order for the short leading group.
  result.extend(_sum_last(row.transpose(1,2)).unbind(0))
  left+=count*size;group=end
 # The first original albedo leaf also received the empty zero-connection edge.
 result[0]=result[0]+torch.zeros_like(result[0])
 return tuple(result)

def _accumulate_partials(carry,parts):
 # The existing32KiB carry reserve covers these <=32 four-lane rows and
 # temporaries. Native ordered RGBA adds each row to the incoming carry;
 # reducing a subtotal first would change rounding and is forbidden.
 if len(parts)>32:raise ValueError('at most32 ordered gradient groups')
 if not parts:return
 if carry.shape!=(3,) or any(p.shape!=(3,) or p.dtype!=carry.dtype or p.device!=carry.device for p in parts):
  raise ValueError('ordered gradient carry requires matching RGB groups')
 rgba=torch.cat((carry,carry.new_zeros(1)))
 values=torch.cat((torch.stack(parts),carry.new_zeros((len(parts),1))),dim=1)
 result=torch.ops.gffx.shading_ordered_rgba(rgba,values)
 carry.copy_(result[:3])


def _own_pending_groups(queue,pending):
 # At a window boundary at most31 complete four-point groups plus one
 # incomplete group remain. New and old copies together fit the32KiB reserve.
 count=sum(len(record['mask']) for record in queue)+(0 if pending is None else len(pending['mask']))
 if len(queue)>31 or count>127:raise ValueError('bounded pending group ownership exceeded')
 queue[:]=[{key:value.clone() for key,value in record.items()} for record in queue]
 return None if pending is None else {key:value.clone() for key,value in pending.items()}


def stream_into(model,camera,out,*,soft=0.,backward=False,tile_rows=None,
                workspace_limit_bytes=67108864,checkpoint=lambda progress:None,group_point_limit=None):
 from gffx.torch import _deferred as d
 report=dict(status='working',complete=False,completed_windows=0,recomputed_windows=0,material_groups=0)
 start=time.perf_counter()
 try:
  albedo,versions=_profile(model,camera,out,backward)
  plan=plan_windows(model,camera,soft=soft,backward=backward,tile_rows=tile_rows,workspace_limit_bytes=workspace_limit_bytes,group_point_limit=group_point_limit)
  report['plan']=plan
  source_ids=tuple(id(v) for v,_ in versions);grad_flags=tuple(v.requires_grad for v,_ in versions)
  prim_ids=tuple(id(p) for p in model.prims)
  metadata=tuple(getattr(camera,k) for k in ('image_height','image_width','near','far','batch_size'))
  destination_ids={k:id(v) for k,v in out.items()};destination_versions={k:v._version for k,v in out.items()}
  def check():
   checkpoint(dict(report))
   current=list(_tensors(tuple(model.prims)))+list(_tensors(vars(camera)))
   if tuple(id(v) for v in current)!=source_ids or tuple(id(p) for p in model.prims)!=prim_ids or tuple(getattr(camera,k) for k in ('image_height','image_width','near','far','batch_size'))!=metadata:
    raise ValueError('stream input binding changed during evaluation')
   if any(v._version!=version for v,version in versions) or tuple(v.requires_grad for v in current)!=grad_flags:
    raise ValueError('stream input changed during evaluation')
   if set(out)!=set(destination_ids) or any(id(out[k])!=destination_ids[k] or out[k]._version!=destination_versions[k] for k in destination_ids):
    raise ValueError('stream destination changed outside the writer')
  check()
  with torch.no_grad():packed,actual_camera=d._prepare(model,camera);d._validate_bindings(packed)
  h,w=camera.image_height,camera.image_width;step=plan['tile_rows'];counts=[]
  windows=[(row,min(step,h-row)) for row in range(0,h,step)]
  for row,height in windows:
   check()
   with torch.no_grad():
    buf=d._buffers(packed,actual_camera,soft,2,window=(row,0,height,w))
    colour=d._shade(packed,actual_camera,buf,point_block_limit=plan['estimate']['principled_point_block_limit'],group_point_limit=plan['estimate']['principled_group_point_limit'])
    out['colour'][:,row:row+height].copy_(colour);out['mask'][:,row:row+height].copy_(buf['mask'])
    destination_versions.update({k:v._version for k,v in out.items()})
    counts.append(int(buf['_covered'].sum()))
    del colour,buf
   report['completed_windows']+=1
  check();report['covered_points']=sum(counts);gradient=None
  if backward:
   gradient=torch.zeros_like(albedo);need=sum(counts)%4 or 4;pending=None
   cotangent=albedo.new_ones(())/(3*h*w)
   from gffx.torch._principled_groups import capacity
   params=packed.materials[0].parameters;q=packed.lights[0].parameters['directions'].shape[-2]
   cohort=capacity(params['budget'],4,q,True,plan['window_allowance_bytes']-plan['carry_reserve_bytes'])
   cohort=min(cohort,plan['estimate']['principled_group_point_limit']//4,max(1,plan['tile_rows']*w//4))
   queue=[];prepared={}
   def flush():
    _accumulate_partials(gradient,_cohort_partials(packed,actual_camera,queue,cotangent,prepared))
    queue.clear()
   for (row,height),expected in zip(reversed(windows),reversed(counts)):
    check()
    with torch.no_grad():
     buf=d._buffers(packed,actual_camera,soft,2,window=(row,0,height,w));selected=buf['_covered']
     if int(selected.sum())!=expected:raise ValueError('window ownership changed during recomputation')
     records={key:buf[key][selected] for key in ('world_position','normal','mask')}
     del buf,selected
    right=expected
    while right:
     have=0 if pending is None else len(pending['mask']);take=min(need-have,right)
     pending={key:torch.cat((value[right-take:right],pending[key])) if pending is not None else value[right-take:right] for key,value in records.items()}
     right-=take
     if len(pending['mask'])==need:
      queue.append(pending);report['material_groups']+=1;pending=None;need=4
      if len(queue)==cohort:flush()
    pending=_own_pending_groups(queue,pending)
    del records
    report['recomputed_windows']+=1
   if pending is not None:raise ValueError('incomplete global material group')
   if queue:flush()
  check()
  if any(not bool(torch.isfinite(v).all()) for v in out.values()) or gradient is not None and not bool(torch.isfinite(gradient).all()):raise ValueError('nonfinite stream output')
  report.update(status='passed',complete=True,elapsed_seconds=time.perf_counter()-start)
  return report,gradient
 except Exception as error:
  report.update(status='stopped',complete=False,error=repr(error),elapsed_seconds=time.perf_counter()-start)
  raise StreamFailure(str(error),report) from error
