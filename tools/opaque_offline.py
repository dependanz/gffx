"""D569 offline original-frame qualification; host-owned, never a speed anchor.

No cropped cameras, range-subtotal additions, implicit retries or mutable receipts.
Rendering stays in the installed library. Each worker owns at most32768pixels.
"""
import hashlib,json,math,os,struct,time
from pathlib import Path
from tools.opaque_frame_store import Writer,Reader,CHANNELS,_encode
PROFILE='offline-global-row-albedo-v1'
MAX_RANGE=32768
DISK_CAP=1024**3
ZERO=struct.pack('<3d',0.,0.,0.).hex()


def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(65536),b''):h.update(b)
 return h.hexdigest()


def seal(path,value):
 path=Path(path);raw=_encode(value)
 with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 return {'path':path.as_posix(),'sha256':hashlib.sha256(raw).hexdigest()}


def read(ref):
 p=Path(ref['path'])
 if p.is_symlink() or p.stat().st_size>1048576:raise ValueError('receipt type/size')
 raw=p.read_bytes()
 if hashlib.sha256(raw).hexdigest()!=ref['sha256']:raise ValueError('receipt hash')
 return json.loads(raw)


def carry_bytes(value):
 try:raw=bytes.fromhex(value);numbers=struct.unpack('<3d',raw)
 except (ValueError,TypeError,struct.error):raise ValueError('exact24byte carry required')
 if any(not math.isfinite(x) for x in numbers):raise ValueError('finite carry required')
 return raw


def make_plan(width,height,identity,*,cotangent=None,row_cap=None,cull_mode=2,initial_carry_hex=ZERO):
 if any(type(x)!=int or x<1 for x in (width,height)) or width*height>1920*1080 or width>MAX_RANGE:raise ValueError('bounded frame dimensions')
 rows=MAX_RANGE//width if row_cap is None else row_cap
 if type(rows)!=int or not 1<=rows<=MAX_RANGE//width:raise ValueError('range pixel cap')
 cotangent=list(cotangent) if cotangent is not None else [1/(width*height*3)]*3+[0.]
 p=dict(version=1,profile=PROFILE,width=width,height=height,identity=identity,cotangent=cotangent,cull_mode=cull_mode,initial_carry_hex=initial_carry_hex,row_cap=rows,ranges=[dict(index=i,y=y,rows=min(rows,height-y),first=y*width,count=min(rows,height-y)*width) for i,y in enumerate(range(0,height,rows))])
 validate_plan(p);return p


def validate_plan(p):
 if p.get('version')!=1 or p.get('profile')!=PROFILE:raise ValueError('offline profile')
 w,h,cap=p['width'],p['height'],p['row_cap']
 if any(type(v)!=int or v<1 for v in (w,h,cap)) or w*h>1920*1080 or w*cap>MAX_RANGE:raise ValueError('range extent')
 expected=[dict(index=i,y=y,rows=min(cap,h-y),first=y*w,count=min(cap,h-y)*w) for i,y in enumerate(range(0,h,cap))]
 if p['ranges']!=expected or len(expected)>64:raise ValueError('range order/extent')
 if type(p['cull_mode'])!=int or p['cull_mode'] not in (1,2,3):raise ValueError('cull mode')
 if len(p['cotangent'])!=4 or any(type(x) not in (int,float) or not math.isfinite(x) for x in p['cotangent']):raise ValueError('cotangent')
 carry_bytes(p['initial_carry_hex']);_encode(p)


def _stable(value):
 # Process addresses are not input identity. Constants, callable names and all
 # tensor bytes ARE. Unsupported constants fail closed rather than use repr.
 from gffx.torch.material import PrincipledBudget
 if isinstance(value,PrincipledBudget):return ['PrincipledBudget',_stable(value._metadata)]
 if isinstance(value,type):return ['type',value.__module__,value.__qualname__]
 if callable(value):return ['callable',value.__module__,value.__qualname__]
 if isinstance(value,dict):return [[_stable(k),_stable(v)] for k,v in value.items()]
 if isinstance(value,(tuple,list)):return [_stable(v) for v in value]
 if value is None or type(value) in (str,int,float,bool):return value
 raise TypeError('unsupported stable input constant '+str(type(value)))


def input_identity(scene,camera):
 from tools.opaque_frame_execution import _input_hashes
 from gffx.torch._opaque_stream import _snapshot
 tensors=[];tree=_snapshot((scene,camera,None),tensors,{})
 return dict(tensors=_input_hashes(tensors),structure=_stable(tree))


def verify_prefix(plan,refs,*,deadline=None,cancel=None):
 from tools.opaque_frame_execution import _check
 validate_plan(plan)
 if len(refs)>len(plan['ranges']):raise ValueError('excess ranges')
 ph=hashlib.sha256(_encode(plan)).hexdigest();carry=plan['initial_carry_hex'];prev=None;inputs=None;streams={k:hashlib.sha256() for k in CHANNELS};covered=0
 for i,ref in enumerate(refs):
  if deadline is not None:_check(deadline,cancel)
  r=read(ref);extent=plan['ranges'][i]
  if r.get('status')!='passed' or r.get('plan_sha256')!=ph or r.get('range')!=extent or r.get('previous')!=prev or r.get('carry_in_hex')!=carry:raise ValueError('range prefix/plan/carry differs')
  carry_bytes(r['carry_out_hex'])
  if inputs is not None and r['inputs']!=inputs:raise ValueError('range inputs differ')
  inputs=r['inputs'];expected=dict(profile=PROFILE,plan_sha256=ph,range=extent,inputs=inputs)
  receipt=r['store'];reader=Reader(receipt['root'],receipt['manifest_sha256'],expected)
  if reader.manifest['width']!=plan['width'] or reader.manifest['height']!=extent['rows']:raise ValueError('range store dimensions')
  import numpy as np
  for offset,count,values in reader.blocks():
   if deadline is not None:_check(deadline,cancel)
   for k,raw in values.items():
    a=np.frombuffer(raw,dtype='<f8')
    if not np.isfinite(a).all():raise ValueError('nonfinite stored value')
    if k=='mask':
     if not ((a>=0)&(a<=1+1e-12)).all():raise ValueError('coverage bounds')
     covered+=int((a>0).sum())
    if k in ('gcolour','gmask'):
     expected_cot=struct.pack('<3d',*plan['cotangent'][:3]) if k=='gcolour' else struct.pack('<d',plan['cotangent'][3])
     if raw!=expected_cot*count:raise ValueError('cotangent bytes')
    streams[k].update(raw)
  prev=ref;carry=r['carry_out_hex']
 return dict(next_index=len(refs),carry_hex=carry,previous=prev,inputs=inputs,covered_pixels=covered,streams={k:dict(sha256=h.hexdigest(),bytes=sum(r['count'] for r in plan['ranges'][:len(refs)])*CHANNELS[k]*8) for k,h in streams.items()})


def run_range(scene,camera,root,plan,index,refs,*,seconds=120,cancel=None):
 import torch
 from gffx.torch import render
 from tools.opaque_frame_execution import _snapshot,_versions,_guard,_check
 if type(seconds) not in (int,float) or not math.isfinite(seconds) or not 0<seconds<=120:raise ValueError('bounded worker deadline')
 deadline=time.perf_counter()+seconds;_check(deadline,cancel)
 prefix=verify_prefix(plan,refs,deadline=deadline,cancel=cancel)
 if type(index)!=int or index!=prefix['next_index'] or not 0<=index<len(plan['ranges']):raise ValueError('next range required')
 if camera.image_width!=plan['width'] or camera.image_height!=plan['height']:raise ValueError('full global camera required')
 inputs=input_identity(scene,camera)
 if prefix['inputs'] is not None and prefix['inputs']!=inputs:raise ValueError('inputs differ from prefix')
 from gffx.torch._opaque_albedo_replay import _profile
 target=_profile(scene,camera);tensors,structure=_snapshot(scene,camera);versions=_versions(tensors)
 active=[t for t in tensors if t.requires_grad]
 if len(active)!=1 or active[0] is not target:raise ValueError('exactly constant albedo target required')
 extent=plan['ranges'][index];ph=hashlib.sha256(_encode(plan)).hexdigest();pinned=dict(profile=PROFILE,plan_sha256=ph,range=extent,inputs=inputs)
 carry=target.new_tensor(struct.unpack('<3d',carry_bytes(prefix['carry_hex'])))
 cot=plan['cotangent'];w=plan['width'];offset=0;_check(deadline,cancel)
 writer=Writer(root,w,extent['rows'],pinned,block_pixels=256)
 try:
  while offset<extent['count']:
   _check(deadline,cancel);_guard(scene,camera,versions,structure)
   gy,x=divmod(extent['first']+offset,w);n=min(256,w-x,extent['count']-offset);region=(gy,x,1,n)
   with torch.no_grad():out=render.opaque_window(scene,camera,region,cull_mode=plan['cull_mode'],tile_pixels=16)
   values={k:out[k].cpu().contiguous().numpy().tobytes() for k in ('colour','mask')};del out
   gc=target.new_tensor(cot[:3]).expand(1,1,n,3).contiguous();gm=target.new_full((1,1,n),cot[3])
   replay=render.opaque_albedo_replay(scene,camera,region,gc,gm,initial_gradient=carry,cull_mode=plan['cull_mode'])
   for k in ('colour','mask'):
    if replay[k].cpu().contiguous().numpy().tobytes()!=values[k]:raise ValueError('recomputed pixels differ')
   carry=replay['albedo_gradient'];del replay,gc,gm
   values.update(gcolour=struct.pack('<3d',*cot[:3])*n,gmask=struct.pack('<d',cot[3])*n)
   writer.append(offset,n,values);offset+=n
  _check(deadline,cancel);_guard(scene,camera,versions,structure)
  receipt=writer.finish();carry_hex=carry.cpu().contiguous().numpy().tobytes().hex();carry_bytes(carry_hex)
  _check(deadline,cancel)
  result=dict(status='passed',plan_sha256=ph,range=extent,inputs=inputs,previous=prefix['previous'],carry_in_hex=prefix['carry_hex'],carry_out_hex=carry_hex,store=receipt)
  return seal(Path(root)/'range.json',result)
 finally:writer.close()


def aggregate(plan,refs,path,*,seconds=120,cancel=None):
 from tools.opaque_frame_execution import _check
 if type(seconds) not in (int,float) or not math.isfinite(seconds) or not 0<seconds<=120:raise ValueError('bounded aggregation deadline')
 deadline=time.perf_counter()+seconds;_check(deadline,cancel)
 p=verify_prefix(plan,refs,deadline=deadline,cancel=cancel)
 if p['next_index']!=len(plan['ranges']):raise ValueError('incomplete campaign')
 result=dict(status='passed',profile=PROFILE,single_worker_anchor=False,plan_sha256=hashlib.sha256(_encode(plan)).hexdigest(),ranges=refs,pixels=plan['width']*plan['height'],covered_pixels=p['covered_pixels'],streams=p['streams'],carry_hex=p['carry_hex'])
 _check(deadline,cancel);seal(path,result);return result
