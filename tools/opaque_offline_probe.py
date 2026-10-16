"""One supervised offline range or aggregation; all setup charged to120seconds."""
import argparse,json,os,time,shutil
from pathlib import Path
from tools import material_loss_cost as cost
from tools import opaque_offline as offline


def validate_report(ref,plan_sha,index,device):
 r=offline.read(ref);s=r.get('supervisor',{})
 if r.get('status')!='passed' or r.get('plan_sha256')!=plan_sha or r.get('index')!=index or r.get('device')!=device:raise ValueError('worker report identity/status')
 if s.get('exit_code')!=0 or s.get('failure') is not None or s.get('monitored_pid')!=r.get('worker_pid'):raise ValueError('worker supervision failed')
 elapsed=s.get('elapsed_seconds');measured=r.get('measured_seconds');peak=s.get('sampled_peak_working_set_bytes')
 if any(type(x) not in (int,float) for x in (elapsed,measured,peak)) or not 0<elapsed<=180 or not 0<measured<=120 or not 0<peak<=(2 if device=='cuda' else 1)*1024**3:raise ValueError('worker resource limits')
 if device=='cuda' and not 0<=r.get('peak_cuda_bytes',-1)<=67108864:raise ValueError('GPU limit')
 return r


def main():
 p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true')
 for k in ('fixtures','output','packet','prefix'):p.add_argument('--'+k,type=Path,required=True)
 for k in ('packet-sha256','prefix-sha256'):p.add_argument('--'+k,required=True)
 p.add_argument('--job',type=int,required=True);p.add_argument('--index',type=int,required=True);p.add_argument('--started',type=float);a=p.parse_args()
 packet=offline.read(dict(path=str(a.packet),sha256=a.packet_sha256));job=packet['jobs'][a.job];device=job['device'];plan=job['plan'];ph=offline.hashlib.sha256(offline._encode(plan)).hexdigest()
 if not a.worker:
  if a.output.exists() or a.output.with_suffix('.range').exists():raise FileExistsError('preserve prior attempt')
  cost.PROFILE=dict(cost.PROFILE,sampled_working_set_limit_bytes=(2 if device=='cuda' and a.index<len(plan['ranges']) else 1)*1024**3)
  args=[]
  for k in ('packet','prefix','packet_sha256','prefix_sha256','job','index'):args+=['--'+k.replace('_','-'),str(getattr(a,k))]
  args+=['--started',str(time.perf_counter())]
  return cost.supervise(a.fixtures,a.output,worker_module='tools.opaque_offline_probe',worker_args=args)
 if a.started is None or not 0<=time.perf_counter()-a.started<120:raise ValueError('whole-worker start required')
 deadline=a.started+120;r=dict(status='working',worker_pid=os.getpid(),device=device,index=a.index,plan_sha256=ph,profile=offline.PROFILE,single_worker_anchor=False,packet_sha256=a.packet_sha256)
 cost.write_report(a.output,r)
 try:
  if packet.get('authorization')!='D569' or packet.get('max_worker_seconds')!=17760 or packet.get('max_watchdog_seconds')!=26640:raise ValueError('approved offline policy required')
  offline.validate_plan(plan)
  for ref in packet['files']+packet['evidence']+[packet['wheel']]:
   if offline.digest(ref['path'])!=ref['sha256']:raise ValueError('frozen source/evidence changed')
  prefix=offline.read(dict(path=str(a.prefix),sha256=a.prefix_sha256))
  if len(prefix)!=a.index or a.index>len(plan['ranges']):raise ValueError('prefix length')
  refs=[validate_report(ref,ph,i,device)['range_receipt'] for i,ref in enumerate(prefix)]
  if shutil.disk_usage(a.output.parent).free<offline.DISK_CAP:raise OSError('insufficient internal campaign storage')
  used=sum(f.stat().st_size for f in Path(packet['campaign_root']).rglob('*') if f.is_file())
  if used+32768*64+1048576>offline.DISK_CAP:raise OSError('campaign storage allowance exhausted')
  if a.index==len(plan['ranges']):
   r['complete']=offline.aggregate(plan,refs,a.output.with_suffix('.complete.json'),seconds=deadline-time.perf_counter())
   r['peak_cuda_bytes']=0
  else:
   import torch
   from dataclasses import replace
   from tools.shading_cuda_closeout import identity
   from tools.corpus_registry import load_entry
   from tools.deferred_flame import create_scene
   from tools.opaque_original_frame_admission import SCENE_SHA
   from gffx.torch.scene import Scene
   from gffx.torch.camera import Camera
   torch.set_num_threads(1);torch.set_num_interop_threads(1)
   r['wheel']=identity(Path(packet['wheel']['path']),packet['wheel']['sha256'])
   blob=load_entry(a.fixtures,'synthetic_flame_m1',allow_development=True)['files']['scene.json']
   if offline.hashlib.sha256(blob).hexdigest()!=SCENE_SHA or plan['identity']['scene']!=SCENE_SHA:raise ValueError('original scene bytes')
   model,cam=create_scene(json.loads(blob),'principled',device=device);m=model.prims[1];m=replace(m,parameters=dict(m.parameters,f0=m.parameters['f0'].repeat(3)));model=Scene([model.prims[0],m,model.prims[2]])
   w,h=plan['width'],plan['height'];camera=Camera.from_intrinsics(cam.fx*h/128,cam.fy*h/128,cam.cx*w/128,cam.cy*h/128,h,w,.1,10.,pose=cam.pose)
   m.parameters['albedo'].requires_grad_()
   if device=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
   r['range_receipt']=offline.run_range(model,camera,a.output.with_suffix('.range'),plan,a.index,refs,seconds=deadline-time.perf_counter())
   if device=='cuda':
    torch.cuda.synchronize();r['peak_cuda_bytes']=torch.cuda.max_memory_allocated()
    if r['peak_cuda_bytes']>67108864:raise MemoryError('actual GPU64MiB limit')
  cost.check_deadline(deadline);r.update(status='passed',measured_seconds=time.perf_counter()-a.started)
 except Exception as e:r.update(status='stopped',error=repr(e));raise
 finally:r['worker_seconds']=time.perf_counter()-a.started;cost.write_report(a.output,r)
 return 0

if __name__=='__main__':raise SystemExit(main())
