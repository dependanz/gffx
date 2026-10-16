"""Frozen serial offline campaign; verified resume, never retry a failed worker."""
import argparse,json,os,subprocess,sys,time,shutil
from pathlib import Path
from tools import opaque_offline as offline
from tools.opaque_offline_probe import validate_report


def main():
 p=argparse.ArgumentParser();p.add_argument('--packet',type=Path,required=True);p.add_argument('--sha256',required=True);a=p.parse_args()
 packet=offline.read(dict(path=str(a.packet),sha256=a.sha256));root=Path(packet['campaign_root']);root.mkdir(exist_ok=True)
 if packet['authorization']!='D569' or len(packet['jobs'])!=4:raise ValueError('approved four-profile packet required')
 start=time.perf_counter();completed=[];total_workers=0;seconds=0.;supervised=0.
 if (root/'campaign-complete.json').exists():raise FileExistsError('campaign already complete')
 try:
  for ji,job in enumerate(packet['jobs']):
   plan=job['plan'];offline.validate_plan(plan)
   if (plan['width'],plan['height']) not in ((512,512),(1920,1080)) or len(plan['ranges'])!=(8 if plan['width']==512 else 64):raise ValueError('approved job extent/range cap')
   ph=offline.hashlib.sha256(offline._encode(plan)).hexdigest();folder=root/f"{job['device']}-{plan['width']}";folder.mkdir(exist_ok=True);prefix=[]
   for i in range(len(plan['ranges'])+1):
    for ref in packet['files']+packet['evidence']+[packet['wheel']]:
     if offline.digest(ref['path'])!=ref['sha256']:raise ValueError('frozen source/evidence changed')
    if total_workers>=148 or seconds>17760 or supervised>26640:raise ValueError('campaign aggregate allowance')
    used=sum(f.stat().st_size for f in root.rglob('*') if f.is_file())
    if used+32768*64+1048576>offline.DISK_CAP or shutil.disk_usage(root).free<offline.DISK_CAP:raise OSError('campaign storage allowance')
    output=folder/f'worker{i:02}.json';prefix_path=folder/f'prefix{i:02}.json'
    if prefix_path.exists():
     if json.loads(prefix_path.read_bytes())!=prefix:raise ValueError('saved prefix differs')
    else:offline.seal(prefix_path,prefix)
    if not output.exists():
     if output.with_suffix('.range').exists():raise ValueError('interrupted range cannot restart unchanged')
     args=[sys.executable,'-X','utf8','-m','tools.opaque_offline_probe','--fixtures',packet['fixtures'],'--output',str(output),'--packet',str(a.packet),'--packet-sha256',a.sha256,'--prefix',str(prefix_path),'--prefix-sha256',offline.digest(prefix_path),'--job',str(ji),'--index',str(i)]
     print(json.dumps(dict(starting=job['device'],width=plan['width'],index=i,total=len(plan['ranges'])+1)),flush=True)
     with output.with_suffix('.log').open('xb') as log:
      process=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT)
     if process.returncode:raise RuntimeError('worker stopped: '+str(output))
    ref=dict(path=output.as_posix(),sha256=offline.digest(output));r=validate_report(ref,ph,i,job['device']);total_workers+=1;seconds+=r['measured_seconds'];supervised+=r['supervisor']['elapsed_seconds']
    print(json.dumps(dict(passed=job['device'],width=plan['width'],index=i,seconds=r['measured_seconds'],peak=r['supervisor']['sampled_peak_working_set_bytes'])),flush=True)
    if i<len(plan['ranges']):prefix.append(ref)
    else:completed.append(r['complete'])
   if ji%2==1:
    cpu,cuda=completed[-2:]
    if cpu['streams']!=cuda['streams'] or cpu['carry_hex']!=cuda['carry_hex'] or cpu['pixels']!=cuda['pixels'] or cpu['covered_pixels']!=cuda['covered_pixels']:raise ValueError('CPU/CUDA complete image or gradient bytes differ')
    offline.seal(root/f"comparison-{plan['width']}.json",dict(status='passed',streams=cpu['streams'],carry_hex=cpu['carry_hex'],pixels=cpu['pixels'],exact_cpu_cuda=True))
  result=dict(status='passed',profile=offline.PROFILE,packet_sha256=a.sha256,workers=total_workers,measured_worker_seconds=seconds,supervised_seconds=supervised,invocation_seconds=time.perf_counter()-start,single_worker_anchor=False,frames=completed)
  offline.seal(root/'campaign-complete.json',result);print(json.dumps({k:v for k,v in result.items() if k!='frames'}),flush=True)
 except Exception as e:
  failure=root/('campaign-stopped-'+str(time.time_ns())+'.json');offline.seal(failure,dict(status='stopped',error=repr(e),completed_workers=total_workers,measured_worker_seconds=seconds,supervised_seconds=supervised,invocation_seconds=time.perf_counter()-start,packet_sha256=a.sha256));raise

if __name__=='__main__':main()
