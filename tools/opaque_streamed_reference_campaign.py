"""Serial, single-use controller and supervised workers for explicit streamed offline profiles."""
import argparse,hashlib,json,os,time
from pathlib import Path
from tools import material_loss_cost as cost
from tools.opaque_streamed_reference_contract import (PROFILE,INPUT_SHA,validate_input,certificate_header,aggregate,verify_files,allocation,digest,validate_certificate,EXTENDED_PROFILE,HOUR_PROFILE,TWO_HOUR_PROFILE,limits_for)
from tools.opaque_reference_profile import worker_profile

def read(path):return json.loads(Path(path).read_bytes())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_manifest(path,expected):
    if sha(path)!=expected:raise ValueError('frozen manifest bytes changed')
    m=read(path)
    if m.get('schema')!='gffx-streamed-campaign-v1' or m.get('profile') not in (PROFILE,EXTENDED_PROFILE,HOUR_PROFILE,TWO_HOUR_PROFILE) or m.get('pieces')!=list(range(22)):raise ValueError('wrong campaign scope')
    checker_options(m)
    if 'execution_order' in m and m['execution_order']!=campaign_order(m):raise ValueError('changed campaign order')
    verify_files(m['files'])
    required=[m[k] for k in ('input','wheel','cpu','cuda')]
    if any(str(Path(p).resolve()) not in {str(Path(r['path']).resolve()) for r in m['files']} for p in required):raise ValueError('unbound required resource')
    if sha(m['input'])!=INPUT_SHA:raise ValueError('wrong frozen input bytes')
    data=read(m['input']);validate_input(data)
    if m.get('allocation')!=[str(v) for v in allocation(data)]:raise ValueError('changed error allocation')
    if m.get('limits')!=limits_for(m['profile']).record():raise ValueError('changed worker limits')
    return m,data

def execution_order(name):
    limits_for(name)
    return [1,0,*range(2,22)] if name in (EXTENDED_PROFILE,HOUR_PROFILE,TWO_HOUR_PROFILE) else list(range(22))


def checker_options(manifest):
    """New proof selection must be named in the frozen hour-profile manifest."""
    if 'checker' not in manifest:return {}
    if manifest.get('checker') not in ('centered-correction-v1','directional-correction-v1','horizon-correction-v1') or manifest.get('profile') not in (HOUR_PROFILE,TWO_HOUR_PROFILE):
        raise ValueError('explicit qualified hour-profile checker required')
    return dict(centered_correction=True,**(dict(directional_correction=True) if manifest['checker'] in ('directional-correction-v1','horizon-correction-v1') else {}),**(dict(horizon_correction=True) if manifest['checker']=='horizon-correction-v1' else {}))


def campaign_order(manifest):
    if checker_options(manifest):
        return [3,2,0,1,*range(4,22)] if manifest['checker'] in ('directional-correction-v1','horizon-correction-v1') else [2,0,1,*range(3,22)]
    return execution_order(manifest.get('profile',PROFILE))


def supervised_launch(fixtures,output,*,reference_profile,piece,launch=None,**kwargs):
    limits=limits_for(reference_profile)
    if type(piece)!=int or not 0<=piece<=22:raise ValueError('campaign worker index required')
    saved=dict(cost.PROFILE)
    try:
        cost.PROFILE.update(cooperative_seconds=limits.worker_seconds if piece<22 else 120,watchdog_seconds=limits.watchdog_seconds if piece<22 else 180,sampled_working_set_limit_bytes=1073741824)
        return (cost.supervise if launch is None else launch)(fixtures,output,**kwargs)
    finally:
        cost.PROFILE.clear();cost.PROFILE.update(saved)

def campaign(manifest,expected,out,*,launch=None):
    # A directory is a single-use campaign claim. Never resume/replace it.
    frozen,_=load_manifest(manifest,expected)
    name=frozen.get('profile',PROFILE);order=campaign_order(frozen);limits=limits_for(name)
    out=Path(out);out.mkdir(exist_ok=False)
    launch=cost.supervise if launch is None else launch
    state=dict(status='running',manifest_sha256=expected,completed_pieces=0,accuracy_admitted=False)
    start=time.perf_counter()
    try:
        for index in [*order,22]:
            cost.check_deadline(start+22*limits.watchdog_seconds+180)
            frozen,data=load_manifest(manifest,expected)
            filename=f'piece{index:02d}.json' if index<22 else 'aggregate.json'
            state.update(active_piece=index,seconds=time.perf_counter()-start)
            cost.write_report(out/'campaign.json',state)
            result=supervised_launch(Path(manifest),out/filename,reference_profile=name,piece=index,launch=launch,worker_module='tools.opaque_streamed_reference_campaign',worker_args=('--manifest-sha',expected,'--piece',str(index)))
            report=read(out/filename)
            if result or report.get('status')!='passed':
                state.update(status='stopped',failed_worker=index,reason=report.get('reason',report.get('error','worker failed')));break
            load_manifest(manifest,expected)
            if index<22:
                validate_certificate(report,certificate_header(data,index,expected,**({} if name==PROFILE else dict(reference_profile=name))))
                state['completed_pieces']+=1
            else:
                if report.get('accuracy_admitted') is not True:raise ValueError('aggregation did not admit complete pixel')
                state.update(status='passed',accuracy_admitted=True)
            cost.write_report(out/'campaign.json',state)
    except Exception as e:state.update(status='stopped',reason=repr(e))
    state['seconds']=time.perf_counter()-start
    cost.write_report(out/'campaign.json',state)
    return 0 if state['status']=='passed' and state['accuracy_admitted'] else 1

def worker(manifest,expected,index,output):
    start=time.perf_counter();deadline=start+120
    row=dict(status='working',worker_pid=os.getpid(),manifest_sha256=expected,piece=index,accuracy_admitted=False)
    try:
        if type(index)!=int or not 0<=index<=22:raise ValueError('bounded campaign worker required')
        m,data=load_manifest(manifest,expected)
        name=m['profile'];limits=limits_for(name);deadline=start+(limits.worker_seconds if index<22 else 120)
        import torch
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        from tools.shading_cuda_closeout import identity
        row['wheel']=identity(Path(m['wheel']),m['wheel_sha256'])
        check=lambda:cost.check_deadline(deadline)
        check()
        if index<22:
            row.update(certificate_header(data,index,expected,**({} if name==PROFILE else dict(reference_profile=name))))
            from tools.opaque_full_material_reference import full_pixel
            last_progress=[0.]
            def progress(item):
                now=time.perf_counter()
                if now-last_progress[0]>=5:
                    last_progress[0]=now;row['progress']=item;row['seconds']=now-start
                    cost.write_report(output,json.loads(json.dumps(row,default=str)))
            result=full_pixel(data,reference_profile=name,streamed_piece=index,checkpoint=check,progress=progress,material_only_inputs=True,tight_discriminant=True,aggregate_residual=True,smooth_partition=(4,4),stable_view_base=True,tight_cosine_slopes=True,**checker_options(m))
            row.update(basis=[[str(v.lo),str(v.hi)] for v in result['basis']],cells=result['cells'],reference_seconds=result['seconds'],reference=result)
        else:
            reports=[read(output.parent/f'piece{i:02d}.json') for i in range(22)]
            ref=aggregate(data,reports,expected,reference_profile=name);row['reference']=ref
            cpu,cuda=read(m['cpu']),read(m['cuda'])
            if cpu.get('status')!='passed' or cuda.get('status')!='passed':raise ValueError('preserved device evidence did not pass')
            # Exact device comparisons are preserved evidence, not the oracle.
            row['checks']=compare_preserved(cpu,cuda,ref)
            row['accuracy_admitted']=all(v['passed'] for v in row['checks'])
            if not row['accuracy_admitted']:raise ValueError('installed rendering exceeds independent allowance')
        load_manifest(manifest,expected);check();row['status']='passed';return 0
    except Exception as e:
        row.update(status='stopped',reason=str(e),details=getattr(e,'details',None));return 1
    finally:
        row['seconds']=time.perf_counter()-start
        cost.write_report(output,json.loads(json.dumps(row,default=str)))

def compare_preserved(cpu,cuda,ref):
    from tools.opaque_shading_intervals import I
    # The CUDA batch records its individual CPU comparisons explicitly.
    cases=cuda['cases']
    selected=next(r for r in cases if r['case']==0)
    if selected['bytes']!=cpu['bytes']:raise ValueError('preserved CPU/CUDA bytes differ')
    flat=lambda x:sum((flat(y) for y in x),[]) if isinstance(x,list) else [x]
    checks=[]
    for key in ('colour','albedo_sum_gradient'):
        values=flat(cpu['values'][key]);bounds=ref[key]
        if len(values)!=len(bounds):raise ValueError('complete comparison shape required')
        for value,bound in zip(values,bounds):
            interval=I(*bound);error=(I(value)-interval).bound;allowance=I('1e-10')+I('1e-9')*I(interval.bound)
            checks.append(dict(quantity=key,value=value,bound=bound,error_upper=str(error),allowance_lower=str(allowance.lo),passed=error<=allowance.lo))
    return checks

def main():
    p=argparse.ArgumentParser();p.add_argument('--fixtures',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--manifest-sha',required=True);p.add_argument('--piece',type=int);p.add_argument('--worker',action='store_true');a=p.parse_args()
    if a.worker:return worker(a.fixtures,a.manifest_sha,a.piece,a.output)
    if a.piece is not None:raise ValueError('only the controller launches individual workers')
    return campaign(a.fixtures,a.manifest_sha,a.output)
if __name__=='__main__':raise SystemExit(main())
