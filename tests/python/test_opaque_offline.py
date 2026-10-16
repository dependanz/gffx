"""Offline ranges preserve global pixels and exact ordered gradient carries."""
import json,os,struct,hashlib
from pathlib import Path
import pytest,torch
from test_opaque_frame_albedo_replay import fixture
from test_opaque_albedo_replay import reference,bits


def test_offline_range_preserves_global_camera_tail_and_carry(tmp_path):
 from tools.opaque_offline import make_plan,run_range,verify_prefix,aggregate
 from tools.opaque_frame_execution import execute,replay
 s,c,a=fixture(3);identity={'test':'global3'};cot=[-.0,-5e-324,.7,.1]
 plan=make_plan(3,3,identity,cotangent=cot,row_cap=2,cull_mode=1)
 full=execute(s,c,tmp_path/'old',identity={},cotangent=cot,cull_mode=1,window_pixels=3,request_pixels=3,albedo_replay=True)
 expected=replay(s,c,full)[0];refs=[]
 for i in range(2):
  s,c,a=fixture(3)  # New object addresses must retain the same stable identity.
  refs.append(run_range(s,c,tmp_path/f'r{i}',plan,i,refs))
 prefix=verify_prefix(plan,refs);assert prefix['next_index']==2
 assert prefix['carry_hex']==bits(expected).hex()
 final=aggregate(plan,refs,tmp_path/'complete.json')
 for k in ('colour','mask'):
  raw=b''.join((tmp_path/f'r{i}'/(k+'.f64')).read_bytes() for i in range(2))
  assert raw==(tmp_path/'old'/(k+'.f64')).read_bytes()
  assert final['streams'][k]['sha256']==hashlib.sha256(raw).hexdigest()
 assert final['pixels']==9 and final['single_worker_anchor'] is False


def test_offline_nonzero_carry_and_varying_cotangents(tmp_path):
 from tools.opaque_offline import make_plan,run_range,verify_prefix
 s,c,a=fixture(2);initial=struct.pack('<3d',-.0,5e-324,-.2).hex();cot=[.2,-.3,5e-324,-.0]
 plan=make_plan(2,2,{},cotangent=cot,row_cap=1,cull_mode=1,initial_carry_hex=initial)
 refs=[];carry=a.new_tensor(struct.unpack('<3d',bytes.fromhex(initial)))
 for y in range(2):
  g=a.new_tensor(cot[:3]).expand(1,1,2,3).contiguous();gm=a.new_full((1,1,2),cot[3])
  from gffx.torch import render
  for x in range(2):
   out=render.opaque_window(s,c,(y,x,1,1),cull_mode=1)
   grad,=torch.autograd.grad((out['colour'],out['mask']),a,(g[:,:,x:x+1],gm[:,:,x:x+1]))
   carry.add_(grad)
  refs.append(run_range(s,c,tmp_path/f'r{y}',plan,y,refs))
 assert verify_prefix(plan,refs)['carry_hex']==bits(carry).hex()


def test_offline_rejects_missing_reordered_corrupt_and_changed_inputs(tmp_path):
 from tools.opaque_offline import make_plan,run_range,verify_prefix,aggregate
 s,c,a=fixture(2);p=make_plan(2,2,{},row_cap=1,cull_mode=1);refs=[run_range(s,c,tmp_path/'r0',p,0,[])]
 with pytest.raises(ValueError):aggregate(p,refs,tmp_path/'incomplete.json')
 assert not (tmp_path/'incomplete.json').exists()
 with pytest.raises(ValueError):verify_prefix(p,refs+refs)
 q=json.loads(json.dumps(p));q['cotangent'][0]=.8
 with pytest.raises(ValueError):verify_prefix(q,refs)
 with torch.no_grad():c.fx.add_(.1)
 with pytest.raises(ValueError,match='input'):run_range(s,c,tmp_path/'wrong',p,1,refs)
 assert not (tmp_path/'wrong').exists()
 f=tmp_path/'r0'/'colour.f64';raw=f.read_bytes();f.write_bytes(bytes([raw[0]^1])+raw[1:])
 with pytest.raises(ValueError):verify_prefix(p,refs)


def test_offline_cancellation_exhaustion_and_pixel_mismatch_never_seal(tmp_path,monkeypatch):
 from tools.opaque_offline import make_plan,run_range
 from gffx.torch import render
 s,c,a=fixture(1);p=make_plan(1,1,{},cull_mode=1)
 for key,kw in [('cancel',{'cancel':lambda:True}),('time',{'seconds':1e-12})]:
  with pytest.raises(TimeoutError):run_range(s,c,tmp_path/key,p,0,[],**kw)
  assert not (tmp_path/key/'range.json').exists()
 old=render.opaque_albedo_replay
 def wrong(*args,**kw):
  out=old(*args,**kw);out['colour'].add_(1e-9);return out
 monkeypatch.setattr(render,'opaque_albedo_replay',wrong)
 with pytest.raises(ValueError,match='recomputed'):run_range(s,c,tmp_path/'bad',p,0,[])
 assert not (tmp_path/'bad'/'range.json').exists()


def test_offline_policy_plan_caps():
 from tools.opaque_offline import make_plan,validate_plan
 for w,h,n in [(512,512,8),(1920,1080,64)]:
  p=make_plan(w,h,{});assert len(p['ranges'])==n
  assert sum(r['count'] for r in p['ranges'])==w*h
  assert all(r['count']<=32768 for r in p['ranges']);validate_plan(p)
  p['ranges'][0]['count']+=1
  with pytest.raises(ValueError):validate_plan(p)
 with pytest.raises(ValueError):make_plan(512,512,{},row_cap=65)


@pytest.mark.parametrize('bad',['carry','predecessor','origin','reorder','camera','cancel_aggregate','memory','mutation'])
def test_offline_receipt_and_resource_negative_controls(tmp_path,monkeypatch,bad):
 from tools.opaque_offline import make_plan,run_range,verify_prefix,aggregate,read,seal
 from gffx.torch import render
 s,c,a=fixture(2);p=make_plan(2,2,{},row_cap=1,cull_mode=1)
 if bad in ('memory','mutation'):
  old=render.opaque_window
  def inject(*args,**kw):
   if bad=='memory':raise MemoryError('injected resource exhaustion')
   out=old(*args,**kw)
   with torch.no_grad():a.add_(1e-4)
   return out
  monkeypatch.setattr(render,'opaque_window',inject)
  with pytest.raises((MemoryError,RuntimeError,ValueError)):run_range(s,c,tmp_path/'bad',p,0,[])
  assert not (tmp_path/'bad'/'range.json').exists();return
 refs=[run_range(s,c,tmp_path/'r0',p,0,[])];refs.append(run_range(s,c,tmp_path/'r1',p,1,refs))
 if bad=='cancel_aggregate':
  with pytest.raises(TimeoutError):aggregate(p,refs,tmp_path/'no.json',cancel=lambda:True)
  assert not (tmp_path/'no.json').exists();return
 if bad=='reorder':
  with pytest.raises(ValueError):verify_prefix(p,list(reversed(refs)))
  return
 row=read(refs[1])
 if bad=='carry':row['carry_in_hex']=struct.pack('<3d',1.,2.,3.).hex()
 elif bad=='predecessor':row['previous']=None
 elif bad=='origin':row['range']['first']=0
 elif bad=='camera':row['inputs']['structure']=['wrong camera']
 refs[1]=seal(tmp_path/'tampered.json',row)
 with pytest.raises(ValueError):verify_prefix(p,refs)


@pytest.mark.parametrize('bad',['failed','pid','time','watchdog','host','gpu','index'])
def test_offline_supervised_report_required(tmp_path,bad):
 from tools.opaque_offline import seal
 from tools.opaque_offline_probe import validate_report
 r=dict(status='passed',plan_sha256='a'*64,index=0,device='cuda',worker_pid=123,measured_seconds=12.,peak_cuda_bytes=123456,supervisor=dict(exit_code=0,failure=None,monitored_pid=123,elapsed_seconds=13.,sampled_peak_working_set_bytes=1000000))
 if bad=='failed':r['status']='stopped'
 elif bad=='pid':r['supervisor']['monitored_pid']=234
 elif bad=='time':r['measured_seconds']=120.01
 elif bad=='watchdog':r['supervisor']['elapsed_seconds']=180.01
 elif bad=='host':r['supervisor']['sampled_peak_working_set_bytes']=2*1024**3+1
 elif bad=='gpu':r['peak_cuda_bytes']=67108865
 elif bad=='index':r['index']=1
 ref=seal(tmp_path/'report.json',r)
 with pytest.raises(ValueError):validate_report(ref,'a'*64,0,'cuda')
