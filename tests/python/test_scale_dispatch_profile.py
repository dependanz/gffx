"""Bounded attribution of current four-point dispatch, not throughput acceptance."""
import os,json,cProfile,pstats,time
import pytest,torch

def test_scale_head_native_dispatch_attribution(record_property):
 output=os.environ.get('GFFX_SCALE_DISPATCH_PROFILE')
 if not output:pytest.skip('explicit attribution output required')
 from pathlib import Path
 from tools.synthetic_flame_fixture import generate
 from tools.deferred_flame import create_scene
 from gffx.torch import render
 from gffx.torch.camera import Camera
 model,cam=create_scene(generate(),'principled',device='cuda')
 camera=Camera.from_intrinsics(cam.fx/4,cam.fy/4,cam.cx/4,cam.cy/4,32,32,.1,10.,pose=cam.pose)
 profile=cProfile.Profile();torch.cuda.synchronize();start=time.perf_counter()
 with torch.no_grad():
  profile.enable();result=render(model,camera,outputs=('colour','mask'));torch.cuda.synchronize();profile.disable()
 elapsed=time.perf_counter()-start;stats=pstats.Stats(profile);rows=[]
 for (file,line,name),(primitive,calls,own,total,_) in stats.stats.items():
  if 'gffx' in file or 'isfinite' in name or 'item' in name:
   rows.append(dict(file=file,line=line,name=name,calls=calls,own_seconds=own,cumulative_seconds=total))
 rows.sort(key=lambda x:x['cumulative_seconds'],reverse=True)
 row=dict(status='measured',size=32,covered=int((result['mask']>0).sum()),profiled_seconds=elapsed,rows=rows[:25],scope='One cProfile-instrumented CUDA forward; attribution only,not uninstrumented timing or speed guarantee.')
 path=Path(output);assert not path.exists();path.write_text(json.dumps(row,indent=2));record_property('dispatch_profile',json.dumps(row))
 assert torch.isfinite(result['colour']).all()
