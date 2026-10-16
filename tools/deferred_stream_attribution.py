"""Bounded64-square attribution only; timings are not qualification calibration."""
import sys,time,json
from pathlib import Path
from tools import deferred_stream_probe as probe,material_loss_cost as cost,shading_cuda_closeout as artifact

def main():
 if '--width' not in sys.argv or sys.argv[sys.argv.index('--width')+1]!='64' or sys.argv[sys.argv.index('--height')+1]!='64':raise ValueError('attribution is bounded to64-square')
 if '--worker' not in sys.argv:
  original=cost.supervise
  def supervise(*args,**kwargs):
   kwargs['worker_module']='tools.deferred_stream_attribution';return original(*args,**kwargs)
  cost.supervise=supervise
  return probe.main()
 import torch
 from gffx.torch import _deferred as d
 from tools import deferred_stream as stream
 samples={}
 def wrap(owner,name):
  original=getattr(owner,name)
  def measured(*args,**kwargs):
   if '--device' in sys.argv and sys.argv[sys.argv.index('--device')+1]=='cuda':torch.cuda.synchronize()
   start=time.perf_counter()
   try:return original(*args,**kwargs)
   finally:
    if '--device' in sys.argv and sys.argv[sys.argv.index('--device')+1]=='cuda':torch.cuda.synchronize()
    row=samples.setdefault(name,dict(calls=0,seconds=0.));row['calls']+=1;row['seconds']+=time.perf_counter()-start
  setattr(owner,name,measured)
 for owner,name in ((d,'_buffers'),(d,'_shade'),(stream,'_cohort_partials'),(torch.ops.gffx,'rasterize'),(torch.ops.gffx,'rasterize_window')):wrap(owner,name)
 try:return probe.main()
 finally:
  p=Path(sys.argv[sys.argv.index('--output')+1]);row=json.loads(p.read_text());row.update(attribution=samples,attribution_source_sha256=artifact.digest(__file__),scope='Instrumented diagnostic only. Nested intervals overlap;do not add them or use these timings as calibration.');p.write_text(json.dumps(row,indent=2))
if __name__=='__main__':raise SystemExit(main())
