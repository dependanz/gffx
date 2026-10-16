"""Independent Decimal checks of the compiled raster edge proof helper.

The tiny test DLL exports edge(6 doubles)->double and sign(9 doubles)->int
from raster_block_bounds.h. It is never installed or used for rendering.
Use the frozen bridge.c and strict-FP command in raster41-edge-oracle1/report.json.
Sampling supplements the analytic bound; it is not a proof over all real inputs.
"""
import argparse,ctypes,decimal,hashlib,json,random
from pathlib import Path

def check(library):
 lib=ctypes.CDLL(str(library));lib.edge.argtypes=[ctypes.c_double]*6;lib.edge.restype=ctypes.c_double
 lib.sign.argtypes=[ctypes.c_double]*9;lib.sign.restype=ctypes.c_int
 rng=random.Random(19414);D=decimal.Decimal.from_float;checked=certificates=0;worst=decimal.Decimal(0)
 with decimal.localcontext() as ctx:
  ctx.prec=120
  for trial in range(2048):
   scale=(2.**20,128.,1e-140,1e-10)[trial%4]
   ax,ay,bx,by=[rng.uniform(-scale,scale) for _ in range(4)]
   l,t=[float(rng.randrange(1024))+.5 for _ in range(2)];r,b=l+31,t+31
   m=max(1.,abs(ax),abs(ay),abs(bx),abs(by),r,b);error=m*m*2.**-44
   sign=lib.sign(ax,ay,bx,by,l,t,r,b,error);certificates+=bool(sign)
   for px,py in ((l,t),(r,t),(l,b),(r,b),((l+r)/2,(t+b)/2),(l+7,t+23)):
    exact=(D(ax)-D(px))*(D(by)-D(py))-(D(ay)-D(py))*(D(bx)-D(px))
    value=lib.edge(ax,ay,bx,by,px,py);ratio=abs(D(value)-exact)/D(error);worst=max(worst,ratio)
    assert ratio<1,(trial,ratio)
    if sign:assert sign*value>0 and decimal.Decimal(sign)*exact>0,trial
    checked+=1
 return dict(status='passed',edge_evaluations=checked,certified_rectangles=certificates,decimal_precision=120,max_error_over_declared_bound=str(worst))

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--library',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 row=check(a.library);row['library_sha256']=hashlib.sha256(a.library.read_bytes()).hexdigest()
 with a.output.open('x',encoding='utf-8') as f:json.dump(row,f,indent=2)
 print(json.dumps(row))
if __name__=='__main__':main()
