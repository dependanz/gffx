"""Bounded installed-wheel affine-radiance demonstration; saved arrays, no substitutions."""
import argparse,hashlib,json,os,time
from pathlib import Path
from tools import material_loss_cost as cost

def main():
    p=argparse.ArgumentParser();p.add_argument('--fixtures',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--worker',action='store_true');a=p.parse_args()
    if not a.worker:return cost.supervise(a.fixtures,a.output,worker_module='tools.opaque_colour_demo')
    start=time.perf_counter();row=dict(status='working',worker_pid=os.getpid(),profile=dict(threads=1,host_bytes=1024**3,cooperative_seconds=120,watchdog_seconds=180))
    try:
        import torch
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        from gffx.torch.render import opaque_integrate
        from tools.shading_cuda_closeout import identity
        row['wheel']=identity(Path(os.environ['GFFX_OPAQUE_WHEEL']),os.environ['GFFX_OPAQUE_WHEEL_SHA'])
        def t(q,grad=False):return torch.tensor(q,dtype=torch.float64,requires_grad=grad)
        def f(q):return torch.tensor(q,dtype=torch.int32)
        verts=[[12.25,12.25,0],[83.75,12.25,0],[83.75,83.75,0],[12.25,83.75,0]]
        fa=[[0,1,2],[0,2,3]];fb=[[0,1,3],[1,2,3]]
        def field(x,y):return [.06+.20*x/96,.10+.46*y/96,.18+.46*x/96]
        panels=[]
        for faces in (fa,fb):
            c=[[field(*verts[i][:2]) for i in face] for face in faces]
            panels.append(opaque_integrate(t(verts),f(faces),t(c),96,96))
            cost.check_deadline(start+120)
        row['diagonal_max_rgba_error']=float((panels[0]-panels[1]).abs().max());assert row['diagonal_max_rgba_error']<=1e-12
        # Two affine depth planes intersect inside the common projected square.
        overlap=verts+[[x,y,(x+.45*y-69.25)/96] for x,y,z in verts]
        faces=fa+[[i+4 for i in face] for face in fb]
        colours=[[field(*overlap[i][:2]) if face_index<2 else [.6+.25*overlap[i][0]/96,.08+.25*overlap[i][1]/96,.035] for i in face] for face_index,face in enumerate(faces)]
        panels.append(opaque_integrate(t(overlap),f(faces),t(colours),96,96));cost.check_deadline(start+120)
        row['panels']=[q.tolist() for q in panels];row['panel_resolution']=[96,96]
        # Closed-form moving depth intersection: red covers x < 2.25, blue the rest.
        v=[[-1,-1,-3.25],[5,-1,2.75],[5,5,2.75],[-1,5,-3.25],[-1,-1,0],[5,-1,0],[5,5,0],[-1,5,0]]
        faces=fa+[[i+4 for i in face] for face in fa];c=[[[1,0,0]]*3]*2+[[[0,0,1]]*3]*2
        vv=t(v,True);cc=t(c,True);ff=f(faces);rgba=opaque_integrate(vv,ff,cc,4,4)
        motion=torch.zeros_like(vv);motion[:4,2]=-1
        derivatives=[]
        for x in range(4):
            one=opaque_integrate(vv,ff,cc,1,1,origin=(x,1));dr=[]
            for channel in range(4):
                gv=torch.autograd.grad(one[0,0,channel],vv,retain_graph=True)[0];dr.append(float((gv*motion).sum()))
            derivatives.append(dr)
            cost.check_deadline(start+120)
        reference=t([[[1,0,0,1],[1,0,0,1],[.25,0,.75,1],[0,0,1,1]]]*4)
        dref=t([[0,0,0,0],[0,0,0,0],[1,0,-1,0],[0,0,0,0]])
        row['intersection']=dict(rgba=rgba.tolist(),derivative=derivatives,coordinate=2.25,max_image_error=float((rgba-reference).abs().max()),max_derivative_error=float((t(derivatives)-dref).abs().max()),derivative_units='linear radiance per unit rightward motion of the depth-intersection boundary')
        assert row['intersection']['max_image_error']<=1e-12 and row['intersection']['max_derivative_error']<=1e-12
        row.update(status='passed',seconds=time.perf_counter()-start,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Actual CPU screen-affine radiance and geometry/depth first derivatives. Display panels are96x96 finite forward examples; independent acceptance uses small analytical fixtures. No nonlinear shader, perspective radiance, CUDA, scale or final SIGGRAPH showcase acceptance.')
        cost.write_report(a.output,row);return 0
    except Exception as e:
        row.update(status='failed',error=repr(e));cost.write_report(a.output,row);raise

if __name__=='__main__':raise SystemExit(main())
