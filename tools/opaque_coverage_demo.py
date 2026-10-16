"""Bounded saved CPU-alpha demonstration; not shaded-renderer qualification."""
import argparse,hashlib,json,os,time
from pathlib import Path
from tools import material_loss_cost as cost

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixtures',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--worker',action='store_true')
    a=p.parse_args()
    if not a.worker:return cost.supervise(a.fixtures,a.output,worker_module='tools.opaque_coverage_demo')
    start=time.perf_counter();r=dict(status='working',worker_pid=os.getpid(),profile=dict(threads=1,host_bytes=1024**3,cooperative_seconds=120,watchdog_seconds=180))
    try:
        import torch
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        from tools.shading_cuda_closeout import identity
        r['wheel']=identity(Path(os.environ['GFFX_OPAQUE_WHEEL']),os.environ['GFFX_OPAQUE_WHEEL_SHA'])
        from gffx.torch import scene
        from gffx.torch.camera import Camera
        from gffx.torch.render import render,opaque_coverage
        def t(x):return torch.tensor(x,dtype=torch.float64)
        faces=torch.tensor([[0,1,2],[0,2,3]],dtype=torch.int32)
        camera=Camera.from_intrinsics(t(4.),t(4.),t(2.),t(2.),4,4,.1,10.)
        model=scene.Scene([scene.Mesh('quad',t([[-1.,-1.,-2.],[1.,-1.,-2.],[1.,1.,-2.],[-1.,1.,-2.]]),faces)])
        r['quad']={name:render(model,camera,outputs=('mask',),**kw)['mask'][0].tolist() for name,kw in [('hard',{}),('soft',dict(soft=.2)),('opaque',dict(coverage='opaque'))]}
        assert r['quad']['hard']==r['quad']['opaque']==[[1.]*4 for _ in range(4)]
        assert min(q for row in r['quad']['soft'] for q in row)<.75
        x=t(2.25).requires_grad_()
        def boundary(x):
            z=x*0.;left=z-1;top=z-1;bottom=z+5
            v=torch.stack([torch.stack([left,top,z]),torch.stack([x,top,z]),torch.stack([x,bottom,z]),torch.stack([left,bottom,z])])
            return opaque_coverage(v,faces,4,4)
        mask=boundary(x);derivative=torch.autograd.functional.jacobian(boundary,x)
        r['boundary']=dict(coverage=mask.detach().tolist(),derivative=derivative.tolist(),coordinate=2.25,derivative_units='coverage per pixel of rightward boundary motion')
        r['boundary']['max_coverage_error']=float((mask-t([1,1,.25,0]).expand(4,4)).detach().abs().max())
        r['boundary']['max_derivative_error']=float((derivative-t([0,0,1,0]).expand(4,4)).abs().max())
        # Use the declared float64 rational-reference tolerance, not new bit identity
        # to closed forms after non-dyadic polygon intersection intermediates.
        assert r['boundary']['max_coverage_error']<=1e-12
        assert r['boundary']['max_derivative_error']<=1e-12
        tiny=opaque_coverage(t([[.0625,.0625,0],[.1875,.0625,0],[.0625,.1875,0]]),faces[:1],1,1)
        assert tiny.item()==1/128
        r['subpixel']=dict(coverage=tiny.item(),expected='1/128',image=tiny.tolist())
        cost.check_deadline(start+120)
        r.update(status='passed',seconds=time.perf_counter()-start,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Actual installed-wheel CPU alpha and first derivative arrays. No shaded colour, CUDA, large-frame performance, final perceptual acceptance or M1 completion claim.')
        cost.write_report(a.output,r);return 0
    except Exception as e:
        r.update(status='failed',error=str(e));cost.write_report(a.output,r);raise

if __name__=='__main__':raise SystemExit(main())
