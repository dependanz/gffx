"""Analytical parallel-camera and public scene controls; no learned assets."""
import os
import unittest
import torch
from gffx.torch.camera import Camera
from gffx.torch import render,scene,material,light
DEVICE=os.environ.get("GFFIT_TEST_DEVICE","cpu")
def t(x,grad=False,device=DEVICE):return torch.tensor(x,dtype=torch.float64,device=device,requires_grad=grad)
def camera():return Camera.from_orthographic(t(2.),t(2.),t(2.),t(2.),4,4,.5,4.5)
def quad(z=-2.,specular=0.):
    v=t([[-2.,-2.,z],[2.,-2.,z],[2.,2.,z],[-2.,2.,z]])
    f=torch.tensor([[0,1,2],[0,2,3]],dtype=torch.int32,device=DEVICE)
    m=scene.Material("m",material.classic,dict(albedo=t([.2,.4,.6]),specular=t(specular),exponent=t(4.)))
    l=scene.Light("l",light.directional,dict(directions=t([[0.,0.,1.]]),irradiance=t([[1.,1.,1.]])))
    return scene.Scene([scene.Mesh("q",v,f,material="m"),m,l])
class OrthographicTests(unittest.TestCase):
    def test_projection_depth_and_inverse(self):
        c=camera();p=t([[.25,-.5,-.5],[.25,-.5,-4.5]])
        q=c.project(p)
        torch.testing.assert_close(q.pixels,t([[2.5,3.],[2.5,3.]]),atol=0,rtol=0)
        torch.testing.assert_close(q.ndc_depth,t([-1.,1.]),atol=0,rtol=0)
        torch.testing.assert_close(c.unproject_points(q.pixels,q.ndc_depth,kind="ndc"),p,atol=0,rtol=0)
        torch.testing.assert_close(c.unproject_points(q.pixels,q.view_depth,kind="view"),p,atol=0,rtol=0)
    def test_weak_perspective_convention_and_gradients(self):
        s=t(1.25,True);tx=t(.125,True);ty=t(-.25,True);p=t([[.25,.5,-2.]],True)
        c=Camera.from_weak_perspective(s,tx,ty,6,8,.5,4.5)
        q=c.project(p).pixels
        torch.testing.assert_close(q,t([[(1+1.25*(.25+.125))*4,(1-1.25*(.5-.25))*3]]),atol=0,rtol=0)
        grads=torch.autograd.grad(q.sum(),(s,tx,ty,p))
        expected=(t(.75),t(5.),t(-3.75),t([[5.,-3.75,0.]]))
        for got,want in zip(grads,expected):torch.testing.assert_close(got,want,atol=1e-14,rtol=0)
    def test_camera_plane_is_defined(self):
        c=camera();p=t([[0.,0.,0.],[0.,0.,1.]])
        self.assertTrue(bool(c.project(p).valid.all()))
        self.assertFalse(bool(c.project(p,in_front_only=True).valid.any()))
        torch.testing.assert_close(c.unproject_points(t([[2.,2.]]),t([0.]),kind="view"),t([[0.,0.,0.]]),atol=0,rtol=0)
    def test_stack_preserves_projection(self):
        c=camera();rig=Camera.stack([c,c]);self.assertEqual(rig.projection_kind,"orthographic")
        p=t([[[.25,.25,-1.]],[[.25,.25,-3.]]]);q=rig.project(p)
        torch.testing.assert_close(q.pixels,t([[[2.5,1.5]],[[2.5,1.5]]]),atol=0,rtol=0)
        perspective=Camera.from_intrinsics(c.fx,c.fy,c.cx,c.cy,4,4,.5,4.5)
        with self.assertRaisesRegex(ValueError,"projection"):Camera.stack([c,perspective])
    def test_invalid_orthographic_inputs(self):
        c=camera();pose=c.pose.clone();pose[0,3,0]=.1
        with self.assertRaisesRegex(ValueError,"affine"):Camera.from_orthographic(c.fx,c.fy,c.cx,c.cy,4,4,.5,4.5,pose)
        with self.assertRaises(ValueError):Camera.from_weak_perspective(t(-1.),0.,0.,4,4,.5,4.5)
        with self.assertRaises(ValueError):Camera.from_weak_perspective(t([1.,2.]),t([0.]),0.,4,4,.5,4.5)
    def test_world_position_and_depth_derivatives(self):
        s=t(1.,True);z=t(-2.,True)
        c=Camera.from_weak_perspective(s,0.,0.,4,4,.5,4.5)
        v=torch.stack((t([-2.,2.,2.,-2.]),t([-2.,-2.,2.,2.]),z.expand(4)),-1)
        f=torch.tensor([[0,1,2],[0,2,3]],dtype=torch.int32,device=DEVICE)
        r=render(scene.Scene([scene.Mesh("q",v,f)]),c,outputs=("world_position","depth"))
        want=(torch.arange(4,dtype=v.dtype,device=DEVICE)+.5-2)/2
        yy,xx=torch.meshgrid(-want,want,indexing="ij")
        torch.testing.assert_close(r["world_position"][0],torch.stack((xx,yy,torch.full_like(xx,-2.)),-1),atol=1e-14,rtol=0)
        gs,gz=torch.autograd.grad(r["world_position"][0,0,0,0]+r["depth"][0,0,0],(s,z))
        torch.testing.assert_close(gs,t(.75),atol=1e-13,rtol=0);torch.testing.assert_close(gz,t(-.5),atol=1e-13,rtol=0)
    def test_parallel_view_specular_is_constant(self):
        c=camera();r=render(quad(specular=.5),c)["colour"]
        n=t([[0.,0.,1.]]);dirs=n[:,None,:]
        brdf=material.classic(n,n,dirs,albedo=t([.2,.4,.6]),specular=t(.5),exponent=t(4.))[0,0]
        torch.testing.assert_close(r,brdf.expand(1,4,4,3),atol=1e-14,rtol=1e-14)
    def test_supersample_and_named_camera(self):
        c=camera();s=quad();s=scene.Scene([*s.prims,scene.Camera("cam",c)])
        a=render(s,"cam",supersample=2);b=render(s,c)
        for k in a:torch.testing.assert_close(a[k],b[k],atol=1e-14,rtol=1e-14)
    def test_opaque_is_explicitly_unqualified(self):
        for outputs in (("mask",),("colour","mask")):
            with self.assertRaisesRegex(NotImplementedError,"orthographic"):render(quad(),camera(),coverage="opaque",outputs=outputs)
    def test_pose_and_intrinsic_finite_differences(self):
        def fn(s,x):
            pose=torch.eye(4,dtype=s.dtype,device=s.device)[None].clone();pose[0,0,3]=x
            return Camera.from_weak_perspective(s,.125,-.25,6,8,.5,4.5,pose).project(t([[.25,.5,-2.]])).pixels
        self.assertTrue(torch.autograd.gradcheck(fn,(t(1.25,True),t(.125,True)),eps=1e-6,atol=1e-8,rtol=1e-6))
    def test_float32_projection(self):
        c=Camera.from_weak_perspective(t(1.25).float(),.125,-.25,6,8,.5,4.5)
        q=c.project(t([[.25,.5,-2.]]).float())
        torch.testing.assert_close(q.pixels,torch.tensor([[5.875,2.0625]],dtype=torch.float32,device=DEVICE),atol=0,rtol=0)
    def test_cpu_cuda_bytes(self):
        if DEVICE!="cuda":self.skipTest("CUDA worker owns byte comparison")
        rows=[]
        for device in ("cpu","cuda"):
            s=t(1.25,True,device);tx=t(.125,True,device);p=t([[.25,.5,-2.]],True,device)
            q=Camera.from_weak_perspective(s,tx,-.25,6,8,.5,4.5).project(p)
            values=[q.pixels,q.ndc_depth,*torch.autograd.grad(q.pixels.sum()+q.ndc_depth.sum(),(s,tx,p))]
            rows.append([x.detach().cpu().contiguous() for x in values])
        from test_deferred_cuda import exact
        for a,b in zip(*rows):exact(a,b,"parallel camera values and gradients")

    def test_render_cpu_cuda_bytes(self):
        if DEVICE!="cuda":self.skipTest("CUDA worker owns rendered byte comparison")
        from test_deferred_cuda import exact
        for dtype in (torch.float32,torch.float64):
            rows=[]
            for device in ("cpu","cuda"):
                def a(v,grad=False):return torch.tensor(v,dtype=dtype,device=device,requires_grad=grad)
                scale=a(.9375,True);tx=a(.0625,True);albedo=a([.21,.43,.67],True)
                v=a([[-2.,-2.,-2.],[2.,-2.,-2.],[2.,2.,-2.],[-2.,2.,-2.]],True)
                f=torch.tensor([[0,1,2],[0,2,3]],dtype=torch.int32,device=device)
                c=Camera.from_weak_perspective(scale,tx,-.125,5,7,.5,4.5)
                m=scene.Material("m",material.classic,dict(albedo=albedo,specular=a(.2),exponent=a(4.)))
                l=scene.Light("l",light.directional,dict(directions=a([[.25,0.,1.]]),irradiance=a([[1.,1.,1.]])))
                r=render(scene.Scene([scene.Mesh("q",v,f,material="m"),m,l]),c,outputs=("colour","mask","depth","world_position","normal"))
                # Select a pixel to avoid an external framework reduction defining this test.
                loss=r["colour"][0,1,2,0]+r["world_position"][0,1,2,0]
                gs,gt,ga,gv=torch.autograd.grad(loss,(scale,tx,albedo,v))
                rows.append(dict(r,scale_gradient=gs,tx_gradient=gt,albedo_gradient=ga,vertices_gradient=gv))
            for key in rows[0]:exact(rows[0][key],rows[1][key],str(dtype)+":"+key)
