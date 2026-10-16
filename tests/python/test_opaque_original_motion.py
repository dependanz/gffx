"""Original-mesh first-VJP diagnostics; only invariances have exact acceptance."""
import os,json,time
from pathlib import Path
from dataclasses import replace
from copy import copy
import torch
from tests.python.test_opaque_full_original import original


def moved(s,c,theta,kind):
    from gffx.torch import scene
    m=s.prims[0];v=m.vertices;n=m.attributes['normals'].values;c=copy(c)
    direction=v.new_tensor([.01,0.,0.])
    if kind in ('vertex_x','joint_translation'):
        v=v+theta*direction
    if kind in ('camera_x','joint_translation'):
        delta=torch.zeros_like(c.pose);delta[0,0,3]=.01;c.pose=c.pose+theta*delta
    if kind=='normal_tilt':
        # Infinitesimal rotation about y; normalization occurs in production.
        n=n+theta*.01*torch.stack((n[:,2],torch.zeros_like(n[:,1]),-n[:,0]),-1)
    if kind=='normal_scale':n=n*(1+theta)
    if kind=='radial_scale':v=c.pose[0,:3,3]+(v-c.pose[0,:3,3])*(1+theta*.01)
    attrs=dict(m.attributes,normals=replace(m.attributes['normals'],values=n))
    return scene.Scene([replace(m,vertices=v,attributes=attrs),*s.prims[1:]]),c


def test_original_motion_diagnostic(record_property):
    from gffx.torch import render
    device=os.environ.get('GFFX_KNOT_DEVICE','cpu');kind=os.environ['GFFX_ORIGINAL_MOTION'];s,c,sha=original(device)
    start=time.perf_counter();theta=c.fx.new_tensor(0.).requires_grad_()
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    def evaluate(t):
        ss,cc=moved(s,c,t,kind)
        return render.opaque_window(ss,cc,(78,101,1,1),cull_mode=render.CULL_BACK)
    out=evaluate(theta);rgb=out['colour'].reshape(3)
    grad=torch.stack([torch.autograd.grad(rgb[i],theta,retain_graph=i<2)[0] for i in range(3)])
    invariant=kind in ('joint_translation','normal_scale','radial_scale')
    row=dict(device=device,direction=kind,scene_sha256=sha,region=[78,101,1,1],colour=rgb.detach().cpu().tolist(),gradient=grad.detach().cpu().tolist(),bytes=dict(colour=rgb.detach().cpu().numpy().tobytes().hex(),gradient=grad.detach().cpu().numpy().tobytes().hex()),independent_derivative=[0,0,0] if invariant else None,scope='Exact transformation invariance only' if invariant else 'Finite-difference screening only; no independent reference or gradient admission')
    p=Path(os.environ['GFFX_FULL_NATIVE']);assert not p.exists()
    p.write_text(json.dumps(row,indent=2),encoding='utf8')
    try:
        if invariant:
            row['checks']=[dict(error_over_limit=abs(x)/1e-10,passed=abs(x)<=1e-10) for x in row['gradient']]
            with torch.no_grad():other=evaluate(theta.new_tensor(.125))
            row['finite_invariance_max_image_change']=(other['colour']-out['colour']).abs().max().item()
        else:
            vals={}
            with torch.no_grad():
                for sign in (-1,1):
                    for h in (2**-10,2**-11):
                        vals[sign*h]=evaluate(theta.new_tensor(sign*h))['colour'].reshape(3)
            coarse=(vals[2**-10]-vals[-2**-10])/(2**-9);fine=(vals[2**-11]-vals[-2**-11])/(2**-10)
            rich=(4*fine-coarse)/3
            row['screening']=dict(steps=[2**-10,2**-11],central_coarse=coarse.cpu().tolist(),central_fine=fine.cpu().tolist(),richardson=rich.cpu().tolist(),gradient_minus_richardson=(grad-rich).detach().cpu().tolist(),not_a_certified_error_bound=True)
    except Exception as error:
        row['diagnostic_failure']=dict(type=type(error).__name__,message=str(error))
        p.write_text(json.dumps(row,indent=2),encoding='utf8')
        raise
    if device=='cuda':torch.cuda.synchronize();row['gpu_peak_bytes']=torch.cuda.max_memory_allocated();assert row['gpu_peak_bytes']<=67108864
    row['seconds']=time.perf_counter()-start
    p.write_text(json.dumps(row,indent=2),encoding='utf8');record_property('original_motion',json.dumps(row))
    if invariant:assert all(x['passed'] for x in row['checks']),row

def test_original_normal_input_vjp(record_property):
    import math,hashlib
    from gffx.torch import render,scene
    device=os.environ.get('GFFX_KNOT_DEVICE','cpu');s,c,sha=original(device);m=s.prims[0];n=m.attributes['normals'].values.detach().requires_grad_();m=replace(m,attributes=dict(m.attributes,normals=replace(m.attributes['normals'],values=n)));s=scene.Scene([m,*s.prims[1:]])
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter();rgb=render.opaque_window(s,c,(78,101,1,1),cull_mode=render.CULL_BACK)['colour'].reshape(3);jac=torch.stack([torch.autograd.grad(rgb[i],n,retain_graph=i<2)[0] for i in range(3)]).cpu().contiguous();normals=n.detach().cpu().tolist();directions=dict(normal_scale=normals,normal_tilt=[[.01*z,0.,-.01*x] for x,y,z in normals]);derivs={key:[math.fsum(a*b for row,d in zip(channel,direction) for a,b in zip(row,d)) for channel in jac.tolist()] for key,direction in directions.items()}
    blob=jac.numpy().tobytes();p=Path(os.environ['GFFX_FULL_NATIVE']);binary=p.with_suffix('.bin');assert not p.exists() and not binary.exists();binary.write_bytes(blob)
    row=dict(device=device,scene_sha256=sha,shape=list(jac.shape),normal_input_jacobian_sha256=hashlib.sha256(blob).hexdigest(),colour_bytes=rgb.detach().cpu().numpy().tobytes().hex(),host_fsum_directional=derivs,seconds=time.perf_counter()-start,scope='Per-input native VJP bytes precede caller-defined parameterization/reduction; host fsum is diagnostic only.')
    if device=='cuda':row['gpu_peak_bytes']=torch.cuda.max_memory_allocated();assert row['gpu_peak_bytes']<=67108864
    p.write_text(json.dumps(row,indent=2),encoding='utf8');record_property('original_normal_vjp',json.dumps(row));assert max(abs(x) for x in derivs['normal_scale'])<=1e-10
