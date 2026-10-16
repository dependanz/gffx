"""Bounded subpixel experiment against the unchanged complete reference."""
def test_original_subpixel_refinement_against_frozen_reference(record_property):
    import json,os,time,math,hashlib
    from pathlib import Path
    from fractions import Fraction as F
    import torch
    from dataclasses import replace
    from gffx.torch import render
    from gffx.torch.camera import Camera
    from gffx.torch._opaque_scene import _project
    from tests.python.test_opaque_full_original import original
    from tools.material_loss_cost import check_deadline
    b=Path('D:/devspace/gffx/qualification-20260913/m1-continuation-20260913');ref=b/'opaque120-20260924/original-campaign1/complete-certificate.json';cert=json.loads(ref.read_bytes());dest=Path(os.environ['GFFX_FULL_NATIVE']);assert not dest.exists();scale=int(os.environ['GFFX_SUBPIXEL_SCALE']);assert scale in (2,4);started=time.perf_counter()
    s,c,scene_sha=original();a=s.prims[1].parameters['albedo'];a.requires_grad_();camera=Camera.from_intrinsics(c.fx*scale,c.fy*scale,c.cx*scale,c.cy*scale,c.image_height*scale,c.image_width*scale,c.near,c.far,pose=c.pose)
    p0=_project(s,c,render.CULL_BACK);p1=_project(s,camera,render.CULL_BACK);assert torch.equal(p1[2][:,:2],p0[2][:,:2]*scale);assert torch.equal(p1[2][:,2:],p0[2][:,2:]);assert torch.equal(p1[4],p0[4]);del p0,p1
    rows=[];record=dict(status='incomplete',scale=scale,reference_sha256=hashlib.sha256(ref.read_bytes()).hexdigest(),scene_sha256=scene_sha,scope='Installed renderer evaluated serially on a dyadic subpixel partition; no production/default/whole-frame qualification',rows=rows)
    try:
        for y in range(scale):
            for x in range(scale):
                check_deadline(started+90);out=render.opaque_window(s,camera,(78*scale+y,101*scale+x,1,1),cull_mode=render.CULL_BACK);v=out['colour'][0,0,0];jac=torch.stack([torch.autograd.grad(v[k],a,retain_graph=k<2)[0] for k in range(3)])
                rows.append(dict(y=y,x=x,rgb=v.tolist(),mask=out['mask'].item(),jacobian=jac.tolist()));del out,v,jac
        count=scale*scale;rgb=[math.fsum(z['rgb'][k] for z in rows)/count for k in range(3)];jac=[[math.fsum(z['jacobian'][i][j] for z in rows)/count for j in range(3)] for i in range(3)];mask=math.fsum(z['mask'] for z in rows)/count;checks=[]
        for label,value,pair in [(f'rgb{k}',rgb[k],cert['value'][k]) for k in range(3)]+[(f'albedo{k}',jac[k][k],cert['albedo_derivative']) for k in range(3)]:
            ends=[dict(error=float(abs(F.from_float(value)-F(q))),limit=float(F('1e-10')+F('1e-9')*abs(F(q))),passed=abs(F.from_float(value)-F(q))<=F('1e-10')+F('1e-9')*abs(F(q))) for q in pair];checks.append(dict(channel=label,value=value,endpoints=ends))
        saved=json.loads((b/'opaque63-20260920/pixel-cpu3-values.json').read_bytes());assert abs(mask-saved['image']['mask'][0][0][0])<=1e-12;assert all(jac[i][j]==0 for i in range(3) for j in range(3) if i!=j)
        passed=all(e['passed'] for z in checks for e in z['endpoints']);record.update(status='passed_diagnostic_only' if passed else 'native_accuracy_failed',rgb=rgb,jacobian=jac,mask=mask,checks=checks);assert passed,'subpixel native experiment does not meet unchanged original tolerance'
    finally:
        record['seconds']=time.perf_counter()-started;dest.write_text(json.dumps(record,indent=2));record_property('original_subpixel',json.dumps(record))
