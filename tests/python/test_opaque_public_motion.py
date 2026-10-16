"""Public scene motion versus complete independently integrated original pixel."""
import hashlib,json,os,struct,time
from pathlib import Path
from fractions import Fraction as Q
import torch


def test_public_original_motion_against_complete_reference(record_property):
    from gffx.torch import render
    from tools.opaque_face_certificate import load_pixel_bundle
    from tests.python.test_opaque_full_original import original
    inp=Path(os.environ['GFFX_FULL_INPUT']);digest=hashlib.sha256(inp.read_bytes()).hexdigest()
    assert digest=='a15bea40cb5cdead7994d828b57f5dcfac9f44fab0292a06e7d0a7b08719f73a'
    data=json.loads(inp.read_bytes());path=Path(os.environ['GFFX_MOTION_REFERENCE']);refsha=os.environ['GFFX_MOTION_REFERENCE_SHA']
    direction=os.environ['GFFX_PUBLIC_MOTION'];physical=direction in ('world_x','camera_x','world_shear_xz')
    world_sha=None
    if physical:
        wp=Path(os.environ['GFFX_WORLD_PROFILE']);world_sha=hashlib.sha256(wp.read_bytes()).hexdigest()
        assert world_sha==os.environ['GFFX_WORLD_PROFILE_SHA']
        profile=json.loads(wp.read_bytes());assert profile['input_sha256']==digest
    reference=load_pixel_bundle(path,sha256=refsha,data=data,input_sha256=digest,world_profile_sha256=world_sha)
    if direction=='camera_x':
        from tools.opaque_camera_reversal import camera_from_world
        reference=camera_from_world(data,profile,reference,input_sha256=digest,world_profile_sha256=world_sha)
    expected={'normal_tilt':'normal_tilt','principal_x':'screen_x','world_x':'world_x','camera_x':'camera_x','world_shear_xz':'world_shear_xz'}
    assert direction in expected and reference['motion']==expected[direction]
    device=os.environ.get('GFFX_KNOT_DEVICE','cpu');s,c,scene_sha=original(device)
    theta=c.fx.new_tensor(0.).requires_grad_()
    if direction=='normal_tilt':
        from dataclasses import replace
        from gffx.torch import scene
        from gffx.torch._reference_channels import expand
        m=s.prims[0];n=m.attributes['normals'].values
        # Same infinitesimal rotation; fixed reduction is part of this caller's parameterization.
        normal=n+expand(theta,n.shape)*.01*torch.stack((n[:,2],torch.zeros_like(n[:,1]),-n[:,0]),-1)
        assert torch.equal(normal.detach().contiguous().view(torch.uint8),n.detach().contiguous().view(torch.uint8))
        s=scene.Scene([replace(m,attributes=dict(m.attributes,normals=replace(m.attributes['normals'],values=normal))),*s.prims[1:]])
    elif direction=='principal_x':
        from copy import copy
        c=copy(c);c.cx=c.cx+theta*.01
    else:
        from dataclasses import replace
        from copy import copy
        from gffx.torch import scene,mesh
        from gffx.torch._reference_channels import expand
        assert scene_sha==profile['scene_sha256'] and c.pose[0].tolist()==profile['camera']['pose']
        m=s.prims[0];v=m.vertices
        assert all(v[int(i)].tolist()==xyz for i,xyz in profile['vertices'].items())
        if direction=='camera_x':
            c=copy(c);velocity=torch.zeros_like(c.pose);velocity[0,0,3]=.01
            c.pose=c.pose+expand(theta,c.pose.shape)*velocity
        else:
            vx=expand(theta,v[:,0].shape)*.01
            if direction=='world_shear_xz':vx=vx*v[:,2]
            vertices=v+torch.stack((vx,torch.zeros_like(vx),torch.zeros_like(vx)),-1)
            assert torch.equal(vertices.detach().contiguous().view(torch.uint8),v.detach().contiguous().view(torch.uint8))
            attributes=dict(m.attributes)
            if direction=='world_shear_xz':
                normals=mesh.vertex_normals(vertices,m.faces,weighting=mesh.WEIGHTING_AREA)
                assert torch.equal(normals.detach().contiguous().view(torch.uint8),m.attributes['normals'].values.detach().contiguous().view(torch.uint8))
                attributes['normals']=replace(m.attributes['normals'],values=normals)
            s=scene.Scene([replace(m,vertices=vertices,attributes=attributes),*s.prims[1:]])
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter();out=render.opaque_window(s,c,(78,101,1,1),cull_mode=render.CULL_BACK)
    rgb=out['colour'].reshape(3);grad=torch.stack([torch.autograd.grad(rgb[k],theta,retain_graph=k<2)[0] for k in range(3)])
    actual=dict(value=rgb.detach().cpu().tolist(),derivative=grad.detach().cpu().tolist());checks=[]
    for quantity,values in actual.items():
        for channel,(value,pair) in enumerate(zip(values,reference[quantity])):
            lo,hi=map(Q,pair);assert hi-lo<=Q('2e-12')
            for endpoint in (lo,hi):
                error=abs(Q(value)-endpoint);limit=Q('1e-10')+Q('1e-9')*abs(endpoint)
                checks.append(dict(quantity=quantity,channel=channel,error_over_limit=float(error/limit),passed=error<=limit))
    result=dict(device=device,direction=direction,reference_motion=reference['motion'],reference_sha256=refsha,scene_sha256=scene_sha,region=[78,101,1,1],actual=actual,checks=checks,bytes=struct.pack('<6d',*actual['value'],*actual['derivative']).hex(),seconds=time.perf_counter()-start,world_profile_sha256=world_sha,scope='Actual public original-scene direction named above with all thirteen visible contributions. Physical shear recomputes area-weighted vertex normals. No arbitrary camera/deformation or full-frame admission.')
    if device=='cuda':
        result['gpu_peak_bytes']=torch.cuda.max_memory_allocated();assert result['gpu_peak_bytes']<=67108864
    target=Path(os.environ['GFFX_FULL_NATIVE']);assert not target.exists();target.write_text(json.dumps(result,indent=2),encoding='utf8');record_property('public_original_motion',json.dumps(result));assert all(r['passed'] for r in checks),checks


def test_extract_original_world_motion_profile(record_property):
    from tests.python.test_opaque_full_original import original
    from gffx.torch._opaque_scene import _project,_normals
    from gffx.torch import render
    data_path=Path(os.environ['GFFX_FULL_INPUT']);data=json.loads(data_path.read_bytes())
    assert hashlib.sha256(data_path.read_bytes()).hexdigest()=='a15bea40cb5cdead7994d828b57f5dcfac9f44fab0292a06e7d0a7b08719f73a'
    s,c,scene_sha=original('cpu')
    with torch.no_grad():
        packed,c,px,keep,clip=_project(s,c,render.CULL_BACK);normals=_normals(packed)
        for row in data['faces']:
            ids=packed.faces[row['face']].long()
            assert ids.tolist()==row['vertices']
            assert px[ids,:2].tolist()==row['corners'] and clip[ids,3].tolist()==row['clip_w']
            assert normals[row['face']].tolist()==row['normal']
            assert (c.pose[0,:3,3]-packed.vertices[ids]).tolist()==row['view']
    assert c.pose[0].tolist()==[[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,3.],[0.,0.,0.,1.]]
    used=data['selected']['original_vertex_ids'];profile=dict(status='verified_source_profile',input_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),scene_sha256=scene_sha,camera=dict(fx=c.fx.item(),fy=c.fy.item(),cx=c.cx.item(),cy=c.cy.item(),pose=c.pose[0].tolist()),vertices={str(i):packed.vertices[i].tolist() for i in used},scope='Frozen actual camera and world vertices for independent motion construction; not a gradient certificate.')
    p=Path(os.environ['GFFX_FULL_NATIVE']);assert not p.exists();p.write_text(json.dumps(profile,indent=2),encoding='utf8');record_property('world_motion_profile',json.dumps(profile))
