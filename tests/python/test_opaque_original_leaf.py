"""Capture public original-mesh input adjoints before caller contractions."""
import os,json,hashlib,time,math
from pathlib import Path
from dataclasses import replace
from copy import copy
import torch
from tests.python.test_opaque_full_original import original

def test_original_leaf_adjoints(record_property):
    from gffx.torch import render,scene
    from gffx.torch._reference_channels import _sum_last
    device=os.environ.get('GFFX_KNOT_DEVICE','cpu');s,c,sha=original(device);m=s.prims[0];v=m.vertices.detach().requires_grad_();n=m.attributes['normals'].values.detach().requires_grad_();pose=c.pose.detach().requires_grad_();c=copy(c);c.pose=pose
    s=scene.Scene([replace(m,vertices=v,attributes=dict(m.attributes,normals=replace(m.attributes['normals'],values=n))),*s.prims[1:]])
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter();rgb=render.opaque_window(s,c,(78,101,1,1),cull_mode=render.CULL_BACK)['colour'].reshape(3);rows=[torch.autograd.grad(rgb[i],(v,n,pose),retain_graph=i<2) for i in range(3)]
    jac={name:torch.stack([row[j] for row in rows]).contiguous() for j,name in enumerate(['vertices','normals','pose'])};p=Path(os.environ['GFFX_FULL_NATIVE']);assert not p.exists();binary={}
    for name,value in jac.items():
        b=value.cpu().numpy().tobytes();f=p.with_name(p.stem+'-'+name+'.bin');assert not f.exists();f.write_bytes(b);binary[name]=dict(path=str(f),sha256=hashlib.sha256(b).hexdigest(),shape=list(value.shape),entries=value.numel())
    directions={};zero=lambda x:torch.zeros_like(x)
    vx=zero(v);vx[:,0]=.01;px=zero(pose);px[0,0,3]=.01
    directions['normal_scale']={'normals':n.detach()};directions['normal_tilt']={'normals':.01*torch.stack((n[:,2],zero(n[:,1]),-n[:,0]),-1)};directions['vertex_x']={'vertices':vx};directions['camera_x']={'pose':px};directions['joint_translation']={'vertices':vx,'pose':px};directions['radial_scale']={'vertices':(v-pose[0,:3,3])*.01}
    contractions={}
    for kind,parts in directions.items():
        terms=[torch.stack([_sum_last((jac[name][ch]*d).reshape(-1)) for ch in range(3)]) for name,d in parts.items()];ordered=terms[0]
        for term in terms[1:]:ordered=ordered+term
        host=[math.fsum(float(g)*float(d) for name,vec in parts.items() for g,d in zip(jac[name][ch].detach().cpu().reshape(-1),vec.detach().cpu().reshape(-1))) for ch in range(3)]
        contractions[kind]=dict(ordered=ordered.detach().cpu().tolist(),ordered_bytes=ordered.detach().cpu().numpy().tobytes().hex(),host_fsum=host,direction_sha256={name:hashlib.sha256(vec.detach().cpu().contiguous().numpy().tobytes()).hexdigest() for name,vec in parts.items()})
    row=dict(device=device,scene_sha256=sha,colour=rgb.detach().cpu().tolist(),colour_bytes=rgb.detach().cpu().numpy().tobytes().hex(),binary=binary,contractions=contractions,seconds=time.perf_counter()-start,scope='Public-input adjoints and explicit ordered caller contractions; no independent nonzero derivative acceptance.')
    if device=='cuda':row['gpu_peak_bytes']=torch.cuda.max_memory_allocated();assert row['gpu_peak_bytes']<=67108864
    p.write_text(json.dumps(row,indent=2),encoding='utf8');record_property('original_leaf',json.dumps(row));assert all(torch.isfinite(x).all() for x in jac.values())


def test_original_camera_chain(record_property,monkeypatch):
    """Locate device differences before changing inverse or composition semantics."""
    from gffx.torch import render
    device=os.environ.get('GFFX_KNOT_DEVICE','cpu');s,c,sha=original(device)
    c.pose=c.pose.detach().requires_grad_();captured=[];adjoints=[]
    def capture(camera):
        inverse=camera.world_to_view();projection=camera.projection();matrix=projection@inverse
        captured.append((matrix,inverse,projection))
        if matrix.requires_grad:
            row={};adjoints.append(row)
            matrix.register_hook(lambda g:row.update(matrix=g.detach().clone()))
            inverse.register_hook(lambda g:row.update(inverse=g.detach().clone()))
        return matrix
    monkeypatch.setattr(type(c),'world_to_clip',capture)
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    rgb=render.opaque_window(s,c,(78,101,1,1),cull_mode=render.CULL_BACK)['colour'].reshape(3)
    rows=[torch.autograd.grad(rgb[k],c.pose,retain_graph=k<2)[0] for k in range(3)]
    assert len(adjoints)==3;matrix,inverse,projection=captured[0]
    data={name:torch.stack([row[name] for row in adjoints]).detach().cpu().contiguous() for name in ['matrix','inverse']}
    data['pose']=torch.stack(rows).detach().cpu().contiguous()
    p=Path(os.environ['GFFX_FULL_NATIVE']);assert not p.exists()
    result=dict(device=device,scene_sha256=sha,values={k:v.tolist() for k,v in data.items()},bytes={k:v.numpy().tobytes().hex() for k,v in data.items()},inputs={k:v.detach().cpu().tolist() for k,v in dict(pose=c.pose,inverse=inverse,projection=projection).items()},colour=rgb.detach().cpu().tolist(),scope='Diagnostic camera composition intermediates; not independent gradient acceptance.')
    if device=='cuda':result['gpu_peak_bytes']=torch.cuda.max_memory_allocated();assert result['gpu_peak_bytes']<=67108864
    p.write_text(json.dumps(result,indent=2),encoding='utf8');record_property('camera_chain',json.dumps(result))
    assert all(torch.isfinite(v).all() for v in data.values())
