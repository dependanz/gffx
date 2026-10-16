"""Captured fixed-head geometry attribution; diagnostic completion is not acceptance."""
import json,os,hashlib,torch

def test_fixed_head_geometry_byte_attribution(record_property):
    from tools.corpus_registry import load_entry
    from tools.deferred_head import create_scene
    from tools.deferred_image_partition import plan_partitioned
    from gffx.torch import render
    data=json.loads(load_entry(Path(os.environ['GFFX_REFERENCE_CORPORA']),'synthetic_face_m1',allow_development=True)['files']['scene.json'])
    names=('world_position','normal','uv','vertex_colour','barycentric','face_index','depth')
    pairs=[]
    for device in ('cpu','cuda'):
        with torch.no_grad():
            model,camera=create_scene(data,'classic',albedo='vertex',device=device)
            crop,plans=plan_partitioned(model,camera,tile_rows=64,outputs=names)
            rows={'input_normals':model.prims[0].attributes['normals'].values,'packed_normals':model.flatten().attributes['normals'].values}
            rows.update(render(model,crop[0],outputs=names,workspace_limit_bytes=plans[0]['allowance']))
            pairs.append({k:v.detach().cpu().contiguous().reshape(-1).clone() for k,v in rows.items()})
    report={}
    for key in pairs[0]:
        a,b=pairs[0][key],pairs[1][key]
        # Finite geometry fields and integer IDs; depth may include equal infinities.
        same=torch.equal(a.view(torch.uint8),b.view(torch.uint8));finite=torch.isfinite(a)&torch.isfinite(b)
        report[key]={'same_bytes':same,'different_elements':int((a!=b).sum()),'max_finite_absolute_error':float((a[finite]-b[finite]).abs().max()) if bool(finite.any()) else None}
    record_property('geometry_byte_attribution',json.dumps(report,sort_keys=True))
    assert all(row['same_bytes'] for row in report.values()), report

from pathlib import Path


def test_deferred_unit_nondyadic_values_and_first_vjp_exact_cpu_cuda():
    from gffx.torch._deferred import _unit
    values=torch.arange(1,100,dtype=torch.float64).reshape(33,3)*.017+torch.tensor([.013,.127,-.031],dtype=torch.float64)
    cot=torch.arange(99,dtype=torch.float64).reshape(33,3).remainder(11)*.125-.375
    results=[]
    for device in ('cpu','cuda'):
        x=values.to(device).requires_grad_();y=_unit(x,'control');g,=torch.autograd.grad(y,x,cot.to(device))
        results.append((y.detach().cpu().contiguous(),g.cpu().contiguous()))
    for a,b in zip(*results):assert torch.equal(a.reshape(-1).view(torch.uint8),b.reshape(-1).view(torch.uint8))
    length=values.square().sum(-1,keepdim=True).sqrt();n=values/length
    expected=(cot-n*(cot*n).sum(-1,keepdim=True))/length
    torch.testing.assert_close(results[0][1],expected,atol=1e-10,rtol=1e-9)


def test_fixed_head_packed_normal_local_attribution(record_property):
    from tools.corpus_registry import load_entry
    from tools.deferred_head import create_scene
    from gffx.torch import scene
    data=json.loads(load_entry(Path(os.environ['GFFX_REFERENCE_CORPORA']),'synthetic_face_m1',allow_development=True)['files']['scene.json'])
    all_rows=[];original_divide=scene.divide;original_sum=scene.sum3
    for device in ('cpu','cuda'):
        captured={}
        def divide(a,b):
            index=len([k for k in captured if k.startswith('a')]);captured['a'+str(index)]=a.detach().cpu().contiguous();captured['b'+str(index)]=b.detach().cpu().contiguous()
            value=original_divide(a,b);captured['result'+str(index)]=value.detach().cpu().contiguous();return value
        def sum3(a):
            value=original_sum(a);captured['sum_input']=a.detach().cpu().contiguous();captured['sum_result']=value.detach().cpu().contiguous();return value
        with torch.no_grad():
            model,_=create_scene(data,'classic',albedo='vertex',device=device)
            scene.divide=divide;scene.sum3=sum3
            try:model.flatten()
            finally:scene.divide=original_divide;scene.sum3=original_sum
        all_rows.append(captured)
    report={}
    for k in all_rows[0]:
        a,b=all_rows[0][k],all_rows[1][k]
        report[k]={'different':int((a!=b).sum()),'max_abs':float((a-b).abs().max())}
    q=all_rows[0]['sum_result'];gpu=q.cuda().sqrt().cpu();cpu=q.sqrt()
    report['common_sqrt']={'different':int((cpu!=gpu).sum()),'max_abs':float((cpu-gpu).abs().max())}
    record_property('packed_local',json.dumps(report,sort_keys=True))
