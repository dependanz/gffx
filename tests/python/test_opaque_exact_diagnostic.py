"""Bounded installed adjoint capture; diagnostic success is not numerical acceptance."""
import importlib.util,json,sys,os
from pathlib import Path
import torch

def test_capture_weighted_scene_gradient_byte_difference(record_property,monkeypatch):
    from gffx.torch import _opaque_scene,render
    from gffx.torch.camera import Camera
    from gffx.torch._deferred import _plan_workload
    from copy import copy
    from tools.material_loss_cost import check_deadline
    import time,numpy as np
    start=time.perf_counter();spec=importlib.util.spec_from_file_location('exact_fixture',Path(__file__).with_name('test_opaque_nonlinear_scene.py'));fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
    cases=[];directory=Path(os.environ['GFFX_OPAQUE_CAPTURE_ROOT']);directory.mkdir()
    for device in ('cpu','cuda','cuda'):
        stages=[];s,c,v,_,_=fixture.fixture(texture=True,rough=True,device=device);v.requires_grad_();t=v.new_tensor;c=Camera.from_intrinsics(t(384),t(384),t(1.47),t(1.48),3,3,.1,10);proxy=copy(c);proxy.image_height=1;proxy.image_width=800;work=_plan_workload(s,proxy,1,67108864-1048576,('colour','mask'),None)
        for pixel in range(9):
            captured={};check_deadline(start+120)
            def trace(frame,event,arg):
                if event=='return' and frame.f_code.co_name=='shade' and frame.f_code.co_filename.endswith('_opaque_scene.py'):
                    for key in ('pixels','clip','q','beta','world','n','unit','values','weights','rgb','alpha','cw'):
                        x=frame.f_locals.get(key)
                        if isinstance(x,torch.Tensor) and x.requires_grad:x.retain_grad();captured[key]=x
                    x=frame.f_locals['packed'].vertices;x.retain_grad();captured['packed_vertices']=x
            old=sys.getprofile();sys.setprofile(trace)
            try:r=_opaque_scene.shade(s,c,outputs=('colour','mask'),cull_mode=render.CULL_NONE,background=None,workspace_limit_bytes=67108864,_region=(*divmod(pixel,3),1,1),_workload=work)
            finally:sys.setprofile(old)
            g,=torch.autograd.grad((r['colour']*t([.2,-.3,.7])).sum()+r['mask'].sum()*.1,v)
            row={k:(x.detach().cpu().clone(),None if x.grad is None else x.grad.detach().cpu().clone()) for k,x in captured.items()};row['final']=(v.detach().cpu(),g.cpu());stages.append(row)
            np.savez(directory/(str(len(cases))+'-'+str(pixel)+'.npz'),**{k+'_'+str(i):x.numpy() for k,z in row.items() for i,x in enumerate(z) if x is not None})
            del captured,r,g
        cases.append(stages)
    rows=[]
    for pixel in range(9):
        stage={}
        for key in cases[0][pixel]:
            stage[key]={}
            for label,index in [('value',0),('gradient',1)]:
                a,b,c=[x[pixel][key][index] for x in cases]
                if a is None:stage[key][label]=None;continue
                stage[key][label]=dict(equal=a.numpy().tobytes()==b.numpy().tobytes(),device_repeat_equal=b.numpy().tobytes()==c.numpy().tobytes(),maximum_absolute=float((a-b).abs().max()) if a.numel() else 0,different=int((a!=b).count_nonzero()),shape=list(a.shape))
        rows.append(stage)
    record_property('exact_stages',json.dumps(dict(pixels=rows,capture_root=str(directory),seconds=time.perf_counter()-start)))
    assert all(row['final']['gradient']['device_repeat_equal'] for row in rows)
    if os.environ.get('GFFX_OPAQUE_REQUIRE_EXACT')=='1':
        assert all(row[k][part]['equal'] for row in rows for k,part in [('rgb','value'),('alpha','value'),('final','gradient')])


def test_projection_stages_are_device_exact(record_property):
    from gffx.torch.camera import Camera
    from gffx.torch import transforms
    cases=[]
    for device in ('cpu','cuda'):
        t=lambda x:torch.tensor(x,dtype=torch.float64,device=device)
        c=Camera.from_intrinsics(t(384),t(384),t(1.47),t(1.48),3,3,.1,10)
        v=t([[-1/128,1/128,-2],[1/128,1/128,-2],[1/128,-1/128,-2],[-1/128,-1/128,-2]])
        matrix=c.world_to_clip();clip=transforms.transform_points(v,matrix.contiguous(),torch.tensor([0,4],dtype=torch.int32,device=device));ndc,_=transforms.perspective_divide(clip)
        cases.append({k:a.cpu() for k,a in dict(projection=c.projection(),inverse=c.world_to_view(),matrix=matrix,clip=clip,ndc=ndc,scalar=c.cx/3,tensor=c.cx/torch.full_like(c.cx,3)).items()})
    result={k:dict(equal=a.numpy().tobytes()==cases[1][k].numpy().tobytes(),cpu=a.tolist(),cuda=cases[1][k].tolist()) for k,a in cases[0].items()}
    record_property('projection_stages',json.dumps(result))
    assert all(x['equal'] for x in result.values())
