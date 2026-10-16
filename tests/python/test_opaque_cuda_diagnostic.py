"""Capture the first differing installed scene adjoint, without changing execution."""
import torch,json,sys,importlib.util
from pathlib import Path

def test_capture_opaque_cuda_scene_adjoint_stages(record_property,monkeypatch):
    from gffx.torch import render
    spec=importlib.util.spec_from_file_location('diag_fixture',Path(__file__).with_name('test_opaque_nlinear_scene.py'.replace('nlinear','nonlinear')));fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
    from gffx.torch import _opaque_scene
    original=_opaque_scene.shade
    cases=[]
    for device in ('cpu','cuda','cuda'):
        captured={}
        def trace(frame,event,arg):
            if event=='return' and frame.f_code.co_name=='shade' and frame.f_code.co_filename.endswith('_opaque_scene.py'):
                for key in ('pixels','clip','q','beta','world','n','unit','values','weights','rgb','alpha','cw','numerator','den'):
                    value=frame.f_locals.get(key)
                    if isinstance(value,torch.Tensor) and value.requires_grad:value.retain_grad();captured[key]=value
                value=frame.f_locals['packed'].vertices
                if value.requires_grad:value.retain_grad();captured['packed_vertices']=value
        model,cam,v,f,b=fixture.fixture(texture=True,rough=True,device=device);v.requires_grad_()
        def instrumented(*args,**kwargs):
            previous=sys.getprofile();sys.setprofile(trace)
            try:return original(*args,**kwargs)
            finally:sys.setprofile(previous)
        monkeypatch.setattr(_opaque_scene,'shade',instrumented)
        o=render.render(model,cam,coverage='opaque',cull_mode=render.CULL_NONE);g,=torch.autograd.grad(o['colour'].sum(),v)
        cases.append({k:(x.detach().cpu(),None if x.grad is None else x.grad.detach().cpu()) for k,x in captured.items()})
        cases[-1]['final']=(v.detach().cpu(),g.cpu())
    rows={}
    for key in cases[0]:
        row={}
        for label,index in [('value',0),('gradient',1)]:
            a,b,c=[x[key][index] for x in cases]
            if a is None:row[label]=None;continue
            row[label]=dict(equal=torch.equal(a,b),device_repeat_equal=torch.equal(b,c),maximum_absolute=float((a-b).abs().max()) if a.numel() else 0,different=int((a!=b).count_nonzero()),shape=list(a.shape))
        rows[key]=row
    record_property('stages',json.dumps(rows))
    print(json.dumps(rows,indent=2))
