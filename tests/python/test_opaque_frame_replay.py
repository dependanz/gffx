"""Finite installed renderer/store composition, separate from dense-frame admission."""
import os,json,time,importlib.util
from pathlib import Path
import torch

def test_frame_store_preserves_scene_pixels_and_ordered_vertex_vjp(tmp_path,record_property):
    from tools.opaque_frame_store import Writer,Reader
    from gffx.torch._opaque_stream import _snapshot,_restore,_evaluate
    from gffx.torch._deferred import _plan_workload
    from gffx.torch import render
    from gffx.torch.camera import Camera
    from copy import copy
    import numpy as np
    spec=importlib.util.spec_from_file_location('frame_fixture',Path(__file__).with_name('test_opaque_nonlinear_scene.py'));f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
    device=os.environ.get('GFFX_KNOT_DEVICE','cpu');s,_,v,_,_=f.fixture(texture=True,rough=True,device=device);v.requires_grad_();t=v.new_tensor;size=3
    c=Camera.from_intrinsics(t(128.*size),t(128.*size),t(size/2-.03),t(size/2-.02),size,size,.1,10.)
    if device=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter();reference=render.render(s,c,coverage='opaque',cull_mode=render.CULL_NONE,opaque_tile_pixels=2)
    gc=t([.2,-.3,.7]).expand(1,size,size,3);gm=t(.1).expand(1,size,size)
    expected=torch.autograd.grad((reference['colour']*gc).sum()+(reference['mask']*gm).sum(),v)[0]
    tensors=[];tree=_snapshot((s,c,None),tensors,{});proxy=copy(c);proxy.image_height=1;proxy.image_width=800
    work=_plan_workload(s,proxy,1,67108864-1048576,('colour','mask'),None)
    config=dict(tree=tree,workload=work,limit=67108864,cull=render.CULL_NONE)
    root=Path(os.environ.get('GFFX_FRAME_STORE_ROOT',str(tmp_path/'frame')));identity={'fixture':'texture-roughness3x3','device':device};writer=Writer(root,size,size,identity,block_pixels=1)
    for offset in range(size*size):
        y,x=divmod(offset,size)
        with torch.no_grad():result=_evaluate(config,tensors,(y,x,1,1))
        for name in result:assert torch.equal(result[name],reference[name][:,y:y+1,x:x+1])
        values={k:a.detach().cpu().contiguous().numpy().tobytes() for k,a in dict(colour=result['colour'],mask=result['mask'],gcolour=gc[:,y:y+1,x:x+1],gmask=gm[:,y:y+1,x:x+1]).items()};writer.append(offset,1,values)
    receipt=writer.finish();accumulated=torch.zeros_like(v);vi=next(i for i,a in enumerate(tensors) if a is v)
    for offset,n,values in Reader(root,receipt['manifest_sha256'],identity).blocks():
        y,x=divmod(offset,size);temporary=[a.detach().requires_grad_(i==vi) for i,a in enumerate(tensors)]
        result=_evaluate(config,temporary,(y,x,1,1));cot={k:t(np.frombuffer(values[k],dtype='<f8').copy()).reshape((1,1,1,3) if k=='gcolour' else (1,1,1)) for k in ('gcolour','gmask')}
        gradient=torch.autograd.grad((result['colour'],result['mask']),temporary[vi],(cot['gcolour'],cot['gmask']),allow_unused=True)[0];accumulated.add_(gradient)
        del temporary,result,gradient,cot
    assert torch.equal(accumulated,expected) and expected.abs().max()>0
    peak=None
    if device=='cuda':torch.cuda.synchronize();peak=torch.cuda.max_memory_allocated();assert peak<67108864
    record_property('frame_replay',json.dumps(dict(device=device,extent=[3,3],pixels=9,receipt=receipt,seconds=time.perf_counter()-start,exact_rgb_mask_and_vertex_vjp=True,vertex_vjp=accumulated.cpu().tolist(),cuda_peak_allocated_bytes=peak,scope='Finite installed scene regional composition with internal disk output/cotangent storage; no production API change,dense-frame timing or arbitrary material accuracy certificate.')))
