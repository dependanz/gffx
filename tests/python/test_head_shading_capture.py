"""Replay the first saved head mismatch through public shading, without a full render."""
import json,os
from pathlib import Path
import torch


def test_saved_head_first_pixel_shading_attribution(monkeypatch,record_property):
    from tools.shading_image_bundle import tensor_from_bytes
    from tools.corpus_registry import load_entry
    from tools.deferred_head import create_scene
    from tools.deferred_image_partition import plan_partitioned
    from gffx.torch import material,shade,scene,render,light
    from gffx.torch._deferred import _light_packet
    root=Path(os.environ['GFFX_HEAD_CAPTURE_ROOT'])
    cpu=tensor_from_bytes((root/(os.environ.get('GFFX_HEAD_CPU_CAPTURE','norm12-head-principled-vertex-cpu2')+'.actual.bin')).read_bytes(),(128,128,3),'float64')[:64]
    gpu=tensor_from_bytes((root/(os.environ.get('GFFX_HEAD_CUDA_CAPTURE','norm12-head-principled-vertex-cuda-row0-1')+'.actual.bin')).read_bytes(),(64,128,3),'float64')
    y,x=(cpu!=gpu).any(-1).nonzero()[0].tolist()
    data=json.loads(load_entry(Path(os.environ['GFFX_REFERENCE_CORPORA']),'synthetic_face_m1',allow_development=True)['files']['scene.json'])
    captures=[]
    with torch.no_grad():
        cpu_model,camera=create_scene(data,'principled',albedo='vertex')
        cameras,plans=plan_partitioned(cpu_model,camera,tile_rows=64,outputs=('normal','world_position','vertex_colour'))
        geometry=render(cpu_model,cameras[0],outputs=('normal','world_position','vertex_colour'),workspace_limit_bytes=plans[0]['allowance'])
        for device in ('cpu','cuda'):
            model,cam=create_scene(data,'principled',albedo='vertex',device=device)
            mat=next(p for p in model.prims if isinstance(p,scene.Material));lamp=next(p for p in model.prims if isinstance(p,scene.Light))
            n=geometry['normal'][0,y,x].reshape(1,3).to(device);point=geometry['world_position'][0,y,x].reshape(1,3).to(device)
            a=geometry['vertex_colour'][0,y,x].reshape(1,3).to(device);v=cam.pose[0,:3,3]-point
            packet=_light_packet(lamp,model.flatten(),0,point)
            captured={};calls={}
            def keep(key,value):
                if isinstance(value,torch.Tensor):captured[key]=value.detach().cpu().clone()
                elif isinstance(value,tuple):
                    for i,z in enumerate(value):keep(key+'/'+str(i),z)
            def wrap(name,fn):
                def invoke(*args,**kwargs):
                    count=calls.get(name,0);calls[name]=count+1;key=name+'/'+str(count)
                    for i,z in enumerate(args):keep(key+'/input'+str(i),z)
                    value=fn(*args,**kwargs);keep(key+'/output',value);return value
                return invoke
            with monkeypatch.context() as patch:
                for name in ('ggx_distribution','smith_visibility','fresnel_schlick'):patch.setattr(shade,name,wrap(name,getattr(shade,name)))
                patch.setattr(material,'_dot',wrap('dot',material._dot))
                patch.setattr(material,'_unit',wrap('unit',material._unit))
                original=material.PrincipledBudget._material_losses
                def losses(owner,*args):return wrap('losses',lambda *q:original(owner,*q))(*args)
                patch.setattr(material.PrincipledBudget,'_material_losses',losses)
                params=dict(mat.parameters,albedo=a)
                brdf=material.principled(n,v,packet['direction'],**params)
                keep('brdf',brdf);keep('colour',light.integrate(n,brdf,packet))
            c=captured['fresnel_schlick/0/input0'].to(device)
            keep('fresnel_power',(1-c).pow(5))
            keep('common_view_log',captures[0]['losses/0/input0'].to(device).log() if captures else captured['losses/0/input0'].to(device).log())
            keep('view_log',captured['losses/0/input0'].to(device).log())
            captures.append(captured)
    report={}
    assert captures[0].keys()==captures[1].keys()
    for k,a in captures[0].items():
        b=captures[1][k];report[k]={'different':int((a!=b).sum()),'max_abs':float((a-b).abs().max()) if a.numel() else 0.}
    record_property('head_pixel_shading',json.dumps({'pixel':[y,x],'stages':report},sort_keys=True))
    record_property('head_pixel_operands',json.dumps({k:v.tolist() for k,v in captures[0].items() if k in ('fresnel_schlick/0/input0','fresnel_schlick/0/input1','fresnel_power','brdf','colour','losses/0/input0','view_log','common_view_log')}))
