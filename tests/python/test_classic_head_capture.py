"""Saved Classic head one-pixel attribution; completion is not numerical acceptance."""
import os,json,math
from pathlib import Path
import torch

def test_classic_head_common_pixel_attribution(monkeypatch,record_property):
    from tools.shading_image_bundle import tensor_from_bytes
    from tools.corpus_registry import load_entry
    from tools.deferred_head import create_scene
    from tools.deferred_image_partition import plan_partitioned
    from gffx.torch import scene,material,shade,light,render
    from gffx.torch._deferred import _light_packet
    root=Path(os.environ['GFFX_HEAD_CAPTURE_ROOT'])
    cpu=tensor_from_bytes((root/'ggx15-head-classic-constant-cpu1.actual.bin').read_bytes(),(128,128,3),'float64')[:64]
    gpu=tensor_from_bytes((root/'ggx15-head-classic-constant-cuda-row0-1.actual.bin').read_bytes(),(64,128,3),'float64')
    y,x=(cpu!=gpu).any(-1).nonzero()[0].tolist()
    data=json.loads(load_entry(Path(os.environ['GFFX_REFERENCE_CORPORA']),'synthetic_face_m1',allow_development=True)['files']['scene.json'])
    rows=[]
    with torch.no_grad():
        base,camera=create_scene(data,'classic');cams,plans=plan_partitioned(base,camera,tile_rows=64,outputs=('normal','world_position'))
        geo=render(base,cams[0],outputs=('normal','world_position'),workspace_limit_bytes=plans[0]['allowance'])
        for device in ('cpu','cuda'):
            model,cam=create_scene(data,'classic',device=device);mat=next(p for p in model.prims if isinstance(p,scene.Material));lamp=next(p for p in model.prims if isinstance(p,scene.Light))
            normal=geo['normal'][0,y,x].reshape(1,3).to(device);point=geo['world_position'][0,y,x].reshape(1,3).to(device);view=cam.pose[0,:3,3]-point
            packet=_light_packet(lamp,model.flatten(),0,point);values={}
            original=shade.blinn_phong
            def capture(q,n):
                result=original(q,n)
                for k,z in dict(q=q,n=n,value=result,power=q.pow(n),square=q*q,scalar_scale=n/(8*math.pi)+1/math.pi,tensor_scale=n/n.new_tensor(8*math.pi)+1/math.pi).items():values[k]=z.cpu().tolist()
                return result
            with monkeypatch.context() as patch:
                patch.setattr(shade,'blinn_phong',capture)
                brdf=material.classic(normal,view,packet['direction'],**mat.parameters)
                values.update(brdf=brdf.cpu().tolist(),colour=light.integrate(normal,brdf,packet).cpu().tolist())
            rows.append(values)
    assert rows[0]['q']==rows[1]['q'] and rows[0]['n']==rows[1]['n']
    record_property('classic_pixel',json.dumps(dict(pixel=[y,x],cpu=rows[0],cuda=rows[1])))
