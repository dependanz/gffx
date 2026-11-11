"""Small portable evaluated-scene bundle, using stdlib ZIP/JSON and raw arrays."""
import json
import zipfile
import torch
from dataclasses import replace
from . import scene,render
from .camera import Camera
from ._scene_builders import resolve_device
from ._deferred import TextureSource


def load(path,*,device='auto',dtype=None):
    device=resolve_device(device);dtype=dtype or torch.float32
    if dtype not in (torch.float32,torch.float64):raise TypeError('scene precision must be float32/float64')
    with zipfile.ZipFile(path) as archive:
        spec=json.loads(archive.read('scene.json'))
        if spec.get('format')!='gffx-scene-1':raise ValueError('unknown scene bundle format')
        def array(record):
            source={'float32':torch.float32,'int32':torch.int32}[record['dtype']]
            value=torch.frombuffer(bytearray(archive.read(record['file'])),dtype=source).reshape(record['shape']).clone()
            return value.to(device=device,dtype=dtype if source.is_floating_point else source)
        textures={}
        for name,record in spec.get('textures',{}).items():
            image=array(record);h,w,c=image.shape
            origin=record.get('origin','top-left')
            if origin not in ('top-left','bottom-left'):raise ValueError('texture origin must be top-left or bottom-left')
            # Bundle UVs use the Blender/OpenGL lower-left convention. Ordinary
            # decoded image files start with their top row; native v=0 samples
            # row zero, so orient the array explicitly before building its mips.
            if origin=='top-left':image=image.flip(0)
            pyramid,offsets=render.texture_pyramid(image.contiguous())
            textures[name]=TextureSource(pyramid,offsets,h,w,c,scalar=(c==1))
        prims=[]
        for record in spec['materials']:
            params={k:(textures[v['texture']] if isinstance(v,dict) and 'texture' in v else v) for k,v in record['parameters'].items()}
            if record.get('kind')=='diffuse':prims.append(scene.Material.diffuse(record['name'],params['base_color'],dtype=dtype,device=device))
            elif record.get('kind')=='dielectric':prims.append(scene.Material.dielectric(record['name'],dtype=dtype,device=device,**params))
            else:prims.append(scene.Material.pbr(record['name'],dtype=dtype,device=device,**params))
        for record in spec['meshes']:
            attributes={key:scene.Attribute(array(value),'face_varying') for key,value in record['attributes'].items()}
            prims.append(scene.Mesh(record['name'],array(record['vertices']),array(record['faces']),attributes,record['material']))
        for record in spec['lights']:
            params=record['parameters']
            if record['kind']=='rectangle':prims.append(scene.Light.rectangle(record['name'],dtype=dtype,device=device,**params))
            else:prims.append(getattr(scene.Light,record['kind'])(record['name'],dtype=dtype,device=device,**params))
        c=spec['camera'];pose=array(c['pose'])[None]
        camera=Camera.from_intrinsics(c['fx'],c['fy'],c['cx'],c['cy'],c['height'],c['width'],c['near'],c['far'],pose=pose)
        prims.append(scene.Camera('camera',camera))
    # Coalesce static face-corner attributes before rendering. The source prims
    # borrow arena views; the composer borrows the same storage for identity
    # transforms. Mutation versions invalidate the normal preparation shortcut.
    meshes=[p for p in prims if isinstance(p,scene.Mesh)]
    if meshes:
        keys=set.intersection(*(set(p.attributes)for p in meshes))
        updates={id(p):dict(p.attributes)for p in meshes}
        from ._reference_channels import divide,sqrt_unit,sum3
        for key in keys:
            if any(p.attributes[key].domain!='face_varying'for p in meshes):continue
            values=[]
            for p in meshes:
                value=p.attributes[key].values
                if key=='normals':
                    identity=torch.eye(3,device=value.device,dtype=value.dtype)[None]
                    value=torch.einsum('bfkc,bcd->bfkd',value[None],identity)
                    value=divide(value,value.abs().amax(-1,keepdim=True))
                    value=divide(value,sqrt_unit(sum3(value.square())).unsqueeze(-1))[0]
                values.append(value)
            arena=torch.cat(values,dim=0);offset=0
            for p,value in zip(meshes,values):
                attribute=scene.Attribute(arena[offset:offset+len(value)],'face_varying');offset+=len(value)
                if key=='normals':object.__setattr__(attribute,'_unit_version',attribute.values._version)
                updates[id(p)][key]=attribute
            del values,arena
        prims=[replace(p,attributes=updates[id(p)])if isinstance(p,scene.Mesh)else p for p in prims]
    return scene.Scene(prims)
