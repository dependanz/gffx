"""Validated in-memory scenes, packed for the existing tensor operations.

See docs/SCENE.md. Eager composition only; no native scene graph or renderer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from collections.abc import Mapping
import torch
from ._reference_channels import sum3, divide, sqrt_unit

__all__ = ['Attribute','Mesh','Material','Light','Camera','Rig','Scene','PackedScene','PackedAttribute']
_DOMAINS={'constant':1,'uniform':2,'vertex':2,'face_varying':3}


def _mapping(value):
    if not isinstance(value,Mapping) or any(not isinstance(k,str) or not k for k in value):
        raise TypeError('scene mappings require nonempty string keys')
    return MappingProxyType(dict(value))


@dataclass(frozen=True)
class Attribute:
    values: torch.Tensor
    domain: str


@dataclass(frozen=True,kw_only=True)
class _Prim:
    parent: str | None = None
    transform: torch.Tensor | None = None


@dataclass(frozen=True)
class Mesh(_Prim):
    name: str
    vertices: torch.Tensor
    faces: torch.Tensor
    attributes: Mapping = field(default_factory=dict)
    material: str | None = None

    def __post_init__(self):
        object.__setattr__(self,'attributes',_mapping(self.attributes))

    @classmethod
    def quad(cls,name,*,center=(0,0,0),u=(1,0,0),v=(0,1,0),material=None,dtype=None,device=None):
        """Author a quad from its center and two full-width oriented spans."""
        from ._scene_builders import quad
        return quad(cls,name,center=center,u=u,v=v,material=material,dtype=dtype,device=device)

    @classmethod
    def sphere(cls,name,*,radius=1.,center=(0,0,0),segments=48,rings=24,material=None,dtype=None,device=None):
        """Author a smooth closed UV sphere with connected radius/center tensors."""
        from ._scene_builders import sphere
        return sphere(cls,name,radius=radius,center=center,segments=segments,rings=rings,
                      material=material,dtype=dtype,device=device)


@dataclass(frozen=True)
class Material(_Prim):
    name: str
    evaluator: object
    parameters: Mapping = field(default_factory=dict)
    emission: object = None

    def __post_init__(self):
        if not callable(self.evaluator):raise TypeError('material evaluator must be callable')
        object.__setattr__(self,'parameters',_mapping(self.parameters))

    @classmethod
    def diffuse(cls,name,albedo,*,dtype=None,device=None):
        from ._scene_builders import diffuse
        return diffuse(cls,name,albedo,dtype=dtype,device=device)

    @classmethod
    def emissive(cls,name,radiance,*,dtype=None,device=None):
        from ._scene_builders import emissive
        return emissive(cls,name,radiance,dtype=dtype,device=device)


@dataclass(frozen=True)
class Light(Material):
    """Light evaluator and parameters, optionally with a visible emitter surface."""
    visible: bool = False

    def __post_init__(self):
        super().__post_init__()
        if type(self.visible) is not bool:raise TypeError('visible must be boolean')

    @classmethod
    def rectangle(cls,name,*,center,u,v,radiance,samples=4,visible=True,dtype=None,device=None):
        from ._scene_builders import rectangle
        return rectangle(cls,name,center=center,u=u,v=v,radiance=radiance,samples=samples,
                         visible=visible,dtype=dtype,device=device)


@dataclass(frozen=True)
class Camera(_Prim):
    name: str
    camera: object

    def __post_init__(self):
        from .camera import Camera as CameraObject
        if not isinstance(self.camera,CameraObject):raise TypeError('camera payload must be gffx.torch.camera.Camera')


@dataclass(frozen=True)
class Rig(_Prim):
    name: str


@dataclass(frozen=True)
class PackedAttribute:
    values: torch.Tensor
    present: torch.Tensor
    domains: tuple


@dataclass(frozen=True)
class PackedScene:
    vertices: torch.Tensor
    faces: torch.Tensor
    vertex_offsets: torch.Tensor
    face_offsets: torch.Tensor
    object_vertex_offsets: torch.Tensor
    object_face_offsets: torch.Tensor
    mesh_names: tuple
    face_material: torch.Tensor
    attributes: Mapping
    world_matrices: Mapping
    materials: tuple
    lights: tuple
    cameras: tuple
    rigs: tuple


def _float(t,name,dtype,device):
    if not isinstance(t,torch.Tensor) or t.layout!=torch.strided:
        raise TypeError(name+' must be a strided tensor')
    if t.dtype not in (torch.float32,torch.float64) or t.dtype!=dtype:
        raise TypeError(name+' must match scene float32/float64 precision')
    if t.device.type not in ('cpu','cuda') or t.device!=device:
        raise ValueError(name+' must match scene CPU/CUDA device; move explicitly')
    if not bool(torch.isfinite(t).all()):raise ValueError(name+' must be finite')


def _batch(t,rank,batch):
    return t.unsqueeze(0).expand((batch,)+tuple(t.shape)) if t.ndim==rank else t.expand((batch,)+tuple(t.shape[1:]))


class Scene:
    """Insertion-ordered prims; borrowed tensors are checked afresh by flatten."""
    def __init__(self,prims):
        self.prims=tuple(prims)
        if any(type(p) not in (Mesh,Material,Light,Camera,Rig) for p in self.prims):
            raise TypeError('scene accepts only typed prims')
        names=[p.name for p in self.prims]
        if any(not isinstance(n,str) or not n for n in names):raise ValueError('prim names must be nonempty strings')
        if len(set(names))!=len(names):raise ValueError('duplicate prim name')

    def to(self,device=None,*,dtype=None):
        """Explicitly transfer a fresh scene, preserving differentiable tensor links."""
        from ._scene_builders import move_scene
        return move_scene(self,device=device,dtype=dtype)

    def flatten(self,*,dtype=None,device=None,batch_size=None):
        if batch_size is not None and (type(batch_size)!=int or batch_size<=0):
            raise ValueError('batch_size must be a positive integer')
        tensors=[t for p in self.prims for t in
                 ((p.vertices,p.transform) if isinstance(p,Mesh) else (p.transform,)) if t is not None]
        first=next((t for t in tensors if isinstance(t,torch.Tensor)),None)
        dtype=dtype if dtype is not None else first.dtype if first is not None else torch.float32
        device=torch.device(device) if device is not None else first.device if first is not None else torch.device('cpu')
        if device.type=='cuda' and device.index is None:device=torch.device('cuda',torch.cuda.current_device())
        if dtype not in (torch.float32,torch.float64):raise TypeError('scene precision must be float32 or float64')
        if device.type not in ('cpu','cuda'):raise ValueError('scene device must be CPU or CUDA')
        sizes=[batch_size or 1];meshes=tuple(p for p in self.prims if type(p)==Mesh)
        materials=tuple(p for p in self.prims if type(p)==Material)
        material_ids={p.name:i for i,p in enumerate(materials)}
        nodes={p.name:p for p in self.prims};channels={}
        for p in self.prims:
            if p.parent is not None and (not isinstance(p.parent,str) or p.parent not in nodes):
                raise ValueError('missing parent for '+p.name)
            if p.transform is not None:
                t=p.transform;_float(t,'transform',dtype,device)
                if t.ndim not in (2,3) or t.shape[-2:]!=(4,4):raise ValueError('transform shape must be [4,4] or [B,4,4]')
                if t.ndim==3:sizes.append(t.shape[0])
                last=torch.tensor([0,0,0,1],dtype=dtype,device=device)
                if not bool((t[...,3,:]==last).all()):raise ValueError('transform must be affine')
                det=torch.linalg.det(t[...,:3,:3])
                if not bool((torch.isfinite(det)&(det!=0)).all()):raise ValueError('transform must be invertible and representable')
            if type(p)!=Mesh:continue
            v,f=p.vertices,p.faces;_float(v,'vertices',dtype,device)
            if v.ndim not in (2,3) or v.shape[-1]!=3:raise ValueError('vertices shape must be [V,3] or [B,V,3]')
            if v.ndim==3:sizes.append(v.shape[0])
            if not isinstance(f,torch.Tensor) or f.layout!=torch.strided or f.dtype!=torch.int32:raise TypeError('faces must be int32 strided tensors')
            if f.device!=device or f.ndim!=2 or f.shape[-1]!=3:raise ValueError('faces device/shape mismatch')
            if bool(((f<0)|(f>=v.shape[-2])).any()):raise ValueError('face index out of range')
            if p.material is not None and (not isinstance(p.material,str) or p.material not in material_ids):raise ValueError('unknown material reference')
            for name,a in p.attributes.items():
                if not isinstance(a,Attribute):raise TypeError('mesh attributes must be Attribute objects')
                if a.domain not in _DOMAINS:raise ValueError('unknown attribute domain')
                t=a.values;_float(t,'attribute '+name,dtype,device);rank=_DOMAINS[a.domain]
                if t.ndim not in (rank,rank+1) or t.shape[-1]<=0:raise ValueError('attribute shape/rank mismatch')
                shape={'constant':(),'uniform':(f.shape[0],),'vertex':(v.shape[-2],),'face_varying':(f.shape[0],3)}[a.domain]
                if tuple(t.shape[-rank:-1])!=shape:raise ValueError('attribute domain shape mismatch')
                if t.ndim==rank+1:sizes.append(t.shape[0])
                c=t.shape[-1]
                if c!=channels.setdefault(name,c) or c!={'uv':2,'colour':3,'normals':3}.get(name,c):raise ValueError('attribute channel count mismatch')
                if name=='normals' and bool((torch.linalg.vector_norm(t,dim=-1)==0).any()):raise ValueError('normals must be nonzero')
        batch=max(sizes)
        if any(n not in (1,batch) for n in sizes) or (batch_size is not None and batch!=batch_size):
            raise ValueError('incompatible scene batches')
        nv=sum(p.vertices.shape[-2] for p in meshes);nf=sum(p.faces.shape[0] for p in meshes)
        if max(nv*batch,nf*batch)>torch.iinfo(torch.int32).max:raise ValueError('packed int32 index overflow')
        identity=torch.eye(4,dtype=dtype,device=device).expand(batch,4,4)
        worlds={};visiting=set()
        # Iterative traversal avoids Python recursion limits for deep hierarchies.
        for p in self.prims:
            chain=[];current=p
            while current.name not in worlds:
                if current.name in visiting:raise ValueError('cycle in scene hierarchy')
                visiting.add(current.name);chain.append(current)
                if current.parent is None:break
                current=nodes[current.parent]
            for node in reversed(chain):
                local=identity if node.transform is None else _batch(node.transform,2,batch)
                worlds[node.name]=(identity if node.parent is None else worlds[node.parent]) @ local
                _float(worlds[node.name],'world transform',dtype,device)
                determinant=torch.linalg.det(worlds[node.name][:,:3,:3])
                if not bool((torch.isfinite(determinant)&(determinant!=0)).all()):
                    raise ValueError('world transform determinant must be finite and nonzero in scene precision')
                visiting.remove(node.name)
        # Packing per mesh first, then flattening the leading frame axis, retains
        # a single frame batch even when it contains several independently posed objects.
        vertices=[];faces=[];vo=[0];fo=[0];face_material=[];attrs={k:[] for k in channels};presence={k:[] for k in channels}
        for p in meshes:
            v=_batch(p.vertices,2,batch);f=p.faces;w=worlds[p.name];a=w[:,:3,:3]
            transformed=v @ a.transpose(-1,-2)+w[:,:3,3].unsqueeze(1)
            _float(transformed,'transformed vertices',dtype,device);vertices.append(transformed)
            faces.append(f+vo[-1]);vo.append(vo[-1]+v.shape[1]);fo.append(fo[-1]+f.shape[0])
            face_material.append(torch.full((f.shape[0],),material_ids.get(p.material,-1),dtype=torch.int32,device=device))
            for name,c in channels.items():
                attr=p.attributes.get(name)
                if attr is None:corners=torch.zeros(batch,f.shape[0],3,c,dtype=dtype,device=device)
                else:
                    values=_batch(attr.values,_DOMAINS[attr.domain],batch)
                    if attr.domain=='constant':corners=values[:,None,None,:].expand(batch,f.shape[0],3,c)
                    elif attr.domain=='uniform':corners=values[:,:,None,:].expand(batch,f.shape[0],3,c)
                    elif attr.domain=='vertex':corners=values[:,f.long(),:]
                    else:corners=values
                    if name=='normals':
                        normal_matrix=torch.linalg.inv(a)*torch.linalg.det(a).sign()[:,None,None]
                        corners=torch.einsum('bfkc,bcd->bfkd',corners,normal_matrix)
                        scale=corners.abs().amax(-1,keepdim=True)
                        corners=divide(corners,scale)
                        corners=divide(corners,sqrt_unit(sum3(corners.square())).unsqueeze(-1))
                    _float(corners,'packed attribute '+name,dtype,device)
                attrs[name].append(corners);presence[name].append(torch.full((f.shape[0],),attr is not None,dtype=torch.bool,device=device))
        def offsets(step):return torch.arange(batch+1,dtype=torch.int32,device=device)*step
        ov=torch.tensor(vo,dtype=torch.int32,device=device)[None,:]+offsets(nv)[:-1,None]
        of=torch.tensor(fo,dtype=torch.int32,device=device)[None,:]+offsets(nf)[:-1,None]
        packed_v=torch.cat(vertices,dim=1).reshape(-1,3) if vertices else torch.empty(0,3,dtype=dtype,device=device)
        packed_f=(torch.cat(faces,dim=0)[None,:,:]+offsets(nv)[:-1,None,None]).reshape(-1,3) if faces else torch.empty(0,3,dtype=torch.int32,device=device)
        packed_attrs={name:PackedAttribute(torch.cat(parts,dim=1).reshape(-1,3,channels[name]),
                       torch.cat(presence[name]).repeat(batch),tuple(p.attributes[name].domain if name in p.attributes else None for p in meshes)) for name,parts in attrs.items()}
        return PackedScene(packed_v,packed_f,offsets(nv),offsets(nf),ov,of,tuple(p.name for p in meshes),
               torch.cat(face_material).repeat(batch) if face_material else torch.empty(0,dtype=torch.int32,device=device),
               MappingProxyType(packed_attrs),MappingProxyType(worlds),materials,
               tuple(p for p in self.prims if type(p)==Light),tuple(p for p in self.prims if type(p)==Camera),tuple(p for p in self.prims if type(p)==Rig))
