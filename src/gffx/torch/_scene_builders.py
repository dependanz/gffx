"""Scene authoring recipes over the existing tensor/native interfaces."""
import math
from dataclasses import fields, replace, is_dataclass
from collections.abc import Mapping
import torch


def resolve_device(device):
    """Resolve explicit setup-time automatic selection; operations never fall back."""
    if device != 'auto':
        return device
    if torch.cuda.is_available():
        from .. import cuda
        if cuda.capabilities()['status'] == 'loaded':
            return torch.device('cuda')
    return torch.device('cpu')


def tensors(*values, dtype=None, device=None):
    first = next((v for v in values if isinstance(v, torch.Tensor)), None)
    dtype = dtype or (first.dtype if first is not None else torch.float32)
    device = resolve_device(device) or (first.device if first is not None else 'cpu')
    if dtype not in (torch.float32, torch.float64):
        raise TypeError('scene constructors require float32 or float64')
    if torch.device(device).type not in ('cpu', 'cuda'):
        raise ValueError('scene device must be cpu or cuda')
    result = tuple(v.to(dtype=dtype, device=device) if isinstance(v, torch.Tensor)
                   else torch.tensor(v, dtype=dtype, device=device) for v in values)
    if any(not bool(torch.isfinite(v).all()) for v in result):
        raise ValueError('scene values must be finite')
    return result


def vectors(*values, **options):
    result = tensors(*values, **options)
    if any(v.shape != (3,) for v in result):
        raise ValueError('expected three-component scene vectors')
    return result


def quad(cls, name, *, center=(0,0,0), u=(1,0,0), v=(0,1,0), material=None, dtype=None, device=None):
    from .scene import Attribute
    c, u, v = vectors(center, u, v, dtype=dtype, device=device)
    n = torch.linalg.cross(u, v)
    if not bool((n != 0).any()):
        raise ValueError('quad spans must be nonzero and not parallel')
    n = n / torch.linalg.vector_norm(n)
    vertices = torch.stack((c-u/2-v/2, c+u/2-v/2, c+u/2+v/2, c-u/2+v/2))
    faces = torch.tensor([[0,1,2],[0,2,3]], dtype=torch.int32, device=c.device)
    return cls(name, vertices, faces, material=material,
               attributes={'normals': Attribute(n.expand(4,3).contiguous(), 'vertex')})


def sphere(cls, name, *, radius=1., center=(0,0,0), segments=48, rings=24, material=None, dtype=None, device=None):
    from .scene import Attribute
    if type(segments) is not int or type(rings) is not int or segments < 3 or rings < 2:
        raise ValueError('sphere requires integer segments>=3 and rings>=2')
    if 2+segments*(rings-1) > 1_000_000:
        raise ValueError('sphere authoring exceeds one million vertices')
    c, r = tensors(center, radius, dtype=dtype, device=device)
    if c.shape != (3,) or r.ndim != 0 or not bool(r > 0):
        raise ValueError('sphere center must be xyz and radius a positive scalar')
    normals = [[0.,1.,0.]]
    for i in range(1,rings):
        p=math.pi*i/rings
        for j in range(segments):
            t=math.tau*j/segments
            normals.append([math.sin(p)*math.cos(t),math.cos(p),math.sin(p)*math.sin(t)])
    normals.append([0.,-1.,0.])
    faces = [[0,1+(j+1)%segments,1+j] for j in range(segments)]
    for i in range(rings-2):
        for j in range(segments):
            a=1+i*segments+j; b=1+i*segments+(j+1)%segments
            c1=1+(i+1)*segments+(j+1)%segments; d=1+(i+1)*segments+j
            faces.extend(([a,b,c1],[a,c1,d]))
    bottom=len(normals)-1
    faces.extend([bottom,1+(rings-2)*segments+j,1+(rings-2)*segments+(j+1)%segments] for j in range(segments))
    n=r.new_tensor(normals)
    return cls(name,n*r+c,torch.tensor(faces,dtype=torch.int32,device=r.device),material=material,
               attributes={'normals':Attribute(n,'vertex')})


def diffuse(cls, name, albedo, *, dtype=None, device=None):
    from . import material
    a, = tensors(albedo, dtype=dtype, device=device)
    if a.ndim == 0: a=a.expand(3)
    if a.shape != (3,) or not bool(((a>=0)&(a<=1)).all()):
        raise ValueError('albedo must be an RGB value in [0,1]')
    return cls(name,material.classic,dict(albedo=a,specular=a.new_zeros(()),exponent=a.new_ones(())))


def emissive(cls, name, radiance, *, dtype=None, device=None):
    r, = tensors(radiance,dtype=dtype,device=device)
    if r.ndim == 0:r=r.expand(3)
    if r.shape != (3,) or not bool((r>=0).all()):
        raise ValueError('emission must be nonnegative RGB radiance')
    result=diffuse(cls,name,r.new_zeros(3))
    return replace(result,emission=r)


def rectangle(cls, name, *, center, u, v, radiance, samples=4, visible=True, dtype=None, device=None):
    from . import light
    c,u,v,r=tensors(center,u,v,radiance,dtype=dtype,device=device)
    if r.ndim==0:r=r.expand(3)
    if any(t.shape!=(3,) for t in (c,u,v,r)):raise ValueError('rectangle requires xyz spans and scalar/RGB radiance')
    if type(samples) is not int or not 1<=samples<=16:
        raise ValueError('rectangle samples must be an integer in [1,16] per axis')
    if type(visible) is not bool:raise TypeError('visible must be boolean')
    if not bool((torch.linalg.cross(u,v)!=0).any()) or not bool((r>=0).all()):
        raise ValueError('rectangle requires nonparallel spans and nonnegative radiance')
    return cls(name,light.rectangle,dict(center=c,u=u,v=v,radiance=r,samples=samples),visible=visible)


def move_scene(scene, *, device=None, dtype=None):
    from .scene import Attribute
    from .camera import Camera
    device = resolve_device(device)
    if dtype is not None and dtype not in (torch.float32,torch.float64):raise TypeError('scene precision must be float32 or float64')
    if device is not None and torch.device(device).type not in ('cpu','cuda'):raise ValueError('scene device must be cpu or cuda')
    if device is None and dtype is None:return scene
    def move(value):
        if isinstance(value,torch.Tensor):
            return value.to(device=device,dtype=dtype if value.is_floating_point() else None)
        if isinstance(value,Attribute):return replace(value,values=move(value.values))
        if isinstance(value,Camera):
            constructor=Camera.from_orthographic if value.projection_kind=='orthographic' else Camera.from_intrinsics
            return constructor(*(move(getattr(value,k)) for k in ('fx','fy','cx','cy')),
                               value.image_height,value.image_width,value.near,value.far,pose=move(value.pose))
        if is_dataclass(value) and not isinstance(value,type):
            return replace(value,**{f.name:move(getattr(value,f.name)) for f in fields(value)})
        if isinstance(value,(list,tuple)):return type(value)(move(v) for v in value)
        if isinstance(value,Mapping):return {k:move(v) for k,v in value.items()}
        return value
    return type(scene)([replace(p,**{f.name:move(getattr(p,f.name)) for f in fields(p)}) for p in scene.prims])


def look_at(cls, eye, target, *, size=(256,256), fov=45., up=(0,1,0), near=.025, far=100., dtype=None, device=None):
    eye,target,up=vectors(eye,target,up,dtype=dtype,device=device)
    if len(size)!=2 or any(type(x) is not int or x<=0 for x in size):
        raise ValueError('size must be positive integer (height,width)')
    if isinstance(fov,bool) or not isinstance(fov,(int,float)) or not 0<fov<179:
        raise ValueError('fov must be a vertical angle in (0,179) degrees')
    forward=target-eye
    if not bool((forward!=0).any()):raise ValueError('eye and target must differ')
    forward=forward/torch.linalg.vector_norm(forward)
    right=torch.linalg.cross(forward,up)
    if not bool((right!=0).any()):raise ValueError('up must not be parallel to viewing direction')
    right=right/torch.linalg.vector_norm(right);vertical=torch.linalg.cross(right,forward)
    rotation=torch.stack((right,vertical,-forward),dim=1)
    upper=torch.cat((rotation,eye[:,None]),dim=1)
    pose=torch.cat((upper,eye.new_tensor([[0,0,0,1]])),dim=0)[None]
    h,w=size;focal=h/(2*math.tan(math.radians(fov)/2))
    return cls.from_intrinsics(focal,focal,w/2,h/2,h,w,near,far,pose=pose)
