"""Forward raster image lighting: owned captures, GGX filtering and diffuse SH.

No tracing service or external renderer is used. Captures render actual geometry
through the same gffx rasterizer. A descriptor updates on every render; capture()
returns a frozen snapshot. Irradiance is an explicit static one-bounce bake.
"""
from dataclasses import dataclass, replace
import math
import torch
import torch.nn.functional as F
from ._raster_material import unit
from .camera import Camera

_AXES=((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1))
_UP=((0,0,1),(0,0,1),(0,0,1),(0,0,1),(0,1,0),(0,1,0))

def _vector(value,like):
    result=torch.as_tensor(value,dtype=like.dtype,device=like.device)
    if result.shape!=(3,) or not bool(torch.isfinite(result).all()):raise ValueError('expected finite xyz vector')
    return result

def _check_image(image,cube=False):
    if not isinstance(image,torch.Tensor) or image.dtype not in (torch.float32,torch.float64):raise TypeError('lighting image must be a float32/float64 tensor')
    if image.device.type not in ('cpu','cuda') or image.ndim!=(4 if cube else 3) or image.shape[-1]!=3:raise ValueError('lighting image must be RGB on CPU/CUDA')
    if cube and (image.shape[0]!=6 or image.shape[1]!=image.shape[2]):raise ValueError('cubemap must be [6,size,size,3]')
    if min(image.shape)<1 or not bool(torch.isfinite(image).all()) or bool((image<0).any()):raise ValueError('radiance must be finite and nonnegative')
    if torch.is_grad_enabled() and image.requires_grad:raise ValueError('raster lighting is forward-only; use no_grad explicitly')

def _basis(like):
    forward=like.new_tensor(_AXES);up=like.new_tensor(_UP)
    right=unit(torch.linalg.cross(forward,up));up=torch.linalg.cross(right,forward)
    return right,up,forward

def cube_directions(size,like):
    a=(torch.arange(size,device=like.device,dtype=like.dtype)+.5)*2/size-1
    x,y=torch.meshgrid(a,-a,indexing='xy');r,u,f=_basis(like)
    return unit(f[:,None,None]+x[None,:,:,None]*r[:,None,None]+y[None,:,:,None]*u[:,None,None])

def _sample_cube(image,direction):
    """Seam-crossing bilinear: taps outside a face are reprojected to neighbours."""
    size=image.shape[1];r,u,f=_basis(direction)
    axis=direction.abs().argmax(-1);face=axis*2+(direction.gather(-1,axis[:,None])[:,0]<0).long()
    d=(direction*f[face]).sum(-1).clamp_min(1e-20)
    x=((direction*r[face]).sum(-1)/d+1)*size/2-.5
    y=(1-(direction*u[face]).sum(-1)/d)*size/2-.5
    ix=x.floor().long();iy=y.floor().long();fx=x-ix;fy=y-iy
    value=torch.zeros_like(direction)
    for ox,oy in ((0,0),(1,0),(0,1),(1,1)):
        px=ix+ox;py=iy+oy
        v=f[face]+((px.to(d.dtype)+.5)*2/size-1)[:,None]*r[face]+(1-(py.to(d.dtype)+.5)*2/size)[:,None]*u[face]
        other_axis=v.abs().argmax(-1);other=other_axis*2+(v.gather(-1,other_axis[:,None])[:,0]<0).long()
        z=(v*f[other]).sum(-1).clamp_min(1e-20)
        xx=(((v*r[other]).sum(-1)/z+1)*size/2).floor().long().clamp(0,size-1)
        yy=((1-(v*u[other]).sum(-1)/z)*size/2).floor().long().clamp(0,size-1)
        weight=(fx if ox else 1-fx)*(fy if oy else 1-fy)
        value+=image[other,yy,xx]*weight[:,None]
    return value

def _atlas(image):
    size=image.shape[1];a=(torch.arange(-1,size+1,device=image.device,dtype=image.dtype)+.5)*2/size-1
    x,y=torch.meshgrid(a,-a,indexing='xy');r,u,f=_basis(image)
    d=f[:,None,None]+x[None,:,:,None]*r[:,None,None]+y[None,:,:,None]*u[:,None,None];d=d.reshape(-1,3)
    axis=d.abs().argmax(-1);face=axis*2+(d.gather(-1,axis[:,None])[:,0]<0).long();z=(d*f[face]).sum(-1)
    xx=(((d*r[face]).sum(-1)/z+1)*size/2).floor().long().clamp(0,size-1);yy=((1-(d*u[face]).sum(-1)/z)*size/2).floor().long().clamp(0,size-1)
    padded=image[face,yy,xx].reshape(6,size+2,size+2,3);padded[:,1:-1,1:-1]=image
    return padded.reshape(6*(size+2),size+2,3).permute(2,0,1)[None].contiguous()

def _sample_atlas(atlas,direction):
    size=atlas.shape[-1]-2;r,u,f=_basis(direction);axis=direction.abs().argmax(-1);face=axis*2+(direction.gather(-1,axis[:,None])[:,0]<0).long()
    z=(direction*f[face]).sum(-1).clamp_min(1e-20)
    x=((direction*r[face]).sum(-1)/z+1)*size/2+1;y=(1-(direction*u[face]).sum(-1)/z)*size/2+1
    grid=torch.stack((x/(size+2)*2-1,(face*(size+2)+y)/(6*(size+2))*2-1),-1)[None,None]
    return F.grid_sample(atlas,grid,align_corners=False,padding_mode='border')[0,:,0].T

def _hammersley(count,like):
    bits=torch.arange(count,device=like.device,dtype=torch.int64);rev=torch.zeros_like(bits)
    for unused in range(32):rev=rev*2+(bits&1);bits=bits>>1
    return torch.stack(((torch.arange(count,device=like.device,dtype=like.dtype)+.5)/count,rev.to(like.dtype)/4294967296),-1)

def _half_vectors(normal,roughness,sequence):
    a=roughness**2;phi=2*math.pi*sequence[:,0]
    cos=torch.sqrt((1-sequence[:,1])/(1+(a*a-1)*sequence[:,1]));sin=torch.sqrt((1-cos*cos).clamp_min(0))
    up=torch.where((normal[:,2].abs()<.999)[:,None],normal.new_tensor([0,0,1]),normal.new_tensor([1,0,0]))
    tangent=unit(torch.linalg.cross(up,normal));bitangent=torch.linalg.cross(normal,tangent)
    return tangent[:,None]*(sin*phi.cos())[None,:,None]+bitangent[:,None]*(sin*phi.sin())[None,:,None]+normal[:,None]*cos[None,:,None]

def _prefilter(cube,samples):
    atlas=_atlas(cube);levels=[cube.contiguous()];size=cube.shape[1];count=int(math.log2(size))+1;sequence=_hammersley(samples,cube)
    for level in range(1,count):
        n=cube_directions(max(1,size>>level),cube).reshape(-1,3);values=[];roughness=level/(count-1)
        for start in range(0,len(n),2048):
            normal=n[start:start+2048];half=_half_vectors(normal,roughness,sequence)
            direction=2*(normal[:,None]*half).sum(-1,keepdim=True)*half-normal[:,None]
            weight=(normal[:,None]*direction).sum(-1).clamp_min(0)
            radiance=_sample_atlas(atlas,direction.reshape(-1,3)).reshape(len(normal),samples,3)
            values.append((radiance*weight[:,:,None]).sum(1)/weight.sum(1).clamp_min(1e-20)[:,None])
        side=max(1,size>>level);levels.append(torch.cat(values).reshape(6,side,side,3))
    return tuple(levels)

def _sh(direction):
    x,y,z=direction.unbind(-1)
    return torch.stack((torch.ones_like(x)*.28209479177387814,.4886025119029199*y,.4886025119029199*z,.4886025119029199*x,1.0925484305920792*x*y,1.0925484305920792*y*z,.31539156525252005*(3*z*z-1),1.0925484305920792*x*z,.5462742152960396*(x*x-y*y)),-1)

def _irradiance_coefficients(cube):
    size=cube.shape[1];directions=cube_directions(size,cube).reshape(-1,3)
    # Exact texel solid angle, not a uniform cube-face weight.
    a=torch.arange(size+1,device=cube.device,dtype=cube.dtype)*2/size-1
    x,y=torch.meshgrid(a,a,indexing='xy');angle=torch.atan2(x*y,torch.sqrt(x*x+y*y+1))
    weight=(angle[1:,1:]-angle[:-1,1:]-angle[1:,:-1]+angle[:-1,:-1]).abs().reshape(-1).repeat(6)
    coefficients=torch.einsum('ni,nj,n->ij',_sh(directions),cube.reshape(-1,3),weight)
    return coefficients*cube.new_tensor([math.pi]+[2*math.pi/3]*3+[math.pi/4]*5)[:,None]

_dfg_cache={}
def _dfg(like):
    key=(like.device,like.dtype)
    if key in _dfg_cache:return _dfg_cache[key]
    size=64;sequence=_hammersley(256,like);rows=[]
    nv=(torch.arange(size,device=like.device,dtype=like.dtype)+.5)/size
    view=torch.stack(((1-nv*nv).sqrt(),torch.zeros_like(nv),nv),-1)
    normal=view.new_tensor([0,0,1]).expand(size,3)
    for i in range(size):
        roughness=(i+.5)/size;half=_half_vectors(normal,roughness,sequence);vh=(view[:,None]*half).sum(-1).clamp_min(0)
        direction=2*vh[:,:,None]*half-view[:,None];nl=direction[:,:,2].clamp_min(0);nh=half[:,:,2]
        a2=roughness**4
        def g1(c):return 2*c/(c+torch.sqrt(a2+(1-a2)*c*c)).clamp_min(1e-20)
        vis=g1(nv[:,None])*g1(nl)*vh/(nh*nv[:,None]).clamp_min(1e-20);vis=torch.where(nl>0,vis,0)
        fc=(1-vh)**5;rows.append(torch.stack(((vis*(1-fc)).mean(-1),(vis*fc).mean(-1)),-1))
    result=torch.stack(rows);_dfg_cache[key]=result;return result

def _brdf(normal,view,roughness):
    nv=(normal*view).sum(-1).clamp(0,1);r=torch.as_tensor(roughness,device=nv.device,dtype=nv.dtype).expand_as(nv)
    # LUT axes are N.V then perceptual roughness; bilinear cell-centre sampling.
    grid=torch.stack((nv*2-1,r*2-1),-1)[None,None]
    return F.grid_sample(_dfg(normal).permute(2,0,1)[None],grid,align_corners=False,padding_mode='border')[0,:,0].T

@dataclass(frozen=True)
class Environment:
    levels: tuple
    irradiance: torch.Tensor

    def __post_init__(self):object.__setattr__(self,'atlases',tuple(_atlas(image) for image in self.levels))

    @classmethod
    def from_cubemap(cls,image,*,samples=64):
        _check_image(image,True)
        if type(samples)is not int or samples<16:raise ValueError('filter samples must be >=16')
        with torch.no_grad():return cls(_prefilter(image.detach().clone(),samples),_irradiance_coefficients(image))

    @classmethod
    def from_latlong(cls,image,*,size=256,samples=64):
        _check_image(image)
        if type(size)is not int or size<4 or size&(size-1):raise ValueError('cube size must be a power of two >=4')
        d=cube_directions(size,image);u=torch.atan2(d[...,1],d[...,0])/(2*math.pi)+.5;v=torch.acos(d[...,2].clamp(-1,1))/math.pi
        # Wrap longitude explicitly, including both bilinear seam taps.
        h,w=image.shape[:2];padded=torch.cat((image[:,-1:],image,image[:,:1]),1)
        gx=(u*w+1)/(w+2)*2-1;grid=torch.stack((gx,v*2-1),-1).reshape(1,6*size,size,2)
        cube=F.grid_sample(padded.permute(2,0,1)[None],grid,align_corners=False,padding_mode='border')[0].permute(1,2,0).reshape(6,size,size,3)
        return cls.from_cubemap(cube,samples=samples)

    @classmethod
    def load(cls,path,*,device='auto',size=256,samples=64):
        """Read scene-linear latlong radiance from a portable scene bundle."""
        import json,zipfile
        from ._scene_builders import resolve_device
        with zipfile.ZipFile(path)as z:
            spec=json.loads(z.read('scene.json'));record=spec['environment']
            if record['dtype']!='float32':raise ValueError('environment bundle must store float32 radiance')
            image=torch.frombuffer(bytearray(z.read(record['file'])),dtype=torch.float32).reshape(record['shape']).clone().to(resolve_device(device))
        return cls.from_latlong(image,size=size,samples=samples)

    @classmethod
    def constant(cls,radiance,*,device=None,dtype=torch.float32):
        rgb=torch.as_tensor(radiance,device=device,dtype=dtype)
        if rgb.ndim==0:rgb=rgb.expand(3)
        return cls.from_cubemap(rgb.expand(6,4,4,3).clone(),samples=16)

    def sample(self,direction,roughness=0.):
        d=direction.reshape(-1,3)
        if not isinstance(roughness,torch.Tensor) or roughness.ndim==0:
            scalar=float(roughness);lod=max(0,min(1,scalar))*(len(self.levels)-1);lo=int(lod);fraction=lod-lo
            value=_sample_atlas(self.atlases[lo],d)
            if fraction:value=value*(1-fraction)+_sample_atlas(self.atlases[min(lo+1,len(self.levels)-1)],d)*fraction
            return value.reshape(direction.shape)
        rough=torch.as_tensor(roughness,device=d.device,dtype=d.dtype).expand(len(d)).clamp(0,1)
        lod=rough*(len(self.levels)-1);low=lod.floor().long();frac=lod-low;out=torch.zeros_like(d)
        for i,image in enumerate(self.levels):
            selected=(low==i)|((low+1)==i)
            if not bool(selected.any()):continue
            weight=torch.where(low[selected]==i,1-frac[selected],frac[selected])
            out[selected]+=_sample_atlas(self.atlases[i],d[selected])*weight[:,None]
        return out.reshape(direction.shape)

    def diffuse(self,normal):return (_sh(normal)@self.irradiance).clamp_min(0)

@dataclass(frozen=True)
class ReflectionProbe:
    position: object
    bounds: object=None
    exclude: tuple=()
    resolution: int=256
    samples: int=64
    environment: object=None

    def capture(self,scene,*,environment=None,irradiance=None,shadows=True,shadow_map_size=512,_context=None):
        from .scene import Scene,Mesh
        from ._raster import render
        like=next(p.vertices for p in scene.prims if isinstance(p,Mesh));position=_vector(self.position,like)
        if type(self.resolution)is not int or self.resolution<4 or self.resolution&(self.resolution-1):raise ValueError('probe resolution must be a power of two >=4')
        filtered=Scene([p for p in scene.prims if p.name not in self.exclude]);images=[];context={} if _context is None else _context
        for direction,up in zip(_AXES,_UP):
            camera=Camera.look_at(position,position+like.new_tensor(direction),up=up,size=(self.resolution,)*2,fov=90,near=.005,far=100,device=like.device,dtype=like.dtype)
            image=render(filtered,camera,supersample=2,shadows=shadows,shadow_map_size=shadow_map_size,environment=environment,irradiance=irradiance,_context=context)
            images.append(image['colour'][0])
        return replace(self,position=position,environment=Environment.from_cubemap(torch.stack(images),samples=self.samples))

    def direction(self,points,direction):
        if self.bounds is None:return direction
        low=_vector(self.bounds[0],points);high=_vector(self.bounds[1],points)
        if not bool((high>low).all()):raise ValueError('probe bounds must have positive extent')
        safe=torch.where(direction.abs()>1e-10,direction,torch.full_like(direction,1e-10))
        t=torch.maximum((low-points)/safe,(high-points)/safe).amin(-1).clamp_min(0)
        corrected=points+direction*t[:,None]-_vector(self.position,points)
        inside=((points>=low)&(points<=high)).all(-1)
        return torch.where(inside[:,None],unit(corrected),direction)

@dataclass(frozen=True)
class PlanarReflection:
    mesh: str
    point: object
    normal: object

    def capture(self,scene,camera,**options):
        from .scene import Scene
        from ._raster import render
        p=_vector(self.point,camera.pose);n=unit(_vector(self.normal,camera.pose))
        if not bool((n!=0).any()):raise ValueError('mirror normal must be nonzero')
        reflection=torch.eye(4,device=p.device,dtype=p.dtype);reflection[:3,:3]-=2*n[:,None]*n[None];reflection[:3,3]=2*(p*n).sum()*n
        pose=reflection@camera.pose;pose=pose.clone();pose[:,:3,0]*=-1
        ctor=Camera.from_orthographic if camera.projection_kind=='orthographic' else Camera.from_intrinsics
        reflected=ctor(camera.fx,camera.fy,camera.cx,camera.cy,camera.image_height,camera.image_width,camera.near,camera.far,pose=pose)
        # Keep only the physical front half-space; camera is virtually behind it.
        side=1 if float(((camera.pose[0,:3,3]-p)*n).sum())>0 else -1
        filtered=Scene([q for q in scene.prims if q.name!=self.mesh])
        image=render(filtered,reflected,_clip_plane=(p,n*side),**options)
        return dict(image=image['colour'][0],camera=reflected)

@dataclass(frozen=True)
class IrradianceVolume:
    bounds: tuple
    coefficients: torch.Tensor
    normal_bias: float=.02
    interpolation: str="trilinear"

    @classmethod
    def bake(cls,scene,*,bounds,resolution=(3,3,3),capture_size=32,environment=None,exclude=(),shadows=True,shadow_map_size=512,progress=None,interpolation="trilinear"):
        from .scene import Scene,Mesh
        from ._raster import render
        if interpolation not in ("normal","trilinear"):raise ValueError("interpolation must be normal or trilinear")
        like=next(p.vertices for p in scene.prims if isinstance(p,Mesh));low=_vector(bounds[0],like);high=_vector(bounds[1],like)
        if not bool((high>low).all()) or len(resolution)!=3 or any(type(x)is not int or x<2 for x in resolution):raise ValueError('volume needs positive bounds and >=2 samples per axis')
        axes=[torch.linspace(float(a),float(b),n,device=like.device,dtype=like.dtype)for a,b,n in zip(low,high,resolution)]
        positions=torch.stack(torch.meshgrid(*axes,indexing='ij'),-1).reshape(-1,3)
        filtered=Scene([p for p in scene.prims if p.name not in exclude]);context={};coefficients=[]
        for i,position in enumerate(positions):
            faces=[]
            for direction,up in zip(_AXES,_UP):
                camera=Camera.look_at(position,position+like.new_tensor(direction),up=up,size=(capture_size,)*2,fov=90,near=.005,far=100,device=like.device,dtype=like.dtype)
                faces.append(render(filtered,camera,supersample=1,shadows=shadows,shadow_map_size=shadow_map_size,environment=environment,_context=context)['colour'][0])
            coefficients.append(_irradiance_coefficients(torch.stack(faces)))
            if progress:progress(dict(stage='diffuse bake',completed=i+1,total=len(positions)))
        return cls((low,high),torch.stack(coefficients).reshape(*resolution,9,3),interpolation=interpolation)

    def __post_init__(self):
        if self.interpolation not in ("normal","trilinear"):raise ValueError("interpolation must be normal or trilinear")
        c=self.coefficients
        if not isinstance(c,torch.Tensor)or c.dtype not in (torch.float32,torch.float64)or c.ndim!=5 or c.shape[-2:]!=(9,3)or min(c.shape[:3])<2:raise ValueError('irradiance grid must be [x>=2,y>=2,z>=2,9,3] float coefficients')
        low=_vector(self.bounds[0],c);high=_vector(self.bounds[1],c)
        if not bool((high>low).all())or not bool(torch.isfinite(c).all())or not math.isfinite(self.normal_bias)or self.normal_bias<0:raise ValueError('invalid irradiance grid bounds/values/bias')
        object.__setattr__(self,'bounds',(low,high))

    def sample(self,points,normal):
        low,high=self.bounds;query=points+normal*self.normal_bias
        size=points.new_tensor(self.coefficients.shape[:3]);q=((query-low)/(high-low)*(size-1)).clamp_min(0);q=torch.minimum(q,size-1)
        base=q.floor().long();fraction=q-base;out=points.new_zeros((len(points),9,3));plain=torch.zeros_like(out);total=points.new_zeros(len(points));shape=self.coefficients.shape[:3]
        for x in (0,1):
            for y in (0,1):
                for z in (0,1):
                    ix=(base[:,0]+x).clamp_max(shape[0]-1);iy=(base[:,1]+y).clamp_max(shape[1]-1);iz=(base[:,2]+z).clamp_max(shape[2]-1)
                    weight=(fraction[:,0]if x else 1-fraction[:,0])*(fraction[:,1]if y else 1-fraction[:,1])*(fraction[:,2]if z else 1-fraction[:,2])
                    coefficient=self.coefficients[ix,iy,iz]
                    if self.interpolation=="normal":
                        plain+=coefficient*weight[:,None,None]
                        position=low+torch.stack((ix,iy,iz),-1)/(size-1)*(high-low)
                        ray=position-points;distance=ray.norm(dim=-1)
                        # A probe behind the receiver cannot describe its front hemisphere.
                        facing=((normal*ray).sum(-1)/distance.clamp_min(1e-10)).clamp_min(0).square()
                        weight=weight*torch.where(distance<1e-8,torch.ones_like(facing),facing)
                    out+=coefficient*weight[:,None,None];total+=weight
        if self.interpolation=="normal":
            out=torch.where((total>1e-10)[:,None,None],out/total.clamp_min(1e-10)[:,None,None],plain)
        value=(_sh(normal)[:,:,None]*out).sum(1).clamp_min(0)
        # Influence uses the surface point; bias only offsets/clamps lookup.
        inside=((points>=low)&(points<=high)).all(-1)
        return value,inside


def surface(points,normal,view,params,*,environment,probes,irradiance,mirror=None,classic=False):
    """Single-scatter split-sum IBL plus spatially interpolated diffuse SH."""
    albedo=params.get('albedo',0) if classic else params.get('base_color',0)
    albedo=torch.as_tensor(albedo,device=points.device,dtype=points.dtype)
    metallic=torch.as_tensor(0 if classic else params.get('metallic',0),device=points.device,dtype=points.dtype).expand(len(points))
    roughness=torch.as_tensor(1 if classic else params.get('roughness',.3),device=points.device,dtype=points.dtype).expand(len(points))
    ior=torch.as_tensor(params.get('ior',1.5),device=points.device,dtype=points.dtype)
    f0=(((ior-1)/(ior+1))**2).expand(len(points))[:,None];f0=f0*(1-metallic[:,None])+albedo*metallic[:,None]
    incoming=environment.diffuse(normal) if environment is not None else torch.zeros_like(points)
    if irradiance is not None:
        baked,inside=irradiance.sample(points,normal);incoming=torch.where(inside[:,None],baked,incoming)
    nv=(normal*view).sum(-1).clamp(0,1)
    fresnel=f0+(1-f0)*(1-nv[:,None])**5
    result=albedo*(1-metallic[:,None])*(1-fresnel)*incoming/math.pi if not classic else albedo*incoming/math.pi
    if classic and not bool(torch.as_tensor(params.get('specular',0)).any()):return result
    direction=unit(2*nv[:,None]*normal-view)
    radiance=environment.sample(direction,params.get('roughness',.3)) if environment is not None else torch.zeros_like(points)
    if probes:
        # Select nearest capture; no hidden blend through unrelated rooms.
        distance=torch.stack([(points-_vector(p.position,points)).square().sum(-1) for p in probes],1);choice=distance.argmin(-1)
        for i,probe in enumerate(probes):
            selected=choice==i
            if bool(selected.any()):radiance[selected]=probe.environment.sample(probe.direction(points[selected],direction[selected]),(params.get('roughness',.3) if torch.as_tensor(params.get('roughness',.3)).ndim==0 else roughness[selected]))
    for entry in (mirror or []):
        selected=entry['selected'];mirror=entry['capture']
        camera=mirror['camera'];homogeneous=torch.cat((points,torch.ones_like(points[:,:1])),-1);clip=homogeneous@camera.world_to_clip()[0].T
        uv=clip[:,:2]/clip[:,3:];grid=torch.stack((uv[:,0],-uv[:,1]),-1)[None,None]
        value=F.grid_sample(mirror['image'].permute(2,0,1)[None],grid,align_corners=False,padding_mode='border')[0,:,0].T
        valid=selected & (clip[:,3]>0)&(uv.abs()<=1).all(-1);radiance=torch.where(valid[:,None],value,radiance)
    ab=_brdf(normal,view,roughness);result+=radiance*(f0*ab[:,:1]+ab[:,1:])
    coat=torch.as_tensor(params.get('coat',0),device=points.device,dtype=points.dtype).expand(len(points))
    if bool((coat>0).any()):
        r=params.get('coat_roughness',.12);ab=_brdf(normal,view,r);cenv=environment.sample(direction,r)if environment is not None else torch.zeros_like(points)
        if probes:
            for i,probe in enumerate(probes):
                selected=choice==i
                if bool(selected.any()):cenv[selected]=probe.environment.sample(probe.direction(points[selected],direction[selected]),r if torch.as_tensor(r).ndim==0 else r[selected])
        cf=.04+.96*(1-nv)**5;result=result*(1-coat*cf)[:,None]+coat[:,None]*cenv*(.04*ab[:,:1]+ab[:,1:])
    return result
