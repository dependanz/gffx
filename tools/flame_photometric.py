"""FLAME fitting example composition over installed gffx and PyTorch.

No model assets, alternate renderer, or copied reference implementation. This
host-side experiment is not a new gffx public API or a completed gffit engine.
"""
from dataclasses import dataclass
import math
import torch
from gffx.torch import mesh,transforms,skeleton,light

REFERENCE_COMMIT='83f84b804938af6fe7472e492ec997e38552069d'

@dataclass(frozen=True)
class FlameData:
    template: torch.Tensor
    blend_basis: torch.Tensor
    pose_basis: torch.Tensor
    joint_regressor: torch.Tensor
    parents: torch.Tensor
    skin_weights: torch.Tensor
    faces: torch.Tensor

def _float(x,shape,reference):
    if not isinstance(x,torch.Tensor) or x.dtype!=reference.dtype or x.device!=reference.device:
        raise ValueError('floating tensor dtype/device mismatch')
    if x.shape!=shape or not bool(torch.isfinite(x).all()):
        raise ValueError('invalid tensor shape or nonfinite input')

def _matrix_value(a,b):
    return _fixed_sum_last(a.unsqueeze(-2)*b.transpose(-1,-2).unsqueeze(-3))

def _fixed_sum_to(value,shape):
    while value.ndim>len(shape):value=_fixed_sum_last(value.movedim(0,-1))
    for dim,n in enumerate(shape):
        if n==1 and value.shape[dim]!=1:value=_fixed_sum_last(value.movedim(dim,-1)).unsqueeze(dim)
    return value

class _ModelMatmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx,a,b):
        ctx.save_for_backward(a,b)
        return _matrix_value(a,b)
    @staticmethod
    def backward(ctx,g):
        a,b=ctx.saved_tensors
        ga=_fixed_sum_to(_matrix_value(g,b.transpose(-1,-2)),a.shape) if ctx.needs_input_grad[0] else None
        gb=_fixed_sum_to(_matrix_value(a.transpose(-1,-2),g),b.shape) if ctx.needs_input_grad[1] else None
        return ga,gb

def _model_matmul(a,b):
    # Bounded example layouts only. Both forward and first VJP use a fixed pair tree.
    return _ModelMatmul.apply(a,b)

def deform(data,coefficients,pose):
    """One frame; coefficients concatenate identity and expression bases.

    Pose is the full joint axis-angle array, including fixed neck/eye entries.
    Assets use [K,V,3] basis layout and ordered parents with root -1.
    """
    v=data.template
    if v.ndim!=2 or v.shape[-1]!=3 or v.dtype not in (torch.float32,torch.float64):
        raise ValueError('invalid template')
    J=data.parents.numel();K=data.blend_basis.shape[0];V=v.shape[0]
    if J<1 or data.parents.shape!=(J,) or data.parents.dtype!=torch.int32 or data.parents.device!=v.device:
        raise ValueError('invalid parents')
    parents=data.parents.detach().cpu().tolist()
    if parents[0]!=-1 or any(p<0 or p>=i for i,p in enumerate(parents[1:],1)):
        raise ValueError('ordered rooted joint tree required')
    for x,shape in [(v,(V,3)),(data.blend_basis,(K,V,3)),(data.pose_basis,((J-1)*9,V,3)),(data.joint_regressor,(J,V)),(data.skin_weights,(V,J)),(coefficients,(K,)),(pose,(J,3))]:
        _float(x,shape,v)
    shaped=mesh.blend_shapes(v.contiguous(),data.blend_basis.contiguous(),coefficients.reshape(1,K).contiguous())[0]
    joints=_model_matmul(data.joint_regressor,shaped)
    rotations=transforms.rodrigues(pose.contiguous())
    features=(rotations[1:]-torch.eye(3,dtype=v.dtype,device=v.device)).reshape(1,-1)
    correction=mesh.blend_shapes(torch.zeros_like(v),data.pose_basis.contiguous(),features.contiguous())[0]
    offsets=torch.stack([joints[0],*[joints[i]-joints[parents[i]] for i in range(1,J)]])
    upper=torch.cat((rotations,offsets[:,:,None]),-1)
    bottom=v.new_tensor([0.,0.,0.,1.]).reshape(1,1,4).expand(J,1,4)
    local=torch.cat((upper,bottom),-2).unsqueeze(0).contiguous()
    world=skeleton.forward_kinematics(local,data.parents.contiguous())
    bind_upper=torch.cat((torch.eye(3,dtype=v.dtype,device=v.device).expand(J,3,3),-joints[:,:,None]),-1)
    inverse_bind=torch.cat((bind_upper,bottom),-2)
    corrected=(shaped+correction).unsqueeze(0).contiguous()
    return mesh.skin_lbs(corrected,_model_matmul(world,inverse_bind).contiguous(),data.skin_weights.contiguous())[0]

def landmarks(data,vertices,face_indices,barycentric):
    return mesh.evaluate_barycentric(vertices.unsqueeze(0).contiguous(),data.faces.contiguous(),face_indices.contiguous(),barycentric.contiguous())[0]

def project(vertices,camera):
    """Reference projected convention: scale(x+tx), -scale(y+ty), -scale*z."""
    _float(camera,(3,),vertices)
    if not bool(camera[0]>0):raise ValueError('camera scale must be positive')
    s,tx,ty=camera;z=s*0;o=z+1
    matrix=torch.stack((s,z,z,s*tx,z,-s,z,-s*ty,z,z,-s,z,z,z,z,o)).reshape(4,4)
    return transforms.transform_points(vertices.contiguous(),matrix.unsqueeze(0).contiguous())[...,:3]

def _fixed_sum_last(value):
    """Pair-tree reduction used by the host example's texture PCA."""
    if value.shape[-1]==0:return value.sum(-1)
    while value.shape[-1]>1:
        stop=value.shape[-1]//2*2
        paired=value[...,:stop:2]+value[...,1:stop:2]
        value=torch.cat((paired,value[...,-1:]),-1) if stop<value.shape[-1] else paired
    return value[...,0]

class _TexturePCA(torch.autograd.Function):
    @staticmethod
    def forward(ctx,mean,basis,coefficients):
        ctx.save_for_backward(basis,coefficients)
        return mean+_fixed_sum_last((basis*coefficients[:,None,None,None]).movedim(0,-1))

    @staticmethod
    def backward(ctx,gradient):
        basis,coefficients=ctx.saved_tensors
        return (gradient if ctx.needs_input_grad[0] else None,
                gradient[None]*coefficients[:,None,None,None] if ctx.needs_input_grad[1] else None,
                _fixed_sum_last((basis*gradient[None]).reshape(basis.shape[0],gradient.numel())) if ctx.needs_input_grad[2] else None)

def texture(mean,basis,coefficients):
    """Caller supplies explicit RGB floating texels, no hidden 255/channel conversion."""
    if mean.ndim!=3 or mean.shape[-1]!=3:raise ValueError('RGB texture required')
    _float(basis,(coefficients.numel(),*mean.shape),mean)
    _float(coefficients,(basis.shape[0],),mean)
    return _TexturePCA.apply(mean,basis,coefficients)

def shade_sh(normals,reference_coefficients):
    """Reference coefficient ordering mapped to gffx irradiance.

    gffx normalizes normals. Unlike upstream's interpolated nonunit normals,
    that is an explicit rendering-model difference, not claimed byte parity.
    """
    _float(reference_coefficients,(9,3),normals)
    # Reference order 1,x,y,z,xy,xz,yz,x2-y2,3z2-1 versus gffx 1,y,z,x,xy,yz,3z2-1,xz,x2-y2.
    coefficients=reference_coefficients[[0,2,3,1,4,6,8,5,7]]
    factors=coefficients.new_tensor([1/math.pi,1,1,1,1,1,1,1,1])
    return light.spherical_harmonics(normals.contiguous(),(coefficients*factors[:,None]).contiguous())

def objective(prediction,target,mask,predicted_landmarks,target_landmarks,parameters):
    """Historical full-image masked L1 denominator and mean landmark distance."""
    _float(target,prediction.shape,prediction);_float(mask,prediction.shape[:-1],prediction)
    _float(target_landmarks,predicted_landmarks.shape,predicted_landmarks)
    if bool((mask<0).any()) or bool((mask>1).any()) or not bool((mask>0).any()):
        raise ValueError('nonempty mask in [0,1] required')
    photometric=8*(mask[...,None]*(prediction-target).abs()).mean()
    landmark=torch.linalg.vector_norm(predicted_landmarks-target_landmarks,dim=-1).mean()
    prior=.5e-4*(parameters['shape'].square().sum()+parameters['expression'].square().sum())
    return dict(photometric=photometric,landmark=landmark,prior=prior,total=photometric+landmark+prior)

def render_image(vertices,faces,face_uv,albedo,lighting,camera,size):
    """Finite single-frame gffx unit-normal hard-SH profile, not opaque/Principled.

    Face UVs use image-row v-down coordinates. Camera landmarks use y-down;
    raster NDC uses y-up. Full normal derivatives are retained. No reference
    detached-normal compatibility or large-model admission is implied.
    """
    from gffx.torch import render
    if type(size)!=int or not 1<=size<=256:raise ValueError('finite profile requires size1..256')
    if vertices.shape[0]>5023 or faces.shape[0]>10000:raise ValueError('finite mesh cap')
    _float(face_uv,(faces.shape[0],3,2),vertices)
    if albedo.ndim!=3 or albedo.shape[-1]!=3 or max(albedo.shape[:2])>512:raise ValueError('finite RGB texture required')
    projected=project(vertices,camera)
    ndc=torch.stack((projected[:,0],-projected[:,1],.5+.05*projected[:,2]),-1).contiguous()
    if bool(((ndc[:,2]<0)|(ndc[:,2]>1)).any()):raise ValueError('depth outside fixed admission interval')
    normals=mesh.vertex_normals(vertices.contiguous(),faces.contiguous())
    if not bool((normals.square().sum(-1)>0).all()):raise ValueError('zero/isolated shading normal')
    index,bary,depth,_=render.rasterize(ndc,faces,size,size,cull_mode=render.CULL_NONE)
    uv=render.interpolate(index,bary,face_uv.contiguous())[0,...,0,:].reshape(-1,2).contiguous()
    normal=render.interpolate(index,bary,normals[faces.long()].contiguous())[0,...,0,:].reshape(-1,3)
    valid=index[0,...,0].reshape(-1)>=0
    normal=torch.where(valid[:,None],normal,normal.new_tensor([0.,0.,1.]))
    pyramid,offsets=render.texture_pyramid(albedo.contiguous(),levels=1)
    samples=render.texture(pyramid,offsets,albedo.shape[0],albedo.shape[1],uv,channel_count=3,filter=render.FILTER_BILINEAR,wrap_u=render.WRAP_BORDER,wrap_v=render.WRAP_BORDER,border=albedo.new_zeros(3))
    colour=samples*shade_sh(normal,lighting)*valid[:,None]
    return dict(colour=colour.reshape(size,size,3),mask=valid.reshape(size,size),depth=depth[0,...,0],profile='gffx-unit-normal-hard-sh-v1')

def optimize_stage(parameters,loss_function,*,iterations,learning_rate,seconds,cancelled=lambda:False):
    """Bounded finite experiment; every post-update loss is freshly evaluated.

    Call under the existing external process/memory watchdog. This is one stage,
    not permission to reset exhausted timers or launch the historical1000steps.
    """
    import time
    if type(iterations)!=int or not 1<=iterations<=64:raise ValueError('stage cap is64iterations')
    if not math.isfinite(seconds) or not 0<seconds<=90:raise ValueError('stage requires0<seconds<=90')
    if not math.isfinite(learning_rate) or not 0<learning_rate<=1:raise ValueError('invalid learning rate')
    values=list(parameters.values())
    if not values or len({id(v) for v in values})!=len(values) or any(not v.is_leaf or not v.requires_grad for v in values):raise ValueError('distinct gradient leaves required')
    deadline=time.perf_counter()+seconds
    def check():
        if cancelled():raise RuntimeError('cancelled fitting stage')
        if time.perf_counter()>=deadline:raise TimeoutError('fitting deadline exhausted')
    check();optimizer=torch.optim.Adam(values,lr=learning_rate)
    history=[]
    for step in range(iterations+1):
        check();optimizer.zero_grad();loss=loss_function()
        if loss.numel()!=1 or not bool(torch.isfinite(loss)):raise ValueError('nonfinite/scalar fitting loss')
        history.append(float(loss.detach()))
        if step==iterations:break
        loss.backward()
        if any(v.grad is None or not bool(torch.isfinite(v.grad).all()) for v in values):raise ValueError('missing/nonfinite fitting derivative')
        check();optimizer.step()
        if any(not bool(torch.isfinite(v).all()) for v in values):raise ValueError('nonfinite parameter update')
    check()
    return dict(status='completed',iterations=iterations,initial_loss=history[0],final_loss=history[-1],losses=history,parameters={k:v.detach().cpu().tolist() for k,v in parameters.items()},seconds=seconds-(deadline-time.perf_counter()))

@dataclass(frozen=True)
class FittingInputs:
    model: FlameData
    face_uv: torch.Tensor
    texture_mean: torch.Tensor
    texture_basis: torch.Tensor
    landmark_faces: torch.Tensor
    landmark_barycentric: torch.Tensor
    contour_faces: torch.Tensor | None = None
    contour_barycentric: torch.Tensor | None = None

def flame_pose(pose):
    """Reference6 controls: global axis-angle then jaw; neck/eyes fixed atzero."""
    _float(pose,(6,),pose)
    zero=torch.zeros_like(pose[:3])
    return torch.stack([pose[:3],zero,pose[3:],zero,zero])

def contour_bin(full_pose):
    """Reference's discrete yaw lookup. No derivative through bin selection."""
    _float(full_pose,(5,3),full_pose)
    rotation=transforms.rodrigues(full_pose[:2].contiguous())
    combined=_model_matmul(rotation[0],rotation[1])
    yaw=torch.atan2(-combined[2,0],torch.sqrt(combined[0,0]**2+combined[1,0]**2))
    angle=int(torch.round(torch.clamp(yaw.detach()*180/math.pi,max=39)))
    return min(78,39-angle) if angle<0 else angle

def evaluate(inputs,parameters,*,size):
    """One model->deformation->landmarks/camera->texture/SH->image evaluation."""
    required={'shape','expression','pose','camera','texture','lighting'}
    if set(parameters)!=required:raise ValueError('explicit six parameter groups required')
    full_pose=flame_pose(parameters['pose'])
    coefficients=torch.cat((parameters['shape'],parameters['expression']))
    vertices=deform(inputs.model,coefficients,full_pose)
    faces=inputs.landmark_faces;bary=inputs.landmark_barycentric
    if (inputs.contour_faces is None)!=(inputs.contour_barycentric is None):
        raise ValueError('contour indices and weights required together')
    if inputs.contour_faces is not None:
        if inputs.contour_faces.shape[0]!=79 or inputs.contour_barycentric.shape!=(79,inputs.contour_faces.shape[1],3):
            raise ValueError('reference79-bin contour shape required')
        index=contour_bin(full_pose)
        faces=torch.cat((inputs.contour_faces[index],faces));bary=torch.cat((inputs.contour_barycentric[index],bary))
    lmks=landmarks(inputs.model,vertices,faces,bary)
    albedo=texture(inputs.texture_mean,inputs.texture_basis,parameters['texture'])
    frame=render_image(vertices,inputs.model.faces,inputs.face_uv,albedo,parameters['lighting'],parameters['camera'],size)
    return dict(frame,landmarks=project(lmks,parameters['camera'])[:,:2],vertices=vertices,albedo=albedo)
