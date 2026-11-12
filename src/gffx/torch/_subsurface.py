"""Raster-only normalized diffuse diffusion; specular and albedo remain sharp.

World-space RGB lengths, same-object/depth bilateral rejection and normalized
finite-support sampling. This is a screen-space approximation, not random walk.
"""
import ctypes, math
import torch


def diffusion_length(radius, albedo):
    """Convert mean free path to the normalized diffusion profile's length.

    Christensen--Burley diffuse-surface fit, equation 6. Relative RGB albedo
    removes the common surface-throughput factor; no renderer code is reused.
    The low-level convolution functions below take the resulting length d.
    """
    a=albedo/albedo.amax(dim=-1,keepdim=True).clamp_min(1e-8)
    return radius/(1.9-a+3.5*(a-.8).square())

def diffuse(image,data,ids,fx,fy,*,perspective=True):
    from . import _raster_accel as accel
    h,w,_=image.shape
    if image.dtype==torch.float32 and accel.available(image.device):
        library=accel._libraries[image.device.type]
        if hasattr(library,'gffx_raster_diffusion_f32'):
            fn=library.gffx_raster_diffusion_f32
            fn.argtypes=[ctypes.c_int]*3+[ctypes.c_float]*2+[ctypes.c_void_p]*5;fn.restype=ctypes.c_int
            tensors=[t.contiguous()for t in [image,data,ids]];out=torch.empty_like(image)
            if image.is_cuda:
                stream=torch.cuda.current_stream(image.device)
                code=fn(1,h,w,float(fx)if perspective else-float(fx),float(fy),*[t.data_ptr()for t in tensors],out.data_ptr(),stream.cuda_stream)
                for t in [*tensors,out]:t.record_stream(stream)
            else:code=fn(0,h,w,float(fx)if perspective else-float(fx),float(fy),*[t.data_ptr()for t in tensors],out.data_ptr(),None)
            if code:raise RuntimeError('subsurface diffusion failed: '+str(code))
            return out
    return reference_diffuse(image,data,ids,fx,fy,perspective=perspective)

def reference_diffuse(image,data,ids,fx,fy,*,perspective=True):
    h,w,_=image.shape;yy,xx=torch.meshgrid(torch.arange(h,device=image.device),torch.arange(w,device=image.device),indexing='ij')
    result=torch.empty_like(image)
    for c in range(3):
        radius=data[...,1+c];z=data[...,0];total=torch.zeros_like(z);den=total.clone()
        for k in range(32):
            r=-math.log(1-((k//4)+.5)/8)*(1 if k%4==0 else 3)*radius
            px=xx+r*float(fx)/(z.clamp_min(1e-8)if perspective else 1)*math.cos(k*2.39996322972865332);py=yy+r*float(fy)/(z.clamp_min(1e-8)if perspective else 1)*math.sin(k*2.39996322972865332)
            ix=px.floor().long();iy=py.floor().long();ax=px-ix;ay=py-iy
            for oy in range(2):
                for ox in range(2):
                    qx=(ix+ox).clamp(0,w-1);qy=(iy+oy).clamp(0,h-1)
                    valid=(ix+ox>=0)&(ix+ox<w)&(iy+oy>=0)&(iy+oy<h)&(ids[qy,qx]==ids)&(data[qy,qx,0]>0)
                    a=(ax if ox else 1-ax)*(ay if oy else 1-ay)*valid
                    a=a*torch.exp(-(data[qy,qx,0]-z).abs()/(radius*2).clamp_min(1e-8))*data[qy,qx,5]
                    total+=a*image[qy,qx,c];den+=a
        value=total/den.clamp_min(1e-20)
        result[...,c]=torch.where((ids>=0)&(z>0)&(radius>0)&(den>1e-12),value,image[...,c])
    return result

def transmit(points,normals,parameters,source_data,size):
    from . import _raster_accel as accel
    if points.dtype!=torch.float32 or not accel.available(points.device):
        raise NotImplementedError('thin subsurface transmission currently requires native float32 CPU/CUDA; set subsurface_transmission=0 for float64')
    library=accel._libraries[points.device.type];fn=library.gffx_raster_transmission_f32
    fn.argtypes=[ctypes.c_int]*4+[ctypes.c_void_p]*8;fn.restype=ctypes.c_int
    f=points.new_zeros((len(points),12));f[:,:3]=parameters['base_color'];f[:,8:11]=parameters['subsurface_radius'];f[:,11]=torch.as_tensor(parameters.get('subsurface_transmission',1),device=points.device,dtype=points.dtype)
    tensors=[t.contiguous()for t in [points,normals,f,*source_data[:3]]];out=torch.empty_like(points)
    if points.is_cuda:
        stream=torch.cuda.current_stream(points.device);code=fn(1,len(points),len(source_data[0]),size,*[t.data_ptr()for t in tensors],out.data_ptr(),stream.cuda_stream)
        for t in [*tensors,out]:t.record_stream(stream)
    else:code=fn(0,len(points),len(source_data[0]),size,*[t.data_ptr()for t in tensors],out.data_ptr(),None)
    if code:raise RuntimeError('subsurface transmission failed: '+str(code))
    return out
