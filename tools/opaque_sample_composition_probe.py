"""Development-only original-profile diagnostic; not a supported renderer or verifier."""
from pathlib import Path
import ctypes,torch
DLL=Path('D:/devspace/gffx/qualification-20260913/m1-continuation-20260913/opaque208-20261003/probe.dll')

_library=ctypes.CDLL(str(DLL))
_function=_library.compose
_function.argtypes=[ctypes.c_int]*4+[ctypes.c_void_p]*20
_function.restype=ctypes.c_int

def compose(packed,camera,clip,normals,owners,q,workload,*,contributions=False,_shared_curves=None):
    from gffx.torch._principled_groups import _prepare_material_curve
    from gffx.torch._deferred import _light_packet
    assert not torch.is_grad_enabled() and camera.batch_size==1
    assert len(packed.materials)==1 and len(packed.lights)==1 and q.device.type=='cpu'
    assert q.dtype==torch.float64 and q.shape[-2:]==(800,4)
    params=packed.materials[0].parameters
    assert params['albedo'].shape==(3,) and params['roughness'].ndim==0 and params['f0'].shape==(3,)
    assert not params.get('two_sided',False)
    state={} if _shared_curves is None else _shared_curves.setdefault('_native_probe',{})
    inputs=[params['roughness'],*params['budget']._data.values(),packed.world_matrices[packed.lights[0].name],*packed.lights[0].parameters.values()]
    signature=tuple((id(x),x._version) for x in inputs if isinstance(x,torch.Tensor))
    if state:assert state['signature']==signature,'diagnostic inputs mutated'
    else:
        state['signature']=signature
        state['curve']=_prepare_material_curve(params['budget'],params['roughness'].reshape(1))
        state['packet']=_light_packet(packed.lights[0],packed,0,packed.vertices[:1])
    cn,curve,partial,mean,width=state['curve'];packet=state['packet']
    assert packet['direction'].shape==(1,1,3) and bool(packet['valid'].all())
    count=q.numel()//3200;assert 1<=count<=16
    weighted=torch.empty_like(q);rgba=q.new_empty((count,4))
    args=[owners,q,packed.vertices,packed.faces,clip,normals,camera.pose[0,:3,3],cn[1:],params['budget']._data['_clog'],curve,partial,mean,width,params['albedo'],params['roughness'],params['f0'],packet['direction'],packet['irradiance'],weighted,rgba]
    values=[x.contiguous() for x in args]
    assert all(x.dtype in (torch.float64,torch.int32) and x.device.type=='cpu' for x in values)
    result=_function(count,len(packed.vertices),len(packed.faces),len(cn)-1,*(x.data_ptr() for x in values));assert result==0,('native probe failure',result)
    rgb=rgba[:,:3].reshape(*q.shape[:-2],3);alpha=rgba[:,3].reshape(q.shape[:-2])
    return (rgb,alpha,weighted) if contributions else (rgb,alpha)
