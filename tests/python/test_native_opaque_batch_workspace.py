"""Installed public workspace queries preserve caller-owned scratch contracts."""
import ctypes as C,os
from pathlib import Path
import gffx,torch,pytest

class Context(C.Structure):
    _fields_=[('struct_size',C.c_uint32),('abi_version',C.c_uint32),('device_type',C.c_uint32),('device_index',C.c_int32),('stream',C.c_void_p),('flags',C.c_uint32),('reserved0',C.c_uint32),('reserved',C.c_uint64*4)]

@pytest.mark.parametrize('name',['gffx_render_opaque_material_quadrature_workspace','gffx_render_opaque_material_batch_workspace'])
def test_material_workspace_queries_match_scratch_and_reject_invalid_metadata(name):
    device=os.environ.get('GFFX_BATCH_DEVICE','cpu')
    from gffx.torch.render import opaque_material_quadrature_batch
    v=torch.empty((0,3),device=device,dtype=torch.float64);f=torch.empty((0,3),device=device,dtype=torch.int32)
    opaque_material_quadrature_batch(v,f,v.new_empty((0,3,0)))
    lib=C.CDLL(str(Path(gffx.__file__).parent/'gffx_core.dll'))
    fn=getattr(lib,name);old=lib.gffx_render_opaque_quadrature_workspace
    for call in (fn,old):call.argtypes=[C.c_uint32,C.POINTER(Context),C.POINTER(C.c_uint64),C.POINTER(C.c_uint64),C.c_void_p];call.restype=C.c_uint32
    ctx=Context(C.sizeof(Context),65536,2 if device=='cuda' else 1,0,None,0,0,(C.c_uint64*4)())
    for dtype in (1,2):
        size=C.c_uint64();align=C.c_uint64();base=C.c_uint64();ba=C.c_uint64()
        assert fn(dtype,C.byref(ctx),C.byref(size),C.byref(align),None)==0
        assert old(dtype,C.byref(ctx),C.byref(base),C.byref(ba),None)==0
        assert 0<size.value==base.value<=1048576 and align.value==ba.value
    assert fn(2,C.byref(ctx),None,C.byref(align),None)!=0
    assert fn(3,C.byref(ctx),C.byref(size),C.byref(align),None)!=0
    ctx.reserved[0]=1
    assert fn(2,C.byref(ctx),C.byref(size),C.byref(align),None)!=0
