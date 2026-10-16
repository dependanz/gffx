"""Independent polynomial flux and visibility controls for native pixel edges."""
import os
import pytest,torch

def call(v,f,edge,cuts=None,curves=None,**kw):
    from gffx.torch.render import opaque_boundary_quadrature
    return opaque_boundary_quadrature(v,f,v.new_empty((len(f),3,0)) if cuts is None else cuts,1,1,edge=edge,curves=curves,**kw)

def fixture():
    device=os.environ.get('GFFX_KNOT_DEVICE','cpu')
    v=torch.tensor([[-2.,-2.,.2],[6.,-2.,.2],[-2.,6.,.2]],dtype=torch.float64,device=device)
    f=torch.tensor([[0,1,2]],dtype=torch.int32,device=device)
    return v,f

@pytest.mark.parametrize('edge',range(4))
def test_native_boundary_polynomial_line_integral(edge):
    v,f=fixture();ids,q=call(v,f,edge);live=ids[0]>=0;q=q[0,live]
    xy=q[:,:3]@v[:,:2];t=xy[:,1 if edge<2 else 0]
    assert len(q)==12
    assert torch.allclose((q[:,3,None]*t[:,None]**torch.arange(7,device=v.device)).sum(0),v.new_tensor([1/(i+1) for i in range(7)]),atol=2e-14,rtol=0)
    assert torch.all((xy[:,0 if edge<2 else 1]-edge%2).abs()<2e-14)


def test_native_boundary_depth_intersection_and_material_split():
    v,f=fixture();v=torch.cat((v,v.clone()));v[3:,2]=v[3:,1]*.2+.1;f=torch.tensor([[0,1,2],[3,4,5]],dtype=torch.int32,device=v.device)
    cuts=(v[f.long(),1]-.25)[...,None].contiguous();ids,q=call(v,f,0,cuts)
    weights=q[0,:,3]
    assert abs(weights[ids[0]==0].sum().item()-.5)<2e-14
    assert abs(weights[ids[0]==1].sum().item()-.5)<2e-14
    assert (ids>=0).sum().item()==36


def test_native_boundary_quadratic_split_and_empty_padding():
    v,f=fixture();b=v[:,1];coef=torch.stack((b[0]*b[0]-.25,b[1]*b[1]-.25,b[2]*b[2]-.25,2*(b[0]*b[1]-.25),2*(b[0]*b[2]-.25),2*(b[1]*b[2]-.25)))[None]
    ids,q=call(v,f,0,curves=coef);live=ids[0]>=0;nodes=q[0,live];y=nodes[:,:3]@b
    assert live.sum().item()==24
    assert abs(nodes[y<.5,3].sum().item()-.5)<2e-14
    ids,q=call(v,f[:0],0);assert (ids==-1).all() and (q==0).all()


def test_native_boundary_refuses_capacity_budget_and_input_derivatives():
    v,f=fixture()
    with pytest.raises(OverflowError):call(v,f.repeat(33,1),0)
    with pytest.raises(ValueError):call(v,f,0,workspace_limit_bytes=1048576)
    with pytest.raises(ValueError):call(v,f,4)
    with pytest.raises(NotImplementedError):call(v.requires_grad_(),f,0)


def test_native_boundary_rejects_edge_coincident_silhouette():
    v,f=fixture();v[:,0]+=2
    with pytest.raises(OverflowError):call(v,f,0)


@pytest.mark.skipif(os.environ.get('GFFX_KNOT_DEVICE','cpu')!='cuda',reason='separate CUDA resource profile')
def test_native_boundary_exact_device_bytes():
    from gffx.torch.render import opaque_boundary_quadrature
    v,f=fixture();v=v.cpu();f=f.cpu();cuts=(v[f.long(),1]-.25)[...,None].contiguous()
    if not torch.cuda.is_available():pytest.skip('CUDA device required')
    for edge in range(4):
        a=opaque_boundary_quadrature(v,f,cuts,1,1,edge=edge)
        b=opaque_boundary_quadrature(v.cuda(),f.cuda(),cuts.cuda(),1,1,edge=edge)
        assert all(torch.equal(x.contiguous().view(torch.uint8),y.cpu().contiguous().view(torch.uint8)) for x,y in zip(a,b))


def test_native_boundary_c_abi_scratch_and_validation():
    import ctypes as C
    from tools.animation_cost import Context,Buffer,View,ABI,tensor_view,bind_native
    v,f=fixture();cuts=v.new_empty((1,3,0));ei,eq=call(v,f,0);lib,_=bind_native()
    ctx=Context();ctx.struct_size=C.sizeof(ctx);ctx.abi_version=ABI;ctx.device_type=2 if v.is_cuda else 1;ctx.device_index=torch.cuda.current_device() if v.is_cuda else 0;ctx.stream=torch.cuda.current_stream().cuda_stream if v.is_cuda else None
    query=lib.gffx_render_opaque_quadrature_workspace;query.argtypes=[C.c_uint32,C.POINTER(Context),C.POINTER(C.c_uint64),C.POINTER(C.c_uint64),C.c_void_p];query.restype=C.c_uint32
    size=C.c_uint64();align=C.c_uint64();assert query(2,C.byref(ctx),C.byref(size),C.byref(align),None)==0
    storage=torch.empty(size.value,dtype=torch.uint8,device=v.device);work=Buffer();work.struct_size=C.sizeof(work);work.abi_version=ABI;work.data=storage.data_ptr();work.capacity_bytes=size.value;work.alignment=align.value;work.device_type=ctx.device_type;work.device_index=ctx.device_index
    ids=torch.full_like(ei,91);q=torch.full_like(eq,91);vv,ff,cc,ii,qq=[tensor_view(t,output=k>=3) for k,t in enumerate((v,f,cuts,ids,q))];ff.dtype=ii.dtype=3
    fun=lib.gffx_render_opaque_boundary_quadrature;fun.argtypes=[C.POINTER(View)]*3+[C.c_int64,C.c_int64,C.c_double,C.c_double,C.c_int32,C.POINTER(Context),C.POINTER(View),C.POINTER(View),C.POINTER(Buffer),C.c_void_p];fun.restype=C.c_uint32
    def run(edge=0):return fun(C.byref(vv),C.byref(ff),C.byref(cc),1,1,0.,0.,edge,C.byref(ctx),C.byref(ii),C.byref(qq),C.byref(work),None)
    for pattern in (0,165,255):
        storage.fill_(pattern);assert run()==0
        assert torch.equal(ids,ei) and torch.equal(q.view(torch.uint8),eq.view(torch.uint8))
    assert run(4)==1
    work.capacity_bytes=size.value-1;assert run()==3;work.capacity_bytes=size.value
    work.data=v.data_ptr();assert run()==1;work.data=storage.data_ptr()
    qq.data=v.data_ptr();assert run()==1;qq.data=q.data_ptr()
    f[0,2]=99;assert run()==1


def test_native_boundary_diagnostic_header_precedes_edge_validation():
    import ctypes as C
    from tools.animation_cost import bind_native
    lib,_=bind_native();fun=lib.gffx_render_opaque_boundary_quadrature
    fun.argtypes=[C.c_void_p]*3+[C.c_int64,C.c_int64,C.c_double,C.c_double,C.c_int32]+[C.c_void_p]*5;fun.restype=C.c_uint32
    bad=C.create_string_buffer(64) # full allocated storage, intentionally zero struct_size
    assert fun(None,None,None,1,1,0.,0.,4,None,None,None,None,C.byref(bad))==6
    assert bad.raw==bytes(64)


@pytest.mark.parametrize('event',['tangent','coincident-cut','segments'])
def test_native_boundary_refuses_unqualified_events_and_final_segment_overflow(event):
    v,f=fixture();y=v[:,1]
    if event=='coincident-cut':
        cuts=v[f.long(),0][...,None].contiguous()
        with pytest.raises(OverflowError):call(v,f,0,cuts)
    elif event=='tangent':
        z=y-.5;c=torch.stack((z[0]*z[0],z[1]*z[1],z[2]*z[2],2*z[0]*z[1],2*z[0]*z[2],2*z[1]*z[2]))[None]
        with pytest.raises(OverflowError):call(v,f,0,curves=c)
    else:
        v=torch.cat((v,v.clone()));v[3:,2]=v[3:,1]*.2+.1;f=torch.tensor([[0,1,2],[3,4,5]],dtype=torch.int32,device=v.device)
        thresholds=torch.arange(1,17,device=v.device,dtype=v.dtype)/34
        cuts=torch.stack((v[:3,1,None]-(thresholds+.5),v[3:,1,None]-thresholds))
        with pytest.raises(OverflowError):call(v,f,0,cuts.contiguous())


def test_native_boundary_degenerate_and_occluded_faces_do_not_contribute():
    v,f=fixture();v=torch.cat((v,v.clone()));v[3:,2]+=.1
    f=torch.tensor([[0,0,1],[0,1,2],[3,4,5]],dtype=torch.int32,device=v.device)
    ids,q=call(v,f,0);assert (ids[ids>=0]==1).all()
    assert abs(q[...,3].sum().item()-1)<2e-14
