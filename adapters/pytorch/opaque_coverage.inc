/* Stable ABI boundary: all geometric arithmetic is native C11. */
Tensor opaque_coverage_call(const Tensor &vertices,const Tensor &faces,int64_t h,int64_t w,double ox,double oy,int64_t limit,const Tensor *gradient) {
    if(vertices.dim()!=2 || vertices.size(1)!=3 || faces.dim()!=2 || faces.size(1)!=3)
        raise_internal("opaque geometry must have shapes [V,3] and [F,3]");
    if(h<=0 || w<=0 || h>1048576 || w>1048576 || limit<=0 || limit>67108864)
        raise_internal("opaque dimensions/workspace budget outside supported limits");
    ViewArena arena;Diagnostic diagnostic;auto context=context_for(vertices);uint64_t bytes=0,alignment=0;
    GFFX_CHECK(gffx_render_opaque_coverage_workspace(dtype_of(vertices),&context,&bytes,&alignment,&diagnostic.buffer),diagnostic.text());
    uint64_t element=dtype_of(vertices)==GFFX_DTYPE_FLOAT64?8:4;
    uint64_t output_count=gradient?static_cast<uint64_t>(vertices.numel()):static_cast<uint64_t>(h)*static_cast<uint64_t>(w);
    if(bytes>static_cast<uint64_t>(limit) || output_count>(static_cast<uint64_t>(limit)-bytes)/element)
        raise_internal("opaque workspace and output exceed workspace budget");
    Tensor workspace=workspace_of(vertices,bytes);auto buffer=workspace_buffer(workspace,bytes);
    std::vector<int64_t> shape=gradient?std::vector<int64_t>{vertices.size(0),3}:std::vector<int64_t>{h,w};
    Tensor output=torch::stable::new_empty(vertices,shape);auto v=arena.read(vertices),f=arena.read(faces),o=arena.write(output);
    if(gradient){auto g=arena.read(*gradient);GFFX_CHECK(gffx_render_opaque_coverage_backward(&v,&f,h,w,ox,oy,&g,&context,&o,&buffer,&diagnostic.buffer),diagnostic.text());}
    else GFFX_CHECK(gffx_render_opaque_coverage(&v,&f,h,w,ox,oy,&context,&o,&buffer,&diagnostic.buffer),diagnostic.text());
    return output;
}
Tensor opaque_coverage(const Tensor &v,const Tensor &f,int64_t h,int64_t w,double ox,double oy,int64_t limit){return opaque_coverage_call(v,f,h,w,ox,oy,limit,nullptr);}
Tensor opaque_coverage_backward(const Tensor &v,const Tensor &f,const Tensor &g,int64_t h,int64_t w,double ox,double oy,int64_t limit){return opaque_coverage_call(v,f,h,w,ox,oy,limit,&g);}
