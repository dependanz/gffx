/* Thin stable ABI conversion; all integration and VJP arithmetic is C11. */
std::tuple<Tensor,Tensor> opaque_integrate_call(const Tensor &v,const Tensor &f,const Tensor &c,int64_t h,int64_t w,double ox,double oy,int64_t limit,const Tensor *g) {
    if(v.dim()!=2||v.size(1)!=3||f.dim()!=2||f.size(1)!=3||c.dim()!=3||c.size(0)!=f.size(0)||c.size(1)!=3||c.size(2)!=3)
        raise_internal("opaque colour shapes require vertices[V,3],faces[F,3],radiance[F,3,3]");
    if(h<=0||w<=0||h>1048576||w>1048576||limit<=0||limit>67108864)raise_internal("opaque colour dimensions/workspace budget invalid");
    ViewArena arena;Diagnostic diagnostic;auto context=context_for(v);uint64_t bytes=0,alignment=0;
    GFFX_CHECK(gffx_render_opaque_integrate_workspace(dtype_of(v),&context,&bytes,&alignment,&diagnostic.buffer),diagnostic.text());
    uint64_t element=dtype_of(v)==GFFX_DTYPE_FLOAT64?8:4;
    uint64_t count=g?static_cast<uint64_t>(v.numel())+static_cast<uint64_t>(c.numel()):4*static_cast<uint64_t>(h)*static_cast<uint64_t>(w);
    if(bytes>static_cast<uint64_t>(limit)||count>(static_cast<uint64_t>(limit)-bytes)/element)raise_internal("opaque colour workspace/output budget exceeded");
    Tensor workspace=workspace_of(v,bytes);auto buffer=workspace_buffer(workspace,bytes);
    Tensor out=torch::stable::new_empty(v,g?std::vector<int64_t>{v.size(0),3}:std::vector<int64_t>{h,w,4});
    Tensor gc=g?torch::stable::new_empty(c,{f.size(0),3,3}):out;
    auto vv=arena.read(v),ff=arena.read(f),cc=arena.read(c),oo=arena.write(out);
    if(g){auto gg=arena.read(*g),cg=arena.write(gc);GFFX_CHECK(gffx_render_opaque_integrate_backward(&vv,&ff,&cc,h,w,ox,oy,&gg,&context,&oo,&cg,&buffer,&diagnostic.buffer),diagnostic.text());}
    else GFFX_CHECK(gffx_render_opaque_integrate(&vv,&ff,&cc,h,w,ox,oy,&context,&oo,&buffer,&diagnostic.buffer),diagnostic.text());
    return {out,gc};
}
Tensor opaque_integrate(const Tensor &v,const Tensor &f,const Tensor &c,int64_t h,int64_t w,double ox,double oy,int64_t limit){return std::get<0>(opaque_integrate_call(v,f,c,h,w,ox,oy,limit,nullptr));}
std::tuple<Tensor,Tensor> opaque_integrate_backward(const Tensor &v,const Tensor &f,const Tensor &c,const Tensor &g,int64_t h,int64_t w,double ox,double oy,int64_t limit){return opaque_integrate_call(v,f,c,h,w,ox,oy,limit,&g);}

std::tuple<Tensor,Tensor> opaque_quadrature_call(const Tensor &v,const Tensor &f,const Tensor &cuts,int64_t h,int64_t w,double ox,double oy,int64_t limit,const Tensor *g,int edge=-1){
 if(v.dim()!=2||v.size(1)!=3||f.dim()!=2||f.size(1)!=3||cuts.dim()!=3||cuts.size(0)!=f.size(0)||!((cuts.size(1)==3&&cuts.size(2)<=16)||(cuts.size(1)==1&&cuts.size(2)>=6&&cuts.size(2)<=51&&(cuts.size(2)-6)%3==0)))raise_internal("opaque quadrature shapes/cut capacity");
 if(h<=0||w<=0||h>1048576||w>1048576||limit<=0||limit>67108864)raise_internal("opaque quadrature dimensions/budget");
 ViewArena arena;Diagnostic d;auto ctx=context_for(v);uint64_t bytes=0,alignment=0;
 GFFX_CHECK(gffx_render_opaque_quadrature_workspace(dtype_of(v),&ctx,&bytes,&alignment,&d.buffer),d.text());
 uint64_t element=dtype_of(v)==GFFX_DTYPE_FLOAT64?8:4;
 uint64_t output=g?(static_cast<uint64_t>(v.numel())+static_cast<uint64_t>(cuts.numel()))*element:static_cast<uint64_t>(h)*static_cast<uint64_t>(w)*800*(4*element+4);
 // Optional private slabs; the minimum C workspace and serial fallback remain.
 if(ctx.device_type==GFFX_DEVICE_CUDA && !g && edge<0 && h*w>1 && bytes<=1048576 && static_cast<uint64_t>(limit)>=4*1048576+output)bytes=4*1048576;
 if(bytes>static_cast<uint64_t>(limit)||output>static_cast<uint64_t>(limit)-bytes)raise_internal("opaque quadrature workspace/output budget exceeded");
 Tensor work=workspace_of(v,bytes);auto buffer=workspace_buffer(work,bytes);
 Tensor a=g?torch::stable::new_empty(v,{v.size(0),3}):torch::stable::new_empty(f,{h*w,800});
 Tensor b=g?torch::stable::new_empty(cuts,{f.size(0),cuts.size(1),cuts.size(2)}):torch::stable::new_empty(v,{h*w,800,4});
 auto vv=arena.read(v),ff=arena.read(f),cc=arena.read(cuts),aa=arena.write(a),bb=arena.write(b);
 if(g){auto gg=arena.read(*g);GFFX_CHECK((edge==-2?gffx_render_opaque_material_quadrature_backward:gffx_render_opaque_quadrature_backward)(&vv,&ff,&cc,h,w,ox,oy,&gg,&ctx,&aa,&bb,&buffer,&d.buffer),d.text());}
 else if(edge>=0)GFFX_CHECK(gffx_render_opaque_boundary_quadrature(&vv,&ff,&cc,h,w,ox,oy,edge,&ctx,&aa,&bb,&buffer,&d.buffer),d.text());
 else GFFX_CHECK((edge==-2?gffx_render_opaque_material_quadrature:gffx_render_opaque_quadrature)(&vv,&ff,&cc,h,w,ox,oy,&ctx,&aa,&bb,&buffer,&d.buffer),d.text());
 return {a,b};
}
std::tuple<Tensor,Tensor> opaque_quadrature(const Tensor &v,const Tensor &f,const Tensor &c,int64_t h,int64_t w,double ox,double oy,int64_t limit){return opaque_quadrature_call(v,f,c,h,w,ox,oy,limit,nullptr);}
std::tuple<Tensor,Tensor> opaque_quadrature_backward(const Tensor &v,const Tensor &f,const Tensor &c,const Tensor &g,int64_t h,int64_t w,double ox,double oy,int64_t limit){return opaque_quadrature_call(v,f,c,h,w,ox,oy,limit,&g);}

std::tuple<Tensor,Tensor> opaque_boundary_quadrature(const Tensor &v,const Tensor &f,const Tensor &c,int64_t h,int64_t w,double ox,double oy,int64_t edge,int64_t limit){if(edge<0||edge>3)raise_internal("opaque boundary edge");return opaque_quadrature_call(v,f,c,h,w,ox,oy,limit,nullptr,(int)edge);}

std::tuple<Tensor,Tensor> opaque_material_quadrature(const Tensor &v,const Tensor &f,const Tensor &c,int64_t h,int64_t w,double ox,double oy,int64_t limit){return opaque_quadrature_call(v,f,c,h,w,ox,oy,limit,nullptr,-2);}
std::tuple<Tensor,Tensor> opaque_material_quadrature_backward(const Tensor &v,const Tensor &f,const Tensor &c,const Tensor &g,int64_t h,int64_t w,double ox,double oy,int64_t limit){return opaque_quadrature_call(v,f,c,h,w,ox,oy,limit,&g,-2);}

std::tuple<Tensor,Tensor,std::vector<int64_t>> opaque_batch_call(const Tensor &v,const Tensor &f,const Tensor &cuts,double ox,double oy,int64_t begin,int64_t batch,int64_t slots,int64_t blocks,int64_t limit,const Tensor *g,int64_t rule=0,int64_t derivatives=3,int32_t *prefix=nullptr){
 if(derivatives<0||derivatives>3||rule<0||rule>4)raise_internal("opaque batch rule");
 const int64_t max_batches=rule==4?80:(rule==3?64:(rule==2?52:16));
 if(v.dim()!=2||v.size(1)!=3||f.dim()!=2||f.size(1)!=3||cuts.dim()!=3||cuts.size(0)!=f.size(0))raise_internal("opaque batch shapes");
 if(begin<0||begin>max_batches*32||batch<0||batch>=max_batches||slots<0||slots>max_batches*800||blocks<0||blocks>max_batches*32||limit<=0||limit>67108864)raise_internal("opaque batch cursor/budget");
 ViewArena arena;Diagnostic d;auto ctx=context_for(v);uint64_t bytes=0,alignment=0;GFFX_CHECK(gffx_render_opaque_material_batch_workspace(dtype_of(v),&ctx,&bytes,&alignment,&d.buffer),d.text());
 uint64_t element=dtype_of(v)==GFFX_DTYPE_FLOAT64?8:4;
 uint64_t output=g?(static_cast<uint64_t>(v.numel())+static_cast<uint64_t>(cuts.numel()))*element:800*(4*element+4);
 if(bytes>static_cast<uint64_t>(limit)||output>static_cast<uint64_t>(limit)-bytes)raise_internal("opaque batch workspace/output budget");
 Tensor work=workspace_of(v,bytes);auto buffer=workspace_buffer(work,bytes);
 Tensor a=g?torch::stable::new_empty(v,{v.size(0),3}):torch::stable::new_empty(f,{1,800});
 Tensor b=g?torch::stable::new_empty(cuts,{f.size(0),cuts.size(1),cuts.size(2)}):torch::stable::new_empty(v,{1,800,4});
 auto vv=arena.read(v),ff=arena.read(f),cc=arena.read(cuts),aa=arena.write(a),bb=arena.write(b);
 int32_t cursor[4]={(int32_t)begin,(int32_t)batch,(int32_t)slots,(int32_t)blocks},receipt[8]={0};
 if(g){auto gg=arena.read(*g);GFFX_CHECK(prefix?(rule==4?gffx_render_opaque_material_prefix_backward80:(rule==3?gffx_render_opaque_material_prefix_backward64:gffx_render_opaque_material_prefix_backward))(&vv,&ff,&cc,ox,oy,cursor,receipt,&gg,&ctx,&aa,&bb,&buffer,&d.buffer,(int32_t)derivatives,prefix):gffx_render_opaque_material_batch_rule_backward(&vv,&ff,&cc,ox,oy,cursor,receipt,&gg,&ctx,&aa,&bb,&buffer,&d.buffer,(int32_t)rule,(int32_t)derivatives),d.text());}
 else GFFX_CHECK(gffx_render_opaque_material_batch_rule(&vv,&ff,&cc,ox,oy,cursor,receipt,&ctx,&aa,&bb,&buffer,&d.buffer,(int32_t)rule),d.text());
 return {a,b,std::vector<int64_t>(receipt,receipt+8)};
}
std::tuple<Tensor,Tensor,std::vector<int64_t>> opaque_material_batch(const Tensor &v,const Tensor &f,const Tensor &c,double ox,double oy,int64_t begin,int64_t batch,int64_t slots,int64_t blocks,int64_t limit){return opaque_batch_call(v,f,c,ox,oy,begin,batch,slots,blocks,limit,nullptr);}
std::tuple<Tensor,Tensor,std::vector<int64_t>> opaque_material_batch_backward(const Tensor &v,const Tensor &f,const Tensor &c,const Tensor &g,double ox,double oy,int64_t begin,int64_t batch,int64_t slots,int64_t blocks,int64_t limit){return opaque_batch_call(v,f,c,ox,oy,begin,batch,slots,blocks,limit,&g);}

std::tuple<Tensor,Tensor,std::vector<int64_t>> opaque_material_batch_rule(const Tensor &v,const Tensor &f,const Tensor &c,double ox,double oy,int64_t begin,int64_t batch,int64_t slots,int64_t blocks,int64_t limit,int64_t rule){return opaque_batch_call(v,f,c,ox,oy,begin,batch,slots,blocks,limit,nullptr,rule);}
std::tuple<Tensor,Tensor,std::vector<int64_t>> opaque_material_batch_rule_backward(const Tensor &v,const Tensor &f,const Tensor &c,const Tensor &g,double ox,double oy,int64_t begin,int64_t batch,int64_t slots,int64_t blocks,int64_t limit,int64_t rule,int64_t derivatives){return opaque_batch_call(v,f,c,ox,oy,begin,batch,slots,blocks,limit,&g,rule,derivatives);}

std::tuple<Tensor,std::vector<int64_t>> opaque_partition_prepare(const Tensor &v,const Tensor &f,const Tensor &cuts,double ox,double oy,int64_t limit){
 ViewArena arena;Diagnostic d;auto ctx=context_for(v);uint64_t bytes=0,alignment=0;
 GFFX_CHECK(gffx_render_opaque_material_batch_workspace(dtype_of(v),&ctx,&bytes,&alignment,&d.buffer),d.text());
 if(limit<=0||limit>67108864||bytes+1664*17*8>(uint64_t)limit)raise_internal("partition storage budget");
 Tensor work=workspace_of(v,bytes),rows=torch::stable::new_empty(v,{1664,17});auto buffer=workspace_buffer(work,bytes);auto vv=arena.read(v),ff=arena.read(f),cc=arena.read(cuts),rr=arena.write(rows);int32_t result[2]={0};
 GFFX_CHECK(gffx_render_opaque_partition_prepare(&vv,&ff,&cc,ox,oy,&ctx,&rr,result,&buffer,&d.buffer),d.text());
 return {rows,std::vector<int64_t>(result,result+2)};
}
std::tuple<Tensor,Tensor> opaque_partition_emit(const Tensor &rows,int64_t count,int64_t begin){
 if(count<0||count>1664||begin<0||begin>count||begin%32)raise_internal("partition range");
 ViewArena arena;Diagnostic d;auto ctx=context_for(rows);Tensor ids=torch::stable::new_empty(rows,{1,800},ScalarType::Int),q=torch::stable::new_empty(rows,{1,800,4});auto rr=arena.read(rows),ii=arena.write(ids),qq=arena.write(q);
 Tensor work=workspace_of(rows,1048576);auto buffer=workspace_buffer(work,1048576);
 GFFX_CHECK(gffx_render_opaque_partition_emit_workspace(&rr,(int32_t)count,(int32_t)begin,&ctx,&ii,&qq,&buffer,&d.buffer),d.text());return {ids,q};
}

std::tuple<Tensor,Tensor,std::vector<int64_t>,std::vector<int64_t>> opaque_material_prefix_backward(const Tensor &v,const Tensor &f,const Tensor &c,const Tensor &g,double ox,double oy,int64_t begin,int64_t batch,int64_t slots,int64_t blocks,int64_t limit,int64_t derivatives,int64_t face,int64_t seen){
 if(face<0||face>512||seen<0||seen>1664)raise_internal("prefix range");int32_t prefix[2]={(int32_t)face,(int32_t)seen};
 auto result=opaque_batch_call(v,f,c,ox,oy,begin,batch,slots,blocks,limit,&g,2,derivatives,prefix);
 return {std::get<0>(result),std::get<1>(result),std::get<2>(result),std::vector<int64_t>(prefix,prefix+2)};
}

std::tuple<Tensor,std::vector<int64_t>> opaque_partition_prepare64(const Tensor &v,const Tensor &f,const Tensor &cuts,double ox,double oy,int64_t limit){
 ViewArena arena;Diagnostic d;auto ctx=context_for(v);uint64_t bytes=0,alignment=0;
 GFFX_CHECK(gffx_render_opaque_material_batch_workspace(dtype_of(v),&ctx,&bytes,&alignment,&d.buffer),d.text());
 if(limit<=0||limit>67108864||bytes+2048*17*8>(uint64_t)limit)raise_internal("partition storage budget");
 Tensor work=workspace_of(v,bytes),rows=torch::stable::new_empty(v,{2048,17});auto buffer=workspace_buffer(work,bytes);auto vv=arena.read(v),ff=arena.read(f),cc=arena.read(cuts),rr=arena.write(rows);int32_t result[2]={0};
 GFFX_CHECK(gffx_render_opaque_partition_prepare64(&vv,&ff,&cc,ox,oy,&ctx,&rr,result,&buffer,&d.buffer),d.text());
 return {rows,std::vector<int64_t>(result,result+2)};
}

std::tuple<Tensor,Tensor> opaque_partition_emit64(const Tensor &rows,int64_t count,int64_t begin){
 if(count<0||count>2048||begin<0||begin>count||begin%32)raise_internal("partition range");
 ViewArena arena;Diagnostic d;auto ctx=context_for(rows);Tensor ids=torch::stable::new_empty(rows,{1,800},ScalarType::Int),q=torch::stable::new_empty(rows,{1,800,4});auto rr=arena.read(rows),ii=arena.write(ids),qq=arena.write(q);
 Tensor work=workspace_of(rows,1048576);auto buffer=workspace_buffer(work,1048576);
 GFFX_CHECK(gffx_render_opaque_partition_emit_workspace64(&rr,(int32_t)count,(int32_t)begin,&ctx,&ii,&qq,&buffer,&d.buffer),d.text());return {ids,q};
}

std::tuple<Tensor,Tensor,std::vector<int64_t>,std::vector<int64_t>> opaque_material_prefix_backward64(const Tensor &v,const Tensor &f,const Tensor &c,const Tensor &g,double ox,double oy,int64_t begin,int64_t batch,int64_t slots,int64_t blocks,int64_t limit,int64_t derivatives,int64_t face,int64_t seen){
 if(face<0||face>512||seen<0||seen>2048)raise_internal("prefix range");int32_t prefix[2]={(int32_t)face,(int32_t)seen};
 auto result=opaque_batch_call(v,f,c,ox,oy,begin,batch,slots,blocks,limit,&g,3,derivatives,prefix);
 return {std::get<0>(result),std::get<1>(result),std::get<2>(result),std::vector<int64_t>(prefix,prefix+2)};
}

std::tuple<Tensor,Tensor,std::vector<int64_t>> opaque_boundary_batch64_call(const Tensor &v,const Tensor &f,const Tensor &cuts,double ox,double oy,int64_t begin,int64_t batch,int64_t slots,int64_t blocks,int64_t limit,const Tensor *g,int64_t edge){
 if(edge<0||edge>3||g)raise_internal("boundary stream edge/gradient");
 const int64_t rule=3,derivatives=0;int32_t *prefix=nullptr;
 const int64_t max_batches=rule==4?80:(rule==3?64:(rule==2?52:16));
 if(v.dim()!=2||v.size(1)!=3||f.dim()!=2||f.size(1)!=3||cuts.dim()!=3||cuts.size(0)!=f.size(0))raise_internal("opaque batch shapes");
 if(begin<0||begin>max_batches*32||batch<0||batch>=max_batches||slots<0||slots>max_batches*800||blocks<0||blocks>max_batches*32||limit<=0||limit>67108864)raise_internal("opaque batch cursor/budget");
 ViewArena arena;Diagnostic d;auto ctx=context_for(v);uint64_t bytes=0,alignment=0;GFFX_CHECK(gffx_render_opaque_material_batch_workspace(dtype_of(v),&ctx,&bytes,&alignment,&d.buffer),d.text());
 uint64_t element=dtype_of(v)==GFFX_DTYPE_FLOAT64?8:4;
 uint64_t output=g?(static_cast<uint64_t>(v.numel())+static_cast<uint64_t>(cuts.numel()))*element:800*(4*element+4);
 if(bytes>static_cast<uint64_t>(limit)||output>static_cast<uint64_t>(limit)-bytes)raise_internal("opaque batch workspace/output budget");
 Tensor work=workspace_of(v,bytes);auto buffer=workspace_buffer(work,bytes);
 Tensor a=g?torch::stable::new_empty(v,{v.size(0),3}):torch::stable::new_empty(f,{1,800});
 Tensor b=g?torch::stable::new_empty(cuts,{f.size(0),cuts.size(1),cuts.size(2)}):torch::stable::new_empty(v,{1,800,4});
 auto vv=arena.read(v),ff=arena.read(f),cc=arena.read(cuts),aa=arena.write(a),bb=arena.write(b);
 int32_t cursor[4]={(int32_t)begin,(int32_t)batch,(int32_t)slots,(int32_t)blocks},receipt[8]={0};
 if(g){auto gg=arena.read(*g);GFFX_CHECK(prefix?(rule==4?gffx_render_opaque_material_prefix_backward80:(rule==3?gffx_render_opaque_material_prefix_backward64:gffx_render_opaque_material_prefix_backward))(&vv,&ff,&cc,ox,oy,cursor,receipt,&gg,&ctx,&aa,&bb,&buffer,&d.buffer,(int32_t)derivatives,prefix):gffx_render_opaque_material_batch_rule_backward(&vv,&ff,&cc,ox,oy,cursor,receipt,&gg,&ctx,&aa,&bb,&buffer,&d.buffer,(int32_t)rule,(int32_t)derivatives),d.text());}
 else GFFX_CHECK(gffx_render_opaque_boundary_batch64(&vv,&ff,&cc,ox,oy,(int32_t)edge,cursor,receipt,&ctx,&aa,&bb,&buffer,&d.buffer),d.text());
 return {a,b,std::vector<int64_t>(receipt,receipt+8)};
}
std::tuple<Tensor,Tensor,std::vector<int64_t>> opaque_boundary_batch64(const Tensor &v,const Tensor &f,const Tensor &c,double ox,double oy,int64_t edge,int64_t begin,int64_t batch,int64_t slots,int64_t blocks,int64_t limit){return opaque_boundary_batch64_call(v,f,c,ox,oy,begin,batch,slots,blocks,limit,nullptr,edge);}

std::tuple<Tensor,std::vector<int64_t>> opaque_partition_prepare80(const Tensor &v,const Tensor &f,const Tensor &cuts,double ox,double oy,int64_t limit){
 ViewArena arena;Diagnostic d;auto ctx=context_for(v);uint64_t bytes=0,alignment=0;
 GFFX_CHECK(gffx_render_opaque_material_batch_workspace(dtype_of(v),&ctx,&bytes,&alignment,&d.buffer),d.text());
 if(limit<=0||limit>67108864||bytes+2560*17*8>(uint64_t)limit)raise_internal("partition storage budget");
 Tensor work=workspace_of(v,bytes),rows=torch::stable::new_empty(v,{2560,17});auto buffer=workspace_buffer(work,bytes);auto vv=arena.read(v),ff=arena.read(f),cc=arena.read(cuts),rr=arena.write(rows);int32_t result[2]={0};
 GFFX_CHECK(gffx_render_opaque_partition_prepare80(&vv,&ff,&cc,ox,oy,&ctx,&rr,result,&buffer,&d.buffer),d.text());
 return {rows,std::vector<int64_t>(result,result+2)};
}

std::tuple<Tensor,Tensor> opaque_partition_emit80(const Tensor &rows,int64_t count,int64_t begin){
 if(count<0||count>2560||begin<0||begin>count||begin%28)raise_internal("partition range");
 ViewArena arena;Diagnostic d;auto ctx=context_for(rows);Tensor ids=torch::stable::new_empty(rows,{1,800},ScalarType::Int),q=torch::stable::new_empty(rows,{1,800,4});auto rr=arena.read(rows),ii=arena.write(ids),qq=arena.write(q);
 Tensor work=workspace_of(rows,1048576);auto buffer=workspace_buffer(work,1048576);
 GFFX_CHECK(gffx_render_opaque_partition_emit_workspace80(&rr,(int32_t)count,(int32_t)begin,&ctx,&ii,&qq,&buffer,&d.buffer),d.text());return {ids,q};
}

std::tuple<Tensor,Tensor,std::vector<int64_t>,std::vector<int64_t>> opaque_material_prefix_backward80(const Tensor &v,const Tensor &f,const Tensor &c,const Tensor &g,double ox,double oy,int64_t begin,int64_t batch,int64_t slots,int64_t blocks,int64_t limit,int64_t derivatives,int64_t face,int64_t seen){
 if(face<0||face>512||seen<0||seen>2560)raise_internal("prefix range");int32_t prefix[2]={(int32_t)face,(int32_t)seen};
 auto result=opaque_batch_call(v,f,c,ox,oy,begin,batch,slots,blocks,limit,&g,4,derivatives,prefix);
 return {std::get<0>(result),std::get<1>(result),std::get<2>(result),std::vector<int64_t>(prefix,prefix+2)};
}

std::tuple<Tensor,Tensor,std::vector<int64_t>> opaque_boundary_batch80_call(const Tensor &v,const Tensor &f,const Tensor &cuts,double ox,double oy,int64_t begin,int64_t batch,int64_t slots,int64_t blocks,int64_t limit,const Tensor *g,int64_t edge){
 if(edge<0||edge>3||g)raise_internal("boundary stream edge/gradient");
 const int64_t rule=4,derivatives=0;int32_t *prefix=nullptr;
 const int64_t max_batches=rule==4?80:(rule==3?64:(rule==2?52:16));
 if(v.dim()!=2||v.size(1)!=3||f.dim()!=2||f.size(1)!=3||cuts.dim()!=3||cuts.size(0)!=f.size(0))raise_internal("opaque batch shapes");
 if(begin<0||begin>max_batches*32||batch<0||batch>=max_batches||slots<0||slots>max_batches*800||blocks<0||blocks>max_batches*32||limit<=0||limit>67108864)raise_internal("opaque batch cursor/budget");
 ViewArena arena;Diagnostic d;auto ctx=context_for(v);uint64_t bytes=0,alignment=0;GFFX_CHECK(gffx_render_opaque_material_batch_workspace(dtype_of(v),&ctx,&bytes,&alignment,&d.buffer),d.text());
 uint64_t element=dtype_of(v)==GFFX_DTYPE_FLOAT64?8:4;
 uint64_t output=g?(static_cast<uint64_t>(v.numel())+static_cast<uint64_t>(cuts.numel()))*element:800*(4*element+4);
 if(bytes>static_cast<uint64_t>(limit)||output>static_cast<uint64_t>(limit)-bytes)raise_internal("opaque batch workspace/output budget");
 Tensor work=workspace_of(v,bytes);auto buffer=workspace_buffer(work,bytes);
 Tensor a=g?torch::stable::new_empty(v,{v.size(0),3}):torch::stable::new_empty(f,{1,800});
 Tensor b=g?torch::stable::new_empty(cuts,{f.size(0),cuts.size(1),cuts.size(2)}):torch::stable::new_empty(v,{1,800,4});
 auto vv=arena.read(v),ff=arena.read(f),cc=arena.read(cuts),aa=arena.write(a),bb=arena.write(b);
 int32_t cursor[4]={(int32_t)begin,(int32_t)batch,(int32_t)slots,(int32_t)blocks},receipt[8]={0};
 if(g){auto gg=arena.read(*g);GFFX_CHECK(prefix?(rule==3?gffx_render_opaque_material_prefix_backward80:gffx_render_opaque_material_prefix_backward)(&vv,&ff,&cc,ox,oy,cursor,receipt,&gg,&ctx,&aa,&bb,&buffer,&d.buffer,(int32_t)derivatives,prefix):gffx_render_opaque_material_batch_rule_backward(&vv,&ff,&cc,ox,oy,cursor,receipt,&gg,&ctx,&aa,&bb,&buffer,&d.buffer,(int32_t)rule,(int32_t)derivatives),d.text());}
 else GFFX_CHECK(gffx_render_opaque_boundary_batch80(&vv,&ff,&cc,ox,oy,(int32_t)edge,cursor,receipt,&ctx,&aa,&bb,&buffer,&d.buffer),d.text());
 return {a,b,std::vector<int64_t>(receipt,receipt+8)};
}

std::tuple<Tensor,Tensor,std::vector<int64_t>> opaque_boundary_batch80(const Tensor &v,const Tensor &f,const Tensor &c,double ox,double oy,int64_t edge,int64_t begin,int64_t batch,int64_t slots,int64_t blocks,int64_t limit){return opaque_boundary_batch80_call(v,f,c,ox,oy,begin,batch,slots,blocks,limit,nullptr,edge);}
