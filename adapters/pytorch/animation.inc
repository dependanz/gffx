/* Included inside register.cpp's anonymous namespace after shared helpers. */
struct AnimationDeviceScope : GffxCudaDeviceScope {
    using GffxCudaDeviceScope::GffxCudaDeviceScope;
};

Tensor animation_gradient_buffer(const Tensor &reference,bool needed) {
    if(needed)return torch::stable::new_empty(reference,reference.sizes());
    return torch::stable::new_empty(reference,std::vector<int64_t>{0});
}
Tensor animation_blend_shapes(const Tensor &t, const Tensor &d, const Tensor &w) {
    AnimationDeviceScope device_scope(t);
    ViewArena arena;
    Diagnostic diagnostic;
    auto context = context_for(t);
    uint64_t bytes=0, alignment=0;
    GFFX_CHECK(gffx_mesh_blend_shapes_workspace(w.size(0),t.size(0),d.size(0),dtype_of(t),
               &context,&bytes,&alignment,&diagnostic.buffer),diagnostic.text());
    Tensor workspace=workspace_of(t,bytes);
    auto buffer=workspace_buffer(workspace,bytes);
    Tensor out=torch::stable::new_empty(t,std::vector<int64_t>{w.size(0),t.size(0),3});
    auto tv=arena.read(t),dv=arena.read(d),wv=arena.read(w),ov=arena.write(out);
    GFFX_CHECK(gffx_mesh_blend_shapes(&tv,&dv,&wv,&context,&ov,&buffer,&diagnostic.buffer),diagnostic.text());
    device_scope.close();return out;
}

std::tuple<Tensor,Tensor,Tensor> animation_blend_shapes_backward(
    const Tensor &t,const Tensor &d,const Tensor &w,const Tensor &g,bool need_t,bool need_d,bool need_w) {
    AnimationDeviceScope device_scope(t);
    ViewArena arena;
    Diagnostic diagnostic;
    auto context=context_for(t);
    uint64_t bytes=0,alignment=0;
    GFFX_CHECK(gffx_mesh_blend_shapes_workspace(w.size(0),t.size(0),d.size(0),dtype_of(t),
               &context,&bytes,&alignment,&diagnostic.buffer),diagnostic.text());
    Tensor workspace=workspace_of(t,bytes);
    auto buffer=workspace_buffer(workspace,bytes);
    Tensor gt=animation_gradient_buffer(t,need_t);
    Tensor gd=animation_gradient_buffer(d,need_d);
    Tensor gw=animation_gradient_buffer(w,need_w);
    auto tv=arena.read(t),dv=arena.read(d),wv=arena.read(w),gv=arena.read(g);
    auto gtv=arena.write(gt),gdv=arena.write(gd),gwv=arena.write(gw);
    GFFX_CHECK(gffx_mesh_blend_shapes_backward(&tv,&dv,&wv,&gv,&context,need_t?&gtv:nullptr,need_d?&gdv:nullptr,need_w?&gwv:nullptr,
               &buffer,&diagnostic.buffer),diagnostic.text());
    device_scope.close();return std::make_tuple(gt,gd,gw);
}

Tensor animation_rodrigues_call(const Tensor &r,const Tensor *gradient) {
    AnimationDeviceScope device_scope(r);
    ViewArena arena;Diagnostic diagnostic;auto context=context_for(r);
    uint64_t bytes=0,alignment=0;
    GFFX_CHECK(gffx_transforms_rodrigues_workspace(r.numel()/3,dtype_of(r),&context,&bytes,&alignment,&diagnostic.buffer),diagnostic.text());
    Tensor workspace=workspace_of(r,bytes);auto buffer=workspace_buffer(workspace,bytes);
    const auto sizes=r.sizes();std::vector<int64_t> shape(sizes.begin(),sizes.end());
    if(gradient==nullptr)shape.push_back(3);
    Tensor out=torch::stable::new_empty(r,shape);auto rv=arena.read(r),ov=arena.write(out);
    if(gradient==nullptr){GFFX_CHECK(gffx_transforms_rodrigues(&rv,&context,&ov,&buffer,&diagnostic.buffer),diagnostic.text());}
    else{auto gv=arena.read(*gradient);GFFX_CHECK(gffx_transforms_rodrigues_backward(&rv,&gv,&context,&ov,&buffer,&diagnostic.buffer),diagnostic.text());}
    device_scope.close();return out;
}
Tensor animation_rodrigues(const Tensor &r){return animation_rodrigues_call(r,nullptr);}
Tensor animation_rodrigues_backward(const Tensor &r,const Tensor &g){return animation_rodrigues_call(r,&g);}

Tensor animation_fk_call(const Tensor &local,const Tensor &parents,const Tensor *gradient) {
    AnimationDeviceScope device_scope(local);
    ViewArena arena;Diagnostic diagnostic;auto context=context_for(local);uint64_t bytes=0,alignment=0;
    GFFX_CHECK(gffx_skeleton_forward_kinematics_workspace(local.size(0),local.size(1),dtype_of(local),&context,&bytes,&alignment,&diagnostic.buffer),diagnostic.text());
    Tensor workspace=workspace_of(local,bytes);auto buffer=workspace_buffer(workspace,bytes);
    Tensor out=torch::stable::new_empty(local,local.sizes());
    auto lv=arena.read(local),pv=arena.read(parents),ov=arena.write(out);
    if(gradient==nullptr){GFFX_CHECK(gffx_skeleton_forward_kinematics(&lv,&pv,&context,&ov,&buffer,&diagnostic.buffer),diagnostic.text());}
    else{auto gv=arena.read(*gradient);GFFX_CHECK(gffx_skeleton_forward_kinematics_backward(&lv,&pv,&gv,&context,&ov,&buffer,&diagnostic.buffer),diagnostic.text());}
    device_scope.close();return out;
}
Tensor animation_forward_kinematics(const Tensor &l,const Tensor &p){return animation_fk_call(l,p,nullptr);}
Tensor animation_forward_kinematics_backward(const Tensor &l,const Tensor &p,const Tensor &g){return animation_fk_call(l,p,&g);}

Tensor animation_skin_lbs(const Tensor &v,const Tensor &t,const Tensor &w) {
    AnimationDeviceScope device_scope(v);
    ViewArena arena;Diagnostic diagnostic;auto context=context_for(v);uint64_t bytes=0,alignment=0;
    GFFX_CHECK(gffx_mesh_skin_lbs_workspace(v.size(0),v.size(1),t.size(1),dtype_of(v),&context,&bytes,&alignment,&diagnostic.buffer),diagnostic.text());
    Tensor workspace=workspace_of(v,bytes);auto buffer=workspace_buffer(workspace,bytes);
    Tensor out=torch::stable::new_empty(v,v.sizes());auto vv=arena.read(v),tv=arena.read(t),wv=arena.read(w),ov=arena.write(out);
    GFFX_CHECK(gffx_mesh_skin_lbs(&vv,&tv,&wv,&context,&ov,&buffer,&diagnostic.buffer),diagnostic.text());device_scope.close();return out;
}
std::tuple<Tensor,Tensor,Tensor> animation_skin_lbs_backward(const Tensor &v,const Tensor &t,const Tensor &w,const Tensor &g,bool need_v,bool need_t,bool need_w) {
    AnimationDeviceScope device_scope(v);
    ViewArena arena;Diagnostic diagnostic;auto context=context_for(v);uint64_t bytes=0,alignment=0;
    GFFX_CHECK(gffx_mesh_skin_lbs_workspace(v.size(0),v.size(1),t.size(1),dtype_of(v),&context,&bytes,&alignment,&diagnostic.buffer),diagnostic.text());
    Tensor workspace=workspace_of(v,bytes);auto buffer=workspace_buffer(workspace,bytes);
    Tensor gv=animation_gradient_buffer(v,need_v),gt=animation_gradient_buffer(t,need_t),gw=animation_gradient_buffer(w,need_w);
    auto vv=arena.read(v),tv=arena.read(t),wv=arena.read(w),gview=arena.read(g),gvv=arena.write(gv),gtv=arena.write(gt),gwv=arena.write(gw);
    GFFX_CHECK(gffx_mesh_skin_lbs_backward(&vv,&tv,&wv,&gview,&context,need_v?&gvv:nullptr,need_t?&gtv:nullptr,need_w?&gwv:nullptr,&buffer,&diagnostic.buffer),diagnostic.text());
    device_scope.close();return std::make_tuple(gv,gt,gw);
}

Tensor animation_evaluate_barycentric(const Tensor &v,const Tensor &f,const Tensor &idx,const Tensor &w) {
    AnimationDeviceScope device_scope(v);
    ViewArena arena;Diagnostic diagnostic;auto context=context_for(v);uint64_t bytes=0,alignment=0;
    GFFX_CHECK(gffx_mesh_evaluate_barycentric_workspace(v.size(0),v.size(1),idx.size(0),dtype_of(v),&context,&bytes,&alignment,&diagnostic.buffer),diagnostic.text());
    Tensor workspace=workspace_of(v,bytes);auto buffer=workspace_buffer(workspace,bytes);
    Tensor out=torch::stable::new_empty(v,std::vector<int64_t>{v.size(0),idx.size(0),3});
    auto vv=arena.read(v),fv=arena.read(f),iv=arena.read(idx),wv=arena.read(w),ov=arena.write(out);
    GFFX_CHECK(gffx_mesh_evaluate_barycentric(&vv,&fv,&iv,&wv,&context,&ov,&buffer,&diagnostic.buffer),diagnostic.text());device_scope.close();return out;
}
std::tuple<Tensor,Tensor> animation_evaluate_barycentric_backward(const Tensor &v,const Tensor &f,const Tensor &idx,const Tensor &w,const Tensor &g,bool need_v,bool need_w) {
    AnimationDeviceScope device_scope(v);
    ViewArena arena;Diagnostic diagnostic;auto context=context_for(v);uint64_t bytes=0,alignment=0;
    GFFX_CHECK(gffx_mesh_evaluate_barycentric_workspace(v.size(0),v.size(1),idx.size(0),dtype_of(v),&context,&bytes,&alignment,&diagnostic.buffer),diagnostic.text());
    Tensor workspace=workspace_of(v,bytes);auto buffer=workspace_buffer(workspace,bytes);
    Tensor gv=animation_gradient_buffer(v,need_v),gw=animation_gradient_buffer(w,need_w);
    auto vv=arena.read(v),fv=arena.read(f),iv=arena.read(idx),wv=arena.read(w),gview=arena.read(g),gvv=arena.write(gv),gwv=arena.write(gw);
    GFFX_CHECK(gffx_mesh_evaluate_barycentric_backward(&vv,&fv,&iv,&wv,&gview,&context,need_v?&gvv:nullptr,need_w?&gwv:nullptr,&buffer,&diagnostic.buffer),diagnostic.text());
    device_scope.close();return std::make_tuple(gv,gw);
}

int64_t animation_workspace_size(const Tensor &reference,int64_t operation,int64_t B,int64_t V,int64_t K) {
    AnimationDeviceScope device_scope(reference);
    Diagnostic diagnostic;auto context=context_for(reference);uint64_t bytes=0,alignment=0;gffx_status status;
    switch(operation) {
        case 1:status=gffx_mesh_blend_shapes_workspace(B,V,K,dtype_of(reference),&context,&bytes,&alignment,&diagnostic.buffer);break;
        case 2:status=gffx_transforms_rodrigues_workspace(V,dtype_of(reference),&context,&bytes,&alignment,&diagnostic.buffer);break;
        case 3:status=gffx_skeleton_forward_kinematics_workspace(B,V,dtype_of(reference),&context,&bytes,&alignment,&diagnostic.buffer);break;
        case 4:status=gffx_mesh_skin_lbs_workspace(B,V,K,dtype_of(reference),&context,&bytes,&alignment,&diagnostic.buffer);break;
        case 5:status=gffx_mesh_evaluate_barycentric_workspace(B,V,K,dtype_of(reference),&context,&bytes,&alignment,&diagnostic.buffer);break;
        default:raise_internal("unknown animation workspace operation");
    }
    GFFX_CHECK(status,diagnostic.text());
    if(bytes>static_cast<uint64_t>(INT64_MAX))raise_internal("animation workspace exceeds tensor extent");
    device_scope.close();return static_cast<int64_t>(bytes);
}
void animation_blend_shapes_out(const Tensor &t,const Tensor &delta,const Tensor &w,Tensor &out,Tensor &scratch) {
    AnimationDeviceScope device_scope(t);
    ViewArena arena;Diagnostic diagnostic;auto c=context_for(t);auto buffer=workspace_buffer(scratch,scratch.numel());
    auto tv=arena.read(t),dv=arena.read(delta),wv=arena.read(w),ov=arena.write(out);
    GFFX_CHECK(gffx_mesh_blend_shapes(&tv,&dv,&wv,&c,&ov,&buffer,&diagnostic.buffer),diagnostic.text());

    device_scope.close();
}
void animation_rodrigues_out(const Tensor &r,Tensor &out,Tensor &scratch) {
    AnimationDeviceScope device_scope(r);
    ViewArena arena;Diagnostic diagnostic;auto c=context_for(r);auto buffer=workspace_buffer(scratch,scratch.numel());
    auto rv=arena.read(r),ov=arena.write(out);
    GFFX_CHECK(gffx_transforms_rodrigues(&rv,&c,&ov,&buffer,&diagnostic.buffer),diagnostic.text());

    device_scope.close();
}
void animation_forward_kinematics_out(const Tensor &local,const Tensor &parents,Tensor &out,Tensor &scratch) {
    AnimationDeviceScope device_scope(local);
    ViewArena arena;Diagnostic diagnostic;auto c=context_for(local);auto buffer=workspace_buffer(scratch,scratch.numel());
    auto lv=arena.read(local),pv=arena.read(parents),ov=arena.write(out);
    GFFX_CHECK(gffx_skeleton_forward_kinematics(&lv,&pv,&c,&ov,&buffer,&diagnostic.buffer),diagnostic.text());

    device_scope.close();
}
void animation_skin_lbs_out(const Tensor &v,const Tensor &t,const Tensor &w,Tensor &out,Tensor &scratch) {
    AnimationDeviceScope device_scope(v);
    ViewArena arena;Diagnostic diagnostic;auto c=context_for(v);auto buffer=workspace_buffer(scratch,scratch.numel());
    auto vv=arena.read(v),tv=arena.read(t),wv=arena.read(w),ov=arena.write(out);
    GFFX_CHECK(gffx_mesh_skin_lbs(&vv,&tv,&wv,&c,&ov,&buffer,&diagnostic.buffer),diagnostic.text());

    device_scope.close();
}
void animation_evaluate_barycentric_out(const Tensor &v,const Tensor &f,const Tensor &idx,const Tensor &w,Tensor &out,Tensor &scratch) {
    AnimationDeviceScope device_scope(v);
    ViewArena arena;Diagnostic diagnostic;auto c=context_for(v);auto buffer=workspace_buffer(scratch,scratch.numel());
    auto vv=arena.read(v),fv=arena.read(f),iv=arena.read(idx),wv=arena.read(w),ov=arena.write(out);
    GFFX_CHECK(gffx_mesh_evaluate_barycentric(&vv,&fv,&iv,&wv,&c,&ov,&buffer,&diagnostic.buffer),diagnostic.text());

    device_scope.close();
}
