/* Stable-ABI allocation/dispatch. Exact tensor validation stays in C. */
Tensor image_gaussian_weights_call(const Tensor &sigma,const Tensor *gradient,int64_t radius){
    if(radius<0||radius>32)raise_internal("Gaussian radius outside [0,32]");
    if(sigma.dim()>1)raise_internal("Gaussian sigma must be scalar or rank one");
    GffxCudaDeviceScope device_scope(sigma);ViewArena arena;Diagnostic diagnostic;
    auto context=context_for(sigma);uint64_t bytes=0,alignment=0;
    GFFX_CHECK(gffx_image_gaussian_weights_workspace(sigma.numel(),dtype_of(sigma),(uint32_t)radius,&context,&bytes,&alignment,&diagnostic.buffer),diagnostic.text());
    Tensor workspace=workspace_of(sigma,bytes);auto buffer=workspace_buffer(workspace,bytes);
    std::vector<int64_t> shape;
    if(sigma.dim()==1)shape.push_back(sigma.size(0));
    if(gradient==nullptr)shape.push_back(2*radius+1);
    Tensor output=torch::stable::new_empty(sigma,shape);auto iv=arena.read(sigma),ov=arena.write(output);
    if(gradient==nullptr){GFFX_CHECK(gffx_image_gaussian_weights(&iv,(uint32_t)radius,&context,&ov,&buffer,&diagnostic.buffer),diagnostic.text());}
    else{auto gv=arena.read(*gradient);GFFX_CHECK(gffx_image_gaussian_weights_backward(&iv,&gv,(uint32_t)radius,&context,&ov,&buffer,&diagnostic.buffer),diagnostic.text());}
    device_scope.close();return output;
}
Tensor image_gaussian_weights(const Tensor &sigma,int64_t radius){return image_gaussian_weights_call(sigma,nullptr,radius);}
Tensor image_gaussian_weights_backward(const Tensor &sigma,const Tensor &gradient,int64_t radius){return image_gaussian_weights_call(sigma,&gradient,radius);}
