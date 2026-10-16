std::tuple<Tensor,Tensor> shading_loss_curve(const Tensor &axis,const Tensor &controls,const Tensor &u){
 if(axis.dim()!=1||u.dim()!=1||axis.size(0)<1||axis.size(0)>512||u.size(0)>32)raise_internal("curve axis/u shape invalid");
 GffxCudaDeviceScope device_scope(u);ViewArena arena;Diagnostic diagnostic;auto context=context_for(u);uint64_t bytes=0,alignment=0;
 GFFX_CHECK(gffx_shading_loss_curve_workspace(u.size(0),axis.size(0),&context,&bytes,&alignment,&diagnostic.buffer),diagnostic.text());
 Tensor workspace=workspace_of(u,bytes);auto buffer=workspace_buffer(workspace,bytes);
 Tensor value=torch::stable::new_empty(u,{u.size(0),10*axis.size(0)-6});Tensor partial=torch::stable::new_empty(u,{u.size(0),10*axis.size(0)-6});
 auto xv=arena.read(axis),bv=arena.read(controls),uv=arena.read(u),yv=arena.write(value),dv=arena.write(partial);
 GFFX_CHECK(gffx_shading_loss_curve(&xv,&bv,&uv,&context,&yv,&dv,&buffer,&diagnostic.buffer),diagnostic.text());device_scope.close();return {value,partial};
}
Tensor shading_loss_curve_backward(const Tensor &partial,const Tensor &gradient){
 if(partial.dim()!=2||partial.size(1)<4||partial.size(1)>5114||(partial.size(1)+6)%10||partial.size(0)>32)raise_internal("curve partial shape invalid");
 GffxCudaDeviceScope device_scope(partial);ViewArena arena;Diagnostic diagnostic;auto context=context_for(partial);uint64_t bytes=0,alignment=0;
 GFFX_CHECK(gffx_shading_loss_curve_workspace(partial.size(0),(partial.size(1)+6)/10,&context,&bytes,&alignment,&diagnostic.buffer),diagnostic.text());
 Tensor workspace=workspace_of(partial,bytes);auto buffer=workspace_buffer(workspace,bytes);Tensor output=torch::stable::new_empty(partial,{partial.size(0)});
 auto pv=arena.read(partial),gv=arena.read(gradient),ov=arena.write(output);
 GFFX_CHECK(gffx_shading_loss_curve_backward(&pv,&gv,&context,&ov,&buffer,&diagnostic.buffer),diagnostic.text());device_scope.close();return output;
}

std::tuple<Tensor,Tensor> shading_loss_query(const Tensor &cn,const Tensor &logs,const Tensor &weights,const Tensor &curve,const Tensor &partial,const Tensor &cosine,const Tensor &owners){
 if(cn.dim()!=1||curve.dim()!=2||cosine.dim()!=1)raise_internal("query extent ranks invalid");
 GffxCudaDeviceScope device_scope(curve);ViewArena arena;Diagnostic diagnostic;auto context=context_for(curve);uint64_t bytes=0,alignment=0;
 GFFX_CHECK(gffx_shading_loss_query_workspace(curve.size(0),cn.size(0),cosine.size(0),&context,&bytes,&alignment,&diagnostic.buffer),diagnostic.text());
 Tensor workspace=workspace_of(curve,bytes);auto buffer=workspace_buffer(workspace,bytes);Tensor query=torch::stable::new_empty(curve,{cosine.size(0),6}),mean=torch::stable::new_empty(curve,{curve.size(0),4});
 auto xv=arena.read(cn),lv=arena.read(logs),wv=arena.read(weights),vv=arena.read(curve),pv=arena.read(partial),cv=arena.read(cosine),ov=arena.read(owners),yv=arena.write(query),mv=arena.write(mean);
 GFFX_CHECK(gffx_shading_loss_query(&xv,&lv,&wv,&vv,&pv,&cv,&ov,&context,&yv,&mv,&buffer,&diagnostic.buffer),diagnostic.text());device_scope.close();return {query,mean};
}
std::tuple<Tensor,Tensor> shading_loss_query_backward(const Tensor &query,const Tensor &mean,const Tensor &owners,const Tensor &gradient,const Tensor &mean_gradient){
 if(query.dim()!=2||mean.dim()!=2)raise_internal("query backward extent ranks invalid");
 GffxCudaDeviceScope device_scope(query);ViewArena arena;Diagnostic diagnostic;auto context=context_for(query);uint64_t bytes=0,alignment=0;
 GFFX_CHECK(gffx_shading_loss_query_workspace(mean.size(0),1,query.size(0),&context,&bytes,&alignment,&diagnostic.buffer),diagnostic.text());
 Tensor workspace=workspace_of(query,bytes);auto buffer=workspace_buffer(workspace,bytes);Tensor u=torch::stable::new_empty(query,{mean.size(0)}),cosine=torch::stable::new_empty(query,{query.size(0)});
 auto qv=arena.read(query),mv=arena.read(mean),ov=arena.read(owners),gv=arena.read(gradient),hv=arena.read(mean_gradient),uv=arena.write(u),cv=arena.write(cosine);
 GFFX_CHECK(gffx_shading_loss_query_backward(&qv,&mv,&ov,&gv,&hv,&context,&uv,&cv,&buffer,&diagnostic.buffer),diagnostic.text());device_scope.close();return {u,cosine};
}

std::tuple<Tensor,Tensor> shading_principled_pairs(const Tensor &cn,const Tensor &logs,const Tensor &curve,const Tensor &partial,const Tensor &mean,const Tensor &width,const Tensor &normal,const Tensor &view,const Tensor &light,const Tensor &albedo,const Tensor &roughness,const Tensor &f0,bool two_sided,bool derivatives){
 if(cn.dim()!=1||normal.dim()!=2||light.dim()!=3)raise_internal("Principled extent ranks invalid");
 GffxCudaDeviceScope device_scope(normal);ViewArena arena;Diagnostic diagnostic;auto context=context_for(normal);uint64_t bytes=0,alignment=0;
 GFFX_CHECK(gffx_shading_principled_pairs_workspace(normal.size(0),light.size(1),derivatives?1:0,&context,&bytes,&alignment,&diagnostic.buffer),diagnostic.text());
 Tensor workspace=workspace_of(normal,bytes);auto buffer=workspace_buffer(workspace,bytes);Tensor value=torch::stable::new_empty(normal,{normal.size(0),light.size(1),3});Tensor jacobian=derivatives?torch::stable::new_empty(normal,{normal.size(0),light.size(1),3,16}):torch::stable::new_empty(normal,{0});
 auto xv=arena.read(cn),lv=arena.read(logs),cv=arena.read(curve),pv=arena.read(partial),mv=arena.read(mean),wv=arena.read(width),nv=arena.read(normal),vv=arena.read(view),ll=arena.read(light),av=arena.read(albedo),rv=arena.read(roughness),fv=arena.read(f0),yv=arena.write(value),jv=arena.write(jacobian);
 GFFX_CHECK(gffx_shading_principled_pairs(&xv,&lv,&cv,&pv,&mv,&wv,&nv,&vv,&ll,&av,&rv,&fv,(two_sided?1:0)|(derivatives?2:0),&context,&yv,&jv,&buffer,&diagnostic.buffer),diagnostic.text());device_scope.close();return {value,jacobian};
}
std::tuple<Tensor,Tensor> shading_principled_pairs_backward(const Tensor &jacobian,const Tensor &gradient){
 if(jacobian.dim()!=4)raise_internal("Principled Jacobian rank invalid");GffxCudaDeviceScope device_scope(jacobian);ViewArena arena;Diagnostic diagnostic;auto context=context_for(jacobian);uint64_t bytes=0,alignment=0;
 GFFX_CHECK(gffx_shading_principled_pairs_workspace(jacobian.size(0),jacobian.size(1),1,&context,&bytes,&alignment,&diagnostic.buffer),diagnostic.text());Tensor workspace=workspace_of(jacobian,bytes);auto buffer=workspace_buffer(workspace,bytes);
 Tensor point=torch::stable::new_empty(jacobian,{jacobian.size(0),16}),light=torch::stable::new_empty(jacobian,{jacobian.size(0),jacobian.size(1),3});auto jv=arena.read(jacobian),gv=arena.read(gradient),pv=arena.write(point),lv=arena.write(light);
 GFFX_CHECK(gffx_shading_principled_pairs_backward(&jv,&gv,&context,&pv,&lv,&buffer,&diagnostic.buffer),diagnostic.text());device_scope.close();return {point,light};
}

/* Separate renderer-internal operation registration. */
std::tuple<Tensor,Tensor> shading_principled_groups(const Tensor &cn,const Tensor &logs,const Tensor &curve,const Tensor &partial,const Tensor &mean,const Tensor &width,const Tensor &normal,const Tensor &view,const Tensor &light,const Tensor &albedo,const Tensor &roughness,const Tensor &f0,bool two_sided,bool derivatives){
 if(cn.dim()!=1||normal.dim()!=2||light.dim()!=3)raise_internal("Principled extent ranks invalid");
 GffxCudaDeviceScope device_scope(normal);ViewArena arena;Diagnostic diagnostic;auto context=context_for(normal);uint64_t bytes=0,alignment=0;
 GFFX_CHECK(gffx_shading_principled_groups_workspace(normal.size(0),light.size(1),derivatives?1:0,&context,&bytes,&alignment,&diagnostic.buffer),diagnostic.text());
 Tensor workspace=workspace_of(normal,bytes);auto buffer=workspace_buffer(workspace,bytes);Tensor value=torch::stable::new_empty(normal,{normal.size(0),light.size(1),3});Tensor jacobian=derivatives?torch::stable::new_empty(normal,{normal.size(0),light.size(1),3,16}):torch::stable::new_empty(normal,{0});
 auto xv=arena.read(cn),lv=arena.read(logs),cv=arena.read(curve),pv=arena.read(partial),mv=arena.read(mean),wv=arena.read(width),nv=arena.read(normal),vv=arena.read(view),ll=arena.read(light),av=arena.read(albedo),rv=arena.read(roughness),fv=arena.read(f0),yv=arena.write(value),jv=arena.write(jacobian);
 GFFX_CHECK(gffx_shading_principled_groups(&xv,&lv,&cv,&pv,&mv,&wv,&nv,&vv,&ll,&av,&rv,&fv,(two_sided?1:0)|(derivatives?2:0),&context,&yv,&jv,&buffer,&diagnostic.buffer),diagnostic.text());device_scope.close();return {value,jacobian};
}
std::tuple<Tensor,Tensor> shading_principled_shared_groups(const Tensor &cn,const Tensor &logs,const Tensor &curve,const Tensor &partial,const Tensor &mean,const Tensor &width,const Tensor &normal,const Tensor &view,const Tensor &light,const Tensor &albedo,const Tensor &roughness,const Tensor &f0,bool two_sided,bool derivatives){
 if(cn.dim()!=1||normal.dim()!=2||light.dim()!=3)raise_internal("Principled extent ranks invalid");
 GffxCudaDeviceScope device_scope(normal);ViewArena arena;Diagnostic diagnostic;auto context=context_for(normal);uint64_t bytes=0,alignment=0;
 GFFX_CHECK(gffx_shading_principled_groups_workspace(normal.size(0),light.size(1),derivatives?1:0,&context,&bytes,&alignment,&diagnostic.buffer),diagnostic.text());
 Tensor workspace=workspace_of(normal,bytes);auto buffer=workspace_buffer(workspace,bytes);Tensor value=torch::stable::new_empty(normal,{normal.size(0),light.size(1),3});Tensor jacobian=derivatives?torch::stable::new_empty(normal,{normal.size(0),light.size(1),3,16}):torch::stable::new_empty(normal,{0});
 auto xv=arena.read(cn),lv=arena.read(logs),cv=arena.read(curve),pv=arena.read(partial),mv=arena.read(mean),wv=arena.read(width),nv=arena.read(normal),vv=arena.read(view),ll=arena.read(light),av=arena.read(albedo),rv=arena.read(roughness),fv=arena.read(f0),yv=arena.write(value),jv=arena.write(jacobian);
 GFFX_CHECK(gffx_shading_principled_shared_groups(&xv,&lv,&cv,&pv,&mv,&wv,&nv,&vv,&ll,&av,&rv,&fv,(two_sided?1:0)|(derivatives?2:0),&context,&yv,&jv,&buffer,&diagnostic.buffer),diagnostic.text());device_scope.close();return {value,jacobian};
}
std::tuple<Tensor,Tensor> shading_principled_groups_backward(const Tensor &jacobian,const Tensor &gradient){
 if(jacobian.dim()!=4)raise_internal("Principled Jacobian rank invalid");GffxCudaDeviceScope device_scope(jacobian);ViewArena arena;Diagnostic diagnostic;auto context=context_for(jacobian);uint64_t bytes=0,alignment=0;
 GFFX_CHECK(gffx_shading_principled_groups_workspace(jacobian.size(0),jacobian.size(1),1,&context,&bytes,&alignment,&diagnostic.buffer),diagnostic.text());Tensor workspace=workspace_of(jacobian,bytes);auto buffer=workspace_buffer(workspace,bytes);
 Tensor point=torch::stable::new_empty(jacobian,{jacobian.size(0),16}),light=torch::stable::new_empty(jacobian,{jacobian.size(0),jacobian.size(1),3});auto jv=arena.read(jacobian),gv=arena.read(gradient),pv=arena.write(point),lv=arena.write(light);
 GFFX_CHECK(gffx_shading_principled_groups_backward(&jv,&gv,&context,&pv,&lv,&buffer,&diagnostic.buffer),diagnostic.text());device_scope.close();return {point,light};
}

Tensor shading_ordered_rgba(const Tensor &carry,const Tensor &values){
 if(carry.dim()!=1||carry.size(0)!=4||values.dim()!=2||values.size(1)!=4)raise_internal("ordered RGBA shape invalid");
 GffxCudaDeviceScope device_scope(carry);ViewArena arena;Diagnostic diagnostic;auto context=context_for(carry);uint64_t bytes=0,alignment=0;
 GFFX_CHECK(gffx_shading_ordered_rgba_workspace(values.size(0),&context,&bytes,&alignment,&diagnostic.buffer),diagnostic.text());
 Tensor workspace=workspace_of(carry,bytes);auto buffer=workspace_buffer(workspace,bytes);Tensor output=torch::stable::new_empty(carry,{4});
 auto av=arena.read(carry),vv=arena.read(values),ov=arena.write(output);
 GFFX_CHECK(gffx_shading_ordered_rgba(&av,&vv,&context,&ov,&buffer,&diagnostic.buffer),diagnostic.text());device_scope.close();return output;
}

Tensor shading_unit_newton(const Tensor &input){
 if(input.dim()!=1)raise_internal("unit Newton shape invalid");
 GffxCudaDeviceScope device_scope(input);ViewArena arena;Diagnostic diagnostic;auto context=context_for(input);uint64_t bytes=0,alignment=0;
 GFFX_CHECK(gffx_shading_unit_newton_workspace(input.size(0),&context,&bytes,&alignment,&diagnostic.buffer),diagnostic.text());
 Tensor workspace=workspace_of(input,bytes);auto buffer=workspace_buffer(workspace,bytes);Tensor output=torch::stable::new_empty(input,{input.size(0)});
 auto xv=arena.read(input),yv=arena.write(output);
 GFFX_CHECK(gffx_shading_unit_newton(&xv,&context,&yv,&buffer,&diagnostic.buffer),diagnostic.text());device_scope.close();return output;
}

Tensor shading_unit_vectors(const Tensor &input){
 if(input.dim()!=2||input.size(1)!=3)raise_internal("unit vectors shape invalid");
 GffxCudaDeviceScope device_scope(input);ViewArena arena;Diagnostic diagnostic;auto context=context_for(input);uint64_t bytes=0,alignment=0;
 GFFX_CHECK(gffx_shading_unit_vectors_workspace(input.size(0),&context,&bytes,&alignment,&diagnostic.buffer),diagnostic.text());
 Tensor workspace=workspace_of(input,bytes);auto buffer=workspace_buffer(workspace,bytes);Tensor output=torch::stable::new_empty(input,{input.size(0),3});
 auto xv=arena.read(input),yv=arena.write(output);
 GFFX_CHECK(gffx_shading_unit_vectors(&xv,&context,&yv,&buffer,&diagnostic.buffer),diagnostic.text());device_scope.close();return output;
}

std::tuple<Tensor,Tensor> shading_delta_brdf(const Tensor &normal,const Tensor &brdf,const Tensor &direction,const Tensor &incoming,const Tensor &cotangent){
 if(brdf.dim()!=3||brdf.size(2)!=3)raise_internal("delta BRDF shape invalid");GffxCudaDeviceScope device_scope(brdf);ViewArena arena;Diagnostic diagnostic;auto context=context_for(brdf);uint64_t bytes=0,alignment=0;
 GFFX_CHECK(gffx_shading_delta_brdf_workspace(brdf.size(0),brdf.size(1),&context,&bytes,&alignment,&diagnostic.buffer),diagnostic.text());
 Tensor workspace=workspace_of(brdf,bytes);auto buffer=workspace_buffer(workspace,bytes);Tensor value=torch::stable::new_empty(brdf,{brdf.size(0),3}),gradient=torch::stable::new_empty(brdf,{brdf.size(0),brdf.size(1),3});
 auto nv=arena.read(normal),bv=arena.read(brdf),lv=arena.read(direction),iv=arena.read(incoming),gv=arena.read(cotangent),vv=arena.write(value),ov=arena.write(gradient);
 GFFX_CHECK(gffx_shading_delta_brdf(&nv,&bv,&lv,&iv,&gv,&context,&vv,&ov,&buffer,&diagnostic.buffer),diagnostic.text());device_scope.close();return {value,gradient};
}

std::tuple<Tensor,Tensor> shading_opaque_samples(const Tensor &owners,const Tensor &q,const Tensor &world,const Tensor &faces,const Tensor &clip,const Tensor &normals,const Tensor &camera,const Tensor &cn,const Tensor &logs,const Tensor &curve,const Tensor &partial,const Tensor &mean,const Tensor &width,const Tensor &albedo,const Tensor &roughness,const Tensor &f0,const Tensor &direction,const Tensor &incoming){
 if(q.dim()!=3||q.size(1)!=800||q.size(2)!=4)raise_internal("opaque sample layout invalid");GffxCudaDeviceScope device_scope(q);ViewArena arena;Diagnostic diagnostic;auto context=context_for(q);uint64_t bytes=0,alignment=0;
 GFFX_CHECK(gffx_shading_opaque_samples_workspace(q.size(0),&context,&bytes,&alignment,&diagnostic.buffer),diagnostic.text());
 Tensor workspace=workspace_of(q,bytes);auto buffer=workspace_buffer(workspace,bytes);Tensor weighted=torch::stable::new_empty(q,{q.size(0),800,4}),rgba=torch::stable::new_empty(q,{q.size(0),4});
 auto ownersv=arena.read(owners),qv=arena.read(q),worldv=arena.read(world),facesv=arena.read(faces),clipv=arena.read(clip),normalsv=arena.read(normals),camerav=arena.read(camera),cnv=arena.read(cn),logsv=arena.read(logs),curvev=arena.read(curve),partialv=arena.read(partial),meanv=arena.read(mean),widthv=arena.read(width),albedov=arena.read(albedo),roughnessv=arena.read(roughness),f0v=arena.read(f0),directionv=arena.read(direction),incomingv=arena.read(incoming),wv=arena.write(weighted),yv=arena.write(rgba);
 GFFX_CHECK(gffx_shading_opaque_samples(&ownersv,&qv,&worldv,&facesv,&clipv,&normalsv,&camerav,&cnv,&logsv,&curvev,&partialv,&meanv,&widthv,&albedov,&roughnessv,&f0v,&directionv,&incomingv,&context,&wv,&yv,&buffer,&diagnostic.buffer),diagnostic.text());device_scope.close();return {weighted,rgba};
}

std::tuple<Tensor,Tensor,Tensor> shading_opaque_albedo(const Tensor &owners,const Tensor &q,const Tensor &world,const Tensor &faces,const Tensor &clip,const Tensor &normals,const Tensor &camera,const Tensor &cn,const Tensor &logs,const Tensor &curve,const Tensor &partial,const Tensor &mean,const Tensor &width,const Tensor &albedo,const Tensor &roughness,const Tensor &f0,const Tensor &direction,const Tensor &incoming,const Tensor &cotangent){
 if(q.dim()!=3||q.size(1)!=800||q.size(2)!=4)raise_internal("opaque sample layout invalid");GffxCudaDeviceScope device_scope(q);ViewArena arena;Diagnostic diagnostic;auto context=context_for(q);uint64_t bytes=0,alignment=0;
 GFFX_CHECK(gffx_shading_opaque_albedo_workspace(q.size(0),&context,&bytes,&alignment,&diagnostic.buffer),diagnostic.text());
 Tensor workspace=workspace_of(q,bytes);auto buffer=workspace_buffer(workspace,bytes);Tensor weighted=torch::stable::new_empty(q,{q.size(0),800,4}),rgba=torch::stable::new_empty(q,{q.size(0),4}),gradient=torch::stable::new_empty(q,{q.size(0),3});
 auto ownersv=arena.read(owners),qv=arena.read(q),worldv=arena.read(world),facesv=arena.read(faces),clipv=arena.read(clip),normalsv=arena.read(normals),camerav=arena.read(camera),cnv=arena.read(cn),logsv=arena.read(logs),curvev=arena.read(curve),partialv=arena.read(partial),meanv=arena.read(mean),widthv=arena.read(width),albedov=arena.read(albedo),roughnessv=arena.read(roughness),f0v=arena.read(f0),directionv=arena.read(direction),incomingv=arena.read(incoming),cotangentv=arena.read(cotangent),wv=arena.write(weighted),yv=arena.write(rgba),gv=arena.write(gradient);
 GFFX_CHECK(gffx_shading_opaque_albedo(&ownersv,&qv,&worldv,&facesv,&clipv,&normalsv,&camerav,&cnv,&logsv,&curvev,&partialv,&meanv,&widthv,&albedov,&roughnessv,&f0v,&directionv,&incomingv,&cotangentv,&context,&wv,&yv,&gv,&buffer,&diagnostic.buffer),diagnostic.text());device_scope.close();return {weighted,rgba,gradient};
}
