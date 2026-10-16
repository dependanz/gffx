/* No GPU needed: additive transfer pointers never read outside an older table. */
#include "plugin_api.h"
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#define CHECK(x) do{if(!(x)){fprintf(stderr,"image tail line %d\n",__LINE__);return 1;}}while(0)
static gffx_status GFFX_CALL forward(const gffx_tensor_view *x,uint32_t mode,const gffx_execution_context *c,
    gffx_tensor_view *y,const gffx_buffer *w,gffx_diagnostic_buffer *d){
    (void)x;(void)mode;(void)c;(void)y;(void)w;(void)d;return GFFX_STATUS_OK;
}
static gffx_status GFFX_CALL backward(const gffx_tensor_view *x,const gffx_tensor_view *g,uint32_t mode,
    const gffx_execution_context *c,gffx_tensor_view *y,const gffx_buffer *w,gffx_diagnostic_buffer *d){
    (void)g;return forward(x,mode,c,y,w,d);
}
static gffx_status GFFX_CALL pc_forward(const gffx_tensor_view *x,const gffx_tensor_view *b,const gffx_tensor_view *u,const gffx_execution_context *c,gffx_tensor_view *y,gffx_tensor_view *dy,const gffx_buffer *w,gffx_diagnostic_buffer *d){
 (void)x;(void)b;(void)u;(void)c;(void)y;(void)dy;(void)w;(void)d;return GFFX_STATUS_OK;
}
static gffx_status GFFX_CALL pc_backward(const gffx_tensor_view *p,const gffx_tensor_view *g,const gffx_execution_context *c,gffx_tensor_view *y,const gffx_buffer *w,gffx_diagnostic_buffer *d){
 return pc_forward(p,g,NULL,c,y,NULL,w,d);
}
static gffx_status GFFX_CALL pq_forward(const gffx_tensor_view *a,const gffx_tensor_view *b,const gffx_tensor_view *c,const gffx_tensor_view *e,const gffx_tensor_view *f,const gffx_tensor_view *g,const gffx_tensor_view *h,const gffx_execution_context *ctx,gffx_tensor_view *y,gffx_tensor_view *dy,const gffx_buffer *w,gffx_diagnostic_buffer *d){return pc_forward(a,b,c,ctx,y,dy,w,d);}
static gffx_status GFFX_CALL pq_backward(const gffx_tensor_view *a,const gffx_tensor_view *b,const gffx_tensor_view *c,const gffx_tensor_view *e,const gffx_tensor_view *f,const gffx_execution_context *ctx,gffx_tensor_view *y,gffx_tensor_view *dy,const gffx_buffer *w,gffx_diagnostic_buffer *d){return pc_forward(a,b,c,ctx,y,dy,w,d);}
static gffx_status GFFX_CALL pp(const gffx_tensor_view *cn,const gffx_tensor_view *logs,const gffx_tensor_view *curve,const gffx_tensor_view *partial,const gffx_tensor_view *mean,const gffx_tensor_view *width,const gffx_tensor_view *normal,const gffx_tensor_view *view,const gffx_tensor_view *light,const gffx_tensor_view *albedo,const gffx_tensor_view *roughness,const gffx_tensor_view *f0,uint32_t options,const gffx_execution_context *context,gffx_tensor_view *value,gffx_tensor_view *jacobian,const gffx_buffer *workspace,gffx_diagnostic_buffer *diagnostic){return GFFX_STATUS_OK;}
static gffx_status GFFX_CALL pp_backward(const gffx_tensor_view *jacobian,const gffx_tensor_view *gradient,const gffx_execution_context *context,gffx_tensor_view *point_gradient,gffx_tensor_view *light_gradient,const gffx_buffer *workspace,gffx_diagnostic_buffer *diagnostic){return GFFX_STATUS_OK;}
static gffx_status GFFX_CALL rw(const gffx_tensor_view *a,const gffx_tensor_view *b,const gffx_tensor_view *c,const gffx_tensor_view *d,int64_t gh,int64_t gw,int64_t r,int64_t col,int64_t h,int64_t w,int64_t k,double blur,uint32_t cull,double eps,const gffx_execution_context *ctx,gffx_tensor_view *fi,gffx_tensor_view *bc,gffx_tensor_view *depth,gffx_tensor_view *distance,const gffx_buffer *ws,gffx_diagnostic_buffer *diag){return GFFX_STATUS_OK;}
int main(void){
    gffx_cuda_operations table;size_t before=offsetof(gffx_cuda_operations,mesh_evaluate_barycentric_backward)+sizeof(table.mesh_evaluate_barycentric_backward);
    memset(&table,0,sizeof(table));table.image_transfer=forward;table.image_transfer_backward=backward;
    CHECK(offsetof(gffx_cuda_operations,image_transfer)==before);
    CHECK(sizeof(gffx_cuda_plugin_api)==88);
    table.struct_size=(uint32_t)before;
    CHECK(!GFFX_CUDA_HAS_OPERATION(&table,image_transfer)&&!GFFX_CUDA_HAS_OPERATION(&table,image_transfer_backward));
    table.struct_size=(uint32_t)(offsetof(gffx_cuda_operations,image_transfer)+sizeof(table.image_transfer)-1);
    CHECK(!GFFX_CUDA_HAS_OPERATION(&table,image_transfer));
    ++table.struct_size;CHECK(GFFX_CUDA_HAS_OPERATION(&table,image_transfer));
    CHECK(!GFFX_CUDA_HAS_OPERATION(&table,image_transfer_backward));
    table.struct_size=sizeof(table);CHECK(GFFX_CUDA_HAS_OPERATION(&table,image_transfer_backward));
    table.image_transfer_backward=NULL;CHECK(!GFFX_CUDA_HAS_OPERATION(&table,image_transfer_backward));
    CHECK(!GFFX_CUDA_HAS_OPERATION((const gffx_cuda_operations *)NULL,image_transfer));
    {
        size_t old_end=offsetof(gffx_cuda_operations,image_transfer_backward)+sizeof(table.image_transfer_backward);
        CHECK(offsetof(gffx_cuda_operations,image_gaussian_weights)==old_end);
        table.image_gaussian_weights=forward;table.image_gaussian_weights_backward=backward;
        table.struct_size=(uint32_t)old_end;
        CHECK(!GFFX_CUDA_HAS_OPERATION(&table,image_gaussian_weights)&&!GFFX_CUDA_HAS_OPERATION(&table,image_gaussian_weights_backward));
        table.struct_size=(uint32_t)(offsetof(gffx_cuda_operations,image_gaussian_weights)+sizeof(table.image_gaussian_weights)-1);
        CHECK(!GFFX_CUDA_HAS_OPERATION(&table,image_gaussian_weights));
        ++table.struct_size;CHECK(GFFX_CUDA_HAS_OPERATION(&table,image_gaussian_weights));
        CHECK(!GFFX_CUDA_HAS_OPERATION(&table,image_gaussian_weights_backward));
        table.struct_size=sizeof(table);CHECK(GFFX_CUDA_HAS_OPERATION(&table,image_gaussian_weights_backward));
        table.image_gaussian_weights_backward=NULL;CHECK(!GFFX_CUDA_HAS_OPERATION(&table,image_gaussian_weights_backward));
    }
    {
        size_t old_end=offsetof(gffx_cuda_operations,image_gaussian_weights_backward)+sizeof(table.image_gaussian_weights_backward);
        CHECK(offsetof(gffx_cuda_operations,shading_loss_curve)==old_end);
        table.shading_loss_curve=pc_forward;table.shading_loss_curve_backward=pc_backward;
        table.struct_size=(uint32_t)old_end;
        CHECK(!GFFX_CUDA_HAS_OPERATION(&table,shading_loss_curve));
        CHECK(!GFFX_CUDA_HAS_OPERATION(&table,shading_loss_curve_backward));
        table.struct_size=(uint32_t)(offsetof(gffx_cuda_operations,shading_loss_curve)+sizeof(table.shading_loss_curve)-1);
        CHECK(!GFFX_CUDA_HAS_OPERATION(&table,shading_loss_curve));
        ++table.struct_size;CHECK(GFFX_CUDA_HAS_OPERATION(&table,shading_loss_curve));
        CHECK(!GFFX_CUDA_HAS_OPERATION(&table,shading_loss_curve_backward));
        table.struct_size=sizeof(table);CHECK(GFFX_CUDA_HAS_OPERATION(&table,shading_loss_curve_backward));
        table.shading_loss_curve_backward=NULL;CHECK(!GFFX_CUDA_HAS_OPERATION(&table,shading_loss_curve_backward));
    }
    {
        size_t old_end=offsetof(gffx_cuda_operations,shading_loss_curve_backward)+sizeof(table.shading_loss_curve_backward);
        CHECK(offsetof(gffx_cuda_operations,shading_loss_query)==old_end);
        table.shading_loss_query=pq_forward;table.shading_loss_query_backward=pq_backward;
        table.struct_size=(uint32_t)old_end;CHECK(!GFFX_CUDA_HAS_OPERATION(&table,shading_loss_query));CHECK(!GFFX_CUDA_HAS_OPERATION(&table,shading_loss_query_backward));
        table.struct_size=(uint32_t)(offsetof(gffx_cuda_operations,shading_loss_query)+sizeof(table.shading_loss_query)-1);CHECK(!GFFX_CUDA_HAS_OPERATION(&table,shading_loss_query));
        ++table.struct_size;CHECK(GFFX_CUDA_HAS_OPERATION(&table,shading_loss_query));CHECK(!GFFX_CUDA_HAS_OPERATION(&table,shading_loss_query_backward));
        table.struct_size=sizeof(table);CHECK(GFFX_CUDA_HAS_OPERATION(&table,shading_loss_query_backward));table.shading_loss_query_backward=NULL;CHECK(!GFFX_CUDA_HAS_OPERATION(&table,shading_loss_query_backward));
    }
    {
        size_t old_end=offsetof(gffx_cuda_operations,shading_loss_query_backward)+sizeof(table.shading_loss_query_backward);
        CHECK(offsetof(gffx_cuda_operations,shading_principled_pairs)==old_end);
        table.shading_principled_pairs=pp;table.shading_principled_pairs_backward=pp_backward;
        table.struct_size=(uint32_t)old_end;CHECK(!GFFX_CUDA_HAS_OPERATION(&table,shading_principled_pairs));CHECK(!GFFX_CUDA_HAS_OPERATION(&table,shading_principled_pairs_backward));
        table.struct_size=(uint32_t)(offsetof(gffx_cuda_operations,shading_principled_pairs)+sizeof(table.shading_principled_pairs)-1);CHECK(!GFFX_CUDA_HAS_OPERATION(&table,shading_principled_pairs));
        ++table.struct_size;CHECK(GFFX_CUDA_HAS_OPERATION(&table,shading_principled_pairs));CHECK(!GFFX_CUDA_HAS_OPERATION(&table,shading_principled_pairs_backward));
        table.struct_size=sizeof(table);CHECK(GFFX_CUDA_HAS_OPERATION(&table,shading_principled_pairs_backward));table.shading_principled_pairs_backward=NULL;CHECK(!GFFX_CUDA_HAS_OPERATION(&table,shading_principled_pairs_backward));
    }
    {
        size_t old_end=offsetof(gffx_cuda_operations,shading_principled_pairs_backward)+sizeof(table.shading_principled_pairs_backward);
        CHECK(offsetof(gffx_cuda_operations,shading_principled_groups)==old_end);
        table.shading_principled_groups=pp;table.shading_principled_groups_backward=pp_backward;
        table.struct_size=(uint32_t)old_end;CHECK(!GFFX_CUDA_HAS_OPERATION(&table,shading_principled_groups));CHECK(!GFFX_CUDA_HAS_OPERATION(&table,shading_principled_groups_backward));
        table.struct_size=(uint32_t)(offsetof(gffx_cuda_operations,shading_principled_groups)+sizeof(table.shading_principled_groups)-1);CHECK(!GFFX_CUDA_HAS_OPERATION(&table,shading_principled_groups));
        ++table.struct_size;CHECK(GFFX_CUDA_HAS_OPERATION(&table,shading_principled_groups));CHECK(!GFFX_CUDA_HAS_OPERATION(&table,shading_principled_groups_backward));
        table.struct_size=sizeof(table);CHECK(GFFX_CUDA_HAS_OPERATION(&table,shading_principled_groups_backward));table.shading_principled_groups_backward=NULL;CHECK(!GFFX_CUDA_HAS_OPERATION(&table,shading_principled_groups_backward));
    }
    {
        size_t old_end=offsetof(gffx_cuda_operations,shading_principled_groups_backward)+sizeof(table.shading_principled_groups_backward);
        CHECK(offsetof(gffx_cuda_operations,render_rasterize_window)==old_end);
        table.render_rasterize_window=rw;table.struct_size=(uint32_t)old_end;
        CHECK(!GFFX_CUDA_HAS_OPERATION(&table,render_rasterize_window));
        table.struct_size=(uint32_t)(old_end+sizeof(table.render_rasterize_window)-1);
        CHECK(!GFFX_CUDA_HAS_OPERATION(&table,render_rasterize_window));
        ++table.struct_size;CHECK(GFFX_CUDA_HAS_OPERATION(&table,render_rasterize_window));
        table.render_rasterize_window=NULL;CHECK(!GFFX_CUDA_HAS_OPERATION(&table,render_rasterize_window));
        CHECK(!GFFX_CUDA_HAS_OPERATION((const gffx_cuda_operations *)NULL,render_rasterize_window));
    }
    return 0;
}
