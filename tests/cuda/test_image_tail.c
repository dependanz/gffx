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
    return 0;
}
