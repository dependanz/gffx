/* Compile/run without CUDA: append-only operation-table compatibility. */
#include "plugin_api.h"
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "animation tail line %d\n", __LINE__); return 1; } } while (0)
static gffx_status GFFX_CALL sentinel(
    const gffx_tensor_view *a,const gffx_tensor_view *b,const gffx_tensor_view *c,
    const gffx_execution_context *context,gffx_tensor_view *out,
    const gffx_buffer *workspace,gffx_diagnostic_buffer *diagnostic) {
    (void)a;(void)b;(void)c;(void)context;(void)out;(void)workspace;(void)diagnostic;
    return GFFX_STATUS_OK;
}
int main(void) {
    gffx_cuda_operations table;
    size_t legacy_size = offsetof(gffx_cuda_operations,reserved)+sizeof(table.reserved);
    memset(&table,0,sizeof(table));
    CHECK(offsetof(gffx_cuda_operations,mesh_blend_shapes)==legacy_size);
    CHECK(sizeof(gffx_cuda_plugin_api)==88);
    table.struct_size=(uint32_t)legacy_size;
    table.mesh_blend_shapes=sentinel;
    CHECK(!GFFX_CUDA_HAS_OPERATION(&table,mesh_blend_shapes));
    table.struct_size=(uint32_t)(offsetof(gffx_cuda_operations,mesh_blend_shapes)+sizeof(table.mesh_blend_shapes)-1);
    CHECK(!GFFX_CUDA_HAS_OPERATION(&table,mesh_blend_shapes));
    ++table.struct_size;
    CHECK(GFFX_CUDA_HAS_OPERATION(&table,mesh_blend_shapes));
    CHECK(!GFFX_CUDA_HAS_OPERATION(&table,mesh_evaluate_barycentric_backward));
    table.struct_size=sizeof(table);table.mesh_blend_shapes=NULL;
    CHECK(!GFFX_CUDA_HAS_OPERATION(&table,mesh_blend_shapes));
    CHECK(!GFFX_CUDA_HAS_OPERATION((const gffx_cuda_operations *)NULL,mesh_blend_shapes));
    return 0;
}
