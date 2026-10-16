#include "plugin_api.h"

#include <cstddef>
#include <cstring>
#include <type_traits>

static_assert(GFFX_CUDA_PLUGIN_ABI_VERSION == GFFX_ABI_VERSION_ENCODE(1, 0),
              "plugin ABI version changed");
static_assert(sizeof(gffx_cuda_plugin_api) == 88, "plugin API layout changed");
static_assert(alignof(gffx_cuda_plugin_api) == 8, "plugin API alignment changed");
static_assert(offsetof(gffx_cuda_plugin_api, struct_size) == 0, "prefix moved");
static_assert(offsetof(gffx_cuda_plugin_api, plugin_abi_version) == 4, "prefix moved");
static_assert(offsetof(gffx_cuda_plugin_api, core_abi_min) == 8, "core range moved");
static_assert(offsetof(gffx_cuda_plugin_api, core_abi_max) == 12, "core range moved");
static_assert(offsetof(gffx_cuda_plugin_api, flags) == 16, "flags moved");
static_assert(offsetof(gffx_cuda_plugin_api, build_identity) == 24, "identity moved");
static_assert(offsetof(gffx_cuda_plugin_api, capabilities_probe) == 32,
              "provider entry moved");
static_assert(std::is_standard_layout<gffx_cuda_plugin_api>::value,
              "plugin API must remain standard layout");
static_assert(GFFX_CUDA_PLUGIN_FLAG_CAPABILITY_PROVIDER == UINT64_C(1), "flag changed");
static_assert(GFFX_CAPABILITY_KEY_CUDA_PLUGIN_PATH == UINT32_C(15), "key changed");
static_assert(GFFX_CAPABILITY_KEY_CUDA_PLUGIN_BUILD_ID == UINT32_C(16), "key changed");
static_assert(GFFX_CAPABILITY_KEY_CUDA_PLUGIN_ABI_VERSION == UINT32_C(17), "key changed");
static_assert(GFFX_CAPABILITY_KEY_CUDA_PLUGIN_COMPATIBLE == UINT32_C(18), "key changed");
static_assert(GFFX_CAPABILITY_KEY_CUDA_DRIVER_STATUS == UINT32_C(19), "key changed");
static_assert(GFFX_CAPABILITY_KEY_CUDA_DRIVER_VERSION == UINT32_C(20), "key changed");
static_assert(GFFX_CAPABILITY_KEY_CUDA_DEVICE_COUNT == UINT32_C(21), "key changed");

static gffx_status GFFX_CALL opaque_stub(const gffx_tensor_view*,const gffx_tensor_view*,const gffx_tensor_view*,int64_t,int64_t,double,double,const gffx_tensor_view*,const gffx_execution_context*,gffx_tensor_view*,gffx_tensor_view*,const gffx_buffer*,gffx_diagnostic_buffer*) { return GFFX_STATUS_OK; }
static gffx_status GFFX_CALL boundary_stub(const gffx_tensor_view*,const gffx_tensor_view*,const gffx_tensor_view*,int64_t,int64_t,double,double,int32_t,const gffx_execution_context*,gffx_tensor_view*,gffx_tensor_view*,const gffx_buffer*,gffx_diagnostic_buffer*) { return GFFX_STATUS_OK; }
int main() {
    gffx_cuda_operations ops = {}; const gffx_cuda_operations *table=&ops;
    ops.render_opaque_quadrature=opaque_stub;
    ops.struct_size=offsetof(gffx_cuda_operations,render_opaque_quadrature);
    if(GFFX_CUDA_HAS_OPERATION(table,render_opaque_quadrature))return __LINE__;
    ops.struct_size=offsetof(gffx_cuda_operations,render_opaque_quadrature)+sizeof(ops.render_opaque_quadrature)-1;
    if(GFFX_CUDA_HAS_OPERATION(table,render_opaque_quadrature))return __LINE__;
    ops.struct_size=sizeof(ops);
    if(!GFFX_CUDA_HAS_OPERATION(table,render_opaque_quadrature))return __LINE__;
    ops.render_opaque_quadrature=nullptr;
    if(GFFX_CUDA_HAS_OPERATION(table,render_opaque_quadrature))return __LINE__;
    ops.render_opaque_boundary=boundary_stub;
    ops.struct_size=offsetof(gffx_cuda_operations,render_opaque_boundary);
    if(GFFX_CUDA_HAS_OPERATION(table,render_opaque_boundary))return __LINE__;
    ops.struct_size=sizeof(ops)-1;
    if(GFFX_CUDA_HAS_OPERATION(table,render_opaque_boundary))return __LINE__;
    ops.struct_size=sizeof(ops);
    if(!GFFX_CUDA_HAS_OPERATION(table,render_opaque_boundary))return __LINE__;
    ops.render_opaque_boundary=nullptr;
    if(GFFX_CUDA_HAS_OPERATION(table,render_opaque_boundary))return __LINE__;
    table=nullptr;
    if(GFFX_CUDA_HAS_OPERATION(table,render_opaque_quadrature))return __LINE__;

    if (std::strcmp(GFFX_CUDA_PLUGIN_HANDSHAKE_SYMBOL,
                    "gffx_cuda_plugin_handshake_v1") != 0) return __LINE__;
    return 0;
}
