#include <gffx/abi.h>
#include <gffx/capabilities.h>
#include <gffx/execution.h>
#include <gffx/status.h>
#include <gffx/tensor.h>

#include <type_traits>

static_assert(std::is_standard_layout<gffx_tensor_view>::value, "tensor view must be standard layout");
static_assert(std::is_standard_layout<gffx_capability_report>::value, "report must be standard layout");

int main() {
    gffx_execution_context context{};
    context.struct_size = static_cast<uint32_t>(sizeof(context));
    context.abi_version = GFFX_ABI_VERSION;
    context.device_type = GFFX_DEVICE_CPU;
    return gffx_get_abi_version() == GFFX_ABI_VERSION ? 0 : 1;
}
