#include <gffx/abi.h>
#include <gffx/capabilities.h>
#include <gffx/execution.h>
#include <gffx/status.h>
#include <gffx/tensor.h>

#include <stddef.h>
#include <stdint.h>

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (0)

_Static_assert(sizeof(void *) == 8, "ABI v1 requires 64-bit pointers");
_Static_assert(sizeof(gffx_diagnostic_buffer) == 64, "diagnostic layout changed");
_Static_assert(sizeof(gffx_tensor_view) == 96, "tensor layout changed");
_Static_assert(sizeof(gffx_execution_context) == 64, "execution layout changed");
_Static_assert(sizeof(gffx_buffer) == 64, "buffer layout changed");
_Static_assert(sizeof(gffx_capability_record) == 96, "capability record layout changed");
_Static_assert(sizeof(gffx_capability_report) == 112, "capability report layout changed");
_Static_assert(_Alignof(gffx_diagnostic_buffer) == 8, "diagnostic alignment changed");
_Static_assert(_Alignof(gffx_tensor_view) == 8, "tensor alignment changed");
_Static_assert(_Alignof(gffx_execution_context) == 8, "execution alignment changed");
_Static_assert(_Alignof(gffx_buffer) == 8, "buffer alignment changed");
_Static_assert(_Alignof(gffx_capability_record) == 8, "record alignment changed");
_Static_assert(_Alignof(gffx_capability_report) == 8, "report alignment changed");

_Static_assert(offsetof(gffx_tensor_view, struct_size) == 0, "prefix offset changed");
_Static_assert(offsetof(gffx_tensor_view, abi_version) == 4, "prefix offset changed");
_Static_assert(offsetof(gffx_tensor_view, data) == 8, "data offset changed");
_Static_assert(offsetof(gffx_tensor_view, shape) == 24, "shape offset changed");
_Static_assert(offsetof(gffx_tensor_view, strides) == 32, "stride offset changed");
_Static_assert(offsetof(gffx_capability_report, records) == 8, "record pointer offset changed");
_Static_assert(offsetof(gffx_capability_report, strings) == 40, "string pointer offset changed");

int main(void) {
    CHECK(GFFX_ABI_VERSION == GFFX_ABI_VERSION_ENCODE(1u, 0u));
    CHECK(GFFX_ABI_VERSION_MAJOR(GFFX_ABI_VERSION) == 1u);
    CHECK(GFFX_ABI_VERSION_MINOR(GFFX_ABI_VERSION) == 0u);
    CHECK(gffx_get_abi_version() == GFFX_ABI_VERSION);

    CHECK(GFFX_STATUS_OK == 0u);
    CHECK(GFFX_STATUS_INVALID_ARGUMENT == 1u);
    CHECK(GFFX_STATUS_UNSUPPORTED == 2u);
    CHECK(GFFX_STATUS_INSUFFICIENT_WORKSPACE == 3u);
    CHECK(GFFX_STATUS_OVERFLOW == 4u);
    CHECK(GFFX_STATUS_BACKEND_FAILURE == 5u);
    CHECK(GFFX_STATUS_ABI_MISMATCH == 6u);
    CHECK(GFFX_STATUS_INTERNAL_ERROR == 7u);

    CHECK(GFFX_DTYPE_FLOAT32 == 1u);
    CHECK(GFFX_DTYPE_FLOAT64 == 2u);
    CHECK(GFFX_DTYPE_INT32 == 3u);
    CHECK(GFFX_DTYPE_UINT32 == 4u);
    CHECK(GFFX_DTYPE_BOOL == 5u);
    CHECK(GFFX_DEVICE_CPU == 1u);
    CHECK(GFFX_DEVICE_CUDA == 2u);
    CHECK(GFFX_MAX_RANK == 64u);
    CHECK(GFFX_TENSOR_READ_ONLY == 1u);
    CHECK(GFFX_TENSOR_OUTPUT == 2u);
    CHECK(GFFX_EXECUTION_ALLOW_NONDETERMINISTIC == 1u);
    CHECK(GFFX_CAPABILITY_RECORD_SENSITIVE == 1u);
    CHECK(GFFX_CAPABILITY_RESULT_STATIC == 1u);
    CHECK(GFFX_CAPABILITY_RESULT_RUNTIME_PROBED == 2u);
    CHECK(GFFX_CAPABILITY_RESULT_OPTIONAL_PROVIDER_ABSENT == 4u);
    CHECK(GFFX_CAPABILITY_RESULT_PARTIAL_FAILURE == 8u);
    CHECK(GFFX_CAPABILITY_PROBE_FULL == 1u);
    CHECK(GFFX_CAPABILITY_PROBE_INCLUDE_SENSITIVE == 2u);
    return 0;
}
