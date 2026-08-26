#ifndef GFFX_STATUS_H
#define GFFX_STATUS_H

#include <gffx/abi.h>

typedef uint32_t gffx_status;

#define GFFX_STATUS_OK UINT32_C(0)
#define GFFX_STATUS_INVALID_ARGUMENT UINT32_C(1)
#define GFFX_STATUS_UNSUPPORTED UINT32_C(2)
#define GFFX_STATUS_INSUFFICIENT_WORKSPACE UINT32_C(3)
#define GFFX_STATUS_OVERFLOW UINT32_C(4)
#define GFFX_STATUS_BACKEND_FAILURE UINT32_C(5)
#define GFFX_STATUS_ABI_MISMATCH UINT32_C(6)
#define GFFX_STATUS_INTERNAL_ERROR UINT32_C(7)

typedef struct gffx_diagnostic_buffer {
    uint32_t struct_size;
    uint32_t abi_version;
    char *data;
    uint64_t capacity_bytes;
    uint64_t required_bytes;
    uint64_t reserved[4];
} gffx_diagnostic_buffer;

#define GFFX_DIAGNOSTIC_INIT \
    { (uint32_t)sizeof(gffx_diagnostic_buffer), GFFX_ABI_VERSION, 0, UINT64_C(0), UINT64_C(0), {0, 0, 0, 0} }

#endif /* GFFX_STATUS_H */
