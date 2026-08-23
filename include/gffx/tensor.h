#ifndef GFFX_TENSOR_H
#define GFFX_TENSOR_H

#include <gffx/status.h>

typedef uint32_t gffx_dtype;

#define GFFX_DTYPE_FLOAT32 UINT32_C(1)
#define GFFX_DTYPE_FLOAT64 UINT32_C(2)
#define GFFX_DTYPE_INT32 UINT32_C(3)
#define GFFX_DTYPE_UINT32 UINT32_C(4)
#define GFFX_DTYPE_BOOL UINT32_C(5)

#define GFFX_MAX_RANK UINT32_C(64)

#define GFFX_TENSOR_READ_ONLY UINT32_C(1)
#define GFFX_TENSOR_OUTPUT UINT32_C(2)

typedef struct gffx_tensor_view {
    uint32_t struct_size;
    uint32_t abi_version;
    void *data;
    uint64_t byte_offset;
    const int64_t *shape;
    const int64_t *strides;
    uint32_t rank;
    gffx_dtype dtype;
    gffx_device_type device_type;
    int32_t device_index;
    uint32_t flags;
    uint32_t reserved0;
    uint64_t reserved[4];
} gffx_tensor_view;

GFFX_EXTERN_C_BEGIN

GFFX_API gffx_status GFFX_CALL gffx_validate_tensor_view(
    const gffx_tensor_view *tensor,
    gffx_diagnostic_buffer *diagnostic
);

GFFX_EXTERN_C_END

#endif /* GFFX_TENSOR_H */
