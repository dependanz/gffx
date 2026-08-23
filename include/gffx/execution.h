#ifndef GFFX_EXECUTION_H
#define GFFX_EXECUTION_H

#include <gffx/status.h>

#define GFFX_EXECUTION_ALLOW_NONDETERMINISTIC UINT32_C(1)

typedef struct gffx_execution_context {
    uint32_t struct_size;
    uint32_t abi_version;
    gffx_device_type device_type;
    int32_t device_index;
    void *stream;
    uint32_t flags;
    uint32_t reserved0;
    uint64_t reserved[4];
} gffx_execution_context;

typedef struct gffx_buffer {
    uint32_t struct_size;
    uint32_t abi_version;
    void *data;
    uint64_t capacity_bytes;
    uint64_t alignment;
    gffx_device_type device_type;
    int32_t device_index;
    uint32_t flags;
    uint32_t reserved0;
    uint64_t reserved[2];
} gffx_buffer;

GFFX_EXTERN_C_BEGIN

GFFX_API gffx_status GFFX_CALL gffx_validate_execution_context(
    const gffx_execution_context *context,
    gffx_diagnostic_buffer *diagnostic
);

GFFX_API gffx_status GFFX_CALL gffx_validate_buffer(
    const gffx_buffer *buffer,
    gffx_diagnostic_buffer *diagnostic
);

GFFX_EXTERN_C_END

#endif /* GFFX_EXECUTION_H */
