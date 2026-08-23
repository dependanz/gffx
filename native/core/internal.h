#ifndef GFFX_CORE_INTERNAL_H
#define GFFX_CORE_INTERNAL_H

#include <gffx/status.h>

#include <stddef.h>

gffx_status gffx_internal_prepare_diagnostic(gffx_diagnostic_buffer *diagnostic);

gffx_status gffx_internal_fail(
    gffx_diagnostic_buffer *diagnostic,
    gffx_status status,
    const char *message
);

gffx_status gffx_internal_validate_header(
    const void *structure,
    uint32_t struct_size,
    uint32_t abi_version,
    uint32_t minimum_size,
    const char *structure_name,
    gffx_diagnostic_buffer *diagnostic
);

int gffx_internal_reserved_u64_is_zero(const uint64_t *values, size_t count);

#endif /* GFFX_CORE_INTERNAL_H */
