#include "internal.h"

#include <string.h>

int gffx_internal_reserved_u64_is_zero(const uint64_t *values, size_t count) {
    size_t index;
    for (index = 0u; index < count; ++index) {
        if (values[index] != UINT64_C(0)) return 0;
    }
    return 1;
}

gffx_status gffx_internal_prepare_diagnostic(gffx_diagnostic_buffer *diagnostic) {
    if (diagnostic == NULL) return GFFX_STATUS_OK;
    if (diagnostic->struct_size < (uint32_t)sizeof(gffx_diagnostic_buffer)) {
        return GFFX_STATUS_ABI_MISMATCH;
    }
    if (GFFX_ABI_VERSION_MAJOR(diagnostic->abi_version) !=
        GFFX_ABI_VERSION_MAJOR(GFFX_ABI_VERSION)) {
        return GFFX_STATUS_ABI_MISMATCH;
    }
    if (!gffx_internal_reserved_u64_is_zero(diagnostic->reserved, 4u)) {
        return GFFX_STATUS_INVALID_ARGUMENT;
    }
    diagnostic->required_bytes = UINT64_C(0);
    if (diagnostic->data != NULL && diagnostic->capacity_bytes != UINT64_C(0)) {
        diagnostic->data[0] = '\0';
    }
    return GFFX_STATUS_OK;
}

gffx_status gffx_internal_fail(
    gffx_diagnostic_buffer *diagnostic,
    gffx_status status,
    const char *message
) {
    size_t message_size = strlen(message);
    uint64_t required = (uint64_t)message_size + UINT64_C(1);
    if (diagnostic != NULL) {
        diagnostic->required_bytes = required;
        if (diagnostic->data != NULL && diagnostic->capacity_bytes != UINT64_C(0)) {
            uint64_t writable = diagnostic->capacity_bytes - UINT64_C(1);
            size_t copy_size = message_size;
            if ((uint64_t)copy_size > writable) copy_size = (size_t)writable;
            if (copy_size != 0u) memcpy(diagnostic->data, message, copy_size);
            diagnostic->data[copy_size] = '\0';
        }
    }
    return status;
}

gffx_status gffx_internal_validate_header(
    const void *structure,
    uint32_t struct_size,
    uint32_t abi_version,
    uint32_t minimum_size,
    const char *structure_name,
    gffx_diagnostic_buffer *diagnostic
) {
    if (structure == NULL) {
        return gffx_internal_fail(diagnostic, GFFX_STATUS_INVALID_ARGUMENT, structure_name);
    }
    if (struct_size < minimum_size) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_ABI_MISMATCH,
            "structure is smaller than the ABI v1 minimum"
        );
    }
    if (GFFX_ABI_VERSION_MAJOR(abi_version) != GFFX_ABI_VERSION_MAJOR(GFFX_ABI_VERSION)) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_ABI_MISMATCH,
            "structure uses an incompatible ABI major version"
        );
    }
    return GFFX_STATUS_OK;
}
