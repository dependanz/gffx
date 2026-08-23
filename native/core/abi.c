#include <gffx/execution.h>
#include <gffx/tensor.h>

#include "internal.h"

#include <limits.h>
#include <stdint.h>

static int gffx_device_is_supported(gffx_device_type device_type) {
    return device_type == GFFX_DEVICE_CPU || device_type == GFFX_DEVICE_CUDA;
}

static gffx_status gffx_validate_device(
    gffx_device_type device_type,
    int32_t device_index,
    gffx_diagnostic_buffer *diagnostic
) {
    if (!gffx_device_is_supported(device_type)) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_UNSUPPORTED,
            "device type is not supported by ABI v1"
        );
    }
    if (device_index < 0) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "device index must be nonnegative"
        );
    }
    if (device_type == GFFX_DEVICE_CPU && device_index != 0) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "the ABI v1 CPU device index must be zero"
        );
    }
    return GFFX_STATUS_OK;
}

static uint64_t gffx_dtype_size(gffx_dtype dtype) {
    switch (dtype) {
        case GFFX_DTYPE_FLOAT32:
        case GFFX_DTYPE_INT32:
        case GFFX_DTYPE_UINT32:
            return UINT64_C(4);
        case GFFX_DTYPE_FLOAT64:
            return UINT64_C(8);
        case GFFX_DTYPE_BOOL:
            return UINT64_C(1);
        default:
            return UINT64_C(0);
    }
}

GFFX_API uint32_t GFFX_CALL gffx_get_abi_version(void) {
    return GFFX_ABI_VERSION;
}

GFFX_API gffx_status GFFX_CALL gffx_validate_tensor_view(
    const gffx_tensor_view *tensor,
    gffx_diagnostic_buffer *diagnostic
) {
    gffx_status status = gffx_internal_prepare_diagnostic(diagnostic);
    uint64_t item_size;
    uint64_t element_count = UINT64_C(1);
    uint64_t expected_stride = UINT64_C(1);
    uint64_t total_bytes;
    uint32_t index;
    int is_empty = 0;

    if (status != GFFX_STATUS_OK) return status;
    if (tensor == NULL) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "tensor view pointer is null"
        );
    }
    status = gffx_internal_validate_header(
        tensor,
        tensor->struct_size,
        tensor->abi_version,
        (uint32_t)sizeof(gffx_tensor_view),
        "tensor view pointer is null",
        diagnostic
    );
    if (status != GFFX_STATUS_OK) return status;
    if (tensor->reserved0 != UINT32_C(0) ||
        !gffx_internal_reserved_u64_is_zero(tensor->reserved, 4u)) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "tensor view reserved fields must be zero"
        );
    }
    if ((tensor->flags & ~(GFFX_TENSOR_READ_ONLY | GFFX_TENSOR_OUTPUT)) != UINT32_C(0)) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_UNSUPPORTED,
            "tensor view contains an unsupported flag"
        );
    }
    if ((tensor->flags & GFFX_TENSOR_READ_ONLY) != UINT32_C(0) &&
        (tensor->flags & GFFX_TENSOR_OUTPUT) != UINT32_C(0)) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "tensor view cannot be both read-only and an output"
        );
    }
    item_size = gffx_dtype_size(tensor->dtype);
    if (item_size == UINT64_C(0)) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_UNSUPPORTED,
            "tensor dtype is not supported by ABI v1"
        );
    }
    status = gffx_validate_device(tensor->device_type, tensor->device_index, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (tensor->rank > GFFX_MAX_RANK) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_UNSUPPORTED,
            "tensor rank exceeds the ABI v1 maximum"
        );
    }
    if (tensor->rank == UINT32_C(0)) {
        if (tensor->shape != NULL || tensor->strides != NULL) {
            return gffx_internal_fail(
                diagnostic,
                GFFX_STATUS_INVALID_ARGUMENT,
                "rank-zero tensors require null shape and stride pointers"
            );
        }
    } else {
        if (tensor->shape == NULL || tensor->strides == NULL) {
            return gffx_internal_fail(
                diagnostic,
                GFFX_STATUS_INVALID_ARGUMENT,
                "positive-rank tensors require shape and stride arrays"
            );
        }
        for (index = 0u; index < tensor->rank; ++index) {
            if (tensor->shape[index] < INT64_C(0)) {
                return gffx_internal_fail(
                    diagnostic,
                    GFFX_STATUS_INVALID_ARGUMENT,
                    "tensor extents must be nonnegative"
                );
            }
            if (tensor->shape[index] == INT64_C(0)) is_empty = 1;
        }
        if (is_empty) {
            element_count = UINT64_C(0);
        } else {
            for (index = 0u; index < tensor->rank; ++index) {
                uint64_t extent = (uint64_t)tensor->shape[index];
                if (extent != UINT64_C(0) && element_count > UINT64_MAX / extent) {
                    return gffx_internal_fail(
                        diagnostic,
                        GFFX_STATUS_OVERFLOW,
                        "tensor element count overflows 64-bit capacity"
                    );
                }
                element_count *= extent;
            }
        }
        if (element_count != UINT64_C(0) && element_count > UINT64_MAX / item_size) {
            return gffx_internal_fail(
                diagnostic,
                GFFX_STATUS_OVERFLOW,
                "tensor byte count overflows 64-bit capacity"
            );
        }
        for (index = tensor->rank; index > 0u; --index) {
            uint64_t extent = (uint64_t)tensor->shape[index - 1u];
            int64_t stride = tensor->strides[index - 1u];
            if (stride <= INT64_C(0)) {
                return gffx_internal_fail(
                    diagnostic,
                    GFFX_STATUS_INVALID_ARGUMENT,
                    "tensor element strides must be positive"
                );
            }
            if (extent > UINT64_C(1)) {
                if (expected_stride > (uint64_t)INT64_MAX) {
                    return gffx_internal_fail(
                        diagnostic,
                        GFFX_STATUS_OVERFLOW,
                        "dense tensor stride exceeds signed 64-bit capacity"
                    );
                }
                if ((uint64_t)stride != expected_stride) {
                    return gffx_internal_fail(
                        diagnostic,
                        GFFX_STATUS_UNSUPPORTED,
                        "ABI v1 accepts only dense C-contiguous tensor views"
                    );
                }
            }
            if (extent != UINT64_C(0) && expected_stride > UINT64_MAX / extent) {
                return gffx_internal_fail(
                    diagnostic,
                    GFFX_STATUS_OVERFLOW,
                    "dense tensor stride arithmetic overflowed"
                );
            }
            expected_stride *= extent;
        }
    }
    if (element_count != UINT64_C(0) && element_count > UINT64_MAX / item_size) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_OVERFLOW,
            "tensor byte count overflows 64-bit capacity"
        );
    }
    total_bytes = element_count * item_size;
    if (tensor->byte_offset > UINT64_MAX - total_bytes) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_OVERFLOW,
            "tensor byte offset and extent overflow 64-bit capacity"
        );
    }
    if ((tensor->byte_offset % item_size) != UINT64_C(0)) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "tensor byte offset is not aligned to its dtype"
        );
    }
    if (element_count != UINT64_C(0) && tensor->data == NULL) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "nonempty tensor view has a null data pointer"
        );
    }
    if (tensor->data == NULL) {
        if (tensor->byte_offset != UINT64_C(0)) {
            return gffx_internal_fail(
                diagnostic,
                GFFX_STATUS_INVALID_ARGUMENT,
                "a null tensor data pointer requires a zero byte offset"
            );
        }
    } else {
        uintptr_t address = (uintptr_t)tensor->data;
        if (tensor->byte_offset > (uint64_t)(UINTPTR_MAX - address)) {
            return gffx_internal_fail(
                diagnostic,
                GFFX_STATUS_OVERFLOW,
                "tensor data pointer plus byte offset overflows"
            );
        }
        address += (uintptr_t)tensor->byte_offset;
        if (total_bytes != UINT64_C(0) &&
            total_bytes - UINT64_C(1) > (uint64_t)(UINTPTR_MAX - address)) {
            return gffx_internal_fail(
                diagnostic,
                GFFX_STATUS_OVERFLOW,
                "tensor addressable byte range overflows"
            );
        }
    }
    return GFFX_STATUS_OK;
}

GFFX_API gffx_status GFFX_CALL gffx_validate_execution_context(
    const gffx_execution_context *context,
    gffx_diagnostic_buffer *diagnostic
) {
    gffx_status status = gffx_internal_prepare_diagnostic(diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (context == NULL) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "execution context pointer is null"
        );
    }
    status = gffx_internal_validate_header(
        context,
        context->struct_size,
        context->abi_version,
        (uint32_t)sizeof(gffx_execution_context),
        "execution context pointer is null",
        diagnostic
    );
    if (status != GFFX_STATUS_OK) return status;
    if (context->reserved0 != UINT32_C(0) ||
        !gffx_internal_reserved_u64_is_zero(context->reserved, 4u)) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "execution context reserved fields must be zero"
        );
    }
    if ((context->flags & ~GFFX_EXECUTION_ALLOW_NONDETERMINISTIC) != UINT32_C(0)) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_UNSUPPORTED,
            "execution context contains an unsupported flag"
        );
    }
    status = gffx_validate_device(context->device_type, context->device_index, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (context->device_type == GFFX_DEVICE_CPU && context->stream != NULL) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "CPU execution does not accept a stream handle"
        );
    }
    return GFFX_STATUS_OK;
}

GFFX_API gffx_status GFFX_CALL gffx_validate_buffer(
    const gffx_buffer *buffer,
    gffx_diagnostic_buffer *diagnostic
) {
    gffx_status status = gffx_internal_prepare_diagnostic(diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (buffer == NULL) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "buffer descriptor pointer is null"
        );
    }
    status = gffx_internal_validate_header(
        buffer,
        buffer->struct_size,
        buffer->abi_version,
        (uint32_t)sizeof(gffx_buffer),
        "buffer descriptor pointer is null",
        diagnostic
    );
    if (status != GFFX_STATUS_OK) return status;
    if (buffer->reserved0 != UINT32_C(0) ||
        !gffx_internal_reserved_u64_is_zero(buffer->reserved, 2u)) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "buffer descriptor reserved fields must be zero"
        );
    }
    if (buffer->flags != UINT32_C(0)) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_UNSUPPORTED,
            "buffer descriptor contains an unsupported flag"
        );
    }
    status = gffx_validate_device(buffer->device_type, buffer->device_index, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (buffer->alignment == UINT64_C(0) ||
        (buffer->alignment & (buffer->alignment - UINT64_C(1))) != UINT64_C(0)) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "buffer alignment must be a nonzero power of two"
        );
    }
    if (buffer->capacity_bytes != UINT64_C(0) && buffer->data == NULL) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "a buffer with nonzero capacity requires a data pointer"
        );
    }
    if (buffer->data != NULL &&
        (((uintptr_t)buffer->data & (uintptr_t)(buffer->alignment - UINT64_C(1))) != 0u)) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "buffer data pointer does not satisfy the declared alignment"
        );
    }
    return GFFX_STATUS_OK;
}
