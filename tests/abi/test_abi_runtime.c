#include <gffx/execution.h>
#include <gffx/status.h>
#include <gffx/tensor.h>

#include <limits.h>
#include <stdint.h>
#include <string.h>

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (0)

static gffx_diagnostic_buffer make_diagnostic(char *data, uint64_t capacity) {
    gffx_diagnostic_buffer diagnostic = {0};
    diagnostic.struct_size = (uint32_t)sizeof(diagnostic);
    diagnostic.abi_version = GFFX_ABI_VERSION;
    diagnostic.data = data;
    diagnostic.capacity_bytes = capacity;
    return diagnostic;
}

static gffx_tensor_view make_tensor(void *data, const int64_t *shape, const int64_t *strides) {
    gffx_tensor_view tensor = {0};
    tensor.struct_size = (uint32_t)sizeof(tensor);
    tensor.abi_version = GFFX_ABI_VERSION;
    tensor.data = data;
    tensor.rank = 2u;
    tensor.shape = shape;
    tensor.strides = strides;
    tensor.dtype = GFFX_DTYPE_FLOAT32;
    tensor.device_type = GFFX_DEVICE_CPU;
    tensor.device_index = 0;
    tensor.flags = GFFX_TENSOR_READ_ONLY;
    return tensor;
}

static int test_diagnostic_and_tensor_validation(void) {
    int64_t shape[2] = {2, 3};
    int64_t strides[2] = {3, 1};
    float values[6] = {0};
    char message[128] = {0};
    gffx_diagnostic_buffer diagnostic = make_diagnostic(message, sizeof(message));
    gffx_tensor_view tensor = make_tensor(values, shape, strides);

    CHECK(gffx_validate_tensor_view(&tensor, &diagnostic) == GFFX_STATUS_OK);
    CHECK(diagnostic.required_bytes == 0u);
    CHECK(message[0] == '\0');

    tensor.strides = (int64_t[2]){1, 2};
    CHECK(gffx_validate_tensor_view(&tensor, &diagnostic) == GFFX_STATUS_UNSUPPORTED);
    CHECK(diagnostic.required_bytes > 1u);
    CHECK(message[0] != '\0');

    tensor = make_tensor(values, shape, strides);
    tensor.shape = (int64_t[2]){-1, 3};
    CHECK(gffx_validate_tensor_view(&tensor, &diagnostic) == GFFX_STATUS_INVALID_ARGUMENT);

    tensor = make_tensor(values, shape, strides);
    tensor.dtype = 99u;
    CHECK(gffx_validate_tensor_view(&tensor, &diagnostic) == GFFX_STATUS_UNSUPPORTED);

    tensor = make_tensor(NULL, shape, strides);
    CHECK(gffx_validate_tensor_view(&tensor, &diagnostic) == GFFX_STATUS_INVALID_ARGUMENT);

    tensor = make_tensor(NULL, (int64_t[2]){0, 3}, strides);
    CHECK(gffx_validate_tensor_view(&tensor, &diagnostic) == GFFX_STATUS_OK);

    tensor = make_tensor(values, (int64_t[2]){INT64_MAX, 2}, strides);
    CHECK(gffx_validate_tensor_view(&tensor, &diagnostic) == GFFX_STATUS_OVERFLOW);

    tensor = make_tensor(values, shape, strides);
    tensor.byte_offset = UINT64_MAX - 3u;
    CHECK(gffx_validate_tensor_view(&tensor, &diagnostic) == GFFX_STATUS_OVERFLOW);

    tensor = make_tensor((void *)(uintptr_t)(UINTPTR_MAX - 3u), shape, strides);
    CHECK(gffx_validate_tensor_view(&tensor, &diagnostic) == GFFX_STATUS_OVERFLOW);

    tensor = make_tensor(values, shape, strides);
    tensor.flags = GFFX_TENSOR_READ_ONLY | GFFX_TENSOR_OUTPUT;
    CHECK(gffx_validate_tensor_view(&tensor, &diagnostic) == GFFX_STATUS_INVALID_ARGUMENT);

    tensor = make_tensor(values, shape, strides);
    tensor.reserved[0] = 1u;
    CHECK(gffx_validate_tensor_view(&tensor, &diagnostic) == GFFX_STATUS_INVALID_ARGUMENT);

    tensor = make_tensor(values, shape, strides);
    tensor.struct_size -= 8u;
    CHECK(gffx_validate_tensor_view(&tensor, &diagnostic) == GFFX_STATUS_ABI_MISMATCH);

    {
        struct extended_tensor {
            gffx_tensor_view base;
            uint64_t future[2];
        } extended = {0};
        extended.base = make_tensor(values, shape, strides);
        extended.base.struct_size = (uint32_t)sizeof(extended);
        CHECK(gffx_validate_tensor_view(&extended.base, &diagnostic) == GFFX_STATUS_OK);
    }

    tensor = make_tensor(values, shape, strides);
    tensor.abi_version = GFFX_ABI_VERSION_ENCODE(2u, 0u);
    CHECK(gffx_validate_tensor_view(&tensor, &diagnostic) == GFFX_STATUS_ABI_MISMATCH);
    return 0;
}

static int test_diagnostic_bounds(void) {
    struct guarded_message {
        char text[5];
        unsigned char canary;
    } guarded = {{0}, 0xA5u};
    gffx_diagnostic_buffer diagnostic = make_diagnostic(guarded.text, sizeof(guarded.text));

    CHECK(gffx_validate_tensor_view(NULL, &diagnostic) == GFFX_STATUS_INVALID_ARGUMENT);
    CHECK(diagnostic.required_bytes > sizeof(guarded.text));
    CHECK(guarded.text[sizeof(guarded.text) - 1u] == '\0');
    CHECK(guarded.canary == 0xA5u);

    diagnostic = make_diagnostic(NULL, 0u);
    CHECK(gffx_validate_tensor_view(NULL, &diagnostic) == GFFX_STATUS_INVALID_ARGUMENT);
    CHECK(diagnostic.required_bytes > 0u);

    diagnostic = make_diagnostic(NULL, 2u);
    CHECK(gffx_validate_tensor_view(NULL, &diagnostic) == GFFX_STATUS_INVALID_ARGUMENT);

    diagnostic = make_diagnostic(guarded.text, sizeof(guarded.text));
    diagnostic.abi_version = GFFX_ABI_VERSION_ENCODE(2u, 0u);
    CHECK(gffx_validate_tensor_view(NULL, &diagnostic) == GFFX_STATUS_ABI_MISMATCH);

    {
        char exact[128] = {0};
        uint64_t exact_size;
        diagnostic = make_diagnostic(NULL, 0u);
        CHECK(gffx_validate_tensor_view(NULL, &diagnostic) == GFFX_STATUS_INVALID_ARGUMENT);
        exact_size = diagnostic.required_bytes;
        CHECK(exact_size > 1u && exact_size <= sizeof(exact));
        diagnostic = make_diagnostic(exact, exact_size);
        CHECK(gffx_validate_tensor_view(NULL, &diagnostic) == GFFX_STATUS_INVALID_ARGUMENT);
        CHECK(diagnostic.required_bytes == exact_size);
        CHECK(exact[exact_size - 1u] == '\0');
        CHECK(strlen(exact) + 1u == exact_size);
    }
    return 0;
}

static int test_execution_and_buffer_validation(void) {
    char message[128] = {0};
    gffx_diagnostic_buffer diagnostic = make_diagnostic(message, sizeof(message));
    gffx_execution_context context = {0};
    unsigned char storage[64] = {0};
    gffx_buffer buffer = {0};

    context.struct_size = (uint32_t)sizeof(context);
    context.abi_version = GFFX_ABI_VERSION;
    context.device_type = GFFX_DEVICE_CPU;
    context.device_index = 0;
    CHECK(gffx_validate_execution_context(&context, &diagnostic) == GFFX_STATUS_OK);

    context.stream = storage;
    CHECK(gffx_validate_execution_context(&context, &diagnostic) == GFFX_STATUS_INVALID_ARGUMENT);
    context.stream = NULL;
    context.flags = 0x80000000u;
    CHECK(gffx_validate_execution_context(&context, &diagnostic) == GFFX_STATUS_UNSUPPORTED);
    context.flags = 0u;
    context.device_type = GFFX_DEVICE_CUDA;
    CHECK(gffx_validate_execution_context(&context, &diagnostic) == GFFX_STATUS_OK);

    buffer.struct_size = (uint32_t)sizeof(buffer);
    buffer.abi_version = GFFX_ABI_VERSION;
    buffer.data = storage;
    buffer.capacity_bytes = sizeof(storage);
    buffer.alignment = 1u;
    buffer.device_type = GFFX_DEVICE_CPU;
    buffer.device_index = 0;
    CHECK(gffx_validate_buffer(&buffer, &diagnostic) == GFFX_STATUS_OK);

    buffer.alignment = 3u;
    CHECK(gffx_validate_buffer(&buffer, &diagnostic) == GFFX_STATUS_INVALID_ARGUMENT);
    buffer.alignment = 1u;
    buffer.data = NULL;
    CHECK(gffx_validate_buffer(&buffer, &diagnostic) == GFFX_STATUS_INVALID_ARGUMENT);
    buffer.capacity_bytes = 0u;
    CHECK(gffx_validate_buffer(&buffer, &diagnostic) == GFFX_STATUS_OK);
    return 0;
}

int main(void) {
    int result = test_diagnostic_and_tensor_validation();
    if (result != 0) return result;
    result = test_diagnostic_bounds();
    if (result != 0) return result;
    return test_execution_and_buffer_validation();
}
