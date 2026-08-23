#include "self_test.h"

#include "internal.h"

#include <stdint.h>
#include <string.h>

/*
 * Private calling-path self-test. See self_test.h: no public API, correctness, or performance
 * claim is attached to anything in this file. Every buffer below is automatic storage owned by
 * this function; the core allocates nothing and retains nothing across the call.
 */

#define GFFX_SELF_TEST_MAX_RECORDS 64u
#define GFFX_SELF_TEST_MAX_STRING_BYTES 2048u

static gffx_diagnostic_buffer gffx_self_test_diagnostic(char *data, uint64_t capacity) {
    gffx_diagnostic_buffer diagnostic = {0};
    diagnostic.struct_size = (uint32_t)sizeof(diagnostic);
    diagnostic.abi_version = GFFX_ABI_VERSION;
    diagnostic.data = data;
    diagnostic.capacity_bytes = capacity;
    return diagnostic;
}

static gffx_capability_report gffx_self_test_report(void) {
    gffx_capability_report report = {0};
    report.struct_size = (uint32_t)sizeof(report);
    report.abi_version = GFFX_ABI_VERSION;
    return report;
}

static uint32_t gffx_self_test_tensor_paths(void) {
    int64_t shape[2] = {2, 3};
    int64_t strides[2] = {3, 1};
    float values[6] = {0};
    char message[128] = {0};
    gffx_diagnostic_buffer diagnostic = gffx_self_test_diagnostic(message, sizeof(message));
    gffx_tensor_view tensor = {0};
    uint32_t checks = UINT32_C(0);

    tensor.struct_size = (uint32_t)sizeof(tensor);
    tensor.abi_version = GFFX_ABI_VERSION;
    tensor.data = values;
    tensor.rank = 2u;
    tensor.shape = shape;
    tensor.strides = strides;
    tensor.dtype = GFFX_DTYPE_FLOAT32;
    tensor.device_type = GFFX_DEVICE_CPU;
    tensor.device_index = 0;
    tensor.flags = GFFX_TENSOR_READ_ONLY;

    if (gffx_validate_tensor_view(&tensor, &diagnostic) == GFFX_STATUS_OK &&
        diagnostic.required_bytes == UINT64_C(0) &&
        message[0] == '\0') {
        checks |= GFFX_SELF_TEST_TENSOR_ACCEPT;
    }

    /* A rank above the ABI v1 maximum must be refused through the same exported entry point. */
    tensor.rank = GFFX_MAX_RANK + 1u;
    if (gffx_validate_tensor_view(&tensor, &diagnostic) == GFFX_STATUS_UNSUPPORTED &&
        diagnostic.required_bytes > UINT64_C(1) &&
        message[0] != '\0') {
        checks |= GFFX_SELF_TEST_TENSOR_REJECT;
    }

    return checks;
}

static uint32_t gffx_self_test_execution_paths(void) {
    unsigned char workspace[64] = {0};
    char message[128] = {0};
    gffx_diagnostic_buffer diagnostic = gffx_self_test_diagnostic(message, sizeof(message));
    gffx_execution_context context = {0};
    gffx_buffer buffer = {0};
    uint32_t checks = UINT32_C(0);

    context.struct_size = (uint32_t)sizeof(context);
    context.abi_version = GFFX_ABI_VERSION;
    context.device_type = GFFX_DEVICE_CPU;
    context.device_index = 0;
    context.stream = NULL;
    context.flags = UINT32_C(0);
    if (gffx_validate_execution_context(&context, &diagnostic) == GFFX_STATUS_OK) {
        checks |= GFFX_SELF_TEST_EXECUTION_ACCEPT;
    }

    buffer.struct_size = (uint32_t)sizeof(buffer);
    buffer.abi_version = GFFX_ABI_VERSION;
    buffer.data = workspace;
    buffer.capacity_bytes = (uint64_t)sizeof(workspace);
    buffer.alignment = UINT64_C(8);
    buffer.device_type = GFFX_DEVICE_CPU;
    buffer.device_index = 0;
    buffer.flags = UINT32_C(0);
    if (gffx_validate_buffer(&buffer, &diagnostic) == GFFX_STATUS_OK) {
        checks |= GFFX_SELF_TEST_BUFFER_ACCEPT;
    }

    return checks;
}

static uint32_t gffx_self_test_diagnostic_paths(void) {
    gffx_tensor_view tensor = {0};
    char narrow[8];
    gffx_diagnostic_buffer truncating;
    uint32_t checks = UINT32_C(0);

    /* An invalid tensor with no diagnostic buffer must still report through the status code. */
    tensor.struct_size = (uint32_t)sizeof(tensor);
    tensor.abi_version = GFFX_ABI_VERSION;
    tensor.dtype = UINT32_C(0);
    tensor.device_type = GFFX_DEVICE_CPU;
    if (gffx_validate_tensor_view(&tensor, NULL) == GFFX_STATUS_UNSUPPORTED) {
        checks |= GFFX_SELF_TEST_DIAGNOSTIC_NULL;
    }

    memset(narrow, 'x', sizeof(narrow));
    truncating = gffx_self_test_diagnostic(narrow, (uint64_t)sizeof(narrow));
    if (gffx_validate_tensor_view(&tensor, &truncating) == GFFX_STATUS_UNSUPPORTED &&
        truncating.required_bytes > (uint64_t)sizeof(narrow) &&
        narrow[sizeof(narrow) - 1u] == '\0' &&
        strlen(narrow) == sizeof(narrow) - 1u) {
        checks |= GFFX_SELF_TEST_DIAGNOSTIC_TRUNCATION;
    }

    return checks;
}

static uint32_t gffx_self_test_capability_paths(void) {
    gffx_capability_record records[GFFX_SELF_TEST_MAX_RECORDS];
    char strings[GFFX_SELF_TEST_MAX_STRING_BYTES];
    char message[192] = {0};
    gffx_diagnostic_buffer diagnostic = gffx_self_test_diagnostic(message, sizeof(message));
    gffx_capability_report sizing = gffx_self_test_report();
    gffx_capability_report filled;
    uint64_t required_records;
    uint64_t required_strings;

    /* Pass one: caller supplies no storage and learns the exact requirement. */
    if (gffx_capabilities_query(&sizing, &diagnostic) != GFFX_STATUS_INSUFFICIENT_WORKSPACE) {
        return UINT32_C(0);
    }
    required_records = sizing.required_record_count;
    required_strings = sizing.required_string_bytes;
    if (required_records == UINT64_C(0) || required_strings == UINT64_C(0)) return UINT32_C(0);
    if (required_records > (uint64_t)GFFX_SELF_TEST_MAX_RECORDS) return UINT32_C(0);
    if (required_strings > (uint64_t)GFFX_SELF_TEST_MAX_STRING_BYTES) return UINT32_C(0);

    /* Pass two: caller-owned automatic storage is filled; the core allocates nothing. */
    memset(records, 0, sizeof(records));
    memset(strings, 0, sizeof(strings));
    filled = gffx_self_test_report();
    filled.records = records;
    filled.record_capacity = (uint64_t)GFFX_SELF_TEST_MAX_RECORDS;
    filled.strings = strings;
    filled.string_capacity_bytes = (uint64_t)GFFX_SELF_TEST_MAX_STRING_BYTES;
    if (gffx_capabilities_query(&filled, &diagnostic) != GFFX_STATUS_OK) return UINT32_C(0);
    if (filled.record_count != required_records) return UINT32_C(0);
    if (filled.string_size_bytes != required_strings) return UINT32_C(0);
    if ((filled.result_flags & GFFX_CAPABILITY_RESULT_STATIC) == UINT32_C(0)) return UINT32_C(0);
    if ((filled.result_flags & GFFX_CAPABILITY_RESULT_RUNTIME_PROBED) != UINT32_C(0)) {
        return UINT32_C(0);
    }
    if (records[0].struct_size != (uint32_t)sizeof(gffx_capability_record)) return UINT32_C(0);
    if (records[0].abi_version != GFFX_ABI_VERSION) return UINT32_C(0);

    return GFFX_SELF_TEST_CAPABILITY_TWO_PASS;
}

GFFX_API gffx_status GFFX_CALL gffx_private_self_test(
    uint32_t *out_checks,
    gffx_diagnostic_buffer *diagnostic
) {
    uint32_t checks = UINT32_C(0);
    gffx_status status = gffx_internal_prepare_diagnostic(diagnostic);

    if (status != GFFX_STATUS_OK) return status;

    if (gffx_get_abi_version() == GFFX_ABI_VERSION) checks |= GFFX_SELF_TEST_ABI_VERSION;
    checks |= gffx_self_test_tensor_paths();
    checks |= gffx_self_test_execution_paths();
    checks |= gffx_self_test_diagnostic_paths();
    checks |= gffx_self_test_capability_paths();

    if (out_checks != NULL) *out_checks = checks;
    if (checks != GFFX_SELF_TEST_ALL) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INTERNAL_ERROR,
            "private calling-path self-test did not complete every check"
        );
    }
    return GFFX_STATUS_OK;
}
