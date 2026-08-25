#include <gffx/capabilities.h>
#include <gffx/tensor.h>

#include "internal.h"
#include "cuda_loader.h"

#include <string.h>

#ifndef GFFX_PACKAGE_VERSION
#define GFFX_PACKAGE_VERSION "unknown"
#endif

#define GFFX_STRINGIFY_INNER(value) #value
#define GFFX_STRINGIFY(value) GFFX_STRINGIFY_INNER(value)

#if defined(_WIN32)
#define GFFX_TARGET_OS_STRING "windows"
#elif defined(__linux__)
#define GFFX_TARGET_OS_STRING "linux"
#elif defined(__APPLE__)
#define GFFX_TARGET_OS_STRING "macos"
#else
#define GFFX_TARGET_OS_STRING "unknown"
#endif

#if defined(_M_X64) || defined(__x86_64__)
#define GFFX_TARGET_ARCH_STRING "x86_64"
#elif defined(_M_ARM64) || defined(__aarch64__)
#define GFFX_TARGET_ARCH_STRING "arm64"
#else
#define GFFX_TARGET_ARCH_STRING "unknown"
#endif

#if defined(_MSC_VER)
#define GFFX_COMPILER_STRING "msvc-" GFFX_STRINGIFY(_MSC_VER)
#elif defined(__clang__)
#define GFFX_COMPILER_STRING "clang-" __clang_version__
#elif defined(__GNUC__)
#define GFFX_COMPILER_STRING "gcc-" __VERSION__
#else
#define GFFX_COMPILER_STRING "unknown"
#endif

#if defined(_WIN32) || \
    (defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
#define GFFX_ENDIANNESS_STRING "little"
#elif defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#define GFFX_ENDIANNESS_STRING "big"
#else
#define GFFX_ENDIANNESS_STRING "unknown"
#endif

typedef struct gffx_capability_spec {
    uint32_t category;
    uint32_t key;
    uint32_t value_type;
    uint32_t flags;
    uint64_t value_u64;
    int64_t value_i64;
    const char *string_value;
} gffx_capability_spec;

#define GFFX_DTYPE_MASK_V1 \
    ((UINT64_C(1) << GFFX_DTYPE_FLOAT32) | \
     (UINT64_C(1) << GFFX_DTYPE_FLOAT64) | \
     (UINT64_C(1) << GFFX_DTYPE_INT32) | \
     (UINT64_C(1) << GFFX_DTYPE_UINT32) | \
     (UINT64_C(1) << GFFX_DTYPE_BOOL))

static const gffx_capability_spec gffx_static_capabilities[] = {
    {GFFX_CAPABILITY_CATEGORY_BUILD, GFFX_CAPABILITY_KEY_ABI_VERSION,
     GFFX_CAPABILITY_VALUE_U64, 0u, GFFX_ABI_VERSION, 0, NULL},
    {GFFX_CAPABILITY_CATEGORY_BUILD, GFFX_CAPABILITY_KEY_PACKAGE_VERSION,
     GFFX_CAPABILITY_VALUE_STRING, 0u, 0u, 0, GFFX_PACKAGE_VERSION},
    {GFFX_CAPABILITY_CATEGORY_BUILD, GFFX_CAPABILITY_KEY_POINTER_BITS,
     GFFX_CAPABILITY_VALUE_U64, 0u, 64u, 0, NULL},
    {GFFX_CAPABILITY_CATEGORY_BUILD, GFFX_CAPABILITY_KEY_TARGET_OS,
     GFFX_CAPABILITY_VALUE_STRING, 0u, 0u, 0, GFFX_TARGET_OS_STRING},
    {GFFX_CAPABILITY_CATEGORY_BUILD, GFFX_CAPABILITY_KEY_TARGET_ARCH,
     GFFX_CAPABILITY_VALUE_STRING, 0u, 0u, 0, GFFX_TARGET_ARCH_STRING},
    {GFFX_CAPABILITY_CATEGORY_BUILD, GFFX_CAPABILITY_KEY_COMPILER,
     GFFX_CAPABILITY_VALUE_STRING, 0u, 0u, 0, GFFX_COMPILER_STRING},
    {GFFX_CAPABILITY_CATEGORY_BUILD, GFFX_CAPABILITY_KEY_ENDIANNESS,
     GFFX_CAPABILITY_VALUE_STRING, 0u, 0u, 0, GFFX_ENDIANNESS_STRING},
    {GFFX_CAPABILITY_CATEGORY_BACKEND, GFFX_CAPABILITY_KEY_CPU_BACKEND,
     GFFX_CAPABILITY_VALUE_BOOL, 0u, 1u, 0, NULL},
    {GFFX_CAPABILITY_CATEGORY_BACKEND, GFFX_CAPABILITY_KEY_CUDA_BACKEND,
     GFFX_CAPABILITY_VALUE_BOOL, 0u, 0u, 0, NULL},
    {GFFX_CAPABILITY_CATEGORY_BUILD, GFFX_CAPABILITY_KEY_DTYPE_MASK,
     GFFX_CAPABILITY_VALUE_U64, 0u, GFFX_DTYPE_MASK_V1, 0, NULL},
    {GFFX_CAPABILITY_CATEGORY_BUILD, GFFX_CAPABILITY_KEY_DEVICE_MASK,
     GFFX_CAPABILITY_VALUE_U64, 0u, UINT64_C(1) << GFFX_DEVICE_CPU, 0, NULL},
    {GFFX_CAPABILITY_CATEGORY_OPERATION, GFFX_CAPABILITY_KEY_OPERATION_COUNT,
     GFFX_CAPABILITY_VALUE_U64, 0u, 0u, 0, NULL}
};

#if defined(GFFX_ENABLE_TEST_PROVIDER)
static const gffx_capability_spec gffx_test_provider_failure_capability = {
    GFFX_CAPABILITY_CATEGORY_BACKEND,
    GFFX_CAPABILITY_KEY_PROVIDER_STATUS,
    GFFX_CAPABILITY_VALUE_STRING,
    0u,
    0u,
    0,
    "test provider: synthetic failure"
};
#endif

static uint64_t gffx_capability_string_size(const gffx_capability_spec *spec) {
    if (spec->value_type != GFFX_CAPABILITY_VALUE_STRING) return UINT64_C(0);
    return (uint64_t)strlen(spec->string_value) + UINT64_C(1);
}

static void gffx_emit_capability(
    const gffx_capability_spec *spec,
    gffx_capability_record *record,
    char *strings,
    uint64_t *string_cursor
) {
    uint64_t string_size = gffx_capability_string_size(spec);
    memset(record, 0, sizeof(*record));
    record->struct_size = (uint32_t)sizeof(*record);
    record->abi_version = GFFX_ABI_VERSION;
    record->category = spec->category;
    record->subject_index = UINT32_C(0);
    record->key = spec->key;
    record->value_type = spec->value_type;
    record->flags = spec->flags;
    record->value_u64 = spec->value_u64;
    record->value_i64 = spec->value_i64;
    if (string_size != UINT64_C(0)) {
        record->string_offset = *string_cursor;
        record->string_size = string_size;
        memcpy(strings + *string_cursor, spec->string_value, (size_t)string_size);
        *string_cursor += string_size;
    }
}

static gffx_status gffx_capabilities_run(
    int full_probe,
    uint32_t probe_flags,
    gffx_capability_report *report,
    gffx_diagnostic_buffer *diagnostic
) {
    const uint64_t static_count =
        (uint64_t)(sizeof(gffx_static_capabilities) / sizeof(gffx_static_capabilities[0]));
#if defined(GFFX_ENABLE_TEST_PROVIDER)
    const uint64_t test_provider_count = full_probe ? UINT64_C(1) : UINT64_C(0);
#else
    const uint64_t test_provider_count = UINT64_C(0);
#endif
    gffx_capability_report cuda_report = {0};
    uint64_t cuda_record_count = UINT64_C(0);
    uint64_t cuda_string_bytes = UINT64_C(0);
    uint32_t cuda_result_flags = UINT32_C(0);
    uint64_t required_records;
    uint64_t required_strings = UINT64_C(0);
    uint64_t string_cursor = UINT64_C(0);
    uint64_t index;
    gffx_status status = gffx_internal_prepare_diagnostic(diagnostic);

    if (status != GFFX_STATUS_OK) return status;
    if (report == NULL) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "capability report pointer is null"
        );
    }
    status = gffx_internal_validate_header(
        report,
        report->struct_size,
        report->abi_version,
        (uint32_t)sizeof(gffx_capability_report),
        "capability report pointer is null",
        diagnostic
    );
    if (status != GFFX_STATUS_OK) return status;
    if (!gffx_internal_reserved_u64_is_zero(report->reserved, 4u)) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "capability report reserved fields must be zero"
        );
    }
    if (report->record_capacity != UINT64_C(0) && report->records == NULL) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "nonzero record capacity requires a record buffer"
        );
    }
    if (report->string_capacity_bytes != UINT64_C(0) && report->strings == NULL) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "nonzero string capacity requires a string buffer"
        );
    }
    if (full_probe) {
        cuda_report.struct_size = (uint32_t)sizeof(cuda_report);
        cuda_report.abi_version = GFFX_ABI_VERSION;
        status = gffx_cuda_loader_probe(probe_flags, &cuda_report, diagnostic);
        if (status != GFFX_STATUS_INSUFFICIENT_WORKSPACE && status != GFFX_STATUS_OK) return status;
        cuda_record_count = cuda_report.required_record_count;
        cuda_string_bytes = cuda_report.required_string_bytes;
        cuda_result_flags = cuda_report.result_flags;
    }
    required_records = static_count + cuda_record_count + test_provider_count;
    for (index = 0u; index < static_count; ++index) {
        required_strings += gffx_capability_string_size(&gffx_static_capabilities[index]);
    }
    if (full_probe) {
        required_strings += cuda_string_bytes;
#if defined(GFFX_ENABLE_TEST_PROVIDER)
        required_strings += gffx_capability_string_size(&gffx_test_provider_failure_capability);
#endif
    }

    report->record_count = UINT64_C(0);
    report->required_record_count = required_records;
    report->string_size_bytes = UINT64_C(0);
    report->required_string_bytes = required_strings;
    report->query_flags = probe_flags;
    report->result_flags = GFFX_CAPABILITY_RESULT_STATIC;
    if (full_probe) {
        report->result_flags |= GFFX_CAPABILITY_RESULT_RUNTIME_PROBED;
        report->result_flags |= cuda_result_flags;
#if defined(GFFX_ENABLE_TEST_PROVIDER)
        report->result_flags |= GFFX_CAPABILITY_RESULT_PARTIAL_FAILURE;
#endif
    }

    if (report->record_capacity < required_records ||
        report->string_capacity_bytes < required_strings ||
        report->records == NULL || report->strings == NULL) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INSUFFICIENT_WORKSPACE,
            "capability report buffers are too small"
        );
    }

    for (index = 0u; index < static_count; ++index) {
        gffx_emit_capability(
            &gffx_static_capabilities[index],
            &report->records[index],
            report->strings,
            &string_cursor
        );
    }
    if (full_probe) {
        memset(&cuda_report, 0, sizeof(cuda_report));
        cuda_report.struct_size = (uint32_t)sizeof(cuda_report);
        cuda_report.abi_version = GFFX_ABI_VERSION;
        cuda_report.records = report->records + static_count;
        cuda_report.record_capacity = report->record_capacity - static_count;
        cuda_report.strings = report->strings + string_cursor;
        cuda_report.string_capacity_bytes = report->string_capacity_bytes - string_cursor;
        status = gffx_cuda_loader_probe(probe_flags, &cuda_report, diagnostic);
        if (status != GFFX_STATUS_OK) {
            report->required_record_count = static_count + cuda_report.required_record_count +
                test_provider_count;
            report->required_string_bytes = string_cursor + cuda_report.required_string_bytes;
            return status;
        }
        for (index = 0u; index < cuda_report.record_count; ++index) {
            if (cuda_report.records[index].value_type == GFFX_CAPABILITY_VALUE_STRING) {
                cuda_report.records[index].string_offset += string_cursor;
            }
        }
        cuda_record_count = cuda_report.record_count;
        cuda_string_bytes = cuda_report.string_size_bytes;
        report->result_flags |= cuda_report.result_flags;
        string_cursor += cuda_string_bytes;
#if defined(GFFX_ENABLE_TEST_PROVIDER)
        gffx_emit_capability(
            &gffx_test_provider_failure_capability,
            &report->records[static_count + cuda_record_count],
            report->strings,
            &string_cursor
        );
#endif
    }
    report->record_count = static_count + cuda_record_count + test_provider_count;
    report->required_record_count = report->record_count;
    report->string_size_bytes = string_cursor;
    report->required_string_bytes = string_cursor;
    return GFFX_STATUS_OK;
}

GFFX_API gffx_status GFFX_CALL gffx_capabilities_query(
    gffx_capability_report *report,
    gffx_diagnostic_buffer *diagnostic
) {
    return gffx_capabilities_run(0, UINT32_C(0), report, diagnostic);
}

GFFX_API gffx_status GFFX_CALL gffx_capabilities_probe(
    uint32_t probe_flags,
    gffx_capability_report *report,
    gffx_diagnostic_buffer *diagnostic
) {
    const uint32_t allowed_flags =
        GFFX_CAPABILITY_PROBE_FULL | GFFX_CAPABILITY_PROBE_INCLUDE_SENSITIVE;
    gffx_status status = gffx_internal_prepare_diagnostic(diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if ((probe_flags & GFFX_CAPABILITY_PROBE_FULL) == UINT32_C(0)) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "the full capability-probe flag is required"
        );
    }
    if ((probe_flags & ~allowed_flags) != UINT32_C(0)) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_UNSUPPORTED,
            "capability probe contains an unsupported flag"
        );
    }
    return gffx_capabilities_run(1, probe_flags, report, diagnostic);
}
