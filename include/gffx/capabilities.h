#ifndef GFFX_CAPABILITIES_H
#define GFFX_CAPABILITIES_H

#include <gffx/status.h>

#define GFFX_CAPABILITY_CATEGORY_BUILD UINT32_C(1)
#define GFFX_CAPABILITY_CATEGORY_HOST UINT32_C(2)
#define GFFX_CAPABILITY_CATEGORY_CPU UINT32_C(3)
#define GFFX_CAPABILITY_CATEGORY_BACKEND UINT32_C(4)
#define GFFX_CAPABILITY_CATEGORY_DRIVER UINT32_C(5)
#define GFFX_CAPABILITY_CATEGORY_DEVICE UINT32_C(6)
#define GFFX_CAPABILITY_CATEGORY_OPERATION UINT32_C(7)

#define GFFX_CAPABILITY_VALUE_U64 UINT32_C(1)
#define GFFX_CAPABILITY_VALUE_I64 UINT32_C(2)
#define GFFX_CAPABILITY_VALUE_STRING UINT32_C(3)
#define GFFX_CAPABILITY_VALUE_BOOL UINT32_C(4)

#define GFFX_CAPABILITY_KEY_ABI_VERSION UINT32_C(1)
#define GFFX_CAPABILITY_KEY_PACKAGE_VERSION UINT32_C(2)
#define GFFX_CAPABILITY_KEY_POINTER_BITS UINT32_C(3)
#define GFFX_CAPABILITY_KEY_TARGET_OS UINT32_C(4)
#define GFFX_CAPABILITY_KEY_TARGET_ARCH UINT32_C(5)
#define GFFX_CAPABILITY_KEY_COMPILER UINT32_C(6)
#define GFFX_CAPABILITY_KEY_ENDIANNESS UINT32_C(7)
#define GFFX_CAPABILITY_KEY_CPU_BACKEND UINT32_C(8)
#define GFFX_CAPABILITY_KEY_CUDA_BACKEND UINT32_C(9)
#define GFFX_CAPABILITY_KEY_DTYPE_MASK UINT32_C(10)
#define GFFX_CAPABILITY_KEY_DEVICE_MASK UINT32_C(11)
#define GFFX_CAPABILITY_KEY_OPERATION_COUNT UINT32_C(12)
#define GFFX_CAPABILITY_KEY_CUDA_PROVIDER_STATUS UINT32_C(13)
#define GFFX_CAPABILITY_KEY_PROVIDER_STATUS UINT32_C(14)

#define GFFX_CAPABILITY_RECORD_SENSITIVE UINT32_C(1)

#define GFFX_CAPABILITY_RESULT_STATIC UINT32_C(1)
#define GFFX_CAPABILITY_RESULT_RUNTIME_PROBED UINT32_C(2)
#define GFFX_CAPABILITY_RESULT_OPTIONAL_PROVIDER_ABSENT UINT32_C(4)
#define GFFX_CAPABILITY_RESULT_PARTIAL_FAILURE UINT32_C(8)

#define GFFX_CAPABILITY_PROBE_FULL UINT32_C(1)
#define GFFX_CAPABILITY_PROBE_INCLUDE_SENSITIVE UINT32_C(2)

typedef struct gffx_capability_record {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t category;
    uint32_t subject_index;
    uint32_t key;
    uint32_t value_type;
    uint32_t flags;
    uint32_t reserved0;
    uint64_t value_u64;
    int64_t value_i64;
    uint64_t string_offset;
    uint64_t string_size;
    uint64_t reserved[4];
} gffx_capability_record;

typedef struct gffx_capability_report {
    uint32_t struct_size;
    uint32_t abi_version;
    gffx_capability_record *records;
    uint64_t record_capacity;
    uint64_t record_count;
    uint64_t required_record_count;
    char *strings;
    uint64_t string_capacity_bytes;
    uint64_t string_size_bytes;
    uint64_t required_string_bytes;
    uint32_t query_flags;
    uint32_t result_flags;
    uint64_t reserved[4];
} gffx_capability_report;

GFFX_EXTERN_C_BEGIN

GFFX_API gffx_status GFFX_CALL gffx_capabilities_query(
    gffx_capability_report *report,
    gffx_diagnostic_buffer *diagnostic
);

GFFX_API gffx_status GFFX_CALL gffx_capabilities_probe(
    uint32_t probe_flags,
    gffx_capability_report *report,
    gffx_diagnostic_buffer *diagnostic
);

GFFX_EXTERN_C_END

#endif /* GFFX_CAPABILITIES_H */
