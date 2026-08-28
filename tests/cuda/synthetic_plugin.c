#include "plugin_api.h"

#include <string.h>
#include <stdlib.h>

#if GFFX_SYNTHETIC_PLUGIN_MODE == 2
GFFX_CUDA_PLUGIN_API uint32_t GFFX_CALL gffx_not_the_handshake(void) {
    return UINT32_C(0);
}
#else
static gffx_status GFFX_CALL synthetic_capabilities(
    uint32_t probe_flags,
    gffx_capability_report *report,
    gffx_diagnostic_buffer *diagnostic
) {
    static const char status_text[] = "synthetic driver available";
    const uint64_t string_bytes = (uint64_t)sizeof(status_text);
    (void)probe_flags;
    (void)diagnostic;
    if (report == NULL || report->struct_size < sizeof(*report) ||
        report->abi_version != GFFX_ABI_VERSION) {
        return GFFX_STATUS_INVALID_ARGUMENT;
    }
    report->record_count = UINT64_C(0);
    report->string_size_bytes = UINT64_C(0);
    report->required_record_count = UINT64_C(2);
    report->required_string_bytes = string_bytes;
    report->result_flags = GFFX_CAPABILITY_RESULT_RUNTIME_PROBED;
    if (report->records == NULL || report->record_capacity < UINT64_C(2) ||
        report->strings == NULL || report->string_capacity_bytes < string_bytes) {
        return GFFX_STATUS_INSUFFICIENT_WORKSPACE;
    }
    memset(report->records, 0, sizeof(*report->records) * 2u);
    report->records[0].struct_size = (uint32_t)sizeof(*report->records);
    report->records[0].abi_version = GFFX_ABI_VERSION;
    report->records[0].category = GFFX_CAPABILITY_CATEGORY_DRIVER;
    report->records[0].key = GFFX_CAPABILITY_KEY_CUDA_DRIVER_STATUS;
    report->records[0].value_type = GFFX_CAPABILITY_VALUE_STRING;
    report->records[0].string_size = string_bytes;
    memcpy(report->strings, status_text, sizeof(status_text));
    report->records[1].struct_size = (uint32_t)sizeof(*report->records);
    report->records[1].abi_version = GFFX_ABI_VERSION;
    report->records[1].category = GFFX_CAPABILITY_CATEGORY_DRIVER;
    report->records[1].key = GFFX_CAPABILITY_KEY_CUDA_DEVICE_COUNT;
    report->records[1].value_type = GFFX_CAPABILITY_VALUE_U64;
    report->records[1].value_u64 = UINT64_C(0);
    report->record_count = UINT64_C(2);
    report->string_size_bytes = string_bytes;
    return GFFX_STATUS_OK;
}

#if GFFX_SYNTHETIC_PLUGIN_MODE == 5
static const gffx_cuda_operations synthetic_operations = {
    (uint32_t)sizeof(gffx_cuda_operations), 0u,
    NULL,
    NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
    NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
    {0, 0, 0, 0}
};
#endif

GFFX_CUDA_PLUGIN_API gffx_status GFFX_CALL gffx_cuda_plugin_handshake_v1(
    uint32_t requested_plugin_abi,
    uint32_t host_core_abi,
    gffx_cuda_plugin_api *api,
    gffx_diagnostic_buffer *diagnostic
) {
#if GFFX_SYNTHETIC_PLUGIN_MODE == 3
    abort();
#endif
    (void)requested_plugin_abi;
    (void)host_core_abi;
    (void)diagnostic;
    if (api == NULL || api->struct_size < sizeof(*api)) {
        return GFFX_STATUS_INVALID_ARGUMENT;
    }
    memset(api, 0, sizeof(*api));
    api->struct_size = (uint32_t)sizeof(*api);
#if GFFX_SYNTHETIC_PLUGIN_MODE == 1
    api->plugin_abi_version = GFFX_ABI_VERSION_ENCODE(2, 0);
#else
    api->plugin_abi_version = GFFX_CUDA_PLUGIN_ABI_VERSION;
#endif
    api->core_abi_min = GFFX_ABI_VERSION;
    api->core_abi_max = GFFX_ABI_VERSION;
#if GFFX_SYNTHETIC_PLUGIN_MODE == 5
    /* A well-formed operation table with every entry NULL, which is what a plugin looks like
     * before any kernel is implemented. The host must accept it: publishing a table is a
     * statement that dispatch exists, not that any operation does. */
    api->flags = GFFX_CUDA_PLUGIN_FLAG_CAPABILITY_PROVIDER |
                 GFFX_CUDA_PLUGIN_FLAG_OPERATION_PROVIDER;
    api->operations = &synthetic_operations;
#elif GFFX_SYNTHETIC_PLUGIN_MODE == 4
    /* Advertises operations and publishes nothing. A build mismatch, and the host must refuse it
     * rather than dispatch through a null table. */
    api->flags = GFFX_CUDA_PLUGIN_FLAG_CAPABILITY_PROVIDER |
                 GFFX_CUDA_PLUGIN_FLAG_OPERATION_PROVIDER;
    api->operations = NULL;
#else
    api->flags = GFFX_CUDA_PLUGIN_FLAG_CAPABILITY_PROVIDER;
#endif
    api->build_identity = "synthetic-cuda-plugin/test-only";
    api->capabilities_probe = synthetic_capabilities;
    return GFFX_STATUS_OK;
}
#endif
