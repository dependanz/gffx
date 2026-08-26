#ifndef GFFX_CUDA_PLUGIN_API_H
#define GFFX_CUDA_PLUGIN_API_H

#include <gffx/capabilities.h>

/* Private host/plugin ABI. This header is intentionally not installed as public API. */
#define GFFX_CUDA_PLUGIN_ABI_VERSION_MAJOR UINT32_C(1)
#define GFFX_CUDA_PLUGIN_ABI_VERSION_MINOR UINT32_C(0)
#define GFFX_CUDA_PLUGIN_ABI_VERSION \
    GFFX_ABI_VERSION_ENCODE( \
        GFFX_CUDA_PLUGIN_ABI_VERSION_MAJOR, \
        GFFX_CUDA_PLUGIN_ABI_VERSION_MINOR \
    )

#define GFFX_CUDA_PLUGIN_HANDSHAKE_SYMBOL "gffx_cuda_plugin_handshake_v1"
#define GFFX_CUDA_PLUGIN_FLAG_CAPABILITY_PROVIDER UINT64_C(1)

#if defined(_WIN32)
#if defined(GFFX_BUILDING_CUDA_PLUGIN)
#define GFFX_CUDA_PLUGIN_API __declspec(dllexport)
#else
#define GFFX_CUDA_PLUGIN_API __declspec(dllimport)
#endif
#else
#define GFFX_CUDA_PLUGIN_API __attribute__((visibility("default")))
#endif

typedef gffx_status (GFFX_CALL *gffx_cuda_plugin_capabilities_fn)(
    uint32_t probe_flags,
    gffx_capability_report *report,
    gffx_diagnostic_buffer *diagnostic
);

typedef struct gffx_cuda_plugin_api {
    uint32_t struct_size;
    uint32_t plugin_abi_version;
    uint32_t core_abi_min;
    uint32_t core_abi_max;
    uint64_t flags;
    const char *build_identity;
    gffx_cuda_plugin_capabilities_fn capabilities_probe;
    uint64_t reserved[6];
} gffx_cuda_plugin_api;

typedef gffx_status (GFFX_CALL *gffx_cuda_plugin_handshake_fn)(
    uint32_t requested_plugin_abi,
    uint32_t host_core_abi,
    gffx_cuda_plugin_api *api,
    gffx_diagnostic_buffer *diagnostic
);

GFFX_EXTERN_C_BEGIN

GFFX_CUDA_PLUGIN_API gffx_status GFFX_CALL gffx_cuda_plugin_handshake_v1(
    uint32_t requested_plugin_abi,
    uint32_t host_core_abi,
    gffx_cuda_plugin_api *api,
    gffx_diagnostic_buffer *diagnostic
);

GFFX_EXTERN_C_END

#endif
