#include "plugin_api.h"
#include "gffx_cuda_ptx.h"

#include <cuda.h>

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#ifndef GFFX_CUDA_BUILD_ID
#define GFFX_CUDA_BUILD_ID "gffx-cuda12/unknown;driver-api"
#endif

typedef struct gffx_cuda_writer {
    gffx_capability_report *report;
    uint64_t record_count;
    uint64_t string_bytes;
    int write;
    int overflow;
} gffx_cuda_writer;

typedef struct gffx_cuda_attribute_spec {
    CUdevice_attribute attribute;
    uint32_t key;
    uint32_t value_type;
} gffx_cuda_attribute_spec;

static const gffx_cuda_attribute_spec gffx_device_attributes[] = {
    {CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
     GFFX_CAPABILITY_KEY_CUDA_COMPUTE_CAPABILITY_MAJOR, GFFX_CAPABILITY_VALUE_U64},
    {CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
     GFFX_CAPABILITY_KEY_CUDA_COMPUTE_CAPABILITY_MINOR, GFFX_CAPABILITY_VALUE_U64},
    {CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
     GFFX_CAPABILITY_KEY_CUDA_MULTIPROCESSOR_COUNT, GFFX_CAPABILITY_VALUE_U64},
    {CU_DEVICE_ATTRIBUTE_WARP_SIZE,
     GFFX_CAPABILITY_KEY_CUDA_WARP_SIZE, GFFX_CAPABILITY_VALUE_U64},
    {CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
     GFFX_CAPABILITY_KEY_CUDA_MAX_THREADS_PER_BLOCK, GFFX_CAPABILITY_VALUE_U64},
    {CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X,
     GFFX_CAPABILITY_KEY_CUDA_MAX_BLOCK_DIM_X, GFFX_CAPABILITY_VALUE_U64},
    {CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y,
     GFFX_CAPABILITY_KEY_CUDA_MAX_BLOCK_DIM_Y, GFFX_CAPABILITY_VALUE_U64},
    {CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z,
     GFFX_CAPABILITY_KEY_CUDA_MAX_BLOCK_DIM_Z, GFFX_CAPABILITY_VALUE_U64},
    {CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X,
     GFFX_CAPABILITY_KEY_CUDA_MAX_GRID_DIM_X, GFFX_CAPABILITY_VALUE_U64},
    {CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y,
     GFFX_CAPABILITY_KEY_CUDA_MAX_GRID_DIM_Y, GFFX_CAPABILITY_VALUE_U64},
    {CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z,
     GFFX_CAPABILITY_KEY_CUDA_MAX_GRID_DIM_Z, GFFX_CAPABILITY_VALUE_U64},
    {CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
     GFFX_CAPABILITY_KEY_CUDA_SHARED_MEMORY_PER_BLOCK, GFFX_CAPABILITY_VALUE_U64},
    {CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK,
     GFFX_CAPABILITY_KEY_CUDA_REGISTERS_PER_BLOCK, GFFX_CAPABILITY_VALUE_U64},
    {CU_DEVICE_ATTRIBUTE_CLOCK_RATE,
     GFFX_CAPABILITY_KEY_CUDA_CLOCK_RATE_KHZ, GFFX_CAPABILITY_VALUE_U64},
    {CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE,
     GFFX_CAPABILITY_KEY_CUDA_MEMORY_CLOCK_RATE_KHZ, GFFX_CAPABILITY_VALUE_U64},
    {CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH,
     GFFX_CAPABILITY_KEY_CUDA_MEMORY_BUS_WIDTH_BITS, GFFX_CAPABILITY_VALUE_U64},
    {CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE,
     GFFX_CAPABILITY_KEY_CUDA_L2_CACHE_BYTES, GFFX_CAPABILITY_VALUE_U64},
    {CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
     GFFX_CAPABILITY_KEY_CUDA_MAX_THREADS_PER_MULTIPROCESSOR, GFFX_CAPABILITY_VALUE_U64},
    {CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING,
     GFFX_CAPABILITY_KEY_CUDA_UNIFIED_ADDRESSING, GFFX_CAPABILITY_VALUE_BOOL},
    {CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY,
     GFFX_CAPABILITY_KEY_CUDA_MANAGED_MEMORY, GFFX_CAPABILITY_VALUE_BOOL},
    {CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS,
     GFFX_CAPABILITY_KEY_CUDA_CONCURRENT_MANAGED_ACCESS, GFFX_CAPABILITY_VALUE_BOOL},
    {CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS,
     GFFX_CAPABILITY_KEY_CUDA_PAGEABLE_MEMORY_ACCESS, GFFX_CAPABILITY_VALUE_BOOL},
    {CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH,
     GFFX_CAPABILITY_KEY_CUDA_COOPERATIVE_LAUNCH, GFFX_CAPABILITY_VALUE_BOOL},
    {CU_DEVICE_ATTRIBUTE_COMPUTE_MODE,
     GFFX_CAPABILITY_KEY_CUDA_COMPUTE_MODE, GFFX_CAPABILITY_VALUE_U64},
    {CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT,
     GFFX_CAPABILITY_KEY_CUDA_KERNEL_TIMEOUT, GFFX_CAPABILITY_VALUE_BOOL},
    {CU_DEVICE_ATTRIBUTE_INTEGRATED,
     GFFX_CAPABILITY_KEY_CUDA_INTEGRATED, GFFX_CAPABILITY_VALUE_BOOL},
    {CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY,
     GFFX_CAPABILITY_KEY_CUDA_CAN_MAP_HOST_MEMORY, GFFX_CAPABILITY_VALUE_BOOL},
    {CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT,
     GFFX_CAPABILITY_KEY_CUDA_ASYNC_ENGINE_COUNT, GFFX_CAPABILITY_VALUE_U64},
    {CU_DEVICE_ATTRIBUTE_ECC_ENABLED,
     GFFX_CAPABILITY_KEY_CUDA_ECC_ENABLED, GFFX_CAPABILITY_VALUE_BOOL},
    {CU_DEVICE_ATTRIBUTE_TCC_DRIVER,
     GFFX_CAPABILITY_KEY_CUDA_TCC_DRIVER, GFFX_CAPABILITY_VALUE_BOOL},
    {CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED,
     GFFX_CAPABILITY_KEY_CUDA_COMPUTE_PREEMPTION, GFFX_CAPABILITY_VALUE_BOOL},
    {CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
     GFFX_CAPABILITY_KEY_CUDA_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, GFFX_CAPABILITY_VALUE_U64},
    {CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR,
     GFFX_CAPABILITY_KEY_CUDA_MAX_BLOCKS_PER_MULTIPROCESSOR, GFFX_CAPABILITY_VALUE_U64},
    {CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED,
     GFFX_CAPABILITY_KEY_CUDA_MEMORY_POOLS_SUPPORTED, GFFX_CAPABILITY_VALUE_BOOL},
    {CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED,
     GFFX_CAPABILITY_KEY_CUDA_GPU_DIRECT_RDMA_SUPPORTED, GFFX_CAPABILITY_VALUE_BOOL}
};

static void gffx_cuda_diagnostic(
    gffx_diagnostic_buffer *diagnostic,
    const char *message
) {
    uint64_t required = (uint64_t)strlen(message) + UINT64_C(1);
    if (diagnostic == NULL || diagnostic->struct_size < sizeof(*diagnostic) ||
        diagnostic->abi_version != GFFX_ABI_VERSION) return;
    diagnostic->required_bytes = required;
    if (diagnostic->data != NULL && diagnostic->capacity_bytes != UINT64_C(0)) {
        size_t copy_size = (size_t)(required <= diagnostic->capacity_bytes
            ? required : diagnostic->capacity_bytes);
        memcpy(diagnostic->data, message, copy_size);
        diagnostic->data[copy_size - 1u] = '\0';
    }
}

static void gffx_cuda_emit_u64(
    gffx_cuda_writer *writer,
    uint32_t category,
    uint32_t subject,
    uint32_t key,
    uint32_t value_type,
    uint64_t value
) {
    if (writer->write && writer->record_count >= writer->report->record_capacity) {
        writer->overflow = 1;
        writer->write = 0;
    }
    if (writer->write) {
        gffx_capability_record *record = &writer->report->records[writer->record_count];
        memset(record, 0, sizeof(*record));
        record->struct_size = (uint32_t)sizeof(*record);
        record->abi_version = GFFX_ABI_VERSION;
        record->category = category;
        record->subject_index = subject;
        record->key = key;
        record->value_type = value_type;
        record->value_u64 = value;
    }
    ++writer->record_count;
}

static void gffx_cuda_emit_string(
    gffx_cuda_writer *writer,
    uint32_t category,
    uint32_t subject,
    uint32_t key,
    uint32_t flags,
    const char *value
) {
    uint64_t bytes = (uint64_t)strlen(value) + UINT64_C(1);
    if (writer->write &&
        (writer->record_count >= writer->report->record_capacity ||
         writer->string_bytes > writer->report->string_capacity_bytes ||
         bytes > writer->report->string_capacity_bytes - writer->string_bytes)) {
        writer->overflow = 1;
        writer->write = 0;
    }
    if (writer->write) {
        gffx_capability_record *record = &writer->report->records[writer->record_count];
        memset(record, 0, sizeof(*record));
        record->struct_size = (uint32_t)sizeof(*record);
        record->abi_version = GFFX_ABI_VERSION;
        record->category = category;
        record->subject_index = subject;
        record->key = key;
        record->value_type = GFFX_CAPABILITY_VALUE_STRING;
        record->flags = flags;
        record->string_offset = writer->string_bytes;
        record->string_size = bytes;
        memcpy(writer->report->strings + writer->string_bytes, value, (size_t)bytes);
    }
    ++writer->record_count;
    writer->string_bytes += bytes;
}

static void gffx_cuda_error_text(CUresult result, char *text, size_t capacity) {
    const char *name = NULL;
    const char *description = NULL;
    (void)cuGetErrorName(result, &name);
    (void)cuGetErrorString(result, &description);
#if defined(_MSC_VER)
    (void)_snprintf_s(text, capacity, _TRUNCATE, "%s%s%s",
        name != NULL ? name : "CUDA_ERROR_UNKNOWN",
        description != NULL ? ": " : "",
        description != NULL ? description : "");
#else
    (void)snprintf(text, capacity, "%s%s%s",
        name != NULL ? name : "CUDA_ERROR_UNKNOWN",
        description != NULL ? ": " : "",
        description != NULL ? description : "");
#endif
}

static void gffx_cuda_format_uuid(const CUuuid *uuid, char *text, size_t capacity) {
    const unsigned char *bytes = (const unsigned char *)uuid->bytes;
#if defined(_MSC_VER)
    (void)_snprintf_s(text, capacity, _TRUNCATE,
#else
    (void)snprintf(text, capacity,
#endif
        "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15]);
}

static void gffx_cuda_collect(gffx_cuda_writer *writer, uint32_t *result_flags) {
    CUresult result = cuInit(0u);
    int driver_version = 0;
    int device_count = 0;
    char status[512] = {0};
    int device_index;

    gffx_cuda_emit_u64(writer, GFFX_CAPABILITY_CATEGORY_BACKEND, 0u,
        GFFX_CAPABILITY_KEY_CUDA_TOOLKIT_VERSION, GFFX_CAPABILITY_VALUE_U64,
        (uint64_t)CUDA_VERSION);
    if (result != CUDA_SUCCESS) {
        gffx_cuda_error_text(result, status, sizeof(status));
        gffx_cuda_emit_u64(writer, GFFX_CAPABILITY_CATEGORY_BACKEND, 0u,
            GFFX_CAPABILITY_KEY_CUDA_BACKEND, GFFX_CAPABILITY_VALUE_BOOL, UINT64_C(0));
        gffx_cuda_emit_string(writer, GFFX_CAPABILITY_CATEGORY_DRIVER, 0u,
            GFFX_CAPABILITY_KEY_CUDA_DRIVER_STATUS, 0u, status);
        gffx_cuda_emit_u64(writer, GFFX_CAPABILITY_CATEGORY_DRIVER, 0u,
            GFFX_CAPABILITY_KEY_CUDA_DEVICE_COUNT, GFFX_CAPABILITY_VALUE_U64, UINT64_C(0));
        *result_flags |= GFFX_CAPABILITY_RESULT_PARTIAL_FAILURE;
        return;
    }

    (void)cuDriverGetVersion(&driver_version);
    result = cuDeviceGetCount(&device_count);
    if (result != CUDA_SUCCESS) {
        gffx_cuda_error_text(result, status, sizeof(status));
        device_count = 0;
        *result_flags |= GFFX_CAPABILITY_RESULT_PARTIAL_FAILURE;
    } else {
        (void)snprintf(status, sizeof(status), "available");
    }
    gffx_cuda_emit_u64(writer, GFFX_CAPABILITY_CATEGORY_BACKEND, 0u,
        GFFX_CAPABILITY_KEY_CUDA_BACKEND, GFFX_CAPABILITY_VALUE_BOOL, UINT64_C(1));
    gffx_cuda_emit_string(writer, GFFX_CAPABILITY_CATEGORY_DRIVER, 0u,
        GFFX_CAPABILITY_KEY_CUDA_DRIVER_STATUS, 0u, status);
    gffx_cuda_emit_u64(writer, GFFX_CAPABILITY_CATEGORY_DRIVER, 0u,
        GFFX_CAPABILITY_KEY_CUDA_DRIVER_VERSION, GFFX_CAPABILITY_VALUE_U64,
        (uint64_t)(driver_version >= 0 ? driver_version : 0));
    gffx_cuda_emit_u64(writer, GFFX_CAPABILITY_CATEGORY_DRIVER, 0u,
        GFFX_CAPABILITY_KEY_CUDA_DEVICE_COUNT, GFFX_CAPABILITY_VALUE_U64,
        (uint64_t)(device_count >= 0 ? device_count : 0));

    for (device_index = 0; device_index < device_count; ++device_index) {
        CUdevice device;
        char name[256] = {0};
        char pci_bus_id[64] = {0};
        char uuid_text[64] = {0};
        CUuuid uuid;
        size_t total_memory = 0u;
        size_t attribute_index;
        uint32_t subject = (uint32_t)device_index + UINT32_C(1);
        if (cuDeviceGet(&device, device_index) != CUDA_SUCCESS) continue;
        if (cuDeviceGetName(name, (int)sizeof(name), device) == CUDA_SUCCESS) {
            gffx_cuda_emit_string(writer, GFFX_CAPABILITY_CATEGORY_DEVICE, subject,
                GFFX_CAPABILITY_KEY_CUDA_DEVICE_NAME, 0u, name);
        }
        if (cuDeviceGetUuid(&uuid, device) == CUDA_SUCCESS) {
            gffx_cuda_format_uuid(&uuid, uuid_text, sizeof(uuid_text));
            gffx_cuda_emit_string(writer, GFFX_CAPABILITY_CATEGORY_DEVICE, subject,
                GFFX_CAPABILITY_KEY_CUDA_DEVICE_UUID, GFFX_CAPABILITY_RECORD_SENSITIVE, uuid_text);
        }
        if (cuDeviceGetPCIBusId(pci_bus_id, (int)sizeof(pci_bus_id), device) == CUDA_SUCCESS) {
            gffx_cuda_emit_string(writer, GFFX_CAPABILITY_CATEGORY_DEVICE, subject,
                GFFX_CAPABILITY_KEY_CUDA_DEVICE_PCI_BUS_ID,
                GFFX_CAPABILITY_RECORD_SENSITIVE, pci_bus_id);
        }
        if (cuDeviceTotalMem(&total_memory, device) == CUDA_SUCCESS) {
            gffx_cuda_emit_u64(writer, GFFX_CAPABILITY_CATEGORY_DEVICE, subject,
                GFFX_CAPABILITY_KEY_CUDA_TOTAL_MEMORY_BYTES, GFFX_CAPABILITY_VALUE_U64,
                (uint64_t)total_memory);
        }
        for (attribute_index = 0u;
             attribute_index < sizeof(gffx_device_attributes) / sizeof(gffx_device_attributes[0]);
             ++attribute_index) {
            int value = 0;
            if (cuDeviceGetAttribute(&value, gffx_device_attributes[attribute_index].attribute,
                                     device) == CUDA_SUCCESS) {
                gffx_cuda_emit_u64(writer, GFFX_CAPABILITY_CATEGORY_DEVICE, subject,
                    gffx_device_attributes[attribute_index].key,
                    gffx_device_attributes[attribute_index].value_type,
                    (uint64_t)(value >= 0 ? value : 0));
            }
        }
    }
}

static gffx_status GFFX_CALL gffx_cuda_capabilities(
    uint32_t probe_flags,
    gffx_capability_report *report,
    gffx_diagnostic_buffer *diagnostic
) {
    gffx_cuda_writer sizing = {0};
    gffx_cuda_writer writer = {0};
    uint32_t result_flags = GFFX_CAPABILITY_RESULT_RUNTIME_PROBED;
    if (report == NULL || report->struct_size < sizeof(*report) ||
        report->abi_version != GFFX_ABI_VERSION) {
        gffx_cuda_diagnostic(diagnostic, "CUDA capability report is invalid");
        return GFFX_STATUS_INVALID_ARGUMENT;
    }
    sizing.report = report;
    gffx_cuda_collect(&sizing, &result_flags);
    report->record_count = UINT64_C(0);
    report->string_size_bytes = UINT64_C(0);
    report->required_record_count = sizing.record_count;
    report->required_string_bytes = sizing.string_bytes;
    report->query_flags = probe_flags;
    report->result_flags = result_flags;
    if (report->records == NULL || report->record_capacity < sizing.record_count ||
        report->strings == NULL || report->string_capacity_bytes < sizing.string_bytes) {
        return GFFX_STATUS_INSUFFICIENT_WORKSPACE;
    }
    writer.report = report;
    writer.write = 1;
    result_flags = GFFX_CAPABILITY_RESULT_RUNTIME_PROBED;
    gffx_cuda_collect(&writer, &result_flags);
    if (writer.overflow) {
        report->required_record_count = writer.record_count;
        report->required_string_bytes = writer.string_bytes;
        return GFFX_STATUS_INSUFFICIENT_WORKSPACE;
    }
    report->record_count = writer.record_count;
    report->string_size_bytes = writer.string_bytes;
    report->required_record_count = writer.record_count;
    report->required_string_bytes = writer.string_bytes;
    report->result_flags = result_flags;
    return GFFX_STATUS_OK;
}

/* Writes the diagnostic and returns the status in one expression, so an error path stays a single
 * line and cannot report a message without also returning the failure it describes. */
static gffx_status gffx_cuda_fail(
    gffx_diagnostic_buffer *diagnostic, gffx_status status, const char *message
) {
    gffx_cuda_diagnostic(diagnostic, message);
    return status;
}

/* ---------------------------------------------------------------------------------------------
 * Kernel module loading and dispatch.
 *
 * The embedded PTX is JIT-compiled by the driver on first use and the resulting module is cached,
 * because compiling it per call would pay the JIT cost on every launch. That cache is process-wide
 * mutable state inside the plugin, which the core scaffold forbids of itself and which is tolerable
 * here only because the plugin is separately loaded and separately inspected. It is the same
 * lifetime question the host faces in deciding whether to keep the plugin mapped, and it is
 * recorded as one unresolved decision rather than two.
 *
 * Loading is per CUDA context, not global: a module handle belongs to the context that created it,
 * so a caller using two contexts must not be handed the first context's module. The cache
 * therefore records which context it was built for and reloads when it changes.
 * --------------------------------------------------------------------------------------------- */

static CUcontext gffx_cuda_module_context = NULL;
static CUmodule gffx_cuda_module = NULL;
static CUfunction gffx_cuda_face_geometry_f32 = NULL;
static CUfunction gffx_cuda_face_geometry_f64 = NULL;
static CUfunction gffx_cuda_validate_faces = NULL;

static gffx_status gffx_cuda_ensure_module(gffx_diagnostic_buffer *diagnostic) {
    CUcontext current = NULL;
    if (cuCtxGetCurrent(&current) != CUDA_SUCCESS || current == NULL) {
        return gffx_cuda_fail(diagnostic, GFFX_STATUS_BACKEND_FAILURE,
                              "no current CUDA context; the caller must establish one");
    }
    if (gffx_cuda_module != NULL && gffx_cuda_module_context == current) {
        return GFFX_STATUS_OK;
    }
    if (cuModuleLoadData(&gffx_cuda_module, gffx_cuda_embedded_ptx) != CUDA_SUCCESS) {
        gffx_cuda_module = NULL;
        return gffx_cuda_fail(diagnostic, GFFX_STATUS_BACKEND_FAILURE,
                              "the driver could not load the embedded PTX module; it is likely "
                              "older than the ISA this plugin was built with");
    }
    if (cuModuleGetFunction(&gffx_cuda_face_geometry_f32, gffx_cuda_module,
                            "gffx_cuda_face_geometry_f32") != CUDA_SUCCESS ||
        cuModuleGetFunction(&gffx_cuda_face_geometry_f64, gffx_cuda_module,
                            "gffx_cuda_face_geometry_f64") != CUDA_SUCCESS ||
        cuModuleGetFunction(&gffx_cuda_validate_faces, gffx_cuda_module,
                            "gffx_cuda_validate_faces") != CUDA_SUCCESS) {
        cuModuleUnload(gffx_cuda_module);
        gffx_cuda_module = NULL;
        return gffx_cuda_fail(diagnostic, GFFX_STATUS_INTERNAL_ERROR,
                              "the embedded PTX module is missing an expected kernel");
    }
    gffx_cuda_module_context = current;
    return GFFX_STATUS_OK;
}

/* Structural validation only.
 *
 * Shapes, dtypes, ranks and null checks are properties of the view and are checked here. Index
 * range is not: those values live in device memory and this is host code. The non-skippable
 * per-call check that EXECUTION_STATE_CONTRACT_V0_1.md requires therefore has no implementation on
 * this path yet, which is an open contract question rather than an oversight, and the kernel does
 * not quietly tolerate a bad index in the meantime.
 */
static gffx_status gffx_cuda_check_device_view(
    const gffx_tensor_view *view, uint32_t rank, gffx_dtype dtype, const char *what,
    gffx_diagnostic_buffer *diagnostic
) {
    if (view == NULL || view->struct_size < sizeof(*view) ||
        view->abi_version != GFFX_ABI_VERSION) {
        return gffx_cuda_fail(diagnostic, GFFX_STATUS_INVALID_ARGUMENT, what);
    }
    if (view->device_type != GFFX_DEVICE_CUDA || view->data == NULL ||
        view->rank != rank || view->dtype != dtype) {
        return gffx_cuda_fail(diagnostic, GFFX_STATUS_INVALID_ARGUMENT, what);
    }
    return GFFX_STATUS_OK;
}

/*
 * Device workspace requirements.
 *
 * Four bytes, for the index-validation status word. The CPU reference for the same operation
 * requires zero, which is precisely why the plugin publishes its own query rather than the host
 * reusing the CPU figure: a device implementation's scratch is its own business, and here the
 * difference is the cost of enforcing a contract the host cannot enforce for device memory.
 */
static gffx_status GFFX_CALL gffx_cuda_op_workspace(
    uint32_t operation, const int64_t *shape, uint32_t shape_count, gffx_dtype dtype,
    const gffx_execution_context *context, uint64_t *required_bytes,
    uint64_t *required_alignment, gffx_diagnostic_buffer *diagnostic
) {
    (void)shape;
    (void)shape_count;
    (void)dtype;
    (void)context;
    if (required_bytes == NULL || required_alignment == NULL) {
        return gffx_cuda_fail(diagnostic, GFFX_STATUS_INVALID_ARGUMENT,
                              "workspace query result pointers must not be null");
    }
    switch (operation) {
        case GFFX_CUDA_OP_MESH_FACE_GEOMETRY:
            *required_bytes = sizeof(int);
            *required_alignment = sizeof(int);
            return GFFX_STATUS_OK;
        default:
            return gffx_cuda_fail(diagnostic, GFFX_STATUS_UNSUPPORTED,
                                  "this plugin implements no such operation");
    }
}

static gffx_status GFFX_CALL gffx_cuda_op_face_geometry(
    const gffx_tensor_view *vertices, const gffx_tensor_view *faces, double eps,
    const gffx_execution_context *context, gffx_tensor_view *unit_normals,
    gffx_tensor_view *areas, gffx_tensor_view *valid, const gffx_buffer *workspace,
    gffx_diagnostic_buffer *diagnostic
) {
    gffx_status status;
    CUfunction kernel;
    CUstream stream;
    long long face_count;
    void *arguments[7];
    unsigned int blocks;
    const unsigned int threads = 256u;
    gffx_dtype dtype;

    if (context == NULL || context->struct_size < sizeof(*context) ||
        context->device_type != GFFX_DEVICE_CUDA) {
        return gffx_cuda_fail(diagnostic, GFFX_STATUS_INVALID_ARGUMENT,
                              "face_geometry on this backend requires a CUDA execution context");
    }
    if (vertices == NULL || vertices->struct_size < sizeof(*vertices)) {
        return gffx_cuda_fail(diagnostic, GFFX_STATUS_INVALID_ARGUMENT,
                              "vertices must be a [V,3] tensor view");
    }
    dtype = vertices->dtype;
    if (dtype != GFFX_DTYPE_FLOAT32 && dtype != GFFX_DTYPE_FLOAT64) {
        return gffx_cuda_fail(diagnostic, GFFX_STATUS_INVALID_ARGUMENT,
                              "vertices must use float32 or float64");
    }
    if (!(eps >= 0.0) || eps != eps) {
        return gffx_cuda_fail(diagnostic, GFFX_STATUS_INVALID_ARGUMENT,
                              "eps must be finite and non-negative");
    }

    status = gffx_cuda_check_device_view(vertices, 2u, dtype, "vertices must be a [V,3] device view",
                                         diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    status = gffx_cuda_check_device_view(faces, 2u, GFFX_DTYPE_INT32,
                                         "faces must be an int32 [F,3] device view", diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    status = gffx_cuda_check_device_view(unit_normals, 2u, dtype,
                                         "unit_normals must be a [F,3] device view", diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    status = gffx_cuda_check_device_view(areas, 1u, dtype, "areas must be an [F] device view",
                                         diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    status = gffx_cuda_check_device_view(valid, 1u, GFFX_DTYPE_BOOL,
                                         "valid must be a bool [F] device view", diagnostic);
    if (status != GFFX_STATUS_OK) return status;

    face_count = (long long)faces->shape[0];
    if (unit_normals->shape[0] != faces->shape[0] || areas->shape[0] != faces->shape[0] ||
        valid->shape[0] != faces->shape[0]) {
        return gffx_cuda_fail(diagnostic, GFFX_STATUS_INVALID_ARGUMENT,
                              "outputs must be sized from the face count");
    }
    if (face_count == 0) return GFFX_STATUS_OK;

    status = gffx_cuda_ensure_module(diagnostic);
    if (status != GFFX_STATUS_OK) return status;

    /* The contract's non-skippable index check, moved to the device because the host cannot read
     * device memory. It runs on the caller's stream before the operation, and the host reads the
     * result, which costs a synchronisation this backend cannot avoid: reporting an error through
     * a synchronous return value requires knowing the answer before returning. That cost is the
     * honest price of the guarantee rather than a reason to drop it. */
    {
        int host_status = 0;
        long long index_count = face_count * 3;
        long long vertex_count = (long long)vertices->shape[0];
        int vertex_count_arg = (int)vertex_count;
        CUdeviceptr status_word;
        void *validate_arguments[4];
        unsigned int validate_blocks;

        if (workspace == NULL || workspace->data == NULL ||
            workspace->capacity_bytes < sizeof(int)) {
            return gffx_cuda_fail(diagnostic, GFFX_STATUS_INSUFFICIENT_WORKSPACE,
                                  "the CUDA backend requires the workspace its query reports, "
                                  "which holds the index-validation status word");
        }
        if (vertex_count > 2147483647LL) {
            return gffx_cuda_fail(diagnostic, GFFX_STATUS_OVERFLOW,
                                  "vertex count exceeds the int32 index range");
        }
        status_word = (CUdeviceptr)(uintptr_t)workspace->data;
        if (cuMemsetD32Async(status_word, 0u, 1u, (CUstream)context->stream) != CUDA_SUCCESS) {
            return gffx_cuda_fail(diagnostic, GFFX_STATUS_BACKEND_FAILURE,
                                  "could not clear the validation status word");
        }
        validate_arguments[0] = (void *)&faces->data;
        validate_arguments[1] = (void *)&index_count;
        validate_arguments[2] = (void *)&vertex_count_arg;
        validate_arguments[3] = (void *)&workspace->data;
        validate_blocks = (unsigned int)((index_count + 255) / 256);
        if (cuLaunchKernel(gffx_cuda_validate_faces, validate_blocks, 1u, 1u, 256u, 1u, 1u, 0u,
                           (CUstream)context->stream, validate_arguments, NULL) != CUDA_SUCCESS) {
            return gffx_cuda_fail(diagnostic, GFFX_STATUS_BACKEND_FAILURE,
                                  "the index validation kernel failed to launch");
        }
        if (cuMemcpyDtoHAsync(&host_status, status_word, sizeof(int),
                              (CUstream)context->stream) != CUDA_SUCCESS ||
            cuStreamSynchronize((CUstream)context->stream) != CUDA_SUCCESS) {
            return gffx_cuda_fail(diagnostic, GFFX_STATUS_BACKEND_FAILURE,
                                  "could not read the validation status word");
        }
        if (host_status != 0) {
            /* No output is written: the operation kernel is never launched. */
            return gffx_cuda_fail(diagnostic, GFFX_STATUS_INVALID_ARGUMENT,
                                  "every face index must lie in [0, V)");
        }
    }

    kernel = (dtype == GFFX_DTYPE_FLOAT64) ? gffx_cuda_face_geometry_f64
                                           : gffx_cuda_face_geometry_f32;
    /* The caller's stream, taken from the execution context. The plugin creates no stream of its
     * own and inserts no synchronisation, so ordering stays the caller's to control. */
    stream = (CUstream)context->stream;
    arguments[0] = (void *)&vertices->data;
    arguments[1] = (void *)&faces->data;
    arguments[2] = (void *)&eps;
    arguments[3] = (void *)&face_count;
    arguments[4] = (void *)&unit_normals->data;
    arguments[5] = (void *)&areas->data;
    arguments[6] = (void *)&valid->data;
    blocks = (unsigned int)((face_count + (long long)threads - 1) / (long long)threads);
    if (cuLaunchKernel(kernel, blocks, 1u, 1u, threads, 1u, 1u, 0u, stream, arguments, NULL)
        != CUDA_SUCCESS) {
        return gffx_cuda_fail(diagnostic, GFFX_STATUS_BACKEND_FAILURE,
                              "the face_geometry kernel failed to launch");
    }
    return GFFX_STATUS_OK;
}

/*
 * The published operation table.
 *
 * Every entry is NULL at this point, which is the honest state: the dispatch path exists and is
 * negotiated, and no CUDA kernel is implemented yet. A NULL entry means unsupported, so a caller
 * asking for a CUDA operation today receives GFFX_STATUS_UNSUPPORTED rather than a silent CPU
 * fallback. Entries are filled in one at a time as kernels land, and nothing outside this table
 * changes when they do.
 *
 * const, so it lives in read-only storage and the file-scope-mutable-state rule is satisfied.
 */
static const gffx_cuda_operations gffx_cuda_operation_table = {
    (uint32_t)sizeof(gffx_cuda_operations),
    0u,
    gffx_cuda_op_workspace,
    gffx_cuda_op_face_geometry, NULL,   /* mesh.face_geometry; backward not yet implemented */
    NULL, NULL,   /* mesh.vertex_normals, backward */
    NULL, NULL,   /* mesh.gather_faces, backward */
    NULL, NULL,   /* transforms.transform_points, backward */
    NULL, NULL,   /* transforms.perspective_divide, backward */
    NULL,         /* mesh.build_edge_topology, no backward by contract */
    NULL, NULL,   /* points.knn, backward */
    NULL, NULL,   /* points.closest_point_on_mesh, backward */
    NULL, NULL,   /* mesh.sample_surface, backward */
    NULL, NULL,   /* render.rasterize, backward */
    NULL, NULL,   /* render.interpolate, backward */
    {0, 0, 0, 0}
};

GFFX_CUDA_PLUGIN_API gffx_status GFFX_CALL gffx_cuda_plugin_handshake_v1(
    uint32_t requested_plugin_abi,
    uint32_t host_core_abi,
    gffx_cuda_plugin_api *api,
    gffx_diagnostic_buffer *diagnostic
) {
    if (api == NULL || api->struct_size < sizeof(*api)) {
        gffx_cuda_diagnostic(diagnostic, "CUDA plugin API buffer is too small");
        return GFFX_STATUS_INVALID_ARGUMENT;
    }
    if (GFFX_ABI_VERSION_MAJOR(requested_plugin_abi) !=
        GFFX_CUDA_PLUGIN_ABI_VERSION_MAJOR) {
        gffx_cuda_diagnostic(diagnostic, "CUDA plugin ABI 1.0 is required");
        return GFFX_STATUS_ABI_MISMATCH;
    }
    if (host_core_abi != GFFX_ABI_VERSION) {
        gffx_cuda_diagnostic(diagnostic, "GFFX core ABI 1.0 is required");
        return GFFX_STATUS_ABI_MISMATCH;
    }
    memset(api, 0, sizeof(*api));
    api->struct_size = (uint32_t)sizeof(*api);
    api->plugin_abi_version = GFFX_CUDA_PLUGIN_ABI_VERSION;
    api->core_abi_min = GFFX_ABI_VERSION;
    api->core_abi_max = GFFX_ABI_VERSION;
    api->flags = GFFX_CUDA_PLUGIN_FLAG_CAPABILITY_PROVIDER |
                 GFFX_CUDA_PLUGIN_FLAG_OPERATION_PROVIDER;
    api->build_identity = GFFX_CUDA_BUILD_ID;
    api->capabilities_probe = gffx_cuda_capabilities;
    api->operations = &gffx_cuda_operation_table;
    return GFFX_STATUS_OK;
}
