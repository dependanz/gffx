/*
 * Native file layer for the PLY reader.
 *
 * This deliberately lives outside native/core. The geometry scaffold is allocation-free and free
 * of process-wide I/O, and runtime.dependency_inspection enforces that; reading a file needs the
 * platform C library, exactly as the CUDA probe loader does. Rather than either forbidding file
 * access or quietly weakening the core gate, this module is excluded from that gate and inspected
 * separately by io_isolation.cmake, which permits stdio and the allocator but still forbids
 * getenv, system, threads, process-wide mutable state, and any framework dependency.
 *
 * The parser itself stays in the core and operates on bytes, so a memory-mapped file, an archive
 * member and a network buffer all share one implementation.
 */

#include <gffx/execution.h>
#include <gffx/io.h>
#include <gffx/status.h>
#include <gffx/tensor.h>

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* A template large enough to exhaust addressable memory is refused rather than truncated. */
#define GFFX_IO_MAX_FILE_BYTES ((int64_t)1 << 34)

static int gffx_io_buffer_valid(const gffx_buffer *buffer) {
    return buffer != NULL && buffer->struct_size == (uint32_t)sizeof(gffx_buffer);
}

GFFX_API gffx_status GFFX_CALL gffx_io_file_read_all(
    const char *path,
    const gffx_execution_context *context,
    gffx_buffer *buffer,
    gffx_diagnostic_buffer *diagnostic
) {
    FILE *handle;
    long end;
    int64_t size;
    void *data;
    size_t read_bytes;
    gffx_status status;

    if (path == NULL || !gffx_io_buffer_valid(buffer)) return GFFX_STATUS_INVALID_ARGUMENT;
    if (context != NULL) {
        status = gffx_validate_execution_context(context, diagnostic);
        if (status != GFFX_STATUS_OK) return status;
    }

    buffer->data = NULL;
    buffer->capacity_bytes = 0u;
    buffer->alignment = 1u;
    buffer->device_type = GFFX_DEVICE_CPU;
    buffer->device_index = 0;

    handle = fopen(path, "rb");
    if (handle == NULL) return GFFX_STATUS_BACKEND_FAILURE;
    if (fseek(handle, 0, SEEK_END) != 0) {
        fclose(handle);
        return GFFX_STATUS_BACKEND_FAILURE;
    }
    end = ftell(handle);
    if (end < 0) {
        fclose(handle);
        return GFFX_STATUS_BACKEND_FAILURE;
    }
    size = (int64_t)end;
    if (size > GFFX_IO_MAX_FILE_BYTES) {
        fclose(handle);
        return GFFX_STATUS_OVERFLOW;
    }
    if (fseek(handle, 0, SEEK_SET) != 0) {
        fclose(handle);
        return GFFX_STATUS_BACKEND_FAILURE;
    }

    /* One extra byte so the allocation is never zero-sized, which malloc may legally refuse. */
    data = malloc((size_t)size + 1u);
    if (data == NULL) {
        fclose(handle);
        return GFFX_STATUS_BACKEND_FAILURE;
    }
    read_bytes = (size > 0) ? fread(data, 1, (size_t)size, handle) : (size_t)0;
    fclose(handle);
    if (read_bytes != (size_t)size) {
        /* A short read means the file changed under us or the stream failed; either way the
         * bytes cannot be trusted, so nothing is handed back. */
        free(data);
        return GFFX_STATUS_BACKEND_FAILURE;
    }

    buffer->data = data;
    buffer->capacity_bytes = (uint64_t)size;
    return GFFX_STATUS_OK;
}

GFFX_API gffx_status GFFX_CALL gffx_io_file_release(gffx_buffer *buffer) {
    if (!gffx_io_buffer_valid(buffer)) return GFFX_STATUS_INVALID_ARGUMENT;
    /* Clearing the descriptor is what makes a second release safe rather than a double free. */
    if (buffer->data != NULL) free(buffer->data);
    buffer->data = NULL;
    buffer->capacity_bytes = 0u;
    return GFFX_STATUS_OK;
}

GFFX_API gffx_status GFFX_CALL gffx_io_ply_probe_file(
    const char *path,
    const gffx_execution_context *context,
    gffx_ply_header *header,
    gffx_diagnostic_buffer *diagnostic
) {
    gffx_buffer buffer = {0};
    gffx_status status;

    buffer.struct_size = (uint32_t)sizeof(buffer);
    buffer.abi_version = GFFX_ABI_VERSION;
    status = gffx_io_file_read_all(path, context, &buffer, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    status = gffx_io_ply_probe(buffer.data, (int64_t)buffer.capacity_bytes, context, header,
                               diagnostic);
    gffx_io_file_release(&buffer);
    return status;
}

GFFX_API gffx_status GFFX_CALL gffx_io_ply_read_file(
    const char *path,
    const gffx_execution_context *context,
    gffx_ply_header *header,
    gffx_tensor_view *vertices,
    gffx_tensor_view *faces,
    gffx_diagnostic_buffer *diagnostic
) {
    gffx_buffer buffer = {0};
    gffx_ply_header local;
    gffx_status status;

    if (header == NULL) return GFFX_STATUS_INVALID_ARGUMENT;
    buffer.struct_size = (uint32_t)sizeof(buffer);
    buffer.abi_version = GFFX_ABI_VERSION;
    status = gffx_io_file_read_all(path, context, &buffer, diagnostic);
    if (status != GFFX_STATUS_OK) return status;

    /* Re-probe the bytes actually read rather than trusting the caller's header, then hand the
     * caller back what this file says, so a template that changed since the probe is caught here
     * instead of silently filling the wrong-sized outputs. */
    local = *header;
    local.struct_size = (uint32_t)sizeof(local);
    local.abi_version = GFFX_ABI_VERSION;
    status = gffx_io_ply_probe(buffer.data, (int64_t)buffer.capacity_bytes, context, &local,
                               diagnostic);
    if (status == GFFX_STATUS_OK) {
        status = gffx_io_ply_read(buffer.data, (int64_t)buffer.capacity_bytes, &local, context,
                                  vertices, faces, NULL, diagnostic);
        if (status == GFFX_STATUS_OK) *header = local;
    }
    gffx_io_file_release(&buffer);
    return status;
}
