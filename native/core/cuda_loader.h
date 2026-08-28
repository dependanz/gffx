#ifndef GFFX_CORE_CUDA_LOADER_H
#define GFFX_CORE_CUDA_LOADER_H

#include <gffx/capabilities.h>

#include "../cuda/plugin_api.h"

gffx_status gffx_cuda_loader_probe(
    uint32_t probe_flags,
    gffx_capability_report *report,
    gffx_diagnostic_buffer *diagnostic
);

/*
 * The negotiated operation table, or NULL when no plugin is present, it publishes no operations,
 * or its table is too small to trust.
 *
 * The plugin is loaded once, lazily, on the first call and then stays mapped for the life of the
 * process. That is deliberate and is the one place GFFX keeps state between calls. The table lives
 * in the plugin's address space, so a pointer into it is valid only while the library is loaded;
 * unloading after each use would mean re-loading and re-JIT-compiling the embedded PTX on every
 * operation, which is not a cost a frame loop can absorb.
 *
 * It is confined here on purpose. runtime.dependency_inspection excludes this translation unit
 * precisely because plugin loading needs platform facilities and cannot be stateless, and nothing
 * outside it gains persistent state. Loading never happens implicitly: this function is called
 * only when an operation has actually been asked for on a CUDA device.
 */
const gffx_cuda_operations *gffx_cuda_loader_operations(void);

#endif
