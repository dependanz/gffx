#ifndef GFFX_CORE_CUDA_LOADER_H
#define GFFX_CORE_CUDA_LOADER_H

#include <gffx/capabilities.h>

gffx_status gffx_cuda_loader_probe(
    uint32_t probe_flags,
    gffx_capability_report *report,
    gffx_diagnostic_buffer *diagnostic
);

#endif
