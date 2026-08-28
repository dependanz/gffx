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
 * No operation-dispatch accessor is published yet, and the omission is deliberate.
 *
 * The negotiated table lives in the plugin's address space, so a pointer to it is valid only while
 * the library stays loaded. gffx_cuda_loader_probe holds its state in a local and does not keep the
 * plugin mapped, so returning that pointer would hand out a dangling one the moment the probe
 * returns. Making it valid means keeping the plugin loaded for the process lifetime, which is
 * exactly the hidden process-wide state this scaffold is built to avoid, and that is a decision to
 * take openly rather than as a side effect of adding dispatch. It is recorded as an unresolved
 * decision in the project plan.
 *
 * The ABI carries operations regardless: the table is negotiated and validated during the probe,
 * so a plugin can publish kernels and the host can already tell whether it did.
 */

#endif
