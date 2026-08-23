#ifndef GFFX_CORE_SELF_TEST_H
#define GFFX_CORE_SELF_TEST_H

/*
 * Private calling-path self-test (Phase 1 Step 5).
 *
 * This header is deliberately NOT installed under include/gffx/, and the entry point it declares
 * is NOT part of the GFFX native ABI v1 public surface. It is compiled only into a private
 * self-test build of the core (GFFX_ENABLE_SELF_TEST) and is absent from the default gffx_core
 * shared library.
 *
 * The self-test proves that the exported calling path and the packaged shared library load and
 * run. It carries no public API, correctness, or performance claim, implements no graphics
 * operation, allocates nothing, and starts no thread. All storage it uses is caller-visible or
 * automatic.
 */

#include <gffx/capabilities.h>
#include <gffx/execution.h>
#include <gffx/status.h>
#include <gffx/tensor.h>

#define GFFX_SELF_TEST_ABI_VERSION UINT32_C(0x00000001)
#define GFFX_SELF_TEST_TENSOR_ACCEPT UINT32_C(0x00000002)
#define GFFX_SELF_TEST_TENSOR_REJECT UINT32_C(0x00000004)
#define GFFX_SELF_TEST_EXECUTION_ACCEPT UINT32_C(0x00000008)
#define GFFX_SELF_TEST_BUFFER_ACCEPT UINT32_C(0x00000010)
#define GFFX_SELF_TEST_DIAGNOSTIC_NULL UINT32_C(0x00000020)
#define GFFX_SELF_TEST_DIAGNOSTIC_TRUNCATION UINT32_C(0x00000040)
#define GFFX_SELF_TEST_CAPABILITY_TWO_PASS UINT32_C(0x00000080)

#define GFFX_SELF_TEST_ALL          \
    (GFFX_SELF_TEST_ABI_VERSION |   \
     GFFX_SELF_TEST_TENSOR_ACCEPT | \
     GFFX_SELF_TEST_TENSOR_REJECT | \
     GFFX_SELF_TEST_EXECUTION_ACCEPT | \
     GFFX_SELF_TEST_BUFFER_ACCEPT | \
     GFFX_SELF_TEST_DIAGNOSTIC_NULL | \
     GFFX_SELF_TEST_DIAGNOSTIC_TRUNCATION | \
     GFFX_SELF_TEST_CAPABILITY_TWO_PASS)

GFFX_EXTERN_C_BEGIN

/*
 * Runs the private calling-path checks. Writes the bitmask of passed checks to out_checks when it
 * is non-null. Returns GFFX_STATUS_OK only when every check in GFFX_SELF_TEST_ALL passed.
 */
GFFX_API gffx_status GFFX_CALL gffx_private_self_test(
    uint32_t *out_checks,
    gffx_diagnostic_buffer *diagnostic
);

GFFX_EXTERN_C_END

#endif /* GFFX_CORE_SELF_TEST_H */
