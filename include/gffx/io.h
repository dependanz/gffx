#ifndef GFFX_IO_H
#define GFFX_IO_H

#include <gffx/execution.h>
#include <gffx/tensor.h>

/*
 * io.ply - optional triangle-template reader for the migration corpus.
 *
 * This reads the PLY subset the migration corpus requires: ASCII and binary little-endian files
 * whose vertex element carries x, y and z, and whose face element carries a triangular
 * vertex_indices list. It is deliberately not a general PLY implementation. A file this subset
 * cannot represent is refused with a precise status, because a mesh that loads wrong produces
 * plausible geometry and wrong results, which is worse than a mesh that does not load.
 *
 * The core performs no file I/O. Both entry points take a caller-owned byte buffer; opening,
 * mapping and reading files belongs to the adapter or the calling application. The reader is
 * therefore usable over a memory-mapped file, an archive member, a network buffer, or a test
 * fixture without a code change.
 *
 * Use is two-pass, because allocation belongs to the caller and counts must be known before
 * buffers exist:
 *
 *   1. gffx_io_ply_probe parses only the header and reports the format, the element counts, and
 *      the offset of the first body byte.
 *   2. The caller allocates vertices[V,3] and faces[F,3].
 *   3. gffx_io_ply_read parses the body into those views. It re-reads the header rather than
 *      trusting the caller's copy for anything it can verify, and rejects a header whose counts
 *      disagree with the supplied output shapes.
 *
 * Property order is not assumed: x, y and z are located by name and may appear at any offset in
 * any order, with arbitrary skipped properties before, between and after them. Element order is
 * not assumed either, so face may precede vertex, and unrelated elements are skipped.
 *
 * ASCII conversion is implemented in-tree rather than through strtod, which the runtime
 * dependency gate excludes. It is exact when the significand fits in 53 bits and the decimal
 * exponent lies in [-22, 22], where the significand and the power of ten are both exactly
 * representable and a single scaling step is correctly rounded; outside that range it is within
 * 2 ulp. Binary little-endian conversion is exact by construction, and is performed by explicit
 * byte assembly rather than by casting the buffer, because the buffer carries no alignment
 * guarantee and the byte order is a property of the format rather than of the host.
 *
 * Indices are checked for int32 range and for being non-negative, but are NOT checked against
 * vertex_count. That is mesh.validate's responsibility, and duplicating it here would create two
 * places that can disagree.
 *
 * On failure no output element is written, so a caller cannot mistake stale buffer contents for
 * parsed geometry. GFFX_STATUS_UNSUPPORTED means a well-formed PLY outside this subset;
 * GFFX_STATUS_INVALID_ARGUMENT means a malformed file, a truncated buffer, or a bad argument.
 */

GFFX_EXTERN_C_BEGIN

#define GFFX_PLY_FORMAT_ASCII                0u
#define GFFX_PLY_FORMAT_BINARY_LITTLE_ENDIAN 1u

typedef struct gffx_ply_header {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t format;
    uint32_t reserved;
    int64_t vertex_count;
    int64_t face_count;
    /* Byte offset of the first body byte, immediately past the end_header line terminator. */
    int64_t data_offset;
} gffx_ply_header;

GFFX_API gffx_status GFFX_CALL gffx_io_ply_probe(
    const void *bytes,
    int64_t length,
    const gffx_execution_context *context,
    gffx_ply_header *header,
    gffx_diagnostic_buffer *diagnostic
);

GFFX_API gffx_status GFFX_CALL gffx_io_ply_read_workspace(
    const gffx_ply_header *header,
    const gffx_execution_context *context,
    uint64_t *required_bytes,
    uint64_t *required_alignment,
    gffx_diagnostic_buffer *diagnostic
);

GFFX_API gffx_status GFFX_CALL gffx_io_ply_read(
    const void *bytes,
    int64_t length,
    const gffx_ply_header *header,
    const gffx_execution_context *context,
    gffx_tensor_view *vertices,
    gffx_tensor_view *faces,
    const gffx_buffer *workspace,
    gffx_diagnostic_buffer *diagnostic
);

GFFX_EXTERN_C_END

#endif /* GFFX_IO_H */
