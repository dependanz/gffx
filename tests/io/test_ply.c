/*
 * Phase 2 Step 3 acceptance fixtures PLY-01..PLY-16 for the optional io.ply triangle-template
 * reader. Fixture numbers match PLY_ACCEPTANCE_V0_1.md in the project record.
 *
 * Every fixture buffer is generated here from committed coordinate literals rather than checked
 * in as a binary asset. That is a redistribution rule before it is a testing preference: the
 * upstream migration corpus commits restricted-rights PLY assets, and a generated octahedron
 * carries no third-party rights at all, so this corpus cannot inherit a licensing question from
 * the corpus it exists to replace.
 *
 * Unlike the core, this test may allocate and may use the hosted C library.
 */

#include <gffx/execution.h>
#include <gffx/io.h>
#include <gffx/status.h>
#include <gffx/tensor.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (0)

static const int64_t pair_strides[2] = {3, 1};

/* The CT-SYN-1 octahedron. Every coordinate is exactly representable, so an ASCII round trip is
 * bit-exact and a float32 output is exact rather than merely close. */
static const double OCTA_VERTICES[18] = {
     1.0,  0.0,  0.0,
    -1.0,  0.0,  0.0,
     0.0,  1.0,  0.0,
     0.0, -1.0,  0.0,
     0.0,  0.0,  1.0,
     0.0,  0.0, -1.0
};
static const int32_t OCTA_FACES[24] = {
    0, 2, 4,   2, 1, 4,   1, 3, 4,   3, 0, 4,
    2, 0, 5,   1, 2, 5,   3, 1, 5,   0, 3, 5
};

/* ---------------------------------------------------------------------- byte buffer builder */

typedef struct {
    unsigned char *data;
    int64_t length;
    int64_t capacity;
} bytebuf;

static void bb_init(bytebuf *b) {
    b->capacity = 256;
    b->length = 0;
    b->data = (unsigned char *)malloc((size_t)b->capacity);
}

static void bb_free(bytebuf *b) {
    free(b->data);
    b->data = NULL;
    b->length = 0;
    b->capacity = 0;
}

static void bb_raw(bytebuf *b, const void *source, int64_t count) {
    if (b->length + count > b->capacity) {
        while (b->length + count > b->capacity) b->capacity *= 2;
        b->data = (unsigned char *)realloc(b->data, (size_t)b->capacity);
    }
    memcpy(b->data + b->length, source, (size_t)count);
    b->length += count;
}

static void bb_text(bytebuf *b, const char *text) {
    bb_raw(b, text, (int64_t)strlen(text));
}

/* Emitted little-endian by explicit byte assembly, so the fixture bytes are the same on a
 * big-endian host and the test exercises the reader's byte order rather than the host's. */
static void bb_u8(bytebuf *b, unsigned value) {
    unsigned char byte = (unsigned char)value;
    bb_raw(b, &byte, 1);
}

static void bb_u32le(bytebuf *b, uint32_t value) {
    unsigned char bytes[4];
    bytes[0] = (unsigned char)(value & 0xffu);
    bytes[1] = (unsigned char)((value >> 8) & 0xffu);
    bytes[2] = (unsigned char)((value >> 16) & 0xffu);
    bytes[3] = (unsigned char)((value >> 24) & 0xffu);
    bb_raw(b, bytes, 4);
}

static void bb_u64le(bytebuf *b, uint64_t value) {
    unsigned char bytes[8];
    int i;
    for (i = 0; i < 8; ++i) bytes[i] = (unsigned char)((value >> (8 * i)) & 0xffu);
    bb_raw(b, bytes, 8);
}

static void bb_f32le(bytebuf *b, double value) {
    float narrowed = (float)value;
    uint32_t bits;
    memcpy(&bits, &narrowed, 4);
    bb_u32le(b, bits);
}

static void bb_f64le(bytebuf *b, double value) {
    uint64_t bits;
    memcpy(&bits, &value, 8);
    bb_u64le(b, bits);
}

static void bb_i32le(bytebuf *b, int32_t value) {
    bb_u32le(b, (uint32_t)value);
}

/* %.17g round-trips a double exactly, and prints an exactly representable coordinate in its
 * shortest faithful form, so the ASCII fixture and the binary fixture describe the same value. */
static void bb_double(bytebuf *b, double value) {
    char text[40];
    sprintf(text, "%.17g", value);
    bb_text(b, text);
}

static void bb_int(bytebuf *b, long value) {
    char text[32];
    sprintf(text, "%ld", value);
    bb_text(b, text);
}

/* -------------------------------------------------------------------------- fixture writers */

typedef struct {
    int binary;          /* 0 ASCII, 1 binary little-endian */
    int double_vertices; /* emit double rather than float vertex properties */
    int extra_before;    /* skipped scalar properties before x */
    int extra_between;   /* skipped scalar property between y and z */
    int extra_after;     /* skipped scalar properties after z */
    int face_first;      /* declare the face element before the vertex element */
    int third_element;   /* declare and emit an unrelated element */
    int crlf;            /* terminate header lines with CRLF */
    int quad_face;       /* emit face 0 with four indices */
    int big_endian;      /* declare binary_big_endian */
    int64_t vertex_count;
    int64_t face_count;
} ply_options;

static ply_options default_options(void) {
    ply_options options;
    memset(&options, 0, sizeof(options));
    options.vertex_count = 6;
    options.face_count = 8;
    return options;
}

static void emit_line(bytebuf *b, const ply_options *options, const char *text) {
    bb_text(b, text);
    bb_text(b, options->crlf ? "\r\n" : "\n");
}

static void emit_vertex_element(bytebuf *b, const ply_options *options) {
    const char *scalar = options->double_vertices ? "double" : "float";
    char line[96];
    int i;
    sprintf(line, "element vertex %ld", (long)options->vertex_count);
    emit_line(b, options, line);
    for (i = 0; i < options->extra_before; ++i) emit_line(b, options, "property uchar quality");
    sprintf(line, "property %s x", scalar);
    emit_line(b, options, line);
    sprintf(line, "property %s y", scalar);
    emit_line(b, options, line);
    if (options->extra_between) emit_line(b, options, "property int marker");
    sprintf(line, "property %s z", scalar);
    emit_line(b, options, line);
    for (i = 0; i < options->extra_after; ++i) emit_line(b, options, "property uchar red");
}

static void emit_face_element(bytebuf *b, const ply_options *options) {
    char line[64];
    sprintf(line, "element face %ld", (long)options->face_count);
    emit_line(b, options, line);
    emit_line(b, options, "property list uchar int vertex_indices");
}

static void emit_vertex_body(bytebuf *b, const ply_options *options) {
    int64_t v;
    int i;
    for (v = 0; v < options->vertex_count; ++v) {
        if (options->binary) {
            for (i = 0; i < options->extra_before; ++i) bb_u8(b, 7u);
            if (options->double_vertices) {
                bb_f64le(b, OCTA_VERTICES[v * 3 + 0]);
                bb_f64le(b, OCTA_VERTICES[v * 3 + 1]);
                if (options->extra_between) bb_i32le(b, -5);
                bb_f64le(b, OCTA_VERTICES[v * 3 + 2]);
            } else {
                bb_f32le(b, OCTA_VERTICES[v * 3 + 0]);
                bb_f32le(b, OCTA_VERTICES[v * 3 + 1]);
                if (options->extra_between) bb_i32le(b, -5);
                bb_f32le(b, OCTA_VERTICES[v * 3 + 2]);
            }
            for (i = 0; i < options->extra_after; ++i) bb_u8(b, 200u);
        } else {
            for (i = 0; i < options->extra_before; ++i) bb_text(b, "7 ");
            bb_double(b, OCTA_VERTICES[v * 3 + 0]);
            bb_text(b, " ");
            bb_double(b, OCTA_VERTICES[v * 3 + 1]);
            bb_text(b, " ");
            if (options->extra_between) bb_text(b, "-5 ");
            bb_double(b, OCTA_VERTICES[v * 3 + 2]);
            for (i = 0; i < options->extra_after; ++i) bb_text(b, " 200");
            bb_text(b, options->crlf ? "\r\n" : "\n");
        }
    }
}

static void emit_face_body(bytebuf *b, const ply_options *options) {
    int64_t f;
    for (f = 0; f < options->face_count; ++f) {
        int arity = (options->quad_face && f == 0) ? 4 : 3;
        int k;
        if (options->binary) {
            bb_u8(b, (unsigned)arity);
            for (k = 0; k < 3; ++k) bb_i32le(b, OCTA_FACES[f * 3 + k]);
            if (arity == 4) bb_i32le(b, OCTA_FACES[f * 3 + 0]);
        } else {
            bb_int(b, arity);
            for (k = 0; k < 3; ++k) {
                bb_text(b, " ");
                bb_int(b, OCTA_FACES[f * 3 + k]);
            }
            if (arity == 4) {
                bb_text(b, " ");
                bb_int(b, OCTA_FACES[f * 3 + 0]);
            }
            bb_text(b, options->crlf ? "\r\n" : "\n");
        }
    }
}

static void emit_third_element(bytebuf *b, const ply_options *options) {
    int64_t i;
    if (!options->third_element) return;
    for (i = 0; i < 2; ++i) {
        if (options->binary) {
            bb_u8(b, 3u);
            bb_f32le(b, 0.5);
        } else {
            emit_line(b, options, "3 0.5");
        }
    }
}

static void build_ply(bytebuf *b, const ply_options *options) {
    bb_init(b);
    emit_line(b, options, "ply");
    if (options->big_endian) emit_line(b, options, "format binary_big_endian 1.0");
    else if (options->binary) emit_line(b, options, "format binary_little_endian 1.0");
    else emit_line(b, options, "format ascii 1.0");
    emit_line(b, options, "comment generated by the gffx acceptance suite");
    emit_line(b, options, "obj_info fixture");
    if (options->face_first) {
        emit_face_element(b, options);
        emit_vertex_element(b, options);
    } else {
        emit_vertex_element(b, options);
        emit_face_element(b, options);
    }
    if (options->third_element) {
        emit_line(b, options, "element edge 2");
        emit_line(b, options, "property uchar kind");
        emit_line(b, options, "property float weight");
    }
    emit_line(b, options, "end_header");
    if (options->face_first) {
        emit_face_body(b, options);
        emit_vertex_body(b, options);
    } else {
        emit_vertex_body(b, options);
        emit_face_body(b, options);
    }
    emit_third_element(b, options);
}

/* ------------------------------------------------------------------------------- harness */

static gffx_execution_context cpu_context(void) {
    gffx_execution_context context = {0};
    context.struct_size = (uint32_t)sizeof(context);
    context.abi_version = GFFX_ABI_VERSION;
    context.device_type = GFFX_DEVICE_CPU;
    context.device_index = 0;
    return context;
}

static gffx_tensor_view make_view(
    void *data, gffx_dtype dtype, uint32_t rank,
    const int64_t *shape, const int64_t *strides, uint32_t flags
) {
    gffx_tensor_view view = {0};
    view.struct_size = (uint32_t)sizeof(view);
    view.abi_version = GFFX_ABI_VERSION;
    view.data = data;
    view.rank = rank;
    view.shape = shape;
    view.strides = strides;
    view.dtype = dtype;
    view.device_type = GFFX_DEVICE_CPU;
    view.device_index = 0;
    view.flags = flags;
    return view;
}

static gffx_ply_header blank_header(void) {
    gffx_ply_header header = {0};
    header.struct_size = (uint32_t)sizeof(header);
    header.abi_version = GFFX_ABI_VERSION;
    return header;
}

/* Reads into caller storage sized from the probe. dtype selects the vertex output type. */
static gffx_status read_octahedron(
    const bytebuf *b, gffx_dtype dtype, void *vertices, int32_t *faces, gffx_ply_header *header
) {
    gffx_execution_context context = cpu_context();
    gffx_diagnostic_buffer diagnostic = {0};
    gffx_tensor_view vertices_view;
    gffx_tensor_view faces_view;
    int64_t vertex_shape[2];
    int64_t face_shape[2];
    gffx_status status;

    diagnostic.struct_size = (uint32_t)sizeof(diagnostic);
    diagnostic.abi_version = GFFX_ABI_VERSION;
    *header = blank_header();
    status = gffx_io_ply_probe(b->data, b->length, &context, header, &diagnostic);
    if (status != GFFX_STATUS_OK) return status;

    vertex_shape[0] = header->vertex_count; vertex_shape[1] = 3;
    face_shape[0] = header->face_count; face_shape[1] = 3;
    vertices_view = make_view(vertices, dtype, 2u, vertex_shape, pair_strides,
                              GFFX_TENSOR_OUTPUT);
    faces_view = make_view(faces, GFFX_DTYPE_INT32, 2u, face_shape, pair_strides,
                           GFFX_TENSOR_OUTPUT);
    return gffx_io_ply_read(b->data, b->length, header, &context, &vertices_view, &faces_view,
                            NULL, &diagnostic);
}

static int faces_match(const int32_t *faces) {
    int i;
    for (i = 0; i < 24; ++i) {
        if (faces[i] != OCTA_FACES[i]) return 0;
    }
    return 1;
}

/* ------------------------------------------------------------------------------ fixtures */

static int test_ply01_ascii_probe(void) {
    bytebuf b;
    gffx_execution_context context = cpu_context();
    gffx_diagnostic_buffer diagnostic = {0};
    gffx_ply_header header = blank_header();
    ply_options options = default_options();
    gffx_status status;

    diagnostic.struct_size = (uint32_t)sizeof(diagnostic);
    diagnostic.abi_version = GFFX_ABI_VERSION;
    build_ply(&b, &options);
    status = gffx_io_ply_probe(b.data, b.length, &context, &header, &diagnostic);
    CHECK(status == GFFX_STATUS_OK);
    CHECK(header.format == GFFX_PLY_FORMAT_ASCII);
    CHECK(header.vertex_count == 6);
    CHECK(header.face_count == 8);
    /* data_offset names the first body byte, so the bytes there begin the first vertex line. */
    CHECK(header.data_offset > 0);
    CHECK(header.data_offset < b.length);
    CHECK(b.data[header.data_offset - 1] == (unsigned char)'\n');
    bb_free(&b);
    return 0;
}

static int test_ply02_ascii_read(void) {
    bytebuf b;
    double vertices[18];
    int32_t faces[24];
    gffx_ply_header header;
    ply_options options = default_options();
    int i;

    build_ply(&b, &options);
    CHECK(read_octahedron(&b, GFFX_DTYPE_FLOAT64, vertices, faces, &header) == GFFX_STATUS_OK);
    for (i = 0; i < 18; ++i) CHECK(vertices[i] == OCTA_VERTICES[i]);
    CHECK(faces_match(faces));
    bb_free(&b);
    return 0;
}

static int test_ply03_binary_read(void) {
    bytebuf ascii;
    bytebuf binary;
    double ascii_vertices[18];
    double binary_vertices[18];
    int32_t ascii_faces[24];
    int32_t binary_faces[24];
    gffx_ply_header ascii_header;
    gffx_ply_header binary_header;
    ply_options options = default_options();

    build_ply(&ascii, &options);
    options.binary = 1;
    build_ply(&binary, &options);

    CHECK(read_octahedron(&ascii, GFFX_DTYPE_FLOAT64, ascii_vertices, ascii_faces,
                          &ascii_header) == GFFX_STATUS_OK);
    CHECK(read_octahedron(&binary, GFFX_DTYPE_FLOAT64, binary_vertices, binary_faces,
                          &binary_header) == GFFX_STATUS_OK);
    CHECK(binary_header.format == GFFX_PLY_FORMAT_BINARY_LITTLE_ENDIAN);
    CHECK(binary_header.vertex_count == 6 && binary_header.face_count == 8);
    /* Bit-identical, not merely close: the coordinates are exactly representable. */
    CHECK(memcmp(ascii_vertices, binary_vertices, sizeof(ascii_vertices)) == 0);
    CHECK(memcmp(ascii_faces, binary_faces, sizeof(ascii_faces)) == 0);
    bb_free(&ascii);
    bb_free(&binary);
    return 0;
}

static int test_ply04_dtypes(void) {
    bytebuf b;
    double wide[18];
    float narrow[18];
    int32_t faces_wide[24];
    int32_t faces_narrow[24];
    gffx_ply_header header;
    ply_options options = default_options();
    int i;

    build_ply(&b, &options);
    CHECK(read_octahedron(&b, GFFX_DTYPE_FLOAT64, wide, faces_wide, &header) == GFFX_STATUS_OK);
    CHECK(read_octahedron(&b, GFFX_DTYPE_FLOAT32, narrow, faces_narrow, &header)
          == GFFX_STATUS_OK);
    for (i = 0; i < 18; ++i) CHECK(narrow[i] == (float)wide[i]);
    bb_free(&b);
    return 0;
}

static int test_ply05_line_endings_and_comments(void) {
    bytebuf plain;
    bytebuf crlf;
    double plain_vertices[18];
    double crlf_vertices[18];
    int32_t plain_faces[24];
    int32_t crlf_faces[24];
    gffx_ply_header header;
    ply_options options = default_options();

    build_ply(&plain, &options);
    options.crlf = 1;
    build_ply(&crlf, &options);
    CHECK(read_octahedron(&plain, GFFX_DTYPE_FLOAT64, plain_vertices, plain_faces, &header)
          == GFFX_STATUS_OK);
    CHECK(read_octahedron(&crlf, GFFX_DTYPE_FLOAT64, crlf_vertices, crlf_faces, &header)
          == GFFX_STATUS_OK);
    CHECK(memcmp(plain_vertices, crlf_vertices, sizeof(plain_vertices)) == 0);
    CHECK(memcmp(plain_faces, crlf_faces, sizeof(plain_faces)) == 0);
    bb_free(&plain);
    bb_free(&crlf);
    return 0;
}

static int test_ply06_extra_properties(void) {
    bytebuf b;
    double vertices[18];
    int32_t faces[24];
    gffx_ply_header header;
    ply_options options = default_options();
    int binary;
    int i;

    for (binary = 0; binary <= 1; ++binary) {
        options = default_options();
        options.binary = binary;
        options.extra_before = 2;
        options.extra_between = 1;
        options.extra_after = 2;
        build_ply(&b, &options);
        CHECK(read_octahedron(&b, GFFX_DTYPE_FLOAT64, vertices, faces, &header)
              == GFFX_STATUS_OK);
        for (i = 0; i < 18; ++i) CHECK(vertices[i] == OCTA_VERTICES[i]);
        CHECK(faces_match(faces));
        bb_free(&b);
    }
    return 0;
}

static int test_ply07_element_order(void) {
    bytebuf b;
    double vertices[18];
    int32_t faces[24];
    gffx_ply_header header;
    ply_options options;
    int binary;
    int i;

    for (binary = 0; binary <= 1; ++binary) {
        options = default_options();
        options.binary = binary;
        options.face_first = 1;
        options.third_element = 1;
        build_ply(&b, &options);
        CHECK(read_octahedron(&b, GFFX_DTYPE_FLOAT64, vertices, faces, &header)
              == GFFX_STATUS_OK);
        for (i = 0; i < 18; ++i) CHECK(vertices[i] == OCTA_VERTICES[i]);
        CHECK(faces_match(faces));
        bb_free(&b);
    }
    return 0;
}

static int test_ply08_quad_face(void) {
    bytebuf b;
    double vertices[18];
    int32_t faces[24];
    gffx_ply_header header;
    ply_options options;
    int binary;

    for (binary = 0; binary <= 1; ++binary) {
        options = default_options();
        options.binary = binary;
        options.quad_face = 1;
        build_ply(&b, &options);
        memset(faces, 0x5a, sizeof(faces));
        CHECK(read_octahedron(&b, GFFX_DTYPE_FLOAT64, vertices, faces, &header)
              == GFFX_STATUS_UNSUPPORTED);
        /* No output written on failure, so the sentinel survives. */
        CHECK(faces[0] == (int32_t)0x5a5a5a5a);
        bb_free(&b);
    }
    return 0;
}

static int test_ply09_big_endian(void) {
    bytebuf b;
    gffx_execution_context context = cpu_context();
    gffx_diagnostic_buffer diagnostic = {0};
    gffx_ply_header header = blank_header();
    ply_options options = default_options();

    diagnostic.struct_size = (uint32_t)sizeof(diagnostic);
    diagnostic.abi_version = GFFX_ABI_VERSION;
    options.binary = 1;
    options.big_endian = 1;
    build_ply(&b, &options);
    CHECK(gffx_io_ply_probe(b.data, b.length, &context, &header, &diagnostic)
          == GFFX_STATUS_UNSUPPORTED);
    bb_free(&b);
    return 0;
}

static int test_ply10_source_width(void) {
    bytebuf single;
    bytebuf twice;
    double from_float[18];
    double from_double[18];
    int32_t faces[24];
    gffx_ply_header header;
    ply_options options = default_options();
    int i;

    options.binary = 1;
    build_ply(&single, &options);
    options.double_vertices = 1;
    build_ply(&twice, &options);
    CHECK(read_octahedron(&single, GFFX_DTYPE_FLOAT64, from_float, faces, &header)
          == GFFX_STATUS_OK);
    CHECK(read_octahedron(&twice, GFFX_DTYPE_FLOAT64, from_double, faces, &header)
          == GFFX_STATUS_OK);
    /* Exactly representable either way, so widening a float source loses nothing. */
    for (i = 0; i < 18; ++i) CHECK(from_float[i] == from_double[i]);
    bb_free(&single);
    bb_free(&twice);
    return 0;
}

static int test_ply11_truncation(void) {
    bytebuf b;
    ply_options options;
    int binary;
    int64_t cut;

    for (binary = 0; binary <= 1; ++binary) {
        options = default_options();
        options.binary = binary;
        build_ply(&b, &options);
        for (cut = 1; cut < b.length; cut += (b.length / 20) + 1) {
            /* A guarded copy: the reader must find the end by bounds check, not by running off
             * the buffer into whatever follows it. */
            unsigned char *guarded = (unsigned char *)malloc((size_t)cut);
            gffx_execution_context context = cpu_context();
            gffx_diagnostic_buffer diagnostic = {0};
            gffx_ply_header header = blank_header();
            double vertices[18];
            int32_t faces[24];
            gffx_tensor_view vertices_view;
            gffx_tensor_view faces_view;
            int64_t vertex_shape[2];
            int64_t face_shape[2];
            gffx_status probe_status;

            diagnostic.struct_size = (uint32_t)sizeof(diagnostic);
            diagnostic.abi_version = GFFX_ABI_VERSION;
            memcpy(guarded, b.data, (size_t)cut);
            probe_status = gffx_io_ply_probe(guarded, cut, &context, &header, &diagnostic);
            if (probe_status == GFFX_STATUS_OK) {
                vertex_shape[0] = header.vertex_count; vertex_shape[1] = 3;
                face_shape[0] = header.face_count; face_shape[1] = 3;
                vertices_view = make_view(vertices, GFFX_DTYPE_FLOAT64, 2u, vertex_shape,
                                          pair_strides, GFFX_TENSOR_OUTPUT);
                faces_view = make_view(faces, GFFX_DTYPE_INT32, 2u, face_shape, pair_strides,
                                       GFFX_TENSOR_OUTPUT);
                CHECK(gffx_io_ply_read(guarded, cut, &header, &context, &vertices_view,
                                       &faces_view, NULL, &diagnostic)
                      == GFFX_STATUS_INVALID_ARGUMENT);
            } else {
                CHECK(probe_status == GFFX_STATUS_INVALID_ARGUMENT);
            }
            free(guarded);
        }
        bb_free(&b);
    }
    return 0;
}

static int test_ply12_malformed_headers(void) {
    gffx_execution_context context = cpu_context();
    gffx_diagnostic_buffer diagnostic = {0};
    gffx_ply_header header = blank_header();
    static const char bad_magic[] =
        "plz\nformat ascii 1.0\nelement vertex 0\nproperty float x\nend_header\n";
    static const char no_end[] =
        "ply\nformat ascii 1.0\nelement vertex 0\nproperty float x\n";
    static const char no_z[] =
        "ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\n"
        "element face 0\nproperty list uchar int vertex_indices\nend_header\n0 0\n";
    static const char list_on_vertex[] =
        "ply\nformat ascii 1.0\nelement vertex 1\nproperty list uchar float x\n"
        "property float y\nproperty float z\nelement face 0\n"
        "property list uchar int vertex_indices\nend_header\n1 0 0 0\n";
    static const char float_index[] =
        "ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\n"
        "property float z\nelement face 1\nproperty list uchar float vertex_indices\n"
        "end_header\n0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n";

    diagnostic.struct_size = (uint32_t)sizeof(diagnostic);
    diagnostic.abi_version = GFFX_ABI_VERSION;
    CHECK(gffx_io_ply_probe(bad_magic, (int64_t)sizeof(bad_magic) - 1, &context, &header,
                            &diagnostic) == GFFX_STATUS_INVALID_ARGUMENT);
    CHECK(gffx_io_ply_probe(no_end, (int64_t)sizeof(no_end) - 1, &context, &header,
                            &diagnostic) == GFFX_STATUS_INVALID_ARGUMENT);
    CHECK(gffx_io_ply_probe(no_z, (int64_t)sizeof(no_z) - 1, &context, &header,
                            &diagnostic) == GFFX_STATUS_INVALID_ARGUMENT);
    CHECK(gffx_io_ply_probe(list_on_vertex, (int64_t)sizeof(list_on_vertex) - 1, &context,
                            &header, &diagnostic) == GFFX_STATUS_UNSUPPORTED);
    CHECK(gffx_io_ply_probe(float_index, (int64_t)sizeof(float_index) - 1, &context, &header,
                            &diagnostic) == GFFX_STATUS_UNSUPPORTED);
    return 0;
}

static int test_ply13_validation(void) {
    bytebuf b;
    gffx_execution_context context = cpu_context();
    gffx_diagnostic_buffer diagnostic = {0};
    gffx_ply_header header = blank_header();
    gffx_ply_header wrong_size;
    double vertices[18];
    int32_t faces[24];
    gffx_tensor_view vertices_view;
    gffx_tensor_view faces_view;
    int64_t vertex_shape[2];
    int64_t face_shape[2];
    int64_t short_shape[2];
    ply_options options = default_options();

    diagnostic.struct_size = (uint32_t)sizeof(diagnostic);
    diagnostic.abi_version = GFFX_ABI_VERSION;
    build_ply(&b, &options);

    CHECK(gffx_io_ply_probe(NULL, 10, &context, &header, &diagnostic)
          == GFFX_STATUS_INVALID_ARGUMENT);
    CHECK(gffx_io_ply_probe(b.data, -1, &context, &header, &diagnostic)
          == GFFX_STATUS_INVALID_ARGUMENT);
    CHECK(gffx_io_ply_probe(b.data, b.length, &context, NULL, &diagnostic)
          == GFFX_STATUS_INVALID_ARGUMENT);
    wrong_size = blank_header();
    wrong_size.struct_size = 4u;
    CHECK(gffx_io_ply_probe(b.data, b.length, &context, &wrong_size, &diagnostic)
          == GFFX_STATUS_INVALID_ARGUMENT);
    CHECK(gffx_io_ply_probe(b.data, b.length, &context, &header, &diagnostic)
          == GFFX_STATUS_OK);

    /* An output shape disagreeing with the header is rejected rather than truncated. */
    short_shape[0] = 4; short_shape[1] = 3;
    vertex_shape[0] = header.vertex_count; vertex_shape[1] = 3;
    face_shape[0] = header.face_count; face_shape[1] = 3;
    vertices_view = make_view(vertices, GFFX_DTYPE_FLOAT64, 2u, short_shape, pair_strides,
                              GFFX_TENSOR_OUTPUT);
    faces_view = make_view(faces, GFFX_DTYPE_INT32, 2u, face_shape, pair_strides,
                           GFFX_TENSOR_OUTPUT);
    CHECK(gffx_io_ply_read(b.data, b.length, &header, &context, &vertices_view, &faces_view,
                           NULL, &diagnostic) == GFFX_STATUS_INVALID_ARGUMENT);

    /* A uint32 face output is rejected: packed indices are int32 everywhere. */
    vertices_view = make_view(vertices, GFFX_DTYPE_FLOAT64, 2u, vertex_shape, pair_strides,
                              GFFX_TENSOR_OUTPUT);
    faces_view = make_view(faces, GFFX_DTYPE_UINT32, 2u, face_shape, pair_strides,
                           GFFX_TENSOR_OUTPUT);
    CHECK(gffx_io_ply_read(b.data, b.length, &header, &context, &vertices_view, &faces_view,
                           NULL, &diagnostic) == GFFX_STATUS_INVALID_ARGUMENT);
    bb_free(&b);
    return 0;
}

static int test_ply14_empty(void) {
    gffx_execution_context context = cpu_context();
    gffx_diagnostic_buffer diagnostic = {0};
    gffx_ply_header header = blank_header();
    gffx_tensor_view vertices_view;
    gffx_tensor_view faces_view;
    int64_t vertex_shape[2];
    int64_t face_shape[2];
    static const char empty[] =
        "ply\nformat ascii 1.0\nelement vertex 0\nproperty float x\nproperty float y\n"
        "property float z\nelement face 0\nproperty list uchar int vertex_indices\n"
        "end_header\n";

    diagnostic.struct_size = (uint32_t)sizeof(diagnostic);
    diagnostic.abi_version = GFFX_ABI_VERSION;
    CHECK(gffx_io_ply_probe(empty, (int64_t)sizeof(empty) - 1, &context, &header, &diagnostic)
          == GFFX_STATUS_OK);
    CHECK(header.vertex_count == 0);
    CHECK(header.face_count == 0);
    vertex_shape[0] = 0; vertex_shape[1] = 3;
    face_shape[0] = 0; face_shape[1] = 3;
    vertices_view = make_view(NULL, GFFX_DTYPE_FLOAT64, 2u, vertex_shape, pair_strides,
                              GFFX_TENSOR_OUTPUT);
    faces_view = make_view(NULL, GFFX_DTYPE_INT32, 2u, face_shape, pair_strides,
                           GFFX_TENSOR_OUTPUT);
    CHECK(gffx_io_ply_read(empty, (int64_t)sizeof(empty) - 1, &header, &context, &vertices_view,
                           &faces_view, NULL, &diagnostic) == GFFX_STATUS_OK);
    return 0;
}

static int test_ply15_ascii_numbers(void) {
    gffx_execution_context context = cpu_context();
    gffx_diagnostic_buffer diagnostic = {0};
    gffx_ply_header header = blank_header();
    gffx_tensor_view vertices_view;
    gffx_tensor_view faces_view;
    double vertices[9];
    int32_t faces[3];
    int64_t vertex_shape[2];
    int64_t face_shape[2];
    /* Row 1 is the exact set. Row 2 exercises signed exponents and a leading dot. Row 3 is
     * outside the exact window and is held to the documented 2 ulp. */
    static const char numeric[] =
        "ply\nformat ascii 1.0\nelement vertex 3\nproperty double x\nproperty double y\n"
        "property double z\nelement face 1\nproperty list uchar int vertex_indices\n"
        "end_header\n"
        "0.5 -0.25 1234.5\n"
        "1e2 -3.5e-2 .75\n"
        "1.2345678901234567e300 -5e-300 9007199254740993\n"
        "3 0 1 2\n";
    static const char malformed[] =
        "ply\nformat ascii 1.0\nelement vertex 1\nproperty double x\nproperty double y\n"
        "property double z\nelement face 0\nproperty list uchar int vertex_indices\n"
        "end_header\n1.0 nan 2.0\n";
    static const char garbage[] =
        "ply\nformat ascii 1.0\nelement vertex 1\nproperty double x\nproperty double y\n"
        "property double z\nelement face 0\nproperty list uchar int vertex_indices\n"
        "end_header\n1.0 12x4 2.0\n";
    double expected_high = 1.2345678901234567e300;
    double relative;

    diagnostic.struct_size = (uint32_t)sizeof(diagnostic);
    diagnostic.abi_version = GFFX_ABI_VERSION;
    CHECK(gffx_io_ply_probe(numeric, (int64_t)sizeof(numeric) - 1, &context, &header,
                            &diagnostic) == GFFX_STATUS_OK);
    vertex_shape[0] = 3; vertex_shape[1] = 3;
    face_shape[0] = 1; face_shape[1] = 3;
    vertices_view = make_view(vertices, GFFX_DTYPE_FLOAT64, 2u, vertex_shape, pair_strides,
                              GFFX_TENSOR_OUTPUT);
    faces_view = make_view(faces, GFFX_DTYPE_INT32, 2u, face_shape, pair_strides,
                           GFFX_TENSOR_OUTPUT);
    CHECK(gffx_io_ply_read(numeric, (int64_t)sizeof(numeric) - 1, &header, &context,
                           &vertices_view, &faces_view, NULL, &diagnostic) == GFFX_STATUS_OK);
    /* Exact: significand within 53 bits, exponent within [-22, 22]. */
    CHECK(vertices[0] == 0.5);
    CHECK(vertices[1] == -0.25);
    CHECK(vertices[2] == 1234.5);
    CHECK(vertices[3] == 100.0);
    CHECK(vertices[4] == -0.035);
    CHECK(vertices[5] == 0.75);
    /* Outside the exact window: held to the documented 2 ulp bound. */
    relative = fabs(vertices[6] - expected_high) / expected_high;
    CHECK(relative <= 2.3e-16);
    CHECK(vertices[7] == -5e-300);
    CHECK(vertices[8] == 9007199254740992.0);

    header = blank_header();
    CHECK(gffx_io_ply_probe(malformed, (int64_t)sizeof(malformed) - 1, &context, &header,
                            &diagnostic) == GFFX_STATUS_OK);
    vertex_shape[0] = 1; face_shape[0] = 0;
    vertices_view = make_view(vertices, GFFX_DTYPE_FLOAT64, 2u, vertex_shape, pair_strides,
                              GFFX_TENSOR_OUTPUT);
    faces_view = make_view(faces, GFFX_DTYPE_INT32, 2u, face_shape, pair_strides,
                           GFFX_TENSOR_OUTPUT);
    /* nan is refused rather than propagated: the geometry kernels would reject it anyway, and a
     * template carrying one is a defect the caller should see here. */
    CHECK(gffx_io_ply_read(malformed, (int64_t)sizeof(malformed) - 1, &header, &context,
                           &vertices_view, &faces_view, NULL, &diagnostic)
          == GFFX_STATUS_INVALID_ARGUMENT);

    header = blank_header();
    CHECK(gffx_io_ply_probe(garbage, (int64_t)sizeof(garbage) - 1, &context, &header,
                            &diagnostic) == GFFX_STATUS_OK);
    CHECK(gffx_io_ply_read(garbage, (int64_t)sizeof(garbage) - 1, &header, &context,
                           &vertices_view, &faces_view, NULL, &diagnostic)
          == GFFX_STATUS_INVALID_ARGUMENT);
    return 0;
}

static int test_ply16_determinism_and_workspace(void) {
    bytebuf b;
    gffx_execution_context context = cpu_context();
    gffx_diagnostic_buffer diagnostic = {0};
    gffx_ply_header header;
    double first[18];
    double second[18];
    int32_t first_faces[24];
    int32_t second_faces[24];
    uint64_t required_bytes = 123u;
    uint64_t required_alignment = 0u;
    ply_options options = default_options();

    diagnostic.struct_size = (uint32_t)sizeof(diagnostic);
    diagnostic.abi_version = GFFX_ABI_VERSION;
    options.binary = 1;
    build_ply(&b, &options);
    CHECK(read_octahedron(&b, GFFX_DTYPE_FLOAT64, first, first_faces, &header)
          == GFFX_STATUS_OK);
    CHECK(gffx_io_ply_read_workspace(&header, &context, &required_bytes, &required_alignment,
                                     &diagnostic) == GFFX_STATUS_OK);
    CHECK(required_bytes == 0u);
    CHECK(read_octahedron(&b, GFFX_DTYPE_FLOAT64, second, second_faces, &header)
          == GFFX_STATUS_OK);
    CHECK(memcmp(first, second, sizeof(first)) == 0);
    CHECK(memcmp(first_faces, second_faces, sizeof(first_faces)) == 0);
    bb_free(&b);
    return 0;
}

int main(void) {
    int result;
    result = test_ply01_ascii_probe(); if (result != 0) return result;
    result = test_ply02_ascii_read(); if (result != 0) return result;
    result = test_ply03_binary_read(); if (result != 0) return result;
    result = test_ply04_dtypes(); if (result != 0) return result;
    result = test_ply05_line_endings_and_comments(); if (result != 0) return result;
    result = test_ply06_extra_properties(); if (result != 0) return result;
    result = test_ply07_element_order(); if (result != 0) return result;
    result = test_ply08_quad_face(); if (result != 0) return result;
    result = test_ply09_big_endian(); if (result != 0) return result;
    result = test_ply10_source_width(); if (result != 0) return result;
    result = test_ply11_truncation(); if (result != 0) return result;
    result = test_ply12_malformed_headers(); if (result != 0) return result;
    result = test_ply13_validation(); if (result != 0) return result;
    result = test_ply14_empty(); if (result != 0) return result;
    result = test_ply15_ascii_numbers(); if (result != 0) return result;
    result = test_ply16_determinism_and_workspace(); if (result != 0) return result;
    return 0;
}
