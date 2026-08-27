/*
 * io.ply - buffer-based PLY triangle-template reader.
 *
 * Governed by the runtime dependency gate like every other core source: no allocation, no file
 * I/O, no process-wide state. The file layer in native/io sits on top of this and supplies the
 * bytes; keeping the parser here is what lets a memory-mapped file, an archive member and a
 * network buffer share one implementation.
 *
 * Behaviour is specified in PLY_ACCEPTANCE_V0_1.md.
 */

#include <gffx/execution.h>
#include <gffx/io.h>
#include <gffx/status.h>
#include <gffx/tensor.h>

#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define PLY_MAX_ELEMENTS 16
#define PLY_MAX_PROPERTIES 64

/* Scalar type codes. Width and class are all the parser needs; the spelling is not retained. */
#define PLY_TYPE_INVALID 0
#define PLY_TYPE_INT 1
#define PLY_TYPE_UINT 2
#define PLY_TYPE_FLOAT 3

typedef struct {
    int type_class;
    int width;
    int is_list;
    int count_class;
    int count_width;
    int coordinate;   /* 0 = x, 1 = y, 2 = z, -1 = not a coordinate */
    int is_indices;   /* the face list property carrying vertex indices */
} ply_property;

typedef struct {
    int kind;         /* 0 = other, 1 = vertex, 2 = face */
    int64_t count;
    int property_count;
    ply_property properties[PLY_MAX_PROPERTIES];
} ply_element;

typedef struct {
    int element_count;
    ply_element elements[PLY_MAX_ELEMENTS];
} ply_layout;

/* --------------------------------------------------------------------------- lexical helpers */

static int ply_is_space(unsigned char c) {
    return c == ' ' || c == '\t' || c == '\r' || c == '\n';
}

static int ply_is_digit(unsigned char c) {
    return c >= '0' && c <= '9';
}

static int ply_token_equals(const unsigned char *bytes, int64_t length, const char *literal) {
    int64_t index = 0;
    while (literal[index] != '\0') {
        if (index >= length) return 0;
        if (bytes[index] != (unsigned char)literal[index]) return 0;
        ++index;
    }
    return index == length;
}

/*
 * Advances past whitespace and reports the next token's bounds. Returns 0 at end of buffer, so a
 * truncated body is detected by the caller running out of tokens rather than by reading past the
 * end.
 */
static int ply_next_token(
    const unsigned char *bytes, int64_t length, int64_t *cursor,
    int64_t *token_start, int64_t *token_length
) {
    int64_t index = *cursor;
    int64_t start;
    while (index < length && ply_is_space(bytes[index])) ++index;
    if (index >= length) {
        *cursor = index;
        return 0;
    }
    start = index;
    while (index < length && !ply_is_space(bytes[index])) ++index;
    *token_start = start;
    *token_length = index - start;
    *cursor = index;
    return 1;
}

/*
 * Header lines are read explicitly rather than through the token scanner, because a property
 * declaration's meaning depends on its position within its line.
 *
 * Returns 0 at end of buffer, 1 for a newline-terminated line, and 2 for a line that ran to the
 * end of the buffer without a terminator. That last case is the difference between a truncated
 * file and an unsupported one: "format ascii 1." cut mid-token would otherwise read as a version
 * this reader does not support, when in fact the file simply stops. Only a terminated line can be
 * judged on its content.
 */
static int ply_next_line(
    const unsigned char *bytes, int64_t length, int64_t *cursor,
    int64_t *line_start, int64_t *line_length
) {
    int64_t index = *cursor;
    int64_t start = index;
    int terminated;
    if (index >= length) return 0;
    while (index < length && bytes[index] != '\n') ++index;
    terminated = (index < length);
    *line_start = start;
    *line_length = index - start;
    /* Trim a CR so CRLF and LF files parse identically. */
    if (*line_length > 0 && bytes[start + *line_length - 1] == '\r') *line_length -= 1;
    if (terminated) ++index;   /* step over the newline itself */
    *cursor = index;
    return terminated ? 1 : 2;
}

static int ply_line_token(
    const unsigned char *bytes, int64_t line_start, int64_t line_length, int ordinal,
    int64_t *token_start, int64_t *token_length
) {
    int64_t cursor = line_start;
    int64_t limit = line_start + line_length;
    int found = 0;
    while (cursor < limit) {
        int64_t start;
        while (cursor < limit && ply_is_space(bytes[cursor])) ++cursor;
        if (cursor >= limit) break;
        start = cursor;
        while (cursor < limit && !ply_is_space(bytes[cursor])) ++cursor;
        if (found == ordinal) {
            *token_start = start;
            *token_length = cursor - start;
            return 1;
        }
        ++found;
    }
    return 0;
}

/* ---------------------------------------------------------------------------- number parsing */

/* Exact powers of ten. Every entry below 1e23 is exactly representable as a double. */
static const double PLY_POW10[23] = {
    1e0,  1e1,  1e2,  1e3,  1e4,  1e5,  1e6,  1e7,  1e8,  1e9,  1e10, 1e11,
    1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19, 1e20, 1e21, 1e22
};

static int ply_parse_int64(
    const unsigned char *bytes, int64_t start, int64_t length, int64_t *out_value
) {
    int64_t index = 0;
    int negative = 0;
    uint64_t magnitude = 0u;
    if (length <= 0) return 0;
    if (bytes[start] == '-') { negative = 1; index = 1; }
    else if (bytes[start] == '+') { index = 1; }
    if (index >= length) return 0;
    for (; index < length; ++index) {
        unsigned char c = bytes[start + index];
        if (!ply_is_digit(c)) return 0;
        if (magnitude > (uint64_t)0x0CCCCCCCCCCCCCCCull) return 0;
        magnitude = magnitude * 10u + (uint64_t)(c - '0');
    }
    if (negative) {
        if (magnitude > (uint64_t)0x8000000000000000ull) return 0;
        *out_value = (magnitude == (uint64_t)0x8000000000000000ull)
            ? (-0x7fffffffffffffffll - 1) : -(int64_t)magnitude;
    } else {
        if (magnitude > (uint64_t)0x7fffffffffffffffull) return 0;
        *out_value = (int64_t)magnitude;
    }
    return 1;
}

/*
 * Decimal to binary conversion.
 *
 * Exact when the significand fits in 53 bits and the decimal exponent lies in [-22, 22]: both
 * operands are then exactly representable and IEEE-754 rounds the single scaling step correctly.
 * Outside that window the scale is built by binary exponentiation over the exact table, which
 * costs a handful of rounded multiplications rather than one per decade.
 *
 * nan and inf spellings are rejected rather than accepted: a template carrying one is a defect
 * the caller should see here, and every geometry kernel would reject the result anyway.
 */
static int ply_parse_double(
    const unsigned char *bytes, int64_t start, int64_t length, double *out_value
) {
    int64_t index = 0;
    int negative = 0;
    uint64_t significand = 0u;
    int significant_digits = 0;
    int fractional_digits = 0;
    int saw_digit = 0;
    int saw_dot = 0;
    int exponent = 0;
    double value;

    if (length <= 0) return 0;
    if (bytes[start] == '-') { negative = 1; index = 1; }
    else if (bytes[start] == '+') { index = 1; }

    for (; index < length; ++index) {
        unsigned char c = bytes[start + index];
        if (c == '.') {
            if (saw_dot) return 0;
            saw_dot = 1;
            continue;
        }
        if (c == 'e' || c == 'E') break;
        if (!ply_is_digit(c)) return 0;
        saw_digit = 1;
        /* Digits beyond the 19th cannot affect a uint64 significand; track their magnitude. */
        if (significant_digits < 19) {
            significand = significand * 10u + (uint64_t)(c - '0');
            if (significand != 0u) ++significant_digits;
            if (saw_dot) ++fractional_digits;
        } else if (!saw_dot) {
            ++exponent;
        }
    }
    if (!saw_digit) return 0;

    if (index < length && (bytes[start + index] == 'e' || bytes[start + index] == 'E')) {
        int64_t written;
        int64_t exponent_start = start + index + 1;
        int64_t exponent_length = length - index - 1;
        if (exponent_length <= 0) return 0;
        if (!ply_parse_int64(bytes, exponent_start, exponent_length, &written)) return 0;
        if (written > 4096 || written < -4096) return 0;
        exponent += (int)written;
    }
    exponent -= fractional_digits;

    value = (double)significand;
    if (exponent >= -22 && exponent <= 22 && significand <= (uint64_t)9007199254740992ull) {
        /* The exact window: one correctly rounded operation. */
        value = (exponent >= 0) ? value * PLY_POW10[exponent] : value / PLY_POW10[-exponent];
    } else {
        int remaining = exponent < 0 ? -exponent : exponent;
        double scale = 1.0;
        double base = 10.0;
        while (remaining > 0) {
            if (remaining & 1) scale *= base;
            base *= base;
            remaining >>= 1;
            if (base > 1e300 && remaining > 1) break;
        }
        if (remaining > 0) {
            /* The residual decades cannot be folded into the squared base without overflowing;
             * apply them directly. */
            while (remaining-- > 0) scale *= 10.0;
        }
        value = (exponent >= 0) ? value * scale : value / scale;
    }

    if (!(value == value) || value > 1.7e308 || value < -1.7e308) return 0;
    *out_value = negative ? -value : value;
    return 1;
}

/* ------------------------------------------------------------------------------ header parse */

static int ply_scalar_type(
    const unsigned char *bytes, int64_t start, int64_t length, int *type_class, int *width
) {
    const unsigned char *token = bytes + start;
    if (ply_token_equals(token, length, "char") || ply_token_equals(token, length, "int8")) {
        *type_class = PLY_TYPE_INT; *width = 1; return 1;
    }
    if (ply_token_equals(token, length, "uchar") || ply_token_equals(token, length, "uint8")) {
        *type_class = PLY_TYPE_UINT; *width = 1; return 1;
    }
    if (ply_token_equals(token, length, "short") || ply_token_equals(token, length, "int16")) {
        *type_class = PLY_TYPE_INT; *width = 2; return 1;
    }
    if (ply_token_equals(token, length, "ushort") || ply_token_equals(token, length, "uint16")) {
        *type_class = PLY_TYPE_UINT; *width = 2; return 1;
    }
    if (ply_token_equals(token, length, "int") || ply_token_equals(token, length, "int32")) {
        *type_class = PLY_TYPE_INT; *width = 4; return 1;
    }
    if (ply_token_equals(token, length, "uint") || ply_token_equals(token, length, "uint32")) {
        *type_class = PLY_TYPE_UINT; *width = 4; return 1;
    }
    if (ply_token_equals(token, length, "float") || ply_token_equals(token, length, "float32")) {
        *type_class = PLY_TYPE_FLOAT; *width = 4; return 1;
    }
    if (ply_token_equals(token, length, "double") || ply_token_equals(token, length, "float64")) {
        *type_class = PLY_TYPE_FLOAT; *width = 8; return 1;
    }
    return 0;
}

/*
 * Parses the header into a layout description. Returns a status so the two distinct failures stay
 * distinguishable: a malformed file is INVALID_ARGUMENT, a well-formed file outside this subset
 * is UNSUPPORTED.
 */
static gffx_status ply_parse_header(
    const unsigned char *bytes, int64_t length, ply_layout *layout,
    uint32_t *out_format, int64_t *out_data_offset
) {
    int64_t cursor = 0;
    int64_t line_start;
    int64_t line_length;
    int64_t token_start;
    int64_t token_length;
    int current = -1;
    int saw_format = 0;
    int saw_end = 0;
    int line_status;

    layout->element_count = 0;

    if (ply_next_line(bytes, length, &cursor, &line_start, &line_length) != 1) {
        return GFFX_STATUS_INVALID_ARGUMENT;
    }
    if (!ply_token_equals(bytes + line_start, line_length, "ply")) {
        return GFFX_STATUS_INVALID_ARGUMENT;
    }

    while ((line_status = ply_next_line(bytes, length, &cursor, &line_start, &line_length))
           != 0) {
        /* An unterminated line means the header itself is cut short, which is a truncation
         * rather than a judgement about the line's content. */
        if (line_status == 2) return GFFX_STATUS_INVALID_ARGUMENT;
        if (!ply_line_token(bytes, line_start, line_length, 0, &token_start, &token_length)) {
            continue;   /* a blank line inside the header carries no meaning */
        }
        if (ply_token_equals(bytes + token_start, token_length, "comment") ||
            ply_token_equals(bytes + token_start, token_length, "obj_info")) {
            continue;
        }
        if (ply_token_equals(bytes + token_start, token_length, "end_header")) {
            saw_end = 1;
            break;
        }
        if (ply_token_equals(bytes + token_start, token_length, "format")) {
            int64_t kind_start;
            int64_t kind_length;
            int64_t version_start;
            int64_t version_length;
            if (!ply_line_token(bytes, line_start, line_length, 1, &kind_start, &kind_length)) {
                return GFFX_STATUS_INVALID_ARGUMENT;
            }
            if (!ply_line_token(bytes, line_start, line_length, 2, &version_start,
                                &version_length)) {
                return GFFX_STATUS_INVALID_ARGUMENT;
            }
            if (!ply_token_equals(bytes + version_start, version_length, "1.0")) {
                return GFFX_STATUS_UNSUPPORTED;
            }
            if (ply_token_equals(bytes + kind_start, kind_length, "ascii")) {
                *out_format = GFFX_PLY_FORMAT_ASCII;
            } else if (ply_token_equals(bytes + kind_start, kind_length,
                                        "binary_little_endian")) {
                *out_format = GFFX_PLY_FORMAT_BINARY_LITTLE_ENDIAN;
            } else if (ply_token_equals(bytes + kind_start, kind_length, "binary_big_endian")) {
                return GFFX_STATUS_UNSUPPORTED;
            } else {
                return GFFX_STATUS_INVALID_ARGUMENT;
            }
            saw_format = 1;
            continue;
        }
        if (ply_token_equals(bytes + token_start, token_length, "element")) {
            int64_t name_start;
            int64_t name_length;
            int64_t count_start;
            int64_t count_length;
            int64_t count_value;
            ply_element *element;
            if (!saw_format) return GFFX_STATUS_INVALID_ARGUMENT;
            if (layout->element_count >= PLY_MAX_ELEMENTS) return GFFX_STATUS_UNSUPPORTED;
            if (!ply_line_token(bytes, line_start, line_length, 1, &name_start, &name_length) ||
                !ply_line_token(bytes, line_start, line_length, 2, &count_start,
                                &count_length)) {
                return GFFX_STATUS_INVALID_ARGUMENT;
            }
            if (!ply_parse_int64(bytes, count_start, count_length, &count_value) ||
                count_value < 0) {
                return GFFX_STATUS_INVALID_ARGUMENT;
            }
            current = layout->element_count++;
            element = &layout->elements[current];
            element->count = count_value;
            element->property_count = 0;
            element->kind = 0;
            if (ply_token_equals(bytes + name_start, name_length, "vertex")) element->kind = 1;
            else if (ply_token_equals(bytes + name_start, name_length, "face")) element->kind = 2;
            continue;
        }
        if (ply_token_equals(bytes + token_start, token_length, "property")) {
            ply_element *element;
            ply_property property;
            int64_t type_start;
            int64_t type_length;
            if (current < 0) return GFFX_STATUS_INVALID_ARGUMENT;
            element = &layout->elements[current];
            if (element->property_count >= PLY_MAX_PROPERTIES) return GFFX_STATUS_UNSUPPORTED;
            memset(&property, 0, sizeof(property));
            property.coordinate = -1;
            if (!ply_line_token(bytes, line_start, line_length, 1, &type_start, &type_length)) {
                return GFFX_STATUS_INVALID_ARGUMENT;
            }
            if (ply_token_equals(bytes + type_start, type_length, "list")) {
                int64_t count_start;
                int64_t count_length;
                int64_t index_start;
                int64_t index_length;
                int64_t name_start;
                int64_t name_length;
                /* A list on the vertex element is well-formed PLY this subset cannot map. */
                if (element->kind == 1) return GFFX_STATUS_UNSUPPORTED;
                if (!ply_line_token(bytes, line_start, line_length, 2, &count_start,
                                    &count_length) ||
                    !ply_line_token(bytes, line_start, line_length, 3, &index_start,
                                    &index_length) ||
                    !ply_line_token(bytes, line_start, line_length, 4, &name_start,
                                    &name_length)) {
                    return GFFX_STATUS_INVALID_ARGUMENT;
                }
                property.is_list = 1;
                if (!ply_scalar_type(bytes, count_start, count_length, &property.count_class,
                                     &property.count_width)) {
                    return GFFX_STATUS_INVALID_ARGUMENT;
                }
                if (!ply_scalar_type(bytes, index_start, index_length, &property.type_class,
                                     &property.width)) {
                    return GFFX_STATUS_INVALID_ARGUMENT;
                }
                if (property.count_class == PLY_TYPE_FLOAT ||
                    property.type_class == PLY_TYPE_FLOAT) {
                    return GFFX_STATUS_UNSUPPORTED;
                }
                if (element->kind == 2 &&
                    (ply_token_equals(bytes + name_start, name_length, "vertex_indices") ||
                     ply_token_equals(bytes + name_start, name_length, "vertex_index"))) {
                    property.is_indices = 1;
                }
            } else {
                int64_t name_start;
                int64_t name_length;
                if (!ply_scalar_type(bytes, type_start, type_length, &property.type_class,
                                     &property.width)) {
                    return GFFX_STATUS_INVALID_ARGUMENT;
                }
                if (!ply_line_token(bytes, line_start, line_length, 2, &name_start,
                                    &name_length)) {
                    return GFFX_STATUS_INVALID_ARGUMENT;
                }
                if (element->kind == 1) {
                    if (ply_token_equals(bytes + name_start, name_length, "x")) {
                        property.coordinate = 0;
                    } else if (ply_token_equals(bytes + name_start, name_length, "y")) {
                        property.coordinate = 1;
                    } else if (ply_token_equals(bytes + name_start, name_length, "z")) {
                        property.coordinate = 2;
                    }
                }
            }
            element->properties[element->property_count++] = property;
            continue;
        }
        /* An unrecognised header directive is a file this reader does not understand. */
        return GFFX_STATUS_INVALID_ARGUMENT;
    }

    if (!saw_end || !saw_format) return GFFX_STATUS_INVALID_ARGUMENT;
    *out_data_offset = cursor;
    return GFFX_STATUS_OK;
}

/* Confirms that a vertex element carries all three coordinates and a face element carries an
 * index list, which the header grammar alone does not guarantee. */
static gffx_status ply_check_layout(const ply_layout *layout) {
    int element;
    int saw_vertex = 0;
    for (element = 0; element < layout->element_count; ++element) {
        const ply_element *entry = &layout->elements[element];
        int property;
        if (entry->kind == 1) {
            int seen[3];
            seen[0] = 0; seen[1] = 0; seen[2] = 0;
            saw_vertex = 1;
            for (property = 0; property < entry->property_count; ++property) {
                int coordinate = entry->properties[property].coordinate;
                if (coordinate >= 0) {
                    if (seen[coordinate]) return GFFX_STATUS_INVALID_ARGUMENT;
                    seen[coordinate] = 1;
                }
            }
            if (!seen[0] || !seen[1] || !seen[2]) return GFFX_STATUS_INVALID_ARGUMENT;
        } else if (entry->kind == 2) {
            int found = 0;
            for (property = 0; property < entry->property_count; ++property) {
                if (entry->properties[property].is_indices) found = 1;
            }
            if (!found && entry->count > 0) return GFFX_STATUS_INVALID_ARGUMENT;
        }
    }
    if (!saw_vertex) return GFFX_STATUS_INVALID_ARGUMENT;
    return GFFX_STATUS_OK;
}

/* ------------------------------------------------------------------------------ body reading */

/* Assembles a little-endian scalar without casting the buffer: the bytes carry no alignment
 * guarantee, and the byte order belongs to the format rather than to the host. */
static int ply_binary_scalar(
    const unsigned char *bytes, int64_t length, int64_t *cursor,
    int type_class, int width, double *out_double, int64_t *out_int
) {
    uint64_t raw = 0u;
    int index;
    if (*cursor + width > length) return 0;
    for (index = 0; index < width; ++index) {
        raw |= ((uint64_t)bytes[*cursor + index]) << (8 * index);
    }
    *cursor += width;
    if (type_class == PLY_TYPE_FLOAT) {
        if (width == 4) {
            uint32_t narrow = (uint32_t)raw;
            float value;
            memcpy(&value, &narrow, 4);
            *out_double = (double)value;
        } else {
            double value;
            memcpy(&value, &raw, 8);
            *out_double = value;
        }
        *out_int = 0;
        return 1;
    }
    if (type_class == PLY_TYPE_INT) {
        int64_t value;
        switch (width) {
            case 1: value = (int64_t)(int8_t)(uint8_t)raw; break;
            case 2: value = (int64_t)(int16_t)(uint16_t)raw; break;
            case 4: value = (int64_t)(int32_t)(uint32_t)raw; break;
            default: value = (int64_t)raw; break;
        }
        *out_int = value;
        *out_double = (double)value;
        return 1;
    }
    *out_int = (int64_t)raw;
    *out_double = (double)raw;
    return 1;
}

static int ply_ascii_scalar(
    const unsigned char *bytes, int64_t length, int64_t *cursor,
    int type_class, double *out_double, int64_t *out_int
) {
    int64_t token_start;
    int64_t token_length;
    if (!ply_next_token(bytes, length, cursor, &token_start, &token_length)) return 0;
    if (type_class == PLY_TYPE_FLOAT) {
        if (!ply_parse_double(bytes, token_start, token_length, out_double)) return 0;
        *out_int = 0;
        return 1;
    }
    if (!ply_parse_int64(bytes, token_start, token_length, out_int)) return 0;
    *out_double = (double)*out_int;
    return 1;
}

static void ply_store_double(void *data, const int64_t *strides, gffx_dtype dtype,
                             int64_t row, int64_t column, double value) {
    int64_t offset = row * strides[0] + column * strides[1];
    if (dtype == GFFX_DTYPE_FLOAT32) ((float *)data)[offset] = (float)value;
    else ((double *)data)[offset] = value;
}

typedef struct {
    const unsigned char *bytes;
    int64_t length;
    int64_t cursor;
    int binary;
} ply_reader;

static int ply_read_scalar(ply_reader *reader, int type_class, int width,
                           double *out_double, int64_t *out_int) {
    if (reader->binary) {
        return ply_binary_scalar(reader->bytes, reader->length, &reader->cursor, type_class,
                                 width, out_double, out_int);
    }
    return ply_ascii_scalar(reader->bytes, reader->length, &reader->cursor, type_class,
                            out_double, out_int);
}

/*
 * Walks the declared elements in order. With store clear this validates only, which is what makes
 * the no-partial-write guarantee achievable; with store set it writes coordinates and indices.
 * Unrelated elements are consumed by width in both passes so the cursor stays aligned with the
 * file rather than assuming vertex and face are the only elements present.
 */
static gffx_status ply_scan_body(
    ply_reader *reader, const ply_layout *layout,
    gffx_tensor_view *vertices, gffx_tensor_view *faces, int store
) {
    int element;
    for (element = 0; element < layout->element_count; ++element) {
        const ply_element *entry = &layout->elements[element];
        int64_t row;
        for (row = 0; row < entry->count; ++row) {
            int property;
            for (property = 0; property < entry->property_count; ++property) {
                const ply_property *descriptor = &entry->properties[property];
                double as_double = 0.0;
                int64_t as_int = 0;
                if (descriptor->is_list) {
                    int64_t arity = 0;
                    int64_t item;
                    if (!ply_read_scalar(reader, descriptor->count_class,
                                         descriptor->count_width, &as_double, &arity)) {
                        return GFFX_STATUS_INVALID_ARGUMENT;
                    }
                    if (arity < 0) return GFFX_STATUS_INVALID_ARGUMENT;
                    if (descriptor->is_indices && arity != 3) {
                        /* Well-formed PLY that this triangle-template subset cannot represent. */
                        return GFFX_STATUS_UNSUPPORTED;
                    }
                    for (item = 0; item < arity; ++item) {
                        if (!ply_read_scalar(reader, descriptor->type_class, descriptor->width,
                                             &as_double, &as_int)) {
                            return GFFX_STATUS_INVALID_ARGUMENT;
                        }
                        if (descriptor->is_indices) {
                            if (as_int < 0 || as_int > 2147483647ll) {
                                return GFFX_STATUS_INVALID_ARGUMENT;
                            }
                            if (store) {
                                ((int32_t *)faces->data)[row * faces->strides[0] +
                                                         item * faces->strides[1]] =
                                    (int32_t)as_int;
                            }
                        }
                    }
                    continue;
                }
                if (!ply_read_scalar(reader, descriptor->type_class, descriptor->width,
                                     &as_double, &as_int)) {
                    return GFFX_STATUS_INVALID_ARGUMENT;
                }
                if (descriptor->coordinate >= 0 && store) {
                    ply_store_double(vertices->data, vertices->strides, vertices->dtype, row,
                                     descriptor->coordinate, as_double);
                }
            }
        }
    }
    return GFFX_STATUS_OK;
}

/* --------------------------------------------------------------------------- public entry points */

static int ply_header_valid(const gffx_ply_header *header) {
    return header != NULL && header->struct_size == (uint32_t)sizeof(gffx_ply_header);
}

GFFX_API gffx_status GFFX_CALL gffx_io_ply_probe(
    const void *bytes,
    int64_t length,
    const gffx_execution_context *context,
    gffx_ply_header *header,
    gffx_diagnostic_buffer *diagnostic
) {
    ply_layout layout;
    uint32_t format = GFFX_PLY_FORMAT_ASCII;
    int64_t data_offset = 0;
    gffx_status status;
    int element;

    (void)diagnostic;
    if (bytes == NULL || length < 0) return GFFX_STATUS_INVALID_ARGUMENT;
    if (!ply_header_valid(header)) return GFFX_STATUS_INVALID_ARGUMENT;
    if (context != NULL) {
        status = gffx_validate_execution_context(context, diagnostic);
        if (status != GFFX_STATUS_OK) return status;
    }

    status = ply_parse_header((const unsigned char *)bytes, length, &layout, &format,
                              &data_offset);
    if (status != GFFX_STATUS_OK) return status;
    status = ply_check_layout(&layout);
    if (status != GFFX_STATUS_OK) return status;

    header->format = format;
    header->reserved = 0u;
    header->vertex_count = 0;
    header->face_count = 0;
    header->data_offset = data_offset;
    for (element = 0; element < layout.element_count; ++element) {
        if (layout.elements[element].kind == 1) {
            header->vertex_count = layout.elements[element].count;
        } else if (layout.elements[element].kind == 2) {
            header->face_count = layout.elements[element].count;
        }
    }
    return GFFX_STATUS_OK;
}

GFFX_API gffx_status GFFX_CALL gffx_io_ply_read_workspace(
    const gffx_ply_header *header,
    const gffx_execution_context *context,
    uint64_t *required_bytes,
    uint64_t *required_alignment,
    gffx_diagnostic_buffer *diagnostic
) {
    (void)context;
    (void)diagnostic;
    if (!ply_header_valid(header)) return GFFX_STATUS_INVALID_ARGUMENT;
    if (required_bytes == NULL || required_alignment == NULL) {
        return GFFX_STATUS_INVALID_ARGUMENT;
    }
    /* The parse is a single forward pass and needs no scratch. */
    *required_bytes = 0u;
    *required_alignment = 1u;
    return GFFX_STATUS_OK;
}

GFFX_API gffx_status GFFX_CALL gffx_io_ply_read(
    const void *bytes,
    int64_t length,
    const gffx_ply_header *header,
    const gffx_execution_context *context,
    gffx_tensor_view *vertices,
    gffx_tensor_view *faces,
    const gffx_buffer *workspace,
    gffx_diagnostic_buffer *diagnostic
) {
    ply_layout layout;
    ply_reader reader;
    uint32_t format = GFFX_PLY_FORMAT_ASCII;
    int64_t data_offset = 0;
    gffx_status status;
    int element;
    int64_t vertex_count = 0;
    int64_t face_count = 0;
    int pass;

    (void)workspace;
    if (bytes == NULL || length < 0) return GFFX_STATUS_INVALID_ARGUMENT;
    if (!ply_header_valid(header)) return GFFX_STATUS_INVALID_ARGUMENT;
    if (vertices == NULL || faces == NULL) return GFFX_STATUS_INVALID_ARGUMENT;
    if (vertices->struct_size != (uint32_t)sizeof(gffx_tensor_view) ||
        faces->struct_size != (uint32_t)sizeof(gffx_tensor_view)) {
        return GFFX_STATUS_INVALID_ARGUMENT;
    }
    if (vertices->rank != 2u || faces->rank != 2u) return GFFX_STATUS_INVALID_ARGUMENT;
    if (vertices->shape[1] != 3 || faces->shape[1] != 3) return GFFX_STATUS_INVALID_ARGUMENT;
    if (vertices->dtype != GFFX_DTYPE_FLOAT32 && vertices->dtype != GFFX_DTYPE_FLOAT64) {
        return GFFX_STATUS_INVALID_ARGUMENT;
    }
    if (faces->dtype != GFFX_DTYPE_INT32) return GFFX_STATUS_INVALID_ARGUMENT;
    if (context != NULL) {
        status = gffx_validate_execution_context(context, diagnostic);
        if (status != GFFX_STATUS_OK) return status;
    }

    /* The header is re-parsed rather than trusted: the caller's copy fixes only the sizes it
     * allocated against, and a disagreement is exactly the case worth catching. */
    status = ply_parse_header((const unsigned char *)bytes, length, &layout, &format,
                              &data_offset);
    if (status != GFFX_STATUS_OK) return status;
    status = ply_check_layout(&layout);
    if (status != GFFX_STATUS_OK) return status;

    for (element = 0; element < layout.element_count; ++element) {
        if (layout.elements[element].kind == 1) vertex_count = layout.elements[element].count;
        else if (layout.elements[element].kind == 2) face_count = layout.elements[element].count;
    }
    if (header->vertex_count != vertex_count || header->face_count != face_count) {
        return GFFX_STATUS_INVALID_ARGUMENT;
    }
    if (vertices->shape[0] != vertex_count || faces->shape[0] != face_count) {
        return GFFX_STATUS_INVALID_ARGUMENT;
    }
    if ((vertex_count > 0 && vertices->data == NULL) ||
        (face_count > 0 && faces->data == NULL)) {
        return GFFX_STATUS_INVALID_ARGUMENT;
    }

    /* Two passes. The first validates without storing, so a defect discovered at the last face
     * cannot leave vertices already written; the second stores into outputs that are known to be
     * reachable. The contract promises that a failed read writes no output, and a single pass
     * cannot keep that promise. */
    for (pass = 0; pass < 2; ++pass) {
        reader.bytes = (const unsigned char *)bytes;
        reader.length = length;
        reader.cursor = data_offset;
        reader.binary = (format == GFFX_PLY_FORMAT_BINARY_LITTLE_ENDIAN);
        status = ply_scan_body(&reader, &layout, vertices, faces, pass == 1);
        if (status != GFFX_STATUS_OK) return status;
    }
    return GFFX_STATUS_OK;
}
