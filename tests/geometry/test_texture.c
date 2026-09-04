/*
 * Phase 4 acceptance fixtures TX-01..TX-16 for render.texture and render.texture_pyramid.
 *
 * Fixture numbers match TEXTURE_ACCEPTANCE_V0_1.md. Failures return the source line.
 *
 * This is a red baseline: it is written before any implementation exists, against the acceptance
 * contract rather than against code. Nothing here may be relaxed to make an implementation pass.
 * Where the contract says a comparison is exact, it is compared exactly - the pyramid reduction
 * and the collapsed bilinear weights both have exactly representable answers, and a tolerance
 * there would silently admit a schedule-dependent CUDA reduction that violates section 2.6.
 */

#include <gffx/execution.h>
#include <gffx/render.h>
#include <gffx/status.h>
#include <gffx/tensor.h>

#include <math.h>
#include <stdint.h>
#include <string.h>

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (0)

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

static void set_component(void *data, gffx_dtype dtype, int64_t index, double value) {
    if (dtype == GFFX_DTYPE_FLOAT64) ((double *)data)[index] = value;
    else ((float *)data)[index] = (float)value;
}

static double get_component(const void *data, gffx_dtype dtype, int64_t index) {
    if (dtype == GFFX_DTYPE_FLOAT64) return ((const double *)data)[index];
    return (double)((const float *)data)[index];
}

static size_t element_size(gffx_dtype dtype) {
    return dtype == GFFX_DTYPE_FLOAT64 ? sizeof(double) : sizeof(float);
}

static gffx_diagnostic_buffer make_diagnostic(void) {
    gffx_diagnostic_buffer diagnostic = {0};
    diagnostic.struct_size = (uint32_t)sizeof(diagnostic);
    diagnostic.abi_version = GFFX_ABI_VERSION;
    return diagnostic;
}

/* ------------------------------------------------------------------ shared invocation helpers */

static gffx_status build_pyramid(
    const void *texture, int64_t height, int64_t width, int64_t channels, int64_t levels,
    gffx_dtype dtype, void *pyramid_out, int32_t *offsets_out, int64_t offsets_capacity,
    int64_t pyramid_capacity
) {
    gffx_execution_context context = cpu_context();
    gffx_diagnostic_buffer diagnostic = make_diagnostic();
    int64_t texture_shape[3];
    int64_t texture_strides[3];
    int64_t pyramid_shape[1];
    int64_t pyramid_strides[1];
    int64_t offsets_shape[1];
    int64_t offsets_strides[1];
    gffx_tensor_view texture_view, pyramid_view, offsets_view;

    texture_shape[0] = height; texture_shape[1] = width; texture_shape[2] = channels;
    texture_strides[0] = width * channels; texture_strides[1] = channels; texture_strides[2] = 1;
    /* The true element capacity of the caller's buffer. Deriving it from the level
     * chain would let the view claim storage the test does not own, which the
     * aliasing check then reports as an overlap with whatever follows on the stack. */
    pyramid_shape[0] = pyramid_capacity;
    pyramid_strides[0] = 1;
    offsets_shape[0] = offsets_capacity;
    offsets_strides[0] = 1;

    texture_view = make_view((void *)texture, dtype, 3u, texture_shape, texture_strides,
                             GFFX_TENSOR_READ_ONLY);
    pyramid_view = make_view(pyramid_out, dtype, 1u, pyramid_shape, pyramid_strides,
                             GFFX_TENSOR_OUTPUT);
    offsets_view = make_view(offsets_out, GFFX_DTYPE_INT32, 1u, offsets_shape, offsets_strides,
                             GFFX_TENSOR_OUTPUT);
    return gffx_render_texture_pyramid(&texture_view, levels, &context, &pyramid_view,
                                       &offsets_view, NULL, &diagnostic);
}

static gffx_status sample_ex(
    const void *pyramid, const int32_t *offsets, int64_t level_count,
    int64_t texture_height, int64_t texture_width,
    const void *coordinates, int64_t count, int64_t channels,
    const void *derivatives, const void *lod,
    uint32_t filter, uint32_t mip_filter, uint32_t wrap_u, uint32_t wrap_v,
    const void *border, gffx_dtype dtype, void *samples_out
) {
    gffx_execution_context context = cpu_context();
    gffx_diagnostic_buffer diagnostic = make_diagnostic();
    int64_t pyramid_shape[1], pyramid_strides[1];
    int64_t offsets_shape[1], offsets_strides[1];
    int64_t coordinate_shape[2], coordinate_strides[2];
    int64_t derivative_shape[2], derivative_strides[2];
    int64_t lod_shape[1], lod_strides[1];
    int64_t border_shape[1], border_strides[1];
    int64_t sample_shape[2], sample_strides[2];
    gffx_tensor_view pyramid_view, offsets_view, coordinate_view, sample_view, border_view;
    gffx_tensor_view derivative_view, lod_view;

    pyramid_shape[0] = offsets[level_count]; pyramid_strides[0] = 1;
    offsets_shape[0] = level_count + 1; offsets_strides[0] = 1;
    coordinate_shape[0] = count; coordinate_shape[1] = 2;
    coordinate_strides[0] = 2; coordinate_strides[1] = 1;
    derivative_shape[0] = count; derivative_shape[1] = 4;
    derivative_strides[0] = 4; derivative_strides[1] = 1;
    lod_shape[0] = count; lod_strides[0] = 1;
    border_shape[0] = channels; border_strides[0] = 1;
    sample_shape[0] = count; sample_shape[1] = channels;
    sample_strides[0] = channels; sample_strides[1] = 1;

    pyramid_view = make_view((void *)pyramid, dtype, 1u, pyramid_shape, pyramid_strides,
                             GFFX_TENSOR_READ_ONLY);
    offsets_view = make_view((void *)offsets, GFFX_DTYPE_INT32, 1u, offsets_shape, offsets_strides,
                             GFFX_TENSOR_READ_ONLY);
    coordinate_view = make_view((void *)coordinates, dtype, 2u, coordinate_shape,
                                coordinate_strides, GFFX_TENSOR_READ_ONLY);
    derivative_view = make_view((void *)derivatives, dtype, 2u, derivative_shape,
                                derivative_strides, GFFX_TENSOR_READ_ONLY);
    lod_view = make_view((void *)lod, dtype, 1u, lod_shape, lod_strides, GFFX_TENSOR_READ_ONLY);
    border_view = make_view((void *)border, dtype, 1u, border_shape, border_strides,
                            GFFX_TENSOR_READ_ONLY);
    sample_view = make_view(samples_out, dtype, 2u, sample_shape, sample_strides,
                            GFFX_TENSOR_OUTPUT);

    return gffx_render_texture(&pyramid_view, &offsets_view,
                               texture_height, texture_width, &coordinate_view,
                               derivatives ? &derivative_view : NULL,
                               lod ? &lod_view : NULL,
                               filter, mip_filter, wrap_u, wrap_v, &border_view,
                               &context, &sample_view, NULL, &diagnostic);
}

/* ------------------------------------------------------------------------ TX-01..TX-04 filters */

static int test_tx01_tx03_nearest_and_bilinear(gffx_dtype dtype) {
    /* 4x4 single channel, value = row*4 + column, so every texel is distinct. */
    unsigned char texture[16 * sizeof(double)];
    unsigned char pyramid[64 * sizeof(double)];
    unsigned char coordinates[8 * sizeof(double)];
    unsigned char samples[4 * sizeof(double)];
    unsigned char border[1 * sizeof(double)];
    int32_t offsets[8] = {0};
    int64_t index;

    for (index = 0; index < 16; ++index) set_component(texture, dtype, index, (double)index);
    set_component(border, dtype, 0, 0.0);
    CHECK(build_pyramid(texture, 4, 4, 1, 1, dtype, pyramid, offsets, 2, 64) == GFFX_STATUS_OK);

    /* TX-01: texel centres under NEAREST return their own texel exactly. */
    set_component(coordinates, dtype, 0, 0.125); set_component(coordinates, dtype, 1, 0.125);
    set_component(coordinates, dtype, 2, 0.875); set_component(coordinates, dtype, 3, 0.875);
    set_component(coordinates, dtype, 4, 0.625); set_component(coordinates, dtype, 5, 0.125);
    CHECK(sample_ex(pyramid, offsets, 1, 4, 4, coordinates, 3, 1, NULL, NULL,
                 GFFX_FILTER_NEAREST, GFFX_MIP_NEAREST,
                 GFFX_WRAP_CLAMP, GFFX_WRAP_CLAMP, border, dtype, samples)
          == GFFX_STATUS_OK);
    CHECK(get_component(samples, dtype, 0) == 0.0);
    CHECK(get_component(samples, dtype, 1) == 15.0);
    CHECK(get_component(samples, dtype, 2) == 2.0);

    /* TX-03: BILINEAR at a texel centre collapses to one weight, so it equals NEAREST exactly. */
    CHECK(sample_ex(pyramid, offsets, 1, 4, 4, coordinates, 3, 1, NULL, NULL,
                 GFFX_FILTER_BILINEAR, GFFX_MIP_NEAREST,
                 GFFX_WRAP_CLAMP, GFFX_WRAP_CLAMP, border, dtype, samples)
          == GFFX_STATUS_OK);
    CHECK(get_component(samples, dtype, 0) == 0.0);
    CHECK(get_component(samples, dtype, 1) == 15.0);
    CHECK(get_component(samples, dtype, 2) == 2.0);
    return 0;
}

static int test_tx02_bilinear_centre(gffx_dtype dtype) {
    /* 2x2 with values 1,2,3,4; the exact centre is their mean, 2.5, exact in both dtypes. */
    unsigned char texture[4 * sizeof(double)];
    unsigned char pyramid[16 * sizeof(double)];
    unsigned char coordinates[2 * sizeof(double)];
    unsigned char samples[1 * sizeof(double)];
    unsigned char border[1 * sizeof(double)];
    int32_t offsets[4] = {0};

    set_component(texture, dtype, 0, 1.0); set_component(texture, dtype, 1, 2.0);
    set_component(texture, dtype, 2, 3.0); set_component(texture, dtype, 3, 4.0);
    set_component(border, dtype, 0, 0.0);
    set_component(coordinates, dtype, 0, 0.5); set_component(coordinates, dtype, 1, 0.5);
    CHECK(build_pyramid(texture, 2, 2, 1, 1, dtype, pyramid, offsets, 2, 16) == GFFX_STATUS_OK);
    CHECK(sample_ex(pyramid, offsets, 1, 2, 2, coordinates, 1, 1, NULL, NULL,
                 GFFX_FILTER_BILINEAR, GFFX_MIP_NEAREST,
                 GFFX_WRAP_CLAMP, GFFX_WRAP_CLAMP, border, dtype, samples)
          == GFFX_STATUS_OK);
    CHECK(get_component(samples, dtype, 0) == 2.5);
    return 0;
}

static int test_tx04_non_square(gffx_dtype dtype) {
    /* 8 wide, 2 tall. u and v must scale by W and H independently; a square assumption fails here.
     * u = 0.5625 is the centre of column 4; v = 0.75 is the centre of row 1. */
    unsigned char texture[16 * sizeof(double)];
    unsigned char pyramid[64 * sizeof(double)];
    unsigned char coordinates[2 * sizeof(double)];
    unsigned char samples[1 * sizeof(double)];
    unsigned char border[1 * sizeof(double)];
    int32_t offsets[8] = {0};
    int64_t index;

    for (index = 0; index < 16; ++index) set_component(texture, dtype, index, (double)index);
    set_component(border, dtype, 0, 0.0);
    set_component(coordinates, dtype, 0, 0.5625); set_component(coordinates, dtype, 1, 0.75);
    CHECK(build_pyramid(texture, 2, 8, 1, 1, dtype, pyramid, offsets, 2, 64) == GFFX_STATUS_OK);
    CHECK(sample_ex(pyramid, offsets, 1, 2, 8, coordinates, 1, 1, NULL, NULL,
                 GFFX_FILTER_NEAREST, GFFX_MIP_NEAREST,
                 GFFX_WRAP_CLAMP, GFFX_WRAP_CLAMP, border, dtype, samples)
          == GFFX_STATUS_OK);
    CHECK(get_component(samples, dtype, 0) == 12.0);
    return 0;
}

/* ------------------------------------------------------------------- TX-05..TX-07 the pyramid */

static int test_tx05_constant_pyramid(gffx_dtype dtype) {
    /* A constant texture must stay exactly constant at every level: the box filter of four equal
     * values is that value, with no rounding. Offsets for 4x4x1 are 16, 20, 21, 22. */
    unsigned char texture[16 * sizeof(double)];
    unsigned char pyramid[64 * sizeof(double)];
    int32_t offsets[8] = {0};
    int64_t index;

    for (index = 0; index < 16; ++index) set_component(texture, dtype, index, 0.375);
    CHECK(build_pyramid(texture, 4, 4, 1, 0, dtype, pyramid, offsets, 4, 64) == GFFX_STATUS_OK);
    CHECK(offsets[0] == 0);
    CHECK(offsets[1] == 16);
    CHECK(offsets[2] == 20);
    CHECK(offsets[3] == 21);
    for (index = 0; index < 21; ++index) {
        CHECK(get_component(pyramid, dtype, index) == 0.375);
    }
    return 0;
}

static int test_tx06_odd_dimensions(gffx_dtype dtype) {
    /* 5x3 must produce 5x3, 2x1, 1x1. The dropped odd row and column are discarded, not
     * reweighted: level 1 texel 0 is the mean of the four texels at rows 0-1, columns 0-1. */
    unsigned char texture[15 * sizeof(double)];
    unsigned char pyramid[64 * sizeof(double)];
    int32_t offsets[8] = {0};
    int64_t index;
    double expected;

    for (index = 0; index < 15; ++index) set_component(texture, dtype, index, (double)index);
    CHECK(build_pyramid(texture, 5, 3, 1, 0, dtype, pyramid, offsets, 4, 64) == GFFX_STATUS_OK);
    CHECK(offsets[1] == 15);      /* 5x3 */
    CHECK(offsets[2] == 15 + 2);  /* 2x1 */
    CHECK(offsets[3] == 15 + 2 + 1);
    expected = (0.0 + 1.0 + 3.0 + 4.0) / 4.0;
    CHECK(get_component(pyramid, dtype, offsets[1]) == expected);
    return 0;
}

static int test_tx07_degenerate_axis(gffx_dtype dtype) {
    /* 1x8: only the wide axis halves, and the axis already at 1 is carried through, so the
     * reduction is a two-texel mean rather than four. Levels are 1x8, 1x4, 1x2, 1x1. */
    unsigned char texture[8 * sizeof(double)];
    unsigned char pyramid[32 * sizeof(double)];
    int32_t offsets[8] = {0};
    int64_t index;

    for (index = 0; index < 8; ++index) set_component(texture, dtype, index, (double)index);
    CHECK(build_pyramid(texture, 1, 8, 1, 0, dtype, pyramid, offsets, 5, 32) == GFFX_STATUS_OK);
    CHECK(offsets[1] == 8);
    CHECK(offsets[2] == 12);
    CHECK(offsets[3] == 14);
    CHECK(offsets[4] == 15);
    CHECK(get_component(pyramid, dtype, offsets[1]) == 0.5);      /* mean of 0 and 1 */
    CHECK(get_component(pyramid, dtype, offsets[1] + 3) == 6.5);  /* mean of 6 and 7 */
    return 0;
}

/* -------------------------------------------------------------------- TX-08..TX-11 the mip level */

static int test_tx08_explicit_lod(gffx_dtype dtype) {
    unsigned char texture[16 * sizeof(double)];
    unsigned char pyramid[64 * sizeof(double)];
    unsigned char coordinates[2 * sizeof(double)];
    unsigned char lod[1 * sizeof(double)];
    unsigned char samples[1 * sizeof(double)];
    unsigned char border[1 * sizeof(double)];
    int32_t offsets[8] = {0};
    int64_t index;
    double level0, level1;

    for (index = 0; index < 16; ++index) set_component(texture, dtype, index, (double)index);
    set_component(border, dtype, 0, 0.0);
    set_component(coordinates, dtype, 0, 0.5); set_component(coordinates, dtype, 1, 0.5);
    CHECK(build_pyramid(texture, 4, 4, 1, 0, dtype, pyramid, offsets, 4, 64) == GFFX_STATUS_OK);

    set_component(lod, dtype, 0, 0.0);
    CHECK(sample_ex(pyramid, offsets, 3, 4, 4, coordinates, 1, 1, NULL, lod,
                 GFFX_FILTER_NEAREST, GFFX_MIP_NEAREST,
                 GFFX_WRAP_CLAMP, GFFX_WRAP_CLAMP, border, dtype, samples)
          == GFFX_STATUS_OK);
    level0 = get_component(samples, dtype, 0);

    set_component(lod, dtype, 0, 1.0);
    CHECK(sample_ex(pyramid, offsets, 3, 4, 4, coordinates, 1, 1, NULL, lod,
                 GFFX_FILTER_NEAREST, GFFX_MIP_NEAREST,
                 GFFX_WRAP_CLAMP, GFFX_WRAP_CLAMP, border, dtype, samples)
          == GFFX_STATUS_OK);
    level1 = get_component(samples, dtype, 0);
    CHECK(level0 != level1);

    /* lod 0.5 under LINEAR is the midpoint of the two level samples. */
    set_component(lod, dtype, 0, 0.5);
    CHECK(sample_ex(pyramid, offsets, 3, 4, 4, coordinates, 1, 1, NULL, lod,
                 GFFX_FILTER_NEAREST, GFFX_MIP_LINEAR,
                 GFFX_WRAP_CLAMP, GFFX_WRAP_CLAMP, border, dtype, samples)
          == GFFX_STATUS_OK);
    CHECK(fabs(get_component(samples, dtype, 0) - 0.5 * (level0 + level1)) < 1e-6);
    return 0;
}

static int test_tx09_tx11_derivative_lod(gffx_dtype dtype) {
    /* rho of 1, 2 and 4 texels per pixel must select levels 0, 1 and 2. With W = H = 4, a
     * du/dx of 0.25 is exactly one texel, so rho = 1 and lod = log2(1) = 0. */
    unsigned char texture[16 * sizeof(double)];
    unsigned char pyramid[64 * sizeof(double)];
    unsigned char coordinates[2 * sizeof(double)];
    unsigned char derivatives[4 * sizeof(double)];
    unsigned char samples[1 * sizeof(double)];
    unsigned char border[1 * sizeof(double)];
    int32_t offsets[8] = {0};
    int64_t index;
    double by_derivative, by_lod;
    unsigned char lod[1 * sizeof(double)];
    double scales[3] = {0.25, 0.5, 1.0};
    double expected_levels[3] = {0.0, 1.0, 2.0};
    size_t k;

    for (index = 0; index < 16; ++index) set_component(texture, dtype, index, (double)index);
    set_component(border, dtype, 0, 0.0);
    set_component(coordinates, dtype, 0, 0.5); set_component(coordinates, dtype, 1, 0.5);
    CHECK(build_pyramid(texture, 4, 4, 1, 0, dtype, pyramid, offsets, 4, 64) == GFFX_STATUS_OK);

    for (k = 0; k < 3; ++k) {
        set_component(derivatives, dtype, 0, scales[k]);
        set_component(derivatives, dtype, 1, 0.0);
        set_component(derivatives, dtype, 2, 0.0);
        set_component(derivatives, dtype, 3, scales[k]);
        CHECK(sample_ex(pyramid, offsets, 3, 4, 4, coordinates, 1, 1, derivatives, NULL,
                     GFFX_FILTER_NEAREST, GFFX_MIP_NEAREST,
                     GFFX_WRAP_CLAMP, GFFX_WRAP_CLAMP, border, dtype, samples)
              == GFFX_STATUS_OK);
        by_derivative = get_component(samples, dtype, 0);

        set_component(lod, dtype, 0, expected_levels[k]);
        CHECK(sample_ex(pyramid, offsets, 3, 4, 4, coordinates, 1, 1, NULL, lod,
                     GFFX_FILTER_NEAREST, GFFX_MIP_NEAREST,
                     GFFX_WRAP_CLAMP, GFFX_WRAP_CLAMP, border, dtype, samples)
              == GFFX_STATUS_OK);
        by_lod = get_component(samples, dtype, 0);
        CHECK(by_derivative == by_lod);
    }

    /* TX-11: derivatives far past the coarsest level clamp rather than reading out of range. */
    set_component(derivatives, dtype, 0, 64.0);
    set_component(derivatives, dtype, 1, 0.0);
    set_component(derivatives, dtype, 2, 0.0);
    set_component(derivatives, dtype, 3, 64.0);
    CHECK(sample_ex(pyramid, offsets, 3, 4, 4, coordinates, 1, 1, derivatives, NULL,
                 GFFX_FILTER_NEAREST, GFFX_MIP_NEAREST,
                 GFFX_WRAP_CLAMP, GFFX_WRAP_CLAMP, border, dtype, samples)
          == GFFX_STATUS_OK);
    set_component(lod, dtype, 0, 2.0);
    by_derivative = get_component(samples, dtype, 0);
    CHECK(sample_ex(pyramid, offsets, 3, 4, 4, coordinates, 1, 1, NULL, lod,
                 GFFX_FILTER_NEAREST, GFFX_MIP_NEAREST,
                 GFFX_WRAP_CLAMP, GFFX_WRAP_CLAMP, border, dtype, samples)
          == GFFX_STATUS_OK);
    CHECK(by_derivative == get_component(samples, dtype, 0));
    return 0;
}

static int test_tx10_zero_derivative(gffx_dtype dtype) {
    /* A stationary sample has rho = 0. log2(0) is negative infinity, so the contract clamps rho
     * before the logarithm; the result must be level 0 and not a NaN or an out-of-range read. */
    unsigned char texture[16 * sizeof(double)];
    unsigned char pyramid[64 * sizeof(double)];
    unsigned char coordinates[2 * sizeof(double)];
    unsigned char derivatives[4 * sizeof(double)];
    unsigned char samples[1 * sizeof(double)];
    unsigned char border[1 * sizeof(double)];
    unsigned char lod[1 * sizeof(double)];
    int32_t offsets[8] = {0};
    int64_t index;
    double by_zero_derivative;

    for (index = 0; index < 16; ++index) set_component(texture, dtype, index, (double)index);
    set_component(border, dtype, 0, 0.0);
    set_component(coordinates, dtype, 0, 0.5); set_component(coordinates, dtype, 1, 0.5);
    for (index = 0; index < 4; ++index) set_component(derivatives, dtype, index, 0.0);
    CHECK(build_pyramid(texture, 4, 4, 1, 0, dtype, pyramid, offsets, 4, 64) == GFFX_STATUS_OK);

    CHECK(sample_ex(pyramid, offsets, 3, 4, 4, coordinates, 1, 1, derivatives, NULL,
                 GFFX_FILTER_BILINEAR, GFFX_MIP_LINEAR,
                 GFFX_WRAP_CLAMP, GFFX_WRAP_CLAMP, border, dtype, samples)
          == GFFX_STATUS_OK);
    by_zero_derivative = get_component(samples, dtype, 0);
    CHECK(by_zero_derivative == by_zero_derivative);  /* not NaN */

    set_component(lod, dtype, 0, 0.0);
    CHECK(sample_ex(pyramid, offsets, 3, 4, 4, coordinates, 1, 1, NULL, lod,
                 GFFX_FILTER_BILINEAR, GFFX_MIP_LINEAR,
                 GFFX_WRAP_CLAMP, GFFX_WRAP_CLAMP, border, dtype, samples)
          == GFFX_STATUS_OK);
    CHECK(by_zero_derivative == get_component(samples, dtype, 0));
    return 0;
}

/* ---------------------------------------------------------------------- TX-12..TX-13 wrapping */

static int test_tx12_tx13_wrap_modes(gffx_dtype dtype) {
    /* 4x1 row 10,20,30,40. Sampling outside [0,1] must resolve through the wrap mode. */
    unsigned char texture[4 * sizeof(double)];
    unsigned char pyramid[16 * sizeof(double)];
    unsigned char coordinates[4 * sizeof(double)];
    unsigned char samples[2 * sizeof(double)];
    unsigned char border[1 * sizeof(double)];
    int32_t offsets[8] = {0};

    set_component(texture, dtype, 0, 10.0); set_component(texture, dtype, 1, 20.0);
    set_component(texture, dtype, 2, 30.0); set_component(texture, dtype, 3, 40.0);
    set_component(border, dtype, 0, -7.0);
    CHECK(build_pyramid(texture, 1, 4, 1, 1, dtype, pyramid, offsets, 2, 16) == GFFX_STATUS_OK);

    /* u = -0.125 is the centre of the texel one to the left of the first; u = 1.125 one to the
     * right of the last. Under NEAREST each wrap mode has a single defined answer. */
    set_component(coordinates, dtype, 0, -0.125); set_component(coordinates, dtype, 1, 0.5);
    set_component(coordinates, dtype, 2, 1.125); set_component(coordinates, dtype, 3, 0.5);

    CHECK(sample_ex(pyramid, offsets, 1, 1, 4, coordinates, 2, 1, NULL, NULL,
                 GFFX_FILTER_NEAREST, GFFX_MIP_NEAREST,
                 GFFX_WRAP_REPEAT, GFFX_WRAP_CLAMP, border, dtype, samples)
          == GFFX_STATUS_OK);
    CHECK(get_component(samples, dtype, 0) == 40.0);
    CHECK(get_component(samples, dtype, 1) == 10.0);

    CHECK(sample_ex(pyramid, offsets, 1, 1, 4, coordinates, 2, 1, NULL, NULL,
                 GFFX_FILTER_NEAREST, GFFX_MIP_NEAREST,
                 GFFX_WRAP_CLAMP, GFFX_WRAP_CLAMP, border, dtype, samples)
          == GFFX_STATUS_OK);
    CHECK(get_component(samples, dtype, 0) == 10.0);
    CHECK(get_component(samples, dtype, 1) == 40.0);

    CHECK(sample_ex(pyramid, offsets, 1, 1, 4, coordinates, 2, 1, NULL, NULL,
                 GFFX_FILTER_NEAREST, GFFX_MIP_NEAREST,
                 GFFX_WRAP_MIRROR, GFFX_WRAP_CLAMP, border, dtype, samples)
          == GFFX_STATUS_OK);
    CHECK(get_component(samples, dtype, 0) == 10.0);
    CHECK(get_component(samples, dtype, 1) == 40.0);

    CHECK(sample_ex(pyramid, offsets, 1, 1, 4, coordinates, 2, 1, NULL, NULL,
                 GFFX_FILTER_NEAREST, GFFX_MIP_NEAREST,
                 GFFX_WRAP_BORDER, GFFX_WRAP_CLAMP, border, dtype, samples)
          == GFFX_STATUS_OK);
    CHECK(get_component(samples, dtype, 0) == -7.0);
    CHECK(get_component(samples, dtype, 1) == -7.0);

    /* TX-13: a bilinear footprint straddling the left edge wraps each tap on its own. At u = 0.0
     * the two taps are the border value and texel 0, weighted equally. */
    set_component(coordinates, dtype, 0, 0.0); set_component(coordinates, dtype, 1, 0.5);
    CHECK(sample_ex(pyramid, offsets, 1, 1, 4, coordinates, 1, 1, NULL, NULL,
                 GFFX_FILTER_BILINEAR, GFFX_MIP_NEAREST,
                 GFFX_WRAP_BORDER, GFFX_WRAP_CLAMP, border, dtype, samples)
          == GFFX_STATUS_OK);
    CHECK(fabs(get_component(samples, dtype, 0) - 0.5 * (-7.0 + 10.0)) < 1e-6);
    return 0;
}

/* ------------------------------------------------------------------- TX-14..TX-15 edge cases */

static int test_tx14_nonfinite(void) {
    /* Non-finite texels propagate through the filter untouched, and a NaN coordinate produces a
     * NaN sample rather than an error. Sanitizing either would hide a caller's bug. */
    double texture[4] = {1.0, 0.0, 0.0, 0.0};
    double pyramid[16] = {0.0};
    double coordinates[2];
    double samples[1];
    double border[1] = {0.0};
    int32_t offsets[8] = {0};

    texture[1] = NAN;
    texture[2] = INFINITY;
    CHECK(build_pyramid(texture, 2, 2, 1, 1, GFFX_DTYPE_FLOAT64, pyramid, offsets, 2, 16)
          == GFFX_STATUS_OK);

    coordinates[0] = 0.75; coordinates[1] = 0.25;   /* the NaN texel */
    CHECK(sample_ex(pyramid, offsets, 1, 2, 2, coordinates, 1, 1, NULL, NULL,
                 GFFX_FILTER_NEAREST, GFFX_MIP_NEAREST,
                 GFFX_WRAP_CLAMP, GFFX_WRAP_CLAMP, border,
                 GFFX_DTYPE_FLOAT64, samples) == GFFX_STATUS_OK);
    CHECK(isnan(samples[0]));

    coordinates[0] = 0.25; coordinates[1] = 0.75;   /* the infinite texel */
    CHECK(sample_ex(pyramid, offsets, 1, 2, 2, coordinates, 1, 1, NULL, NULL,
                 GFFX_FILTER_NEAREST, GFFX_MIP_NEAREST,
                 GFFX_WRAP_CLAMP, GFFX_WRAP_CLAMP, border,
                 GFFX_DTYPE_FLOAT64, samples) == GFFX_STATUS_OK);
    CHECK(isinf(samples[0]));

    coordinates[0] = NAN; coordinates[1] = 0.5;
    CHECK(sample_ex(pyramid, offsets, 1, 2, 2, coordinates, 1, 1, NULL, NULL,
                 GFFX_FILTER_BILINEAR, GFFX_MIP_NEAREST,
                 GFFX_WRAP_CLAMP, GFFX_WRAP_CLAMP, border,
                 GFFX_DTYPE_FLOAT64, samples) == GFFX_STATUS_OK);
    CHECK(isnan(samples[0]));
    return 0;
}

static int test_tx15_validation(void) {
    double texture[4] = {1.0, 2.0, 3.0, 4.0};
    double pyramid[16] = {0.0};
    double coordinates[2] = {0.5, 0.5};
    double derivatives[4] = {0.0, 0.0, 0.0, 0.0};
    double lod[1] = {0.0};
    double samples[1];
    double border[1] = {0.0};
    int32_t offsets[8] = {0};

    CHECK(build_pyramid(texture, 2, 2, 1, 1, GFFX_DTYPE_FLOAT64, pyramid, offsets, 2, 16)
          == GFFX_STATUS_OK);

    /* derivatives and lod are mutually exclusive: supplying both is a caller error, not a
     * precedence question the library should resolve silently. */
    CHECK(sample_ex(pyramid, offsets, 1, 2, 2, coordinates, 1, 1, derivatives, lod,
                 GFFX_FILTER_NEAREST, GFFX_MIP_NEAREST,
                 GFFX_WRAP_CLAMP, GFFX_WRAP_CLAMP, border,
                 GFFX_DTYPE_FLOAT64, samples) == GFFX_STATUS_INVALID_ARGUMENT);

    /* Unknown enum values are rejected rather than defaulted. */
    CHECK(sample_ex(pyramid, offsets, 1, 2, 2, coordinates, 1, 1, NULL, NULL,
                 999u, GFFX_MIP_NEAREST,
                 GFFX_WRAP_CLAMP, GFFX_WRAP_CLAMP, border,
                 GFFX_DTYPE_FLOAT64, samples) == GFFX_STATUS_INVALID_ARGUMENT);
    CHECK(sample_ex(pyramid, offsets, 1, 2, 2, coordinates, 1, 1, NULL, NULL,
                 GFFX_FILTER_NEAREST, GFFX_MIP_NEAREST,
                 999u, GFFX_WRAP_CLAMP, border,
                 GFFX_DTYPE_FLOAT64, samples) == GFFX_STATUS_INVALID_ARGUMENT);

    /* An int32 texture has no defined filter result. */
    CHECK(build_pyramid(texture, 2, 2, 1, 1, GFFX_DTYPE_INT32, pyramid, offsets, 2, 16)
          == GFFX_STATUS_UNSUPPORTED);

    /* A negative level count is invalid; zero means the full chain and is valid. */
    CHECK(build_pyramid(texture, 2, 2, 1, -1, GFFX_DTYPE_FLOAT64, pyramid, offsets, 2, 16)
          == GFFX_STATUS_INVALID_ARGUMENT);
    return 0;
}

/* ---------------------------------------------------------------- TX-16 determinism and workspace */

static int test_tx16_determinism(gffx_dtype dtype) {
    unsigned char texture[16 * sizeof(double)];
    unsigned char pyramid[64 * sizeof(double)];
    unsigned char coordinates[8 * sizeof(double)];
    unsigned char first[4 * sizeof(double)];
    unsigned char second[4 * sizeof(double)];
    unsigned char border[1 * sizeof(double)];
    int32_t offsets[8] = {0};
    int64_t index;
    size_t bytes = 4u * element_size(dtype);
    uint32_t filters[2] = {GFFX_FILTER_NEAREST, GFFX_FILTER_BILINEAR};
    uint32_t mips[2] = {GFFX_MIP_NEAREST, GFFX_MIP_LINEAR};
    size_t f, m;

    for (index = 0; index < 16; ++index) set_component(texture, dtype, index, (double)index * 0.7);
    set_component(border, dtype, 0, 0.0);
    for (index = 0; index < 8; ++index) {
        set_component(coordinates, dtype, index, 0.1 + 0.11 * (double)index);
    }
    CHECK(build_pyramid(texture, 4, 4, 1, 0, dtype, pyramid, offsets, 4, 64) == GFFX_STATUS_OK);

    for (f = 0; f < 2; ++f) {
        for (m = 0; m < 2; ++m) {
            CHECK(sample_ex(pyramid, offsets, 3, 4, 4, coordinates, 4, 1, NULL, NULL, filters[f], mips[m],
                         GFFX_WRAP_REPEAT, GFFX_WRAP_REPEAT, border, dtype, first)
                  == GFFX_STATUS_OK);
            CHECK(sample_ex(pyramid, offsets, 3, 4, 4, coordinates, 4, 1, NULL, NULL, filters[f], mips[m],
                         GFFX_WRAP_REPEAT, GFFX_WRAP_REPEAT, border, dtype, second)
                  == GFFX_STATUS_OK);
            CHECK(memcmp(first, second, bytes) == 0);
        }
    }
    return 0;
}

static int test_workspace_query(void) {
    uint64_t required_bytes = UINT64_MAX;
    uint64_t required_alignment = 0;
    gffx_execution_context context = cpu_context();
    gffx_diagnostic_buffer diagnostic = make_diagnostic();

    CHECK(gffx_render_texture_workspace(16, 1, GFFX_DTYPE_FLOAT64, &context, &required_bytes,
                                        &required_alignment, &diagnostic) == GFFX_STATUS_OK);
    CHECK(required_bytes == 0u);
    CHECK(gffx_render_texture_pyramid_workspace(4, 4, 1, GFFX_DTYPE_FLOAT64, &context,
                                                &required_bytes, &required_alignment, &diagnostic)
          == GFFX_STATUS_OK);
    CHECK(required_bytes == 0u);
    return 0;
}


/* TX-16 gradients. Central differences at 1e-6 in float64 only; float32 has too little precision
 * for a finite-difference comparison at this tolerance, which is why the contract states float64.
 * BILINEAR is differentiable in the coordinate almost everywhere, and NEAREST is not - it must
 * return exactly zero rather than a small wrong number. */

static gffx_status sample_backward_ex(
    const void *pyramid, const int32_t *offsets, int64_t level_count,
    int64_t texture_height, int64_t texture_width,
    const void *coordinates, int64_t count, int64_t channels,
    const void *grad_samples, uint32_t filter,
    void *grad_pyramid_out, void *grad_coordinates_out
) {
    gffx_execution_context context = cpu_context();
    gffx_diagnostic_buffer diagnostic = make_diagnostic();
    int64_t pyramid_shape[1], pyramid_strides[1];
    int64_t offsets_shape[1], offsets_strides[1];
    int64_t coordinate_shape[2], coordinate_strides[2];
    int64_t sample_shape[2], sample_strides[2];
    int64_t border_shape[1], border_strides[1];
    double border[4] = {0.0, 0.0, 0.0, 0.0};
    gffx_tensor_view pyramid_view, offsets_view, coordinate_view, grad_sample_view;
    gffx_tensor_view grad_pyramid_view, grad_coordinate_view, border_view;

    pyramid_shape[0] = offsets[level_count]; pyramid_strides[0] = 1;
    offsets_shape[0] = level_count + 1; offsets_strides[0] = 1;
    coordinate_shape[0] = count; coordinate_shape[1] = 2;
    coordinate_strides[0] = 2; coordinate_strides[1] = 1;
    sample_shape[0] = count; sample_shape[1] = channels;
    sample_strides[0] = channels; sample_strides[1] = 1;
    border_shape[0] = channels; border_strides[0] = 1;

    pyramid_view = make_view((void *)pyramid, GFFX_DTYPE_FLOAT64, 1u, pyramid_shape,
                             pyramid_strides, GFFX_TENSOR_READ_ONLY);
    offsets_view = make_view((void *)offsets, GFFX_DTYPE_INT32, 1u, offsets_shape, offsets_strides,
                             GFFX_TENSOR_READ_ONLY);
    coordinate_view = make_view((void *)coordinates, GFFX_DTYPE_FLOAT64, 2u, coordinate_shape,
                                coordinate_strides, GFFX_TENSOR_READ_ONLY);
    grad_sample_view = make_view((void *)grad_samples, GFFX_DTYPE_FLOAT64, 2u, sample_shape,
                                 sample_strides, GFFX_TENSOR_READ_ONLY);
    border_view = make_view(border, GFFX_DTYPE_FLOAT64, 1u, border_shape, border_strides,
                            GFFX_TENSOR_READ_ONLY);
    grad_pyramid_view = make_view(grad_pyramid_out, GFFX_DTYPE_FLOAT64, 1u, pyramid_shape,
                                  pyramid_strides, GFFX_TENSOR_OUTPUT);
    grad_coordinate_view = make_view(grad_coordinates_out, GFFX_DTYPE_FLOAT64, 2u,
                                     coordinate_shape, coordinate_strides,
                                     GFFX_TENSOR_OUTPUT);

    return gffx_render_texture_backward(
        &pyramid_view, &offsets_view, texture_height, texture_width, &coordinate_view, NULL, NULL,
        filter, GFFX_MIP_NEAREST, GFFX_WRAP_CLAMP, GFFX_WRAP_CLAMP, &border_view,
        &grad_sample_view, &context, &grad_pyramid_view, &grad_coordinate_view,
        NULL, &diagnostic);
}

static int test_tx16_gradients(void) {
    double texture[16];
    double pyramid[64] = {0.0};
    double coordinates[2] = {0.3125, 0.6875};
    double grad_samples[1] = {1.0};
    double grad_pyramid[64] = {0.0};
    double grad_coordinates[2] = {0.0, 0.0};
    double samples[1];
    double border[1] = {0.0};
    int32_t offsets[8] = {0};
    const double step = 1e-6;
    int64_t index;
    int axis;
    double weight_sum = 0.0;

    for (index = 0; index < 16; ++index) texture[index] = 0.3 + 0.61 * (double)index;
    CHECK(build_pyramid(texture, 4, 4, 1, 1, GFFX_DTYPE_FLOAT64, pyramid, offsets, 2, 64)
          == GFFX_STATUS_OK);

    CHECK(sample_backward_ex(pyramid, offsets, 1, 4, 4, coordinates, 1, 1, grad_samples,
                          GFFX_FILTER_BILINEAR, grad_pyramid, grad_coordinates)
          == GFFX_STATUS_OK);

    /* Each coordinate component against a central difference. */
    for (axis = 0; axis < 2; ++axis) {
        double plus, minus, numeric, saved = coordinates[axis];
        coordinates[axis] = saved + step;
        CHECK(sample_ex(pyramid, offsets, 1, 4, 4, coordinates, 1, 1, NULL, NULL, GFFX_FILTER_BILINEAR,
                     GFFX_MIP_NEAREST, GFFX_WRAP_CLAMP, GFFX_WRAP_CLAMP, border,
                     GFFX_DTYPE_FLOAT64, samples) == GFFX_STATUS_OK);
        plus = samples[0];
        coordinates[axis] = saved - step;
        CHECK(sample_ex(pyramid, offsets, 1, 4, 4, coordinates, 1, 1, NULL, NULL, GFFX_FILTER_BILINEAR,
                     GFFX_MIP_NEAREST, GFFX_WRAP_CLAMP, GFFX_WRAP_CLAMP, border,
                     GFFX_DTYPE_FLOAT64, samples) == GFFX_STATUS_OK);
        minus = samples[0];
        coordinates[axis] = saved;
        numeric = (plus - minus) / (2.0 * step);
        CHECK(fabs(grad_coordinates[axis] - numeric) < 1e-6);
    }

    /* grad_pyramid is the scatter of the bilinear weights, so with an upstream gradient of 1 it
     * sums to exactly 1 across the four taps and is zero everywhere else. Checking the sum rather
     * than each weight is what catches a scatter that drops or double-counts a tap. */
    for (index = 0; index < offsets[1]; ++index) weight_sum += grad_pyramid[index];
    CHECK(fabs(weight_sum - 1.0) < 1e-12);

    /* NEAREST: exactly zero, not merely small. */
    grad_coordinates[0] = 1.0; grad_coordinates[1] = 1.0;
    CHECK(sample_backward_ex(pyramid, offsets, 1, 4, 4, coordinates, 1, 1, grad_samples,
                          GFFX_FILTER_NEAREST, grad_pyramid, grad_coordinates)
          == GFFX_STATUS_OK);
    CHECK(grad_coordinates[0] == 0.0);
    CHECK(grad_coordinates[1] == 0.0);
    return 0;
}

static int test_tx16_pyramid_gradient(void) {
    /* The transpose of the box filter. Seeding one level-1 texel with 1.0 must deposit exactly
     * 0.25 into each of the four level-0 texels beneath it and nothing anywhere else. Checking the
     * placement rather than only the total is what catches a transpose that scatters to the right
     * count of texels in the wrong positions. */
    double texture[16];
    double pyramid[64] = {0.0};
    double grad_pyramid[64] = {0.0};
    double grad_texture[16] = {0.0};
    int32_t offsets[8] = {0};
    gffx_execution_context context = cpu_context();
    gffx_diagnostic_buffer diagnostic = make_diagnostic();
    int64_t offsets_shape[1], offsets_strides[1];
    int64_t pyramid_shape[1], pyramid_strides[1];
    int64_t texture_shape[3], texture_strides[3];
    gffx_tensor_view offsets_view, grad_pyramid_view, grad_texture_view;
    int64_t index;
    double total = 0.0;

    for (index = 0; index < 16; ++index) texture[index] = (double)index;
    CHECK(build_pyramid(texture, 4, 4, 1, 0, GFFX_DTYPE_FLOAT64, pyramid, offsets, 4, 64)
          == GFFX_STATUS_OK);

    /* Level 1 is 2x2; its texel 0 covers level-0 texels 0, 1, 4 and 5. */
    grad_pyramid[offsets[1]] = 1.0;

    offsets_shape[0] = 4; offsets_strides[0] = 1;
    pyramid_shape[0] = offsets[3]; pyramid_strides[0] = 1;
    texture_shape[0] = 4; texture_shape[1] = 4; texture_shape[2] = 1;
    texture_strides[0] = 4; texture_strides[1] = 1; texture_strides[2] = 1;

    offsets_view = make_view(offsets, GFFX_DTYPE_INT32, 1u, offsets_shape, offsets_strides,
                             GFFX_TENSOR_READ_ONLY);
    grad_pyramid_view = make_view(grad_pyramid, GFFX_DTYPE_FLOAT64, 1u, pyramid_shape,
                                  pyramid_strides, GFFX_TENSOR_READ_ONLY);
    grad_texture_view = make_view(grad_texture, GFFX_DTYPE_FLOAT64, 3u, texture_shape,
                                  texture_strides, GFFX_TENSOR_OUTPUT);

    CHECK(gffx_render_texture_pyramid_backward(&offsets_view, 4, 4, 1, &grad_pyramid_view,
                                               &context, &grad_texture_view, NULL, &diagnostic)
          == GFFX_STATUS_OK);

    CHECK(grad_texture[0] == 0.25);
    CHECK(grad_texture[1] == 0.25);
    CHECK(grad_texture[4] == 0.25);
    CHECK(grad_texture[5] == 0.25);
    for (index = 0; index < 16; ++index) total += grad_texture[index];
    CHECK(total == 1.0);
    return 0;
}

static int test_tx16_pyramid_gradient_multilevel(void) {
    /* Two levels of reduction, which is where a transpose written as a level-by-level cascade
     * goes wrong: level 2 of a 4x4 pyramid is a single texel covering all sixteen level-0 texels,
     * so seeding it with 1.0 must deposit exactly 1/16 in every one of them. A cascade that
     * stops after one step deposits nothing here while still passing the single-level fixture. */
    double texture[16];
    double pyramid[64] = {0.0};
    double grad_pyramid[64] = {0.0};
    double grad_texture[16] = {0.0};
    int32_t offsets[8] = {0};
    gffx_execution_context context = cpu_context();
    gffx_diagnostic_buffer diagnostic = make_diagnostic();
    int64_t offsets_shape[1], offsets_strides[1];
    int64_t pyramid_shape[1], pyramid_strides[1];
    int64_t texture_shape[3], texture_strides[3];
    gffx_tensor_view offsets_view, grad_pyramid_view, grad_texture_view;
    int64_t index;
    double total = 0.0;

    for (index = 0; index < 16; ++index) texture[index] = (double)index;
    CHECK(build_pyramid(texture, 4, 4, 1, 0, GFFX_DTYPE_FLOAT64, pyramid, offsets, 4, 64)
          == GFFX_STATUS_OK);
    grad_pyramid[offsets[2]] = 1.0;

    offsets_shape[0] = 4; offsets_strides[0] = 1;
    pyramid_shape[0] = offsets[3]; pyramid_strides[0] = 1;
    texture_shape[0] = 4; texture_shape[1] = 4; texture_shape[2] = 1;
    texture_strides[0] = 4; texture_strides[1] = 1; texture_strides[2] = 1;

    offsets_view = make_view(offsets, GFFX_DTYPE_INT32, 1u, offsets_shape, offsets_strides,
                             GFFX_TENSOR_READ_ONLY);
    grad_pyramid_view = make_view(grad_pyramid, GFFX_DTYPE_FLOAT64, 1u, pyramid_shape,
                                  pyramid_strides, GFFX_TENSOR_READ_ONLY);
    grad_texture_view = make_view(grad_texture, GFFX_DTYPE_FLOAT64, 3u, texture_shape,
                                  texture_strides, GFFX_TENSOR_OUTPUT);

    CHECK(gffx_render_texture_pyramid_backward(&offsets_view, 4, 4, 1, &grad_pyramid_view,
                                               &context, &grad_texture_view, NULL, &diagnostic)
          == GFFX_STATUS_OK);
    for (index = 0; index < 16; ++index) {
        CHECK(grad_texture[index] == 0.0625);
        total += grad_texture[index];
    }
    CHECK(total == 1.0);
    return 0;
}

int main(void) {
    int result;
    gffx_dtype dtypes[2] = {GFFX_DTYPE_FLOAT32, GFFX_DTYPE_FLOAT64};
    size_t index;

    for (index = 0u; index < 2u; ++index) {
        gffx_dtype dtype = dtypes[index];
        result = test_tx01_tx03_nearest_and_bilinear(dtype); if (result != 0) return result;
        result = test_tx02_bilinear_centre(dtype); if (result != 0) return result;
        result = test_tx04_non_square(dtype); if (result != 0) return result;
        result = test_tx05_constant_pyramid(dtype); if (result != 0) return result;
        result = test_tx06_odd_dimensions(dtype); if (result != 0) return result;
        result = test_tx07_degenerate_axis(dtype); if (result != 0) return result;
        result = test_tx08_explicit_lod(dtype); if (result != 0) return result;
        result = test_tx09_tx11_derivative_lod(dtype); if (result != 0) return result;
        result = test_tx10_zero_derivative(dtype); if (result != 0) return result;
        result = test_tx12_tx13_wrap_modes(dtype); if (result != 0) return result;
        result = test_tx16_determinism(dtype); if (result != 0) return result;
    }
    result = test_tx14_nonfinite(); if (result != 0) return result;
    result = test_tx15_validation(); if (result != 0) return result;
    result = test_tx16_gradients(); if (result != 0) return result;
    result = test_tx16_pyramid_gradient(); if (result != 0) return result;
    result = test_tx16_pyramid_gradient_multilevel(); if (result != 0) return result;
    result = test_workspace_query(); if (result != 0) return result;
    return 0;
}
