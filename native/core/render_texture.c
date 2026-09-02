/*
 * render.texture_pyramid - Phase 4 CPU reference kernels.
 *
 * Level 0 is the input copied bit-for-bit; level l+1 is the arithmetic mean of each two-by-two
 * block of level l. The summation order is fixed as TEXTURE_ACCEPTANCE_V0_1.md section 2.1 writes
 * it, left to right then top to bottom, because floating-point addition is not associative: a
 * reduction free to reassociate would pass a tolerance test and silently violate the bit-identity
 * claim of section 2.6, which is a contract term rather than a test-strictness preference.
 *
 * An odd dimension drops its trailing row or column rather than reweighting, and a dimension that
 * has reached 1 is carried through unhalved, so the reduction there is a two-texel mean. Levels
 * occupy one contiguous buffer addressed by level_offsets, which is what lets a streaming caller
 * hold a pyramid as a single tensor and build it once.
 *
 * The backward is the transpose of that box filter. It is computed as a direct scatter from each
 * level to level 0 rather than as a level-by-level cascade: a cascade would have to mutate the
 * incoming gradient in place so a level-2 contribution could reach level 0 through level 1, which
 * would either corrupt a caller-owned input or need scratch the contract puts at zero bytes. The
 * direct form is exact rather than an approximation because a level-l texel's footprint in level 0
 * is a rectangle over which the accumulated weight is uniform. Both directions need zero
 * workspace.
 */

#include <gffx/execution.h>
#include <gffx/render.h>
#include <gffx/tensor.h>

#include "internal.h"
#include "mesh_common.h"

#include <math.h>
#include <stdint.h>

/* Upper bound on channels, so the mip blend can hold one level's sample on the
 * stack rather than allocating. Four covers RGBA; a larger texture is rejected
 * with a diagnostic rather than silently truncated. */
#define GFFX_TEXTURE_MAX_CHANNELS 4

/* Dimensions of level l given level 0, following section 2.1's halving rule. */
static void gffx_texture_level_extent(
    int64_t height, int64_t width, int64_t level, int64_t *out_height, int64_t *out_width
) {
    int64_t level_index;
    int64_t h = height;
    int64_t w = width;
    for (level_index = 0; level_index < level; ++level_index) {
        h = h > INT64_C(1) ? h / INT64_C(2) : INT64_C(1);
        w = w > INT64_C(1) ? w / INT64_C(2) : INT64_C(1);
    }
    *out_height = h;
    *out_width = w;
}

/*
 * The number of levels in a full chain. Section 2.1 states floor(log2(max(H,W))) + 1; this counts
 * the halvings directly rather than calling log2, because the floating-point logarithm of an exact
 * power of two is the one case where a rounding error changes the answer by a whole level.
 */
static int64_t gffx_texture_full_level_count(int64_t height, int64_t width) {
    int64_t count = INT64_C(1);
    int64_t h = height;
    int64_t w = width;
    while (h > INT64_C(1) || w > INT64_C(1)) {
        h = h > INT64_C(1) ? h / INT64_C(2) : INT64_C(1);
        w = w > INT64_C(1) ? w / INT64_C(2) : INT64_C(1);
        ++count;
    }
    return count;
}

GFFX_API gffx_status GFFX_CALL gffx_render_texture_pyramid_workspace(
    int64_t texture_height,
    int64_t texture_width,
    int64_t channel_count,
    gffx_dtype dtype,
    const gffx_execution_context *context,
    uint64_t *required_bytes,
    uint64_t *required_alignment,
    gffx_diagnostic_buffer *diagnostic
) {
    gffx_status status = gffx_internal_prepare_diagnostic(diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (required_bytes == NULL || required_alignment == NULL) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "workspace query result pointers must not be null"
        );
    }
    if (texture_height < INT64_C(0) || texture_width < INT64_C(0) ||
        channel_count < INT64_C(0)) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "texture extents and channel count must be nonnegative"
        );
    }
    if (dtype != GFFX_DTYPE_FLOAT32 && dtype != GFFX_DTYPE_FLOAT64) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_UNSUPPORTED,
            "render.texture_pyramid supports the float32 and float64 dtypes"
        );
    }
    status = gffx_validate_execution_context(context, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (context->device_type != GFFX_DEVICE_CPU) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_UNSUPPORTED,
            "render.texture_pyramid implements only the CPU backend in this phase"
        );
    }
    /* Section 2.9: zero bytes in both directions. The reduction reads and writes the caller's
     * pyramid buffer in place, level by level, and needs no scratch of its own. */
    *required_bytes = UINT64_C(0);
    *required_alignment = UINT64_C(1);
    return GFFX_STATUS_OK;
}

GFFX_API gffx_status GFFX_CALL gffx_render_texture_pyramid(
    const gffx_tensor_view *texture,
    int64_t levels,
    const gffx_execution_context *context,
    gffx_tensor_view *pyramid,
    gffx_tensor_view *level_offsets,
    const gffx_buffer *workspace,
    gffx_diagnostic_buffer *diagnostic
) {
    gffx_status status;
    int64_t height;
    int64_t width;
    int64_t channels;
    int64_t level_count;
    int64_t level;
    int64_t total_elements;
    int32_t *offset_data;

    if (context != NULL && context->struct_size >= sizeof(*context) &&
        context->device_type == GFFX_DEVICE_CUDA) {
        /* No CUDA provider publishes this operation yet. Returning UNSUPPORTED keeps the missing
         * kernel visible rather than silently running the CPU path on device memory. */
        return gffx_internal_fail(
            diagnostic, GFFX_STATUS_UNSUPPORTED,
            "the CUDA provider does not implement this operation");
    }
    status = gffx_internal_prepare_diagnostic(diagnostic);
    if (status != GFFX_STATUS_OK) return status;

    if (texture == NULL || texture->rank != 3u || texture->shape == NULL) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "texture must be an [H,W,C] view"
        );
    }
    status = gffx_validate_tensor_view(texture, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (texture->dtype != GFFX_DTYPE_FLOAT32 && texture->dtype != GFFX_DTYPE_FLOAT64) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_UNSUPPORTED,
            "render.texture_pyramid supports the float32 and float64 dtypes"
        );
    }
    status = gffx_validate_execution_context(context, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (context->device_type != GFFX_DEVICE_CPU ||
        texture->device_type != GFFX_DEVICE_CPU) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_UNSUPPORTED,
            "render.texture_pyramid implements only the CPU backend in this phase"
        );
    }
    if (workspace != NULL) {
        status = gffx_validate_buffer(workspace, diagnostic);
        if (status != GFFX_STATUS_OK) return status;
    }

    height = texture->shape[0];
    width = texture->shape[1];
    channels = texture->shape[2];
    if (height <= INT64_C(0) || width <= INT64_C(0) || channels <= INT64_C(0)) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "texture extents and channel count must be positive"
        );
    }
    if (levels < INT64_C(0)) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "level count must be nonnegative; zero requests the full chain"
        );
    }
    level_count = gffx_texture_full_level_count(height, width);
    if (levels != INT64_C(0)) {
        if (levels > level_count) {
            return gffx_internal_fail(
                diagnostic,
                GFFX_STATUS_INVALID_ARGUMENT,
                "level count exceeds the full chain for this texture"
            );
        }
        level_count = levels;
    }

    if (pyramid == NULL || pyramid->rank != 1u || pyramid->shape == NULL) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "pyramid must be a rank-1 output view"
        );
    }
    if (level_offsets == NULL || level_offsets->rank != 1u || level_offsets->shape == NULL ||
        level_offsets->shape[0] != level_count + INT64_C(1)) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "level offsets must be an [L+1] output view"
        );
    }
    status = gffx_validate_tensor_view(pyramid, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    status = gffx_validate_tensor_view(level_offsets, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (pyramid->dtype != texture->dtype) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "pyramid must match the texture dtype"
        );
    }
    if (level_offsets->dtype != GFFX_DTYPE_INT32) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "level offsets must be an int32 view"
        );
    }
    if ((pyramid->flags & GFFX_TENSOR_OUTPUT) == UINT32_C(0) ||
        (level_offsets->flags & GFFX_TENSOR_OUTPUT) == UINT32_C(0)) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "operation outputs must carry the output flag"
        );
    }
    if (pyramid->device_type != GFFX_DEVICE_CPU ||
        level_offsets->device_type != GFFX_DEVICE_CPU) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_UNSUPPORTED,
            "render.texture_pyramid implements only the CPU backend in this phase"
        );
    }
    if (gffx_mesh_views_overlap(pyramid, texture) ||
        gffx_mesh_views_overlap(level_offsets, texture) ||
        gffx_mesh_views_overlap(pyramid, level_offsets)) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "outputs may not alias an input or another output"
        );
    }

    /* Offsets first: the total is needed to check the pyramid's capacity before any write. */
    offset_data = (int32_t *)gffx_mesh_elements(level_offsets);
    total_elements = INT64_C(0);
    for (level = 0; level < level_count; ++level) {
        int64_t level_height;
        int64_t level_width;
        gffx_texture_level_extent(height, width, level, &level_height, &level_width);
        offset_data[level] = (int32_t)total_elements;
        total_elements += level_height * level_width * channels;
        if (total_elements > (int64_t)INT32_MAX) {
            return gffx_internal_fail(
                diagnostic,
                GFFX_STATUS_INVALID_ARGUMENT,
                "pyramid element count exceeds the int32 offset range"
            );
        }
    }
    offset_data[level_count] = (int32_t)total_elements;
    if (pyramid->shape[0] < total_elements) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "pyramid view is smaller than the level chain requires"
        );
    }

    if (texture->dtype == GFFX_DTYPE_FLOAT64) {
        const double *source = (const double *)gffx_mesh_elements_const(texture);
        double *target = (double *)gffx_mesh_elements(pyramid);
        int64_t index;
        for (index = 0; index < height * width * channels; ++index) target[index] = source[index];
        for (level = 1; level < level_count; ++level) {
            int64_t previous_height;
            int64_t previous_width;
            int64_t level_height;
            int64_t level_width;
            int64_t y;
            int64_t x;
            int64_t c;
            const double *previous;
            double *current;
            gffx_texture_level_extent(height, width, level - INT64_C(1), &previous_height,
                                      &previous_width);
            gffx_texture_level_extent(height, width, level, &level_height, &level_width);
            previous = target + offset_data[level - INT64_C(1)];
            current = target + offset_data[level];
            for (y = 0; y < level_height; ++y) {
                for (x = 0; x < level_width; ++x) {
                    /* A dimension already at 1 is carried through, so the block collapses to two
                     * texels or to one; the divisor follows the block actually read. */
                    int64_t y0 = previous_height > INT64_C(1) ? y * INT64_C(2) : y;
                    int64_t x0 = previous_width > INT64_C(1) ? x * INT64_C(2) : x;
                    int64_t y1 = previous_height > INT64_C(1) ? y0 + INT64_C(1) : y0;
                    int64_t x1 = previous_width > INT64_C(1) ? x0 + INT64_C(1) : x0;
                    double divisor = 1.0;
                    if (previous_height > INT64_C(1)) divisor *= 2.0;
                    if (previous_width > INT64_C(1)) divisor *= 2.0;
                    for (c = 0; c < channels; ++c) {
                        /* Fixed order: left to right, then top to bottom. */
                        double sum = previous[(y0 * previous_width + x0) * channels + c];
                        if (x1 != x0) sum += previous[(y0 * previous_width + x1) * channels + c];
                        if (y1 != y0) {
                            sum += previous[(y1 * previous_width + x0) * channels + c];
                            if (x1 != x0) {
                                sum += previous[(y1 * previous_width + x1) * channels + c];
                            }
                        }
                        current[(y * level_width + x) * channels + c] = sum / divisor;
                    }
                }
            }
        }
    } else {
        const float *source = (const float *)gffx_mesh_elements_const(texture);
        float *target = (float *)gffx_mesh_elements(pyramid);
        int64_t index;
        for (index = 0; index < height * width * channels; ++index) target[index] = source[index];
        for (level = 1; level < level_count; ++level) {
            int64_t previous_height;
            int64_t previous_width;
            int64_t level_height;
            int64_t level_width;
            int64_t y;
            int64_t x;
            int64_t c;
            const float *previous;
            float *current;
            gffx_texture_level_extent(height, width, level - INT64_C(1), &previous_height,
                                      &previous_width);
            gffx_texture_level_extent(height, width, level, &level_height, &level_width);
            previous = target + offset_data[level - INT64_C(1)];
            current = target + offset_data[level];
            for (y = 0; y < level_height; ++y) {
                for (x = 0; x < level_width; ++x) {
                    int64_t y0 = previous_height > INT64_C(1) ? y * INT64_C(2) : y;
                    int64_t x0 = previous_width > INT64_C(1) ? x * INT64_C(2) : x;
                    int64_t y1 = previous_height > INT64_C(1) ? y0 + INT64_C(1) : y0;
                    int64_t x1 = previous_width > INT64_C(1) ? x0 + INT64_C(1) : x0;
                    float divisor = 1.0f;
                    if (previous_height > INT64_C(1)) divisor *= 2.0f;
                    if (previous_width > INT64_C(1)) divisor *= 2.0f;
                    for (c = 0; c < channels; ++c) {
                        /* Accumulated in float32, not promoted to double and rounded back. The
                         * contract's exactness claims hold in the stated dtype, and promoting
                         * would make the CPU disagree with a device kernel that does not. */
                        float sum = previous[(y0 * previous_width + x0) * channels + c];
                        if (x1 != x0) sum += previous[(y0 * previous_width + x1) * channels + c];
                        if (y1 != y0) {
                            sum += previous[(y1 * previous_width + x0) * channels + c];
                            if (x1 != x0) {
                                sum += previous[(y1 * previous_width + x1) * channels + c];
                            }
                        }
                        current[(y * level_width + x) * channels + c] = sum / divisor;
                    }
                }
            }
        }
    }
    return GFFX_STATUS_OK;
}

GFFX_API gffx_status GFFX_CALL gffx_render_texture_pyramid_backward(
    const gffx_tensor_view *level_offsets,
    int64_t texture_height,
    int64_t texture_width,
    int64_t channel_count,
    const gffx_tensor_view *grad_pyramid,
    const gffx_execution_context *context,
    gffx_tensor_view *grad_texture,
    const gffx_buffer *workspace,
    gffx_diagnostic_buffer *diagnostic
) {
    gffx_status status;
    int64_t level_count;
    int64_t level;
    const int32_t *offset_data;

    if (context != NULL && context->struct_size >= sizeof(*context) &&
        context->device_type == GFFX_DEVICE_CUDA) {
        return gffx_internal_fail(
            diagnostic, GFFX_STATUS_UNSUPPORTED,
            "the CUDA provider does not implement this operation");
    }
    status = gffx_internal_prepare_diagnostic(diagnostic);
    if (status != GFFX_STATUS_OK) return status;

    if (level_offsets == NULL || level_offsets->rank != 1u || level_offsets->shape == NULL ||
        level_offsets->shape[0] < INT64_C(2)) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "level offsets must be an [L+1] view"
        );
    }
    status = gffx_validate_tensor_view(level_offsets, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (level_offsets->dtype != GFFX_DTYPE_INT32) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "level offsets must be an int32 view"
        );
    }
    if (grad_pyramid == NULL || grad_pyramid->rank != 1u || grad_pyramid->shape == NULL) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "pyramid gradient must be a rank-1 view"
        );
    }
    status = gffx_validate_tensor_view(grad_pyramid, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (grad_pyramid->dtype != GFFX_DTYPE_FLOAT32 &&
        grad_pyramid->dtype != GFFX_DTYPE_FLOAT64) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_UNSUPPORTED,
            "render.texture_pyramid supports the float32 and float64 dtypes"
        );
    }
    status = gffx_validate_execution_context(context, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (context->device_type != GFFX_DEVICE_CPU) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_UNSUPPORTED,
            "render.texture_pyramid implements only the CPU backend in this phase"
        );
    }
    if (workspace != NULL) {
        status = gffx_validate_buffer(workspace, diagnostic);
        if (status != GFFX_STATUS_OK) return status;
    }
    if (texture_height <= INT64_C(0) || texture_width <= INT64_C(0) ||
        channel_count <= INT64_C(0)) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "texture extents and channel count must be positive"
        );
    }
    if (grad_texture == NULL || grad_texture->rank != 3u || grad_texture->shape == NULL ||
        grad_texture->shape[0] != texture_height ||
        grad_texture->shape[1] != texture_width ||
        grad_texture->shape[2] != channel_count) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "texture gradient must be an [H,W,C] output view"
        );
    }
    status = gffx_validate_tensor_view(grad_texture, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (grad_texture->dtype != grad_pyramid->dtype) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "texture gradient must match the pyramid gradient dtype"
        );
    }
    if ((grad_texture->flags & GFFX_TENSOR_OUTPUT) == UINT32_C(0)) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "operation outputs must carry the output flag"
        );
    }
    if (grad_texture->device_type != GFFX_DEVICE_CPU) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_UNSUPPORTED,
            "render.texture_pyramid implements only the CPU backend in this phase"
        );
    }
    if (gffx_mesh_views_overlap(grad_texture, grad_pyramid) ||
        gffx_mesh_views_overlap(grad_texture, level_offsets)) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "outputs may not alias an input or another output"
        );
    }

    offset_data = (const int32_t *)gffx_mesh_elements_const(level_offsets);
    level_count = level_offsets->shape[0] - INT64_C(1);
    if (offset_data[level_count] > grad_pyramid->shape[0]) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "pyramid gradient is smaller than the level offsets describe"
        );
    }

    /*
     * The transpose, done as a direct scatter to level 0 rather than as a level-by-level cascade.
     * A cascade would have to mutate the incoming gradient in place to let level 2 reach level 0
     * through level 1, which would either corrupt a caller-owned input or need scratch the
     * contract puts at zero bytes.
     *
     * The direct form is available because the footprint of a level-l texel in level 0 is an exact
     * rectangle: each step down doubles an extent only where that extent was above 1, and the
     * divisor at that step is exactly the number of texels the step adds. The accumulated weight
     * is therefore uniform over the footprint and equal to one over its area, which is what makes
     * a single pass exact rather than an approximation of the cascade.
     */
    if (grad_pyramid->dtype == GFFX_DTYPE_FLOAT64) {
        const double *source = (const double *)gffx_mesh_elements_const(grad_pyramid);
        double *target = (double *)gffx_mesh_elements(grad_texture);
        int64_t index;
        for (index = 0; index < texture_height * texture_width * channel_count; ++index) {
            target[index] = source[index];
        }
        for (level = INT64_C(1); level < level_count; ++level) {
            int64_t coarse_height;
            int64_t coarse_width;
            int64_t y;
            int64_t x;
            int64_t c;
            const double *coarse = source + offset_data[level];
            gffx_texture_level_extent(texture_height, texture_width, level, &coarse_height,
                                      &coarse_width);
            for (y = 0; y < coarse_height; ++y) {
                for (x = 0; x < coarse_width; ++x) {
                    int64_t y_begin = y;
                    int64_t y_end = y + INT64_C(1);
                    int64_t x_begin = x;
                    int64_t x_end = x + INT64_C(1);
                    int64_t step;
                    int64_t fy;
                    int64_t fx;
                    double share;
                    for (step = level; step > INT64_C(0); --step) {
                        int64_t fine_height;
                        int64_t fine_width;
                        gffx_texture_level_extent(texture_height, texture_width,
                                                  step - INT64_C(1), &fine_height, &fine_width);
                        if (fine_height > INT64_C(1)) {
                            y_begin *= INT64_C(2);
                            y_end *= INT64_C(2);
                        }
                        if (fine_width > INT64_C(1)) {
                            x_begin *= INT64_C(2);
                            x_end *= INT64_C(2);
                        }
                    }
                    share = 1.0 / (double)((y_end - y_begin) * (x_end - x_begin));
                    for (fy = y_begin; fy < y_end; ++fy) {
                        for (fx = x_begin; fx < x_end; ++fx) {
                            for (c = 0; c < channel_count; ++c) {
                                target[(fy * texture_width + fx) * channel_count + c] +=
                                    coarse[(y * coarse_width + x) * channel_count + c] * share;
                            }
                        }
                    }
                }
            }
        }
    } else {
        const float *source = (const float *)gffx_mesh_elements_const(grad_pyramid);
        float *target = (float *)gffx_mesh_elements(grad_texture);
        int64_t index;
        for (index = 0; index < texture_height * texture_width * channel_count; ++index) {
            target[index] = source[index];
        }
        for (level = INT64_C(1); level < level_count; ++level) {
            int64_t coarse_height;
            int64_t coarse_width;
            int64_t y;
            int64_t x;
            int64_t c;
            const float *coarse = source + offset_data[level];
            gffx_texture_level_extent(texture_height, texture_width, level, &coarse_height,
                                      &coarse_width);
            for (y = 0; y < coarse_height; ++y) {
                for (x = 0; x < coarse_width; ++x) {
                    int64_t y_begin = y;
                    int64_t y_end = y + INT64_C(1);
                    int64_t x_begin = x;
                    int64_t x_end = x + INT64_C(1);
                    int64_t step;
                    int64_t fy;
                    int64_t fx;
                    float share;
                    for (step = level; step > INT64_C(0); --step) {
                        int64_t fine_height;
                        int64_t fine_width;
                        gffx_texture_level_extent(texture_height, texture_width,
                                                  step - INT64_C(1), &fine_height, &fine_width);
                        if (fine_height > INT64_C(1)) {
                            y_begin *= INT64_C(2);
                            y_end *= INT64_C(2);
                        }
                        if (fine_width > INT64_C(1)) {
                            x_begin *= INT64_C(2);
                            x_end *= INT64_C(2);
                        }
                    }
                    share = 1.0f / (float)((y_end - y_begin) * (x_end - x_begin));
                    for (fy = y_begin; fy < y_end; ++fy) {
                        for (fx = x_begin; fx < x_end; ++fx) {
                            for (c = 0; c < channel_count; ++c) {
                                target[(fy * texture_width + fx) * channel_count + c] +=
                                    coarse[(y * coarse_width + x) * channel_count + c] * share;
                            }
                        }
                    }
                }
            }
        }
    }
    return GFFX_STATUS_OK;
}

/* ------------------------------------------------------------------------- render.texture */

/*
 * 2^-126, the clamp section 2.4 puts on rho before the logarithm. It is FLT_MIN, chosen by the
 * contract so a stationary sample yields a finite lod of -126 that then clamps to 0 rather than
 * the negative infinity a naive log2(0) would produce.
 */
#define GFFX_TEXTURE_RHO_FLOOR 1.1754943508222875e-38

/*
 * Wrap one texel index. Returns 1 when the index lands inside the level and writes it to *out,
 * and 0 when the caller must substitute the border value. Wrapping is applied to indices rather
 * than to the coordinate, which is what makes a bilinear footprint straddling an edge wrap each of
 * its four taps on its own instead of clamping the footprint as a unit.
 */
static int gffx_texture_wrap(int64_t index, int64_t extent, uint32_t mode, int64_t *out) {
    int64_t period;
    int64_t folded;
    switch (mode) {
        case GFFX_WRAP_REPEAT:
            *out = ((index % extent) + extent) % extent;
            return 1;
        case GFFX_WRAP_CLAMP:
            *out = index < INT64_C(0) ? INT64_C(0)
                 : (index >= extent ? extent - INT64_C(1) : index);
            return 1;
        case GFFX_WRAP_MIRROR:
            period = extent * INT64_C(2);
            folded = ((index % period) + period) % period;
            *out = folded >= extent ? period - INT64_C(1) - folded : folded;
            return 1;
        default: /* GFFX_WRAP_BORDER */
            if (index < INT64_C(0) || index >= extent) return 0;
            *out = index;
            return 1;
    }
}

static int gffx_texture_valid_filter(uint32_t value) {
    return value == GFFX_FILTER_NEAREST || value == GFFX_FILTER_BILINEAR;
}

static int gffx_texture_valid_mip(uint32_t value) {
    return value == GFFX_MIP_NEAREST || value == GFFX_MIP_LINEAR;
}

static int gffx_texture_valid_wrap(uint32_t value) {
    return value == GFFX_WRAP_REPEAT || value == GFFX_WRAP_CLAMP ||
           value == GFFX_WRAP_MIRROR || value == GFFX_WRAP_BORDER;
}

/*
 * Level selection. lod is already clamped to [0, L-1]. Under NEAREST one level carries the whole
 * weight; under LINEAR the pair straddling lod is blended by its fractional part, combined in
 * ascending level order as section 2.5 requires.
 */
static void gffx_texture_select_levels(
    double lod, int64_t level_count, uint32_t mip_filter,
    int64_t *first, int64_t *second, double *second_weight
) {
    double base;
    if (mip_filter == GFFX_MIP_NEAREST) {
        int64_t chosen = (int64_t)(lod + 0.5);
        if (chosen > level_count - INT64_C(1)) chosen = level_count - INT64_C(1);
        *first = chosen;
        *second = chosen;
        *second_weight = 0.0;
        return;
    }
    base = floor(lod);
    *first = (int64_t)base;
    *second = *first + INT64_C(1);
    if (*second > level_count - INT64_C(1)) *second = level_count - INT64_C(1);
    *second_weight = lod - base;
}

/* Level extents and element offset for one level, shared by forward and backward. */
static void gffx_texture_level_geometry(
    const int32_t *offsets, int64_t height, int64_t width, int64_t level,
    int64_t *level_height, int64_t *level_width, int64_t *element_offset
) {
    gffx_texture_level_extent(height, width, level, level_height, level_width);
    *element_offset = (int64_t)offsets[level];
}

#define GFFX_TEXTURE_DEFINE_SAMPLER(SUFFIX, SCALAR, FLOOR_FN)                                     \
static void gffx_texture_sample_level_##SUFFIX(                                                   \
    const SCALAR *level, int64_t level_height, int64_t level_width, int64_t channels,             \
    SCALAR u, SCALAR v, uint32_t filter, uint32_t wrap_u, uint32_t wrap_v,                        \
    const SCALAR *border, SCALAR *out                                                             \
) {                                                                                               \
    int64_t c;                                                                                    \
    if (filter == GFFX_FILTER_NEAREST) {                                                          \
        int64_t x = (int64_t)FLOOR_FN(u * (SCALAR)level_width);                                   \
        int64_t y = (int64_t)FLOOR_FN(v * (SCALAR)level_height);                                  \
        int64_t xi;                                                                               \
        int64_t yi;                                                                               \
        int inside = gffx_texture_wrap(x, level_width, wrap_u, &xi) &                             \
                     gffx_texture_wrap(y, level_height, wrap_v, &yi);                             \
        for (c = 0; c < channels; ++c) {                                                          \
            out[c] = inside ? level[(yi * level_width + xi) * channels + c] : border[c];           \
        }                                                                                         \
        return;                                                                                   \
    }                                                                                             \
    {                                                                                             \
        SCALAR fx = u * (SCALAR)level_width - (SCALAR)0.5;                                        \
        SCALAR fy = v * (SCALAR)level_height - (SCALAR)0.5;                                       \
        int64_t x0 = (int64_t)FLOOR_FN(fx);                                                       \
        int64_t y0 = (int64_t)FLOOR_FN(fy);                                                       \
        SCALAR a = fx - (SCALAR)x0;                                                               \
        SCALAR b = fy - (SCALAR)y0;                                                               \
        int64_t xi0, xi1, yi0, yi1;                                                               \
        int in_x0 = gffx_texture_wrap(x0, level_width, wrap_u, &xi0);                             \
        int in_x1 = gffx_texture_wrap(x0 + INT64_C(1), level_width, wrap_u, &xi1);                \
        int in_y0 = gffx_texture_wrap(y0, level_height, wrap_v, &yi0);                            \
        int in_y1 = gffx_texture_wrap(y0 + INT64_C(1), level_height, wrap_v, &yi1);               \
        for (c = 0; c < channels; ++c) {                                                          \
            SCALAR t00 = (in_x0 && in_y0) ? level[(yi0 * level_width + xi0) * channels + c]       \
                                          : border[c];                                            \
            SCALAR t10 = (in_x1 && in_y0) ? level[(yi0 * level_width + xi1) * channels + c]       \
                                          : border[c];                                            \
            SCALAR t01 = (in_x0 && in_y1) ? level[(yi1 * level_width + xi0) * channels + c]       \
                                          : border[c];                                            \
            SCALAR t11 = (in_x1 && in_y1) ? level[(yi1 * level_width + xi1) * channels + c]       \
                                          : border[c];                                            \
            /* Fixed accumulation order, section 2.3: (x0,y0), (x1,y0), (x0,y1), (x1,y1). */      \
            SCALAR sum = ((SCALAR)1 - a) * ((SCALAR)1 - b) * t00;                                 \
            sum += a * ((SCALAR)1 - b) * t10;                                                     \
            sum += ((SCALAR)1 - a) * b * t01;                                                     \
            sum += a * b * t11;                                                                   \
            out[c] = sum;                                                                         \
        }                                                                                         \
    }                                                                                             \
}

GFFX_TEXTURE_DEFINE_SAMPLER(f64, double, floor)
GFFX_TEXTURE_DEFINE_SAMPLER(f32, float, floorf)

/*
 * Backward through one level. Scatters the filter weights into grad_level and accumulates the
 * coordinate gradient. NEAREST contributes exactly zero to the coordinate gradient rather than a
 * small value, because it is piecewise constant and its derivative is genuinely zero almost
 * everywhere; returning an approximation there would invent a slope that does not exist.
 */
#define GFFX_TEXTURE_DEFINE_BACKWARD(SUFFIX, SCALAR, FLOOR_FN)                                    \
static void gffx_texture_backward_level_##SUFFIX(                                                 \
    const SCALAR *level, SCALAR *grad_level,                                                      \
    int64_t level_height, int64_t level_width, int64_t channels,                                  \
    SCALAR u, SCALAR v, uint32_t filter, uint32_t wrap_u, uint32_t wrap_v,                        \
    const SCALAR *grad_sample, SCALAR level_weight,                                               \
    SCALAR *grad_u, SCALAR *grad_v                                                                \
) {                                                                                               \
    int64_t c;                                                                                    \
    if (filter == GFFX_FILTER_NEAREST) {                                                          \
        int64_t x = (int64_t)FLOOR_FN(u * (SCALAR)level_width);                                   \
        int64_t y = (int64_t)FLOOR_FN(v * (SCALAR)level_height);                                  \
        int64_t xi;                                                                               \
        int64_t yi;                                                                               \
        if (gffx_texture_wrap(x, level_width, wrap_u, &xi) &                                      \
            gffx_texture_wrap(y, level_height, wrap_v, &yi)) {                                    \
            for (c = 0; c < channels; ++c) {                                                      \
                grad_level[(yi * level_width + xi) * channels + c] +=                             \
                    level_weight * grad_sample[c];                                                \
            }                                                                                     \
        }                                                                                         \
        return;                                                                                   \
    }                                                                                             \
    {                                                                                             \
        SCALAR fx = u * (SCALAR)level_width - (SCALAR)0.5;                                        \
        SCALAR fy = v * (SCALAR)level_height - (SCALAR)0.5;                                       \
        int64_t x0 = (int64_t)FLOOR_FN(fx);                                                       \
        int64_t y0 = (int64_t)FLOOR_FN(fy);                                                       \
        SCALAR a = fx - (SCALAR)x0;                                                               \
        SCALAR b = fy - (SCALAR)y0;                                                               \
        int64_t xi0, xi1, yi0, yi1;                                                               \
        int in_x0 = gffx_texture_wrap(x0, level_width, wrap_u, &xi0);                             \
        int in_x1 = gffx_texture_wrap(x0 + INT64_C(1), level_width, wrap_u, &xi1);                \
        int in_y0 = gffx_texture_wrap(y0, level_height, wrap_v, &yi0);                            \
        int in_y1 = gffx_texture_wrap(y0 + INT64_C(1), level_height, wrap_v, &yi1);               \
        SCALAR w00 = ((SCALAR)1 - a) * ((SCALAR)1 - b);                                           \
        SCALAR w10 = a * ((SCALAR)1 - b);                                                         \
        SCALAR w01 = ((SCALAR)1 - a) * b;                                                         \
        SCALAR w11 = a * b;                                                                       \
        for (c = 0; c < channels; ++c) {                                                          \
            SCALAR g = level_weight * grad_sample[c];                                             \
            SCALAR t00 = (in_x0 && in_y0) ? level[(yi0 * level_width + xi0) * channels + c]       \
                                          : (SCALAR)0;                                            \
            SCALAR t10 = (in_x1 && in_y0) ? level[(yi0 * level_width + xi1) * channels + c]       \
                                          : (SCALAR)0;                                            \
            SCALAR t01 = (in_x0 && in_y1) ? level[(yi1 * level_width + xi0) * channels + c]       \
                                          : (SCALAR)0;                                            \
            SCALAR t11 = (in_x1 && in_y1) ? level[(yi1 * level_width + xi1) * channels + c]       \
                                          : (SCALAR)0;                                            \
            /* A border tap has no texel to receive gradient; its value is a constant. */         \
            if (in_x0 && in_y0) grad_level[(yi0 * level_width + xi0) * channels + c] += w00 * g;  \
            if (in_x1 && in_y0) grad_level[(yi0 * level_width + xi1) * channels + c] += w10 * g;  \
            if (in_x0 && in_y1) grad_level[(yi1 * level_width + xi0) * channels + c] += w01 * g;  \
            if (in_x1 && in_y1) grad_level[(yi1 * level_width + xi1) * channels + c] += w11 * g;  \
            /* Analytic derivative of the bilinear weights; the chain rule brings in the level    \
             * extent because a and b are fractions of a texel, not of the [0,1] coordinate. */   \
            *grad_u += g * (SCALAR)level_width *                                                  \
                       (((SCALAR)1 - b) * (t10 - t00) + b * (t11 - t01));                         \
            *grad_v += g * (SCALAR)level_height *                                                 \
                       (((SCALAR)1 - a) * (t01 - t00) + a * (t11 - t10));                         \
        }                                                                                         \
    }                                                                                             \
}

GFFX_TEXTURE_DEFINE_BACKWARD(f64, double, floor)
GFFX_TEXTURE_DEFINE_BACKWARD(f32, float, floorf)

GFFX_API gffx_status GFFX_CALL gffx_render_texture_workspace(
    int64_t sample_count,
    int64_t channel_count,
    gffx_dtype dtype,
    const gffx_execution_context *context,
    uint64_t *required_bytes,
    uint64_t *required_alignment,
    gffx_diagnostic_buffer *diagnostic
) {
    gffx_status status = gffx_internal_prepare_diagnostic(diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (required_bytes == NULL || required_alignment == NULL) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "workspace query result pointers must not be null"
        );
    }
    if (sample_count < INT64_C(0) || channel_count < INT64_C(0)) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "sample and channel counts must be nonnegative"
        );
    }
    if (dtype != GFFX_DTYPE_FLOAT32 && dtype != GFFX_DTYPE_FLOAT64) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_UNSUPPORTED,
            "render.texture supports the float32 and float64 dtypes"
        );
    }
    status = gffx_validate_execution_context(context, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (context->device_type != GFFX_DEVICE_CPU) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_UNSUPPORTED,
            "render.texture implements only the CPU backend in this phase"
        );
    }
    /* Section 2.9: zero on the CPU. The CUDA path will need four bytes for a coordinate
     * validation status word, which its own query will report when that backend exists. */
    *required_bytes = UINT64_C(0);
    *required_alignment = UINT64_C(1);
    return GFFX_STATUS_OK;
}

/*
 * Shared argument checking for the forward and backward entry points. Everything both directions
 * need agree on lives here so the two cannot drift into accepting different inputs.
 */
static gffx_status gffx_texture_check_common(
    const gffx_tensor_view *pyramid,
    const gffx_tensor_view *level_offsets,
    const gffx_tensor_view *coordinates,
    const gffx_tensor_view *derivatives,
    const gffx_tensor_view *lod,
    uint32_t filter,
    uint32_t mip_filter,
    uint32_t wrap_u,
    uint32_t wrap_v,
    const gffx_tensor_view *border,
    const gffx_execution_context *context,
    const gffx_buffer *workspace,
    gffx_diagnostic_buffer *diagnostic,
    int64_t *out_count,
    int64_t *out_levels
) {
    gffx_status status;
    int64_t level_count;

    if (pyramid == NULL || pyramid->rank != 1u || pyramid->shape == NULL) {
        return gffx_internal_fail(
            diagnostic, GFFX_STATUS_INVALID_ARGUMENT, "pyramid must be a rank-1 view");
    }
    status = gffx_validate_tensor_view(pyramid, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (pyramid->dtype != GFFX_DTYPE_FLOAT32 && pyramid->dtype != GFFX_DTYPE_FLOAT64) {
        return gffx_internal_fail(
            diagnostic, GFFX_STATUS_UNSUPPORTED,
            "render.texture supports the float32 and float64 dtypes");
    }
    if (level_offsets == NULL || level_offsets->rank != 1u || level_offsets->shape == NULL ||
        level_offsets->shape[0] < INT64_C(2)) {
        return gffx_internal_fail(
            diagnostic, GFFX_STATUS_INVALID_ARGUMENT, "level offsets must be an [L+1] view");
    }
    status = gffx_validate_tensor_view(level_offsets, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (level_offsets->dtype != GFFX_DTYPE_INT32) {
        return gffx_internal_fail(
            diagnostic, GFFX_STATUS_INVALID_ARGUMENT, "level offsets must be an int32 view");
    }
    level_count = level_offsets->shape[0] - INT64_C(1);

    if (coordinates == NULL || coordinates->rank != 2u || coordinates->shape == NULL ||
        coordinates->shape[1] != INT64_C(2)) {
        return gffx_internal_fail(
            diagnostic, GFFX_STATUS_INVALID_ARGUMENT, "coordinates must be an [N,2] view");
    }
    status = gffx_validate_tensor_view(coordinates, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (coordinates->dtype != pyramid->dtype) {
        return gffx_internal_fail(
            diagnostic, GFFX_STATUS_INVALID_ARGUMENT,
            "coordinates must match the pyramid dtype");
    }

    /* Section 2.4: at most one of the two. Supplying both is a caller error rather than a
     * precedence question the library should resolve on its own. */
    if (derivatives != NULL && lod != NULL) {
        return gffx_internal_fail(
            diagnostic, GFFX_STATUS_INVALID_ARGUMENT,
            "derivatives and lod are mutually exclusive");
    }
    if (derivatives != NULL) {
        if (derivatives->rank != 2u || derivatives->shape == NULL ||
            derivatives->shape[0] != coordinates->shape[0] ||
            derivatives->shape[1] != INT64_C(4)) {
            return gffx_internal_fail(
                diagnostic, GFFX_STATUS_INVALID_ARGUMENT, "derivatives must be an [N,4] view");
        }
        status = gffx_validate_tensor_view(derivatives, diagnostic);
        if (status != GFFX_STATUS_OK) return status;
        if (derivatives->dtype != pyramid->dtype) {
            return gffx_internal_fail(
                diagnostic, GFFX_STATUS_INVALID_ARGUMENT,
                "derivatives must match the pyramid dtype");
        }
    }
    if (lod != NULL) {
        if (lod->rank != 1u || lod->shape == NULL ||
            lod->shape[0] != coordinates->shape[0]) {
            return gffx_internal_fail(
                diagnostic, GFFX_STATUS_INVALID_ARGUMENT, "lod must be an [N] view");
        }
        status = gffx_validate_tensor_view(lod, diagnostic);
        if (status != GFFX_STATUS_OK) return status;
        if (lod->dtype != pyramid->dtype) {
            return gffx_internal_fail(
                diagnostic, GFFX_STATUS_INVALID_ARGUMENT, "lod must match the pyramid dtype");
        }
    }

    /* Unknown enum values are rejected rather than defaulted, so a caller that passes a value
     * from another library's vocabulary learns it here instead of silently getting clamping. */
    if (!gffx_texture_valid_filter(filter) || !gffx_texture_valid_mip(mip_filter) ||
        !gffx_texture_valid_wrap(wrap_u) || !gffx_texture_valid_wrap(wrap_v)) {
        return gffx_internal_fail(
            diagnostic, GFFX_STATUS_INVALID_ARGUMENT,
            "filter, mip_filter and wrap modes must be declared enum values");
    }
    if (border == NULL) {
        if (wrap_u == GFFX_WRAP_BORDER || wrap_v == GFFX_WRAP_BORDER) {
            return gffx_internal_fail(
                diagnostic, GFFX_STATUS_INVALID_ARGUMENT,
                "the border wrap mode requires a border value");
        }
    } else {
        if (border->rank != 1u || border->shape == NULL) {
            return gffx_internal_fail(
                diagnostic, GFFX_STATUS_INVALID_ARGUMENT, "border must be a [C] view");
        }
        status = gffx_validate_tensor_view(border, diagnostic);
        if (status != GFFX_STATUS_OK) return status;
        if (border->dtype != pyramid->dtype) {
            return gffx_internal_fail(
                diagnostic, GFFX_STATUS_INVALID_ARGUMENT,
                "border must match the pyramid dtype");
        }
    }

    status = gffx_validate_execution_context(context, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (context->device_type != GFFX_DEVICE_CPU ||
        pyramid->device_type != GFFX_DEVICE_CPU ||
        coordinates->device_type != GFFX_DEVICE_CPU) {
        return gffx_internal_fail(
            diagnostic, GFFX_STATUS_UNSUPPORTED,
            "render.texture implements only the CPU backend in this phase");
    }
    if (workspace != NULL) {
        status = gffx_validate_buffer(workspace, diagnostic);
        if (status != GFFX_STATUS_OK) return status;
    }
    *out_count = coordinates->shape[0];
    *out_levels = level_count;
    return GFFX_STATUS_OK;
}

#define GFFX_TEXTURE_DEFINE_FORWARD(SUFFIX, SCALAR, SQRT_FN, LOG2_FN)                             \
static void gffx_texture_forward_##SUFFIX(                                                        \
    const SCALAR *pyramid, const int32_t *offsets, int64_t level_count,                           \
    int64_t height, int64_t width, int64_t channels,                                              \
    const SCALAR *coordinates, int64_t count,                                                     \
    const SCALAR *derivatives, const SCALAR *lod_values,                                          \
    uint32_t filter, uint32_t mip_filter, uint32_t wrap_u, uint32_t wrap_v,                       \
    const SCALAR *border, SCALAR *samples                                                         \
) {                                                                                               \
    int64_t n;                                                                                    \
    for (n = 0; n < count; ++n) {                                                                 \
        SCALAR u = coordinates[n * INT64_C(2)];                                                    \
        SCALAR v = coordinates[n * INT64_C(2) + INT64_C(1)];                                       \
        SCALAR *out = samples + n * channels;                                                      \
        double lod = 0.0;                                                                          \
        int64_t first;                                                                             \
        int64_t second;                                                                            \
        double blend;                                                                              \
        int64_t c;                                                                                 \
        int64_t h0, w0, off0, h1, w1, off1;                                                        \
        /* A non-finite coordinate cannot be converted to a texel index without undefined          \
         * behaviour, and section 2.4 requires a NaN coordinate to produce a NaN sample rather     \
         * than an error, so it is resolved here before any index arithmetic. */                   \
        if (!(u == u) || !(v == v) || u * (SCALAR)0 != (SCALAR)0 || v * (SCALAR)0 != (SCALAR)0) {  \
            for (c = 0; c < channels; ++c) out[c] = (SCALAR)NAN;                           \
            continue;                                                                              \
        }                                                                                          \
        if (derivatives != NULL) {                                                                 \
            SCALAR dudx = derivatives[n * INT64_C(4)];                                             \
            SCALAR dvdx = derivatives[n * INT64_C(4) + INT64_C(1)];                                \
            SCALAR dudy = derivatives[n * INT64_C(4) + INT64_C(2)];                                \
            SCALAR dvdy = derivatives[n * INT64_C(4) + INT64_C(3)];                                \
            SCALAR ax = dudx * (SCALAR)width;                                                      \
            SCALAR bx = dvdx * (SCALAR)height;                                                     \
            SCALAR ay = dudy * (SCALAR)width;                                                      \
            SCALAR by = dvdy * (SCALAR)height;                                                     \
            SCALAR rx = SQRT_FN(ax * ax + bx * bx);                                                \
            SCALAR ry = SQRT_FN(ay * ay + by * by);                                                \
            SCALAR rho = rx > ry ? rx : ry;                                                        \
            if (!(rho > (SCALAR)GFFX_TEXTURE_RHO_FLOOR)) rho = (SCALAR)GFFX_TEXTURE_RHO_FLOOR;     \
            lod = (double)LOG2_FN(rho);                                                            \
        } else if (lod_values != NULL) {                                                           \
            lod = (double)lod_values[n];                                                           \
        }                                                                                          \
        if (!(lod > 0.0)) lod = 0.0;                                                               \
        if (lod > (double)(level_count - INT64_C(1))) lod = (double)(level_count - INT64_C(1));    \
        gffx_texture_select_levels(lod, level_count, mip_filter, &first, &second, &blend);         \
        gffx_texture_level_geometry(offsets, height, width, first, &h0, &w0, &off0);               \
        gffx_texture_sample_level_##SUFFIX(pyramid + off0, h0, w0, channels, u, v, filter,         \
                                           wrap_u, wrap_v, border, out);                           \
        if (blend > 0.0 && second != first) {                                                      \
            SCALAR coarse[GFFX_TEXTURE_MAX_CHANNELS];                                              \
            gffx_texture_level_geometry(offsets, height, width, second, &h1, &w1, &off1);          \
            gffx_texture_sample_level_##SUFFIX(pyramid + off1, h1, w1, channels, u, v, filter,     \
                                               wrap_u, wrap_v, border, coarse);                    \
            /* Ascending level order, section 2.5. */                                              \
            for (c = 0; c < channels; ++c) {                                                       \
                out[c] = out[c] * (SCALAR)(1.0 - blend) + coarse[c] * (SCALAR)blend;               \
            }                                                                                      \
        }                                                                                          \
    }                                                                                              \
}

GFFX_TEXTURE_DEFINE_FORWARD(f64, double, sqrt, log2)
GFFX_TEXTURE_DEFINE_FORWARD(f32, float, sqrtf, log2f)

GFFX_API gffx_status GFFX_CALL gffx_render_texture(
    const gffx_tensor_view *pyramid,
    const gffx_tensor_view *level_offsets,
    int64_t texture_height,
    int64_t texture_width,
    const gffx_tensor_view *coordinates,
    const gffx_tensor_view *derivatives,
    const gffx_tensor_view *lod,
    uint32_t filter,
    uint32_t mip_filter,
    uint32_t wrap_u,
    uint32_t wrap_v,
    const gffx_tensor_view *border,
    const gffx_execution_context *context,
    gffx_tensor_view *samples,
    const gffx_buffer *workspace,
    gffx_diagnostic_buffer *diagnostic
) {
    gffx_status status;
    int64_t count;
    int64_t level_count;
    int64_t channels;
    const int32_t *offset_data;

    if (context != NULL && context->struct_size >= sizeof(*context) &&
        context->device_type == GFFX_DEVICE_CUDA) {
        return gffx_internal_fail(
            diagnostic, GFFX_STATUS_UNSUPPORTED,
            "the CUDA provider does not implement this operation");
    }
    status = gffx_internal_prepare_diagnostic(diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    status = gffx_texture_check_common(pyramid, level_offsets, coordinates, derivatives, lod,
                                       filter, mip_filter, wrap_u, wrap_v, border, context,
                                       workspace, diagnostic, &count, &level_count);
    if (status != GFFX_STATUS_OK) return status;

    if (samples == NULL || samples->rank != 2u || samples->shape == NULL ||
        samples->shape[0] != count) {
        return gffx_internal_fail(
            diagnostic, GFFX_STATUS_INVALID_ARGUMENT, "samples must be an [N,C] output view");
    }
    status = gffx_validate_tensor_view(samples, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    channels = samples->shape[1];
    if (channels <= INT64_C(0) || channels > GFFX_TEXTURE_MAX_CHANNELS) {
        return gffx_internal_fail(
            diagnostic, GFFX_STATUS_INVALID_ARGUMENT,
            "channel count must be positive and within the supported bound");
    }
    if (samples->dtype != pyramid->dtype) {
        return gffx_internal_fail(
            diagnostic, GFFX_STATUS_INVALID_ARGUMENT, "samples must match the pyramid dtype");
    }
    if ((samples->flags & GFFX_TENSOR_OUTPUT) == UINT32_C(0)) {
        return gffx_internal_fail(
            diagnostic, GFFX_STATUS_INVALID_ARGUMENT,
            "operation outputs must carry the output flag");
    }
    if (samples->device_type != GFFX_DEVICE_CPU) {
        return gffx_internal_fail(
            diagnostic, GFFX_STATUS_UNSUPPORTED,
            "render.texture implements only the CPU backend in this phase");
    }
    if (gffx_mesh_views_overlap(samples, pyramid) ||
        gffx_mesh_views_overlap(samples, coordinates)) {
        return gffx_internal_fail(
            diagnostic, GFFX_STATUS_INVALID_ARGUMENT,
            "outputs may not alias an input or another output");
    }
    if (border != NULL && border->shape[0] != channels) {
        return gffx_internal_fail(
            diagnostic, GFFX_STATUS_INVALID_ARGUMENT, "border must carry one value per channel");
    }
    if (texture_height <= INT64_C(0) || texture_width <= INT64_C(0)) {
        return gffx_internal_fail(
            diagnostic, GFFX_STATUS_INVALID_ARGUMENT, "texture extents must be positive");
    }
    offset_data = (const int32_t *)gffx_mesh_elements_const(level_offsets);
    /* The extents, the channel count and the offsets must describe the same level 0, which is the
     * one cross-check available against a caller passing a pyramid built for another texture. */
    if ((int64_t)offset_data[1] - (int64_t)offset_data[0] !=
        texture_height * texture_width * channels) {
        return gffx_internal_fail(
            diagnostic, GFFX_STATUS_INVALID_ARGUMENT,
            "level offsets disagree with the texture extents and channel count");
    }
    if ((int64_t)offset_data[level_count] > pyramid->shape[0]) {
        return gffx_internal_fail(
            diagnostic, GFFX_STATUS_INVALID_ARGUMENT,
            "pyramid is smaller than the level offsets describe");
    }
    if (count == INT64_C(0)) return GFFX_STATUS_OK;

    if (pyramid->dtype == GFFX_DTYPE_FLOAT64) {
        gffx_texture_forward_f64(
            (const double *)gffx_mesh_elements_const(pyramid), offset_data, level_count,
            texture_height, texture_width, channels,
            (const double *)gffx_mesh_elements_const(coordinates), count,
            derivatives ? (const double *)gffx_mesh_elements_const(derivatives) : NULL,
            lod ? (const double *)gffx_mesh_elements_const(lod) : NULL,
            filter, mip_filter, wrap_u, wrap_v,
            border ? (const double *)gffx_mesh_elements_const(border) : NULL,
            (double *)gffx_mesh_elements(samples));
    } else {
        gffx_texture_forward_f32(
            (const float *)gffx_mesh_elements_const(pyramid), offset_data, level_count,
            texture_height, texture_width, channels,
            (const float *)gffx_mesh_elements_const(coordinates), count,
            derivatives ? (const float *)gffx_mesh_elements_const(derivatives) : NULL,
            lod ? (const float *)gffx_mesh_elements_const(lod) : NULL,
            filter, mip_filter, wrap_u, wrap_v,
            border ? (const float *)gffx_mesh_elements_const(border) : NULL,
            (float *)gffx_mesh_elements(samples));
    }
    return GFFX_STATUS_OK;
}

#define GFFX_TEXTURE_DEFINE_BACKWARD_ENTRY(SUFFIX, SCALAR, SQRT_FN, LOG2_FN)                      \
static void gffx_texture_backward_##SUFFIX(                                                       \
    const SCALAR *pyramid, const int32_t *offsets, int64_t level_count,                           \
    int64_t height, int64_t width, int64_t channels,                                              \
    const SCALAR *coordinates, int64_t count,                                                     \
    const SCALAR *derivatives, const SCALAR *lod_values,                                          \
    uint32_t filter, uint32_t mip_filter, uint32_t wrap_u, uint32_t wrap_v,                       \
    const SCALAR *grad_samples, SCALAR *grad_pyramid, SCALAR *grad_coordinates                    \
) {                                                                                               \
    int64_t n;                                                                                    \
    int64_t total = (int64_t)offsets[level_count];                                                \
    for (n = 0; n < total; ++n) grad_pyramid[n] = (SCALAR)0;                                      \
    for (n = 0; n < count * INT64_C(2); ++n) grad_coordinates[n] = (SCALAR)0;                     \
    for (n = 0; n < count; ++n) {                                                                 \
        SCALAR u = coordinates[n * INT64_C(2)];                                                    \
        SCALAR v = coordinates[n * INT64_C(2) + INT64_C(1)];                                       \
        const SCALAR *g = grad_samples + n * channels;                                             \
        double lod = 0.0;                                                                          \
        int64_t first, second;                                                                     \
        double blend;                                                                              \
        int64_t h0, w0, off0, h1, w1, off1;                                                        \
        if (!(u == u) || !(v == v) || u * (SCALAR)0 != (SCALAR)0 ||                                \
            v * (SCALAR)0 != (SCALAR)0) {                                                          \
            continue;                                                                              \
        }                                                                                          \
        if (derivatives != NULL) {                                                                 \
            SCALAR ax = derivatives[n * INT64_C(4)] * (SCALAR)width;                               \
            SCALAR bx = derivatives[n * INT64_C(4) + INT64_C(1)] * (SCALAR)height;                 \
            SCALAR ay = derivatives[n * INT64_C(4) + INT64_C(2)] * (SCALAR)width;                  \
            SCALAR by = derivatives[n * INT64_C(4) + INT64_C(3)] * (SCALAR)height;                 \
            SCALAR rx = SQRT_FN(ax * ax + bx * bx);                                                \
            SCALAR ry = SQRT_FN(ay * ay + by * by);                                                \
            SCALAR rho = rx > ry ? rx : ry;                                                        \
            if (!(rho > (SCALAR)GFFX_TEXTURE_RHO_FLOOR)) rho = (SCALAR)GFFX_TEXTURE_RHO_FLOOR;     \
            lod = (double)LOG2_FN(rho);                                                            \
        } else if (lod_values != NULL) {                                                           \
            lod = (double)lod_values[n];                                                           \
        }                                                                                          \
        if (!(lod > 0.0)) lod = 0.0;                                                               \
        if (lod > (double)(level_count - INT64_C(1))) lod = (double)(level_count - INT64_C(1));    \
        gffx_texture_select_levels(lod, level_count, mip_filter, &first, &second, &blend);         \
        gffx_texture_level_geometry(offsets, height, width, first, &h0, &w0, &off0);               \
        /* lod comes from nondifferentiable inputs, so the level blend weights are constants and   \
         * the coordinate gradient is just their weighted sum of per-level gradients. */           \
        gffx_texture_backward_level_##SUFFIX(                                                      \
            pyramid + off0, grad_pyramid + off0, h0, w0, channels, u, v, filter, wrap_u, wrap_v,   \
            g, (SCALAR)(1.0 - ((blend > 0.0 && second != first) ? blend : 0.0)),                   \
            grad_coordinates + n * INT64_C(2), grad_coordinates + n * INT64_C(2) + INT64_C(1));    \
        if (blend > 0.0 && second != first) {                                                      \
            gffx_texture_level_geometry(offsets, height, width, second, &h1, &w1, &off1);          \
            gffx_texture_backward_level_##SUFFIX(                                                  \
                pyramid + off1, grad_pyramid + off1, h1, w1, channels, u, v, filter, wrap_u,       \
                wrap_v, g, (SCALAR)blend,                                                          \
                grad_coordinates + n * INT64_C(2),                                                 \
                grad_coordinates + n * INT64_C(2) + INT64_C(1));                                   \
        }                                                                                          \
    }                                                                                              \
}

GFFX_TEXTURE_DEFINE_BACKWARD_ENTRY(f64, double, sqrt, log2)
GFFX_TEXTURE_DEFINE_BACKWARD_ENTRY(f32, float, sqrtf, log2f)

GFFX_API gffx_status GFFX_CALL gffx_render_texture_backward(
    const gffx_tensor_view *pyramid,
    const gffx_tensor_view *level_offsets,
    int64_t texture_height,
    int64_t texture_width,
    const gffx_tensor_view *coordinates,
    const gffx_tensor_view *derivatives,
    const gffx_tensor_view *lod,
    uint32_t filter,
    uint32_t mip_filter,
    uint32_t wrap_u,
    uint32_t wrap_v,
    const gffx_tensor_view *border,
    const gffx_tensor_view *grad_samples,
    const gffx_execution_context *context,
    gffx_tensor_view *grad_pyramid,
    gffx_tensor_view *grad_coordinates,
    const gffx_buffer *workspace,
    gffx_diagnostic_buffer *diagnostic
) {
    gffx_status status;
    int64_t count;
    int64_t level_count;
    int64_t channels;
    const int32_t *offset_data;

    if (context != NULL && context->struct_size >= sizeof(*context) &&
        context->device_type == GFFX_DEVICE_CUDA) {
        return gffx_internal_fail(
            diagnostic, GFFX_STATUS_UNSUPPORTED,
            "the CUDA provider does not implement this operation");
    }
    status = gffx_internal_prepare_diagnostic(diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    status = gffx_texture_check_common(pyramid, level_offsets, coordinates, derivatives, lod,
                                       filter, mip_filter, wrap_u, wrap_v, border, context,
                                       workspace, diagnostic, &count, &level_count);
    if (status != GFFX_STATUS_OK) return status;

    if (grad_samples == NULL || grad_samples->rank != 2u || grad_samples->shape == NULL ||
        grad_samples->shape[0] != count) {
        return gffx_internal_fail(
            diagnostic, GFFX_STATUS_INVALID_ARGUMENT, "sample gradient must be an [N,C] view");
    }
    status = gffx_validate_tensor_view(grad_samples, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    channels = grad_samples->shape[1];
    if (channels <= INT64_C(0) || channels > GFFX_TEXTURE_MAX_CHANNELS) {
        return gffx_internal_fail(
            diagnostic, GFFX_STATUS_INVALID_ARGUMENT,
            "channel count must be positive and within the supported bound");
    }
    if (grad_pyramid == NULL || grad_pyramid->rank != 1u || grad_pyramid->shape == NULL ||
        grad_coordinates == NULL || grad_coordinates->rank != 2u ||
        grad_coordinates->shape == NULL || grad_coordinates->shape[0] != count ||
        grad_coordinates->shape[1] != INT64_C(2)) {
        return gffx_internal_fail(
            diagnostic, GFFX_STATUS_INVALID_ARGUMENT,
            "gradients must be a rank-1 pyramid view and an [N,2] coordinate view");
    }
    status = gffx_validate_tensor_view(grad_pyramid, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    status = gffx_validate_tensor_view(grad_coordinates, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (grad_pyramid->dtype != pyramid->dtype || grad_coordinates->dtype != pyramid->dtype ||
        grad_samples->dtype != pyramid->dtype) {
        return gffx_internal_fail(
            diagnostic, GFFX_STATUS_INVALID_ARGUMENT, "gradients must match the pyramid dtype");
    }
    if ((grad_pyramid->flags & GFFX_TENSOR_OUTPUT) == UINT32_C(0) ||
        (grad_coordinates->flags & GFFX_TENSOR_OUTPUT) == UINT32_C(0)) {
        return gffx_internal_fail(
            diagnostic, GFFX_STATUS_INVALID_ARGUMENT,
            "operation outputs must carry the output flag");
    }
    if (grad_pyramid->device_type != GFFX_DEVICE_CPU ||
        grad_coordinates->device_type != GFFX_DEVICE_CPU) {
        return gffx_internal_fail(
            diagnostic, GFFX_STATUS_UNSUPPORTED,
            "render.texture implements only the CPU backend in this phase");
    }
    if (gffx_mesh_views_overlap(grad_pyramid, pyramid) ||
        gffx_mesh_views_overlap(grad_coordinates, coordinates) ||
        gffx_mesh_views_overlap(grad_pyramid, grad_coordinates)) {
        return gffx_internal_fail(
            diagnostic, GFFX_STATUS_INVALID_ARGUMENT,
            "outputs may not alias an input or another output");
    }
    if (texture_height <= INT64_C(0) || texture_width <= INT64_C(0)) {
        return gffx_internal_fail(
            diagnostic, GFFX_STATUS_INVALID_ARGUMENT, "texture extents must be positive");
    }
    offset_data = (const int32_t *)gffx_mesh_elements_const(level_offsets);
    if ((int64_t)offset_data[1] - (int64_t)offset_data[0] !=
        texture_height * texture_width * channels) {
        return gffx_internal_fail(
            diagnostic, GFFX_STATUS_INVALID_ARGUMENT,
            "level offsets disagree with the texture extents and channel count");
    }
    if ((int64_t)offset_data[level_count] > grad_pyramid->shape[0]) {
        return gffx_internal_fail(
            diagnostic, GFFX_STATUS_INVALID_ARGUMENT,
            "pyramid gradient is smaller than the level offsets describe");
    }
    if (count == INT64_C(0)) return GFFX_STATUS_OK;

    if (pyramid->dtype == GFFX_DTYPE_FLOAT64) {
        gffx_texture_backward_f64(
            (const double *)gffx_mesh_elements_const(pyramid), offset_data, level_count,
            texture_height, texture_width, channels,
            (const double *)gffx_mesh_elements_const(coordinates), count,
            derivatives ? (const double *)gffx_mesh_elements_const(derivatives) : NULL,
            lod ? (const double *)gffx_mesh_elements_const(lod) : NULL,
            filter, mip_filter, wrap_u, wrap_v,
            (const double *)gffx_mesh_elements_const(grad_samples),
            (double *)gffx_mesh_elements(grad_pyramid),
            (double *)gffx_mesh_elements(grad_coordinates));
    } else {
        gffx_texture_backward_f32(
            (const float *)gffx_mesh_elements_const(pyramid), offset_data, level_count,
            texture_height, texture_width, channels,
            (const float *)gffx_mesh_elements_const(coordinates), count,
            derivatives ? (const float *)gffx_mesh_elements_const(derivatives) : NULL,
            lod ? (const float *)gffx_mesh_elements_const(lod) : NULL,
            filter, mip_filter, wrap_u, wrap_v,
            (const float *)gffx_mesh_elements_const(grad_samples),
            (float *)gffx_mesh_elements(grad_pyramid),
            (float *)gffx_mesh_elements(grad_coordinates));
    }
    return GFFX_STATUS_OK;
}
