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

#include <stdint.h>

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
