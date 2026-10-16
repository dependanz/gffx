/* Batch-one16x16 spatial cooperative rejection. Exactly 256 lanes, as launched by the provider.
   Stable face compaction and the established rectangle predicate preserve each
   pixel's original arithmetic/order. 20484 bytes static shared memory per block;
   no global allocation. Padded tail lanes must reach every barrier. */
#define GFFX_CUDA_RASTERIZE_COOPERATIVE(SUFFIX, SCALAR, INFINITY_VALUE) \
extern "C" __global__ void gffx_cuda_rasterize_cooperative_##SUFFIX( \
    const SCALAR *__restrict__ ndc_vertices, const int *__restrict__ faces, \
    const int *__restrict__ face_offsets, long long image_height, long long image_width, \
    long long faces_per_pixel, double blur_squared, unsigned int cull_mode, double eps, \
    long long pixel_count, int *__restrict__ face_index, SCALAR *__restrict__ barycentric, \
    SCALAR *__restrict__ depth, SCALAR *__restrict__ signed_distance, \
    long long global_height,long long global_width,long long row_offset,long long column_offset \
) { \
    long long blocks_per_row=(image_width-1)/16+1; \
    long long first_column=((long long)blockIdx.x%blocks_per_row)*16; \
    long long first_row=((long long)blockIdx.x/blocks_per_row)*16; \
    long long column=first_column+threadIdx.x%16; \
    long long row=first_row+threadIdx.x/16; \
    long long pixel=row*image_width+column; \
    int active=row<image_height && column<image_width; \
    /* Inactive row/column lanes still participate in every block barrier. */ \
    long long batch = 0; /* provider admits batch-one only, including padded lanes */ \
    long long pixel_base = active ? pixel * faces_per_pixel : 0; \
    long long first_face = (long long)face_offsets[batch]; \
    long long last_face = (long long)face_offsets[batch + 1]; \
    double half_width = (double)global_width * 0.5; \
    double half_height = (double)global_height * 0.5; \
    double px = (double)(column+column_offset) + 0.5; \
    double py = (double)(row+row_offset) + 0.5; \
    for (long long slot = 0; active && slot < faces_per_pixel; ++slot) { \
        long long entry = pixel_base + slot; \
        face_index[entry] = -1; \
        depth[entry] = (INFINITY_VALUE); \
        signed_distance[entry] = (INFINITY_VALUE); \
        barycentric[entry * 3 + 0] = (SCALAR)0; \
        barycentric[entry * 3 + 1] = (SCALAR)0; \
        barycentric[entry * 3 + 2] = (SCALAR)0; \
    } \
    __shared__ double projected[9][256]; \
    __shared__ int retained[256],order[256],count; \
    long long last_column=first_column+15<image_width?first_column+15:image_width-1; \
    long long last_row=first_row+15<image_height?first_row+15:image_height-1; \
    double left=(double)(column_offset+first_column)+.5; \
    double right=(double)(column_offset+last_column)+.5; \
    double top=(double)(row_offset+first_row)+.5,bottom=(double)(row_offset+last_row)+.5; \
    for(long long chunk=first_face;chunk<last_face;chunk+=256){ \
        int lane=threadIdx.x;long long face=chunk+lane;retained[lane]=0; \
        if(face<last_face){ \
            long long i0 = (long long)faces[face * 3 + 0]; \
            long long i1 = (long long)faces[face * 3 + 1]; \
            long long i2 = (long long)faces[face * 3 + 2]; \
            double nx0 = (double)ndc_vertices[i0 * 3 + 0]; \
            double ny0 = (double)ndc_vertices[i0 * 3 + 1]; \
            double nz0 = (double)ndc_vertices[i0 * 3 + 2]; \
            double nx1 = (double)ndc_vertices[i1 * 3 + 0]; \
            double ny1 = (double)ndc_vertices[i1 * 3 + 1]; \
            double nz1 = (double)ndc_vertices[i1 * 3 + 2]; \
            double nx2 = (double)ndc_vertices[i2 * 3 + 0]; \
            double ny2 = (double)ndc_vertices[i2 * 3 + 1]; \
            double nz2 = (double)ndc_vertices[i2 * 3 + 2]; \
            double ax = (nx0 + 1.0) * half_width; \
            double ay = (1.0 - ny0) * half_height; \
            double bx = (nx1 + 1.0) * half_width; \
            double by = (1.0 - ny1) * half_height; \
            double cx = (nx2 + 1.0) * half_width; \
            double cy = (1.0 - ny2) * half_height; \
 \
            projected[0][lane]=ax;projected[1][lane]=ay;projected[2][lane]=nz0; \
            projected[3][lane]=bx;projected[4][lane]=by;projected[5][lane]=nz1; \
            projected[6][lane]=cx;projected[7][lane]=cy;projected[8][lane]=nz2; \
            retained[lane]=!gffx_raster_block_excluded(gffx_raster_make_box(ax,ay,bx,by,cx,cy,(double)global_width,(double)global_height),ax,ay,bx,by,cx,cy,left,top,right,bottom,blur_squared); \
        } \
        __syncthreads(); \
        if(lane==0){count=0;for(int k=0;k<256;++k)if(retained[k])order[count++]=k;} \
        __syncthreads(); \
        for(int selected=0;active && selected<count;++selected){ \
            int source=order[selected];long long face=chunk+source; \
            double ax=projected[0][source],ay=projected[1][source],nz0=projected[2][source]; \
            double bx=projected[3][source],by=projected[4][source],nz1=projected[5][source]; \
            double cx=projected[6][source],cy=projected[7][source],nz2=projected[8][source]; \
        double e0 = (bx - px) * (cy - py) - (by - py) * (cx - px); \
        double e1 = (cx - px) * (ay - py) - (cy - py) * (ax - px); \
        double e2 = (ax - px) * (by - py) - (ay - py) * (bx - px); \
        double area2 = e0 + e1 + e2; \
        double w0, w1, w2, fragment_depth, distance_squared, signed_value; \
        int inside; \
        long long position; \
        if (!(fabs(area2) > eps)) continue; \
        if (cull_mode == 2u && -area2 <= 0.0) continue; \
        if (cull_mode == 3u && -area2 > 0.0) continue; \
        w0 = e0 / area2; \
        w1 = e1 / area2; \
        w2 = e2 / area2; \
        inside = (w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0); \
        if (!inside && gffx_raster_far(gffx_raster_make_box(ax,ay,bx,by,cx,cy,(double)global_width,(double)global_height),px,py,px,py,blur_squared)) continue; \
        distance_squared = gffx_cuda_boundary_distance_squared(px, py, ax, ay, bx, by, cx, cy); \
        if (!inside && distance_squared > blur_squared) continue; \
        fragment_depth = w0 * nz0 + w1 * nz1 + w2 * nz2; \
        signed_value = inside ? -distance_squared : distance_squared; \
        position = faces_per_pixel; \
        while (position > 0) { \
            long long above = pixel_base + position - 1; \
            double previous = (double)depth[above]; \
            if (face_index[above] < 0 || fragment_depth < previous) { \
                --position; \
            } else { \
                break; \
            } \
        } \
        if (position >= faces_per_pixel) continue; \
        for (long long slot = faces_per_pixel - 1; slot > position; --slot) { \
            long long to = pixel_base + slot; \
            long long from = pixel_base + slot - 1; \
            face_index[to] = face_index[from]; \
            depth[to] = depth[from]; \
            signed_distance[to] = signed_distance[from]; \
            barycentric[to * 3 + 0] = barycentric[from * 3 + 0]; \
            barycentric[to * 3 + 1] = barycentric[from * 3 + 1]; \
            barycentric[to * 3 + 2] = barycentric[from * 3 + 2]; \
        } \
        { \
            long long entry = pixel_base + position; \
            face_index[entry] = (int)face; \
            depth[entry] = (SCALAR)fragment_depth; \
            signed_distance[entry] = (SCALAR)signed_value; \
            barycentric[entry * 3 + 0] = (SCALAR)w0; \
            barycentric[entry * 3 + 1] = (SCALAR)w1; \
            barycentric[entry * 3 + 2] = (SCALAR)w2; \
        } \
        } \
        __syncthreads(); \
    } \
}

GFFX_CUDA_RASTERIZE_COOPERATIVE(f32, float, __int_as_float(0x7f800000))
GFFX_CUDA_RASTERIZE_COOPERATIVE(f64, double, __longlong_as_double(0x7ff0000000000000LL))
#undef GFFX_CUDA_RASTERIZE_COOPERATIVE
