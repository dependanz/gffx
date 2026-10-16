/* Internal bounded continuation accounting. No geometry, storage or hidden state.
 * Caller must bind immutable inputs and consume receipts in order. This checks
 * recomputed prefixes; it is not authentication of an arbitrary caller token. */
#ifndef GFFX_OPAQUE_BATCH_CURSOR_H
#define GFFX_OPAQUE_BATCH_CURSOR_H
#ifndef GFFX_OPAQUE_LOCAL
#define GFFX_OPAQUE_LOCAL static
#endif
typedef struct {
    int begin,batch,prefix_slots,prefix_blocks;
    int seen,emitted,slots,blocks,total_slots,total_blocks;
    int full,failed,max_batches;
} QuadBatch;
GFFX_OPAQUE_LOCAL int quad_batch_init_profile(QuadBatch *b,int begin,int batch,int slots,int blocks,int max_batches){
    QuadBatch empty={0};*b=empty;b->max_batches=max_batches;
    if(max_batches!=16&&max_batches!=52&&max_batches!=64&&max_batches!=80){b->failed=1;return -1;}
    if(begin<0||begin>max_batches*32||batch<0||batch>=max_batches||slots<0||slots>batch*800||blocks<0||blocks>batch*32||begin>blocks||((batch==0)!=(begin==0))){b->failed=1;return -1;}
    b->begin=begin;b->batch=batch;b->prefix_slots=slots;b->prefix_blocks=blocks;return 0;
}
GFFX_OPAQUE_LOCAL int quad_batch_init(QuadBatch *b,int begin,int batch,int slots,int blocks){
    return quad_batch_init_profile(b,begin,batch,slots,blocks,16);
}
/* 0 skip an already committed piece; 1 emit whole piece; 2 yield before it;
 * -1 refuses the entire receipt, including any partial output already written. */
GFFX_OPAQUE_LOCAL int quad_batch_offer(QuadBatch *b,int slots,int blocks){
    if(b->failed||b->full||slots<1||slots>800||blocks<1||blocks>32){b->failed=1;return -1;}
    if(b->seen==b->begin&&(b->total_slots!=b->prefix_slots||b->total_blocks!=b->prefix_blocks)){b->failed=1;return -1;}
    if(b->total_slots>b->max_batches*800-slots||b->total_blocks>b->max_batches*32-blocks){b->failed=1;return -1;}
    if(b->seen>=b->begin&&(b->slots>800-slots||b->blocks>32-blocks)){
        if(b->batch==b->max_batches-1){b->failed=1;return -1;}b->full=1;return 2;
    }
    b->total_slots+=slots;b->total_blocks+=blocks;b->seen++;
    if(b->seen<=b->begin)return 0;
    b->emitted++;b->slots+=slots;b->blocks+=blocks;return 1;
}
GFFX_OPAQUE_LOCAL int quad_batch_finish(QuadBatch *b){
    if(b->failed||b->seen<b->begin||(b->seen==b->begin&&(b->total_slots!=b->prefix_slots||b->total_blocks!=b->prefix_blocks))||(b->full&&!b->emitted)){b->failed=1;return -1;}
    return 0;
}
#endif
