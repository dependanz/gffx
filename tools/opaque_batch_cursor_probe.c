/* Research harness only, deliberately separate from the installed renderer. */
#include "../native/core/opaque_batch_cursor.h"
#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif
EXPORT int batch_probe(const int *items,int count,int begin,int batch,int slots,int blocks,int *out,int *actions){
    QuadBatch b;if(count<0||count>1024||quad_batch_init(&b,begin,batch,slots,blocks))return 1;
    for(int i=0;i<count;i++)actions[i]=-2;
    for(int i=0;i<count;i++){int a=quad_batch_offer(&b,items[2*i],items[2*i+1]);actions[i]=a;if(a<0)return 2;if(a==2)break;}
    if(quad_batch_finish(&b))return 3;
    out[0]=b.seen;out[1]=!b.full;out[2]=b.slots;out[3]=b.blocks;out[4]=b.total_slots;out[5]=b.total_blocks;out[6]=b.emitted;out[7]=b.failed;return 0;
}
