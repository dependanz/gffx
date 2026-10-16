#ifndef GFFX_OPAQUE_SAMPLES_ARGS_H
#define GFFX_OPAQUE_SAMPLES_ARGS_H
#include <stdint.h>
typedef struct {const void *data[18];int64_t count[18];double *weighted,*rgba;int pixels,nv,nf,nodes;} GffxOpaqueSamples;
#endif
