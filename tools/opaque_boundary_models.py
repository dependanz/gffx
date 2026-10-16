"""Thin ctypes access to explicitly built, bounded C11 research models only."""
import ctypes
import os
from pathlib import Path

MODELS={'complete':0,'sampled_boundary':1,'micro_edge':2}


def evaluate(model,vertices,faces,colours,velocity,*,height,width,origin=(0,0)):
    if model not in MODELS:raise ValueError('unknown research model')
    if type(height)!=int or type(width)!=int or not 0<height<=8 or not 0<width<=8:raise ValueError('bounded 8x8 profile')
    if len(vertices)>64 or len(faces)>32 or len(velocity)!=len(vertices) or len(colours)!=len(faces):raise ValueError('bounded geometry/profile shape')
    if any(len(row)!=3 for rows in (vertices,velocity,faces,colours) for row in rows):raise ValueError('three coordinates/channels per row')
    if any(type(i)!=int or not 0<=i<len(vertices) for f in faces for i in f):raise ValueError('face index')
    if len(origin)!=2:raise ValueError('origin shape')
    path=Path(os.environ['GFFX_OPAQUE_MODEL_DLL']).resolve()
    library=ctypes.CDLL(str(path))
    real=ctypes.c_double;integer=ctypes.c_int
    array=lambda rows,kind:(kind*sum(len(r) for r in rows))(*(kind(float(x)) if kind is real else x for r in rows for x in r))
    v=array(vertices,real);d=array(velocity,real);f=array(faces,integer);rgb=array(colours,real)
    n=height*width*4;image=(real*n)();gradient=(real*n)()
    fn=library.opaque_model
    fn.argtypes=[integer,integer,ctypes.POINTER(real),ctypes.POINTER(real),integer,ctypes.POINTER(integer),ctypes.POINTER(real),integer,integer,real,real,ctypes.POINTER(real),ctypes.POINTER(real)]
    fn.restype=integer
    code=fn(MODELS[model],len(vertices),v,d,len(faces),f,rgb,height,width,*map(float,origin),image,gradient)
    if code:raise ValueError(f'native research profile refused: {code}')
    return {'image':[list(image[i:i+4]) for i in range(0,n,4)],
            'derivative':[list(gradient[i:i+4]) for i in range(0,n,4)]}
