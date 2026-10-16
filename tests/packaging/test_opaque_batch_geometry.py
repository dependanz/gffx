"""Finite native streamed geometry prototype; independent area/ordering controls."""
import ctypes,os,math
import numpy as np
import pytest

def call(v,f,c,*,cursor=(0,0,0,0),origin=(0.,0.)):
    lib=ctypes.CDLL(os.environ['GFFX_BATCH_GEOMETRY_DLL']);fn=lib.batch_geometry;ptr=ctypes.c_void_p
    fn.argtypes=[ptr,ctypes.c_int,ptr,ctypes.c_int,ptr,ctypes.c_int,ctypes.c_double,ctypes.c_double,ptr,ptr,ptr,ptr];fn.restype=ctypes.c_int
    v=np.ascontiguousarray(v,dtype=np.float64);f=np.ascontiguousarray(f,dtype=np.int32);c=np.ascontiguousarray(c,dtype=np.float64)
    ids=np.full((1,800),-1,dtype=np.int32);q=np.zeros((1,800,4));state=np.array(cursor,dtype=np.int32);receipt=np.zeros(8,dtype=np.int32)
    code=fn(v.ctypes.data,len(v),f.ctypes.data,len(f),c.ctypes.data,c.shape[1],*origin,state.ctypes.data,ids.ctypes.data,q.ctypes.data,receipt.ctypes.data)
    return code,ids,q,receipt.tolist()

def wedge(n=32):
    v=[[.5,.5,0.]]+[[.5+2*math.cos(2*math.pi*i/n),.5+2*math.sin(2*math.pi*i/n),0.] for i in range(n)]
    f=[[0,i+1,(i+1)%n+1] for i in range(n)]
    return np.array(v),np.array(f,dtype=np.int32),np.zeros((n,6))

def test_streamed_pixel_preserves_all_faces_and_closed_unit_area():
    v,f,c=wedge();cursor=(0,0,0,0);mass=0.;faces=set();batches=0
    while True:
        code,ids,q,r=call(v,f,c,cursor=cursor);assert code==0
        next_,done,slots,blocks,total_slots,total_blocks,emitted,work=r
        assert 0<slots<=800 and 0<blocks<=32 and total_slots<=12800 and total_blocks<=512
        assert np.all(ids[0,:slots]>=0) and np.all(ids[0,slots:]==-1) and np.all(q[0,slots:]==0)
        assert np.all(q[0,:slots,3]>=0) and np.max(np.abs(q[0,:slots,:3].sum(1)-1))<1e-12
        faces.update(ids[0,:slots]);mass+=math.fsum(q[0,:slots,3]);batches+=1
        if done:break
        cursor=(next_,batches,total_slots,total_blocks)
        assert batches<16
    assert batches>1 and faces==set(range(32)) and abs(mass-1)<1e-12

def test_streamed_geometry_rejects_changed_prefix_before_acceptance():
    v,f,c=wedge();code,_,_,r=call(v,f,c);assert code==0 and not r[1]
    assert call(v,f,c,cursor=(r[0],1,r[4]-1,r[5]))[0]!=0

def test_streamed_empty_pixel_is_complete_and_zero():
    v,f,c=wedge();code,ids,q,r=call(v,f,c,origin=(10.,10.))
    assert code==0 and r[:7]==[0,1,0,0,0,0,0] and np.all(ids==-1) and np.all(q==0)

def test_streamed_pixel_keeps_candidate_refusal():
    v,f,c=wedge(33);assert call(v,f,c)[0]!=0


def derivative(v,f,c,gradient,*,cursor=(0,0,0,0),vertex_seed=-1,cut_seed=-1):
    lib=ctypes.CDLL(os.environ['GFFX_BATCH_GEOMETRY_DLL']);fn=lib.batch_derivative;ptr=ctypes.c_void_p
    fn.argtypes=[ptr,ctypes.c_int,ptr,ctypes.c_int,ptr,ctypes.c_int,ptr,ctypes.c_int64,ctypes.c_int64,ptr,ptr,ptr];fn.restype=ctypes.c_int
    v=np.ascontiguousarray(v,dtype=np.float64);f=np.ascontiguousarray(f,dtype=np.int32);c=np.ascontiguousarray(c,dtype=np.float64);g=np.ascontiguousarray(gradient,dtype=np.float64)
    state=np.array(cursor,dtype=np.int32);receipt=np.zeros(8,dtype=np.int32);value=np.zeros(1)
    code=fn(v.ctypes.data,len(v),f.ctypes.data,len(f),c.ctypes.data,c.shape[1],state.ctypes.data,vertex_seed,cut_seed,g.ctypes.data,value.ctypes.data,receipt.ctypes.data)
    return code,value[0],receipt.tolist()


def batches(v,f,c):
    cursor=(0,0,0,0)
    for index in range(16):
        code,ids,q,r=call(v,f,c,cursor=cursor);assert code==0
        yield cursor,ids,q,r
        if r[1]:return
        cursor=(r[0],index+1,r[4],r[5])
    pytest.fail('partial pixel reached the finite batch ceiling')


@pytest.mark.parametrize('scale',[1.,1e-20,-1e-20])
def test_streamed_material_translation_retains_signed_and_tiny_boundary_flux(scale):
    v,f,c=wedge();colour=v[:,0].copy();value=gradient=0.;count=0
    # Translate every vertex: material colour becomes x-delta on the fixed unit pixel.
    for cursor,ids,q,r in batches(v,f,c):
        count+=1;g=np.zeros_like(q);n=r[2];col=colour[f[ids[0,:n]]]
        values=np.sum(q[0,:n,:3]*col,axis=1)*scale
        g[0,:n,:3]=q[0,:n,3,None]*col*scale;g[0,:n,3]=values
        value+=math.fsum(q[0,:n,3]*values)
        for seed in range(0,len(v)*3,3):
            code,d,rr=derivative(v,f,c,g,cursor=cursor,vertex_seed=seed)
            assert code==0 and rr[:7]==r[:7];gradient+=d
    assert count>1 and abs(value/scale-.5)<2e-12 and abs(gradient/scale+1)<2e-12


def test_streamed_curved_horizon_has_closed_area_and_boundary_motion():
    from decimal import Decimal,localcontext
    v=[]
    for i in range(7):
        x=(i%3)*.25+.0625;y=(i//3)*.25+.0625
        v.extend([[x,y,0],[x+.125,y,0],[x,y+.125,0]])
    v=np.array(v);f=np.arange(21,dtype=np.int32).reshape(7,3);c=np.tile([-.25,.75,.65,.5,.5,1.5],(7,1))
    area=full=ga=gb=0.;count=0
    for cursor,ids,q,r in batches(v,f,c):
        count+=1;n=r[2];beta=q[0,:n,:3];side=(beta[:,1]+beta[:,2]-.25-.1*beta[:,2]**2)>0
        g=np.zeros_like(q);g[0,:n,3]=side
        area+=math.fsum(q[0,:n,3]*side);full+=math.fsum(q[0,:n,3])
        for face in range(7):
            for k in range(6):
                code,d,rr=derivative(v,f,c,g,cursor=cursor,cut_seed=face*6+k)
                assert code==0 and rr[:7]==r[:7];ga+=d*(-1 if k<3 else -2)
                if k==2:gb-=d
    with localcontext() as ctx:
        ctx.prec=80;a=Decimal('.25');b=Decimal('.1');root=2*a/(1+(1-4*a*b).sqrt());det=Decimal(7)/64
        expected=float(det*(Decimal('.5')-a*root+root**2/2-b*root**3/3))
        ea=float(-det*root);eb=float(-det*root**3/3)
    assert count==4 and abs(full-7/128)<1e-13
    assert abs(area-expected)<1e-12 and abs(ga-ea)<1e-12 and abs(gb-eb)<1e-12


def test_exact_separation_orientation_matches_rational_signs():
    from fractions import Fraction as F
    import random
    lib=ctypes.CDLL(os.environ['GFFX_BATCH_GEOMETRY_DLL']);fn=lib.orientation_sign
    fn.argtypes=[ctypes.c_void_p];fn.restype=ctypes.c_int
    rng=random.Random(156);cases=[]
    for _ in range(300):cases.append([rng.uniform(-8,8) for _ in range(6)])
    for exponent in (-100,0,100):
        h=2.**exponent
        for delta in (0.,2.**-50,-2.**-50):
            cases.append([.5*h,.5*h,2*h,2*h,(3+delta)*h,(3-delta)*h])
    for values in cases:
        a,b,c,d,e,f=map(F,values);det=(c-a)*(f-b)-(d-b)*(e-a)
        x=np.array(values);assert fn(x.ctypes.data)==(det>0)-(det<0)
    x=np.array([0.,0.,1e-300,0.,0.,1e-300]);assert fn(x.ctypes.data)==2
