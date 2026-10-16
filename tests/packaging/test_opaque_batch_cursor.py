"""C11 prototype continuation accounting; not installed renderer qualification."""
import ctypes,os
import pytest

def run(items,*,begin=0,batch=0,prefix=(0,0)):
    lib=ctypes.CDLL(os.environ['GFFX_BATCH_PROBE_DLL'])
    fn=lib.batch_probe;ptr=ctypes.POINTER(ctypes.c_int)
    fn.argtypes=[ptr,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ptr,ptr];fn.restype=ctypes.c_int
    data=(ctypes.c_int*(2*len(items)))(*(n for pair in items for n in pair));out=(ctypes.c_int*8)();actions=(ctypes.c_int*len(items))()
    code=fn(data,len(items),begin,batch,*prefix,out,actions)
    return code,list(out),list(actions)

def test_batches_cover_every_piece_once_in_order():
    items=[(44,1)]*37+[(288,5),(40,1)]
    begin=batch=slots=blocks=0;seen=[]
    while True:
        code,out,actions=run(items,begin=begin,batch=batch,prefix=(slots,blocks));assert code==0
        next_,done,local_slots,local_blocks,slots,blocks,emitted,_=out
        assert local_slots<=800 and local_blocks<=32 and emitted==actions.count(1)
        seen.extend(i for i,a in enumerate(actions) if a==1)
        assert next_>begin;begin=next_;batch+=1
        if done:break
        assert batch<16
    assert seen==list(range(len(items))) and slots==1956 and blocks==43

def test_batch_never_splits_a_piece_and_keeps_exact_capacity():
    code,out,actions=run([(400,16),(400,16),(1,1)])
    assert code==0 and out[:4]==[2,0,800,32] and actions==[1,1,2]
    code,out,actions=run([(400,16),(400,16),(1,1)],begin=2,batch=1,prefix=(800,32))
    assert code==0 and out[:6]==[3,1,1,1,801,33] and actions==[0,0,1]

@pytest.mark.parametrize('kw',[dict(begin=2,batch=1,prefix=(799,32)),dict(begin=2,batch=1,prefix=(800,31)),dict(begin=4,batch=1,prefix=(1200,48))])
def test_forged_or_unreachable_prefix_cannot_complete(kw):
    assert run([(400,16)]*3,**kw)[0]!=0

@pytest.mark.parametrize('items,kw',[([(0,1)],{}),([(801,1)],{}),([(1,33)],{}),([(1,0)],{}),([(1,1)],dict(batch=16)),([(1,1)],dict(begin=-1)),([(1,1)],dict(begin=1,batch=0,prefix=(1,1)))])
def test_invalid_extent_and_cursor_refuse(items,kw):
    assert run(items,**kw)[0]!=0

def test_sixteenth_batch_can_complete_but_cannot_yield_again():
    items=[(800,32)]*16
    code,out,_=run(items,begin=15,batch=15,prefix=(12000,480))
    assert code==0 and out[:6]==[16,1,800,32,12800,512]
    assert run(items+[(1,1)],begin=15,batch=15,prefix=(12000,480))[0]!=0

def test_empty_stream_has_one_complete_receipt_and_no_emission():
    code,out,actions=run([])
    assert code==0 and out==[0,1,0,0,0,0,0,0] and actions==[]
