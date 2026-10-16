"""Caller-side continuation safeguards; no renderer accuracy claim."""
import numpy as np
import pytest

def fixture():
    from tools.opaque_batch_protocol import PixelStream
    inputs={'vertices':np.array([[0.,0.,0.]])};stream=PixelStream(inputs,{'origin':[0,0],'profile':'test'},deadline=100.,clock=lambda:1.)
    ids=np.full((1,800),-1,dtype=np.int32);q=np.zeros((1,800,4));ids[0,0]=0;q[0,0]=[1,0,0,1]
    return inputs,stream,ids,q

def test_stream_completion_is_atomic_and_summation_retains_order():
    inputs,s,ids,q=fixture()
    with pytest.raises(RuntimeError,match='incomplete'):s.result()
    for i,value in enumerate((1e16,1.,-1e16,1e-20)):
        cursor=s.begin();r=[i+1,int(i==3),1,1,i+1,i+1,1,5]
        s.consume(cursor,r,ids,q,np.array([[value,0.,0.,1.]]))
        if i<3:
            with pytest.raises(RuntimeError,match='incomplete'):s.result()
    assert s.result()[0]==1e-20 and s.result()[3]==4 and s.work==20
    with pytest.raises(RuntimeError):s.begin()

def test_changed_inputs_and_cancelled_streams_cannot_resume():
    inputs,s,ids,q=fixture();inputs['vertices'][0,0]=1
    with pytest.raises(RuntimeError,match='changed'):s.begin()
    _,s,_,_=fixture();s.cancel()
    with pytest.raises(RuntimeError,match='cancel'):s.begin()
    with pytest.raises(RuntimeError):s.result()

def test_stream_detects_mutation_after_begin():
    inputs,s,ids,q=fixture();cursor=s.begin();inputs['vertices'][0,0]=1
    with pytest.raises(RuntimeError,match='changed'):s.consume(cursor,[1,1,1,1,1,1,1,1],ids,q,np.ones((1,4)))
    with pytest.raises(RuntimeError):s.result()

@pytest.mark.parametrize('receipt',[[2,1,1,1,1,1,1,0],[1,1,1,1,2,1,1,0],[1,1,801,1,801,1,1,0],[1,2,1,1,1,1,1,0],[1,1,1,1,1,1,1,-1]])
def test_inconsistent_receipt_poisoning_is_terminal(receipt):
    _,s,ids,q=fixture();cursor=s.begin()
    with pytest.raises((ValueError,RuntimeError)):s.consume(cursor,receipt,ids,q,np.ones((1,4)))
    with pytest.raises(RuntimeError):s.result()

def test_duplicate_receipt_cannot_replay_a_contribution():
    _,s,ids,q=fixture();cursor=s.begin();s.consume(cursor,[1,0,1,1,1,1,1,0],ids,q,np.ones((1,4)))
    with pytest.raises(RuntimeError):s.consume(cursor,[1,0,1,1,1,1,1,0],ids,q,np.ones((1,4)))
    with pytest.raises(RuntimeError):s.result()

def test_sixteenth_partial_batch_and_shared_work_deadline_refuse():
    _,s,ids,q=fixture()
    for i in range(15):s.consume(s.begin(),[i+1,0,1,1,i+1,i+1,1,0],ids,q,np.ones((1,4)))
    with pytest.raises(RuntimeError):s.consume(s.begin(),[16,0,1,1,16,16,1,0],ids,q,np.ones((1,4)))
    _,s,ids,q=fixture();s.clock=lambda:101.
    with pytest.raises(TimeoutError):s.begin()
    _,s,ids,q=fixture()
    with pytest.raises(RuntimeError):s.consume(s.begin(),[1,1,1,1,1,1,1,2000001],ids,q,np.ones((1,4)))

@pytest.mark.parametrize('kind',['padding','nan','lost-slot'])
def test_partial_or_nonfinite_sample_payload_is_refused(kind):
    _,s,ids,q=fixture()
    if kind=='padding':q[0,799,3]=1
    elif kind=='nan':q[0,0,3]=float('nan')
    else:ids[0,0]=-1
    with pytest.raises(ValueError):s.consume(s.begin(),[1,1,1,1,1,1,1,0],ids,q,np.ones((1,4)))
    with pytest.raises(RuntimeError):s.result()


def test_protocol_consumes_real_native_batches_before_exposing_pixel():
    import importlib.util
    from pathlib import Path
    from tools.opaque_batch_protocol import PixelStream
    spec=importlib.util.spec_from_file_location('batch_geometry_controls',Path(__file__).with_name('test_opaque_batch_geometry.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    v,f,c=m.wedge();s=PixelStream({'v':v,'f':f,'c':c},{'origin':[0.,0.]},deadline=100.,clock=lambda:1.)
    for cursor,ids,q,r in m.batches(v,f,c):
        assert s.begin()==cursor;n=r[2];colour=(q[0,:n,:3]*v[f[ids[0,:n]],0]).sum(1)
        contributions=np.zeros((n,4));contributions[:,0]=colour*q[0,:n,3];contributions[:,3]=q[0,:n,3]
        s.consume(cursor,r,ids,q,contributions)
    assert abs(s.result()[0]-.5)<1e-12 and abs(s.result()[3]-1)<1e-12 and s.cursor[1]>1
