"""Closed polynomial controls for factored first-VJP probe evaluation."""
import pytest
@pytest.mark.parametrize('chunk',[1,2,4])
def test_streamed_motion_keeps_direct_and_sample_chain_rules(chunk):
    import torch
    from tools.opaque_streamed_motion import evaluate
    t=lambda x:torch.tensor(x,dtype=torch.float64);theta=t(.25).requires_grad_();x=t([0,.25,.75,1]);w=t([.125,.375,.375,.125]);q=torch.stack((x+theta,x*0,x*0,w*(1+theta)),-1)
    def field(p):return torch.stack((p[:,0]*theta,p[:,0]**2,theta.expand(len(p))),-1)*p[:,3,None]
    r=evaluate(q,theta,field,chunk=chunk)
    assert r['derivative']==[1.4375,2.546875,1.5]
    assert r['value']==[.234375,.83984375,.3125]
    assert r['sample_vjp_calls']==3

def test_streamed_motion_handles_no_direct_parameter_path():
    import torch
    from tools.opaque_streamed_motion import evaluate
    theta=torch.tensor(.25,dtype=torch.float64,requires_grad=True);q=torch.stack((theta+1,theta*0,theta*0,theta*0+1))[None]
    result=evaluate(q,theta,lambda p:p[:,:3]*p[:,3,None])
    assert result['derivative']==[1.,0.,0.]

def test_streamed_motion_requires_supported_bounded_profile():
    import torch
    from tools.opaque_streamed_motion import evaluate
    theta=torch.tensor(0.,requires_grad=True)
    with pytest.raises(ValueError):evaluate(torch.ones(801,4),theta,lambda p:p[:,:3])
    with pytest.raises(ValueError):evaluate(torch.ones(4,4),theta,lambda p:p[:,:3],chunk=8)


def test_complete_pixel_face_sum_keeps_signed_tiny_contributions():
    from tools.opaque_streamed_motion import sum_faces
    tiny=2.**-60
    rows=[dict(face=i,value=v,derivative=d,sample_vjp_calls=3) for i,v,d in [(10,[1e16,1e16,-1e16],[1e16,1e16,-1e16]),(11,[1.,tiny,-tiny],[tiny,1.,-1.]),(12,[-1e16,-1e16,1e16],[-1e16,-1e16,1e16])]]
    result=sum_faces(rows,expected_faces=[10,11,12])
    assert result['value']==[1.,tiny,-tiny] and result['derivative']==[tiny,1.,-1.] and result['sample_vjp_calls']==9
    assert sum_faces(rows[::-1],expected_faces=[10,11,12])==result


@pytest.mark.parametrize('bad',['missing','duplicate','unexpected','nan_value','inf_derivative','channels','expected_duplicate'])
def test_complete_pixel_face_sum_refuses_missing_owners_or_invalid_values(bad):
    from tools.opaque_streamed_motion import sum_faces
    from copy import deepcopy
    rows=[dict(face=i,value=[0.]*3,derivative=[0.]*3,sample_vjp_calls=3) for i in (10,11,12)];expected=[10,11,12]
    if bad=='missing':rows.pop()
    if bad=='duplicate':rows[2]=deepcopy(rows[0])
    if bad=='unexpected':rows[2]['face']=13
    if bad=='nan_value':rows[0]['value'][0]=float('nan')
    if bad=='inf_derivative':rows[0]['derivative'][0]=float('inf')
    if bad=='channels':rows[0]['derivative'].pop()
    if bad=='expected_duplicate':expected[2]=10
    with pytest.raises(ValueError):sum_faces(rows,expected_faces=expected)
