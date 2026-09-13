"""Bounded image cost progression is explicit and cannot silently allocate a full frame."""
import pytest


def test_image_cost_profile_is_finite_and_requires_matching_previous_stage():
    from tools.image_cost_probe import PROFILE,check_previous
    assert PROFILE['sizes']==(32,128,256)
    assert PROFILE['warmups']==1 and PROFILE['repetitions']==3
    assert PROFILE['cooperative_seconds']==120 and PROFILE['watchdog_seconds']==180
    assert check_previous(32,False,None,'source','wheel') is None
    with pytest.raises(ValueError,match='previous'):
        check_previous(128,False,None,'source','wheel')
    previous={'status':'passed','side':32,'device':'cpu','driver_sha256':'source',
              'wheel':{'sha256':'wheel'},'supervisor':{'exit_code':0,'failure':None}}
    assert check_previous(128,False,previous,'source','wheel')==previous
    for key,value in (('status','stopped'),('side',128),('device','cuda'),('driver_sha256','changed')):
        bad={**previous,key:value}
        with pytest.raises(ValueError):check_previous(128,False,bad,'source','wheel')


@pytest.mark.parametrize('side',[0,64,512,1080,1920,True])
def test_image_cost_refuses_undeclared_size_before_tensor_work(side):
    from tools.image_cost_probe import check_previous
    with pytest.raises(ValueError,match='size'):
        check_previous(side,False,None,'source','wheel')
