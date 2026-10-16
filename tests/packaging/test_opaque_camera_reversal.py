"""Exact camera/world parameter reversal, never a relabeled numerical trial."""
from copy import deepcopy
from fractions import Fraction as Q
import pytest


def fixture():
    data=dict(material=dict(roughness=1.,f0=[.04]*3,albedo=[.2,.4,.6],direction=[1.,1.,2.]),scene_sha256='a'*64,selected=dict(original_vertex_ids=[0,1,2]),faces=[dict(vertices=[0,1,2],clip_w=[2,3,5],normal=[[1,2,3]]*3)])
    profile=dict(status='verified_source_profile',input_sha256='b'*64,scene_sha256='a'*64,camera=dict(fx=2,fy=3,cx=0,cy=0,pose=[[1,0,0,0],[0,1,0,0],[0,0,1,3],[0,0,0,1]]),vertices={'0':[1,0,1],'1':[0,1,0],'2':[-1,0,-2]})
    result=dict(status='qualified_pixel',motion='world_x',input_sha256='b'*64,world_profile_sha256='c'*64,pieces=list(range(13)),faces=list(range(6)),value=[['1','1.000000000001']]*3,derivative=[['-2e-1000','-1e-1000'],['-1','-.999999999999'],['0','1e-1000']])
    return data,profile,result


def test_camera_reversal_preserves_all_values_and_signed_tiny_derivatives():
    from tools.opaque_camera_reversal import camera_from_world
    data,profile,world=fixture();before=deepcopy(world)
    result=camera_from_world(data,profile,world,input_sha256='b'*64,world_profile_sha256='c'*64)
    assert world==before and result['motion']=='camera_x' and result['proof']=='exact_aligned_relative_translation_reversal'
    assert result['value']==world['value'] and result['pieces']==world['pieces']
    for got,(lo,hi) in zip(result['derivative'],world['derivative']):assert list(map(Q,got))==[-Q(hi),-Q(lo)]


@pytest.mark.parametrize('bad',['status','motion','input','profile','camera','normal_motion','view_motion','triangle_motion','screen_motion','inverse_w_motion','wide','nan','pieces','material'])
def test_camera_reversal_refuses_unproved_or_mismatched_source(monkeypatch,bad):
    import tools.opaque_camera_reversal as m
    data,profile,world=fixture()
    if bad=='status':world['status']='unresolved'
    if bad=='motion':world['motion']='world_shear_xz'
    if bad=='input':world['input_sha256']='d'*64
    if bad=='profile':world['world_profile_sha256']='d'*64
    if bad=='camera':profile['camera']['pose'][0][0]=2
    if bad=='wide':world['derivative'][0]=['0','3e-12']
    if bad=='nan':world['value'][0]=['NaN','1']
    if bad=='material':data['material']['direction']=[1,0,0]
    if bad=='pieces':world['pieces'].pop()
    if bad in ('normal_motion','view_motion','triangle_motion','screen_motion','inverse_w_motion'):
        old=m.aligned_world_motion
        def wrong(*a,**kw):
            r=old(*a,**kw)
            if kw['kind']=='camera_x':
                if bad=='screen_motion':r[bad][0][0]+=Q(1,10**1000)
                elif bad=='inverse_w_motion':r['faces'][0][bad][0]+=Q(1,10**1000)
                else:r['faces'][0][bad][0][0]+=Q(1,10**1000)
            return r
        monkeypatch.setattr(m,'aligned_world_motion',wrong)
    with pytest.raises(ValueError):m.camera_from_world(data,profile,world,input_sha256='b'*64,world_profile_sha256='c'*64)
