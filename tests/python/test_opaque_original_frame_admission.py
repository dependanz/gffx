"""Contract tests for the D504 original-opaque startup exception. No rendering."""
import sys,unittest,importlib.util,math
from pathlib import Path

HERE=Path(__file__).parent

def module():
    from tools import opaque_original_frame_admission
    return opaque_original_frame_admission


def current(width=128,height=128):
    return dict(driver_sha256='d'*64,scene_sha256='293d0f876b91be535fa09b78423b2f5bde56c36d2462241cec3d4acc20967b39',source_sha256={'runner':'s'*64},ownership_profile='public-window-pixel-replay-v2',wheel={'sha256':'w'*64},estimate={'principled_point_block_limit':4,'principled_group_point_limit':4},profile=dict(width=width,height=height,coverage='opaque',material='principled',dtype='float64',device='cpu',vertices=5002,faces=10000,outputs=['colour','mask'],fitted_parameters=['constant_RGB_albedo'],backward=True,cull_mode=2,workspace_limit_bytes=67108864,threads=1))

def anchor(size,seconds):
    r=current(size,size);r.update(status='passed',render_executed=True,measured_seconds=seconds,supervisor={'exit_code':0,'failure':None})
    return dict(path=f'/verified/frame{size}.json',sha256='a'*64,report=r)

class AdmissionTests(unittest.TestCase):
    def test_only_approved_startups_need_no_timing_anchors(self):
        m=module()
        for size in (128,256):
            r=m.time_admission([],current(size,size),remaining_seconds=110)
            self.assertEqual(r['policy'],'D504-original-opaque-startup-v1');self.assertTrue(r['startup']);self.assertIsNone(r['forecast_seconds']);self.assertFalse(r['accuracy_admitted']);self.assertFalse(r['storage_admitted'])
    def test_large_frame_requires_two_matching_complete_anchors(self):
        m=module()
        with self.assertRaisesRegex(ValueError,'two complete anchors'):m.time_admission([],current(512,512),remaining_seconds=110)
        r=m.time_admission([anchor(128,1),anchor(256,3)],current(512,512),remaining_seconds=110)
        self.assertEqual(r['forecast_seconds'],62);self.assertFalse(r['startup'])
    def test_forecast_must_fit_remaining_whole_worker_time(self):
        m=module()
        for remaining in (62,61):
            with self.assertRaisesRegex(ValueError,'remaining'):m.time_admission([anchor(128,1),anchor(256,3)],current(512,512),remaining_seconds=remaining)
    def test_profiles_cannot_borrow_startup_exception(self):
        m=module()
        for key,value in [('coverage','soft'),('material','classic'),('dtype','float32'),('device','mps'),('vertices',5001),('faces',9999),('cull_mode',1),('threads',2),('workspace_limit_bytes',67108865),('backward',1),('outputs',['colour']),('fitted_parameters',['vertices'])]:
            with self.subTest(key=key):
                r=current();r['profile'][key]=value
                with self.assertRaises(ValueError):m.time_admission([],r,remaining_seconds=110)
        for field in ('scene_sha256','ownership_profile'):
            r=current();r[field]='other'
            with self.assertRaises(ValueError):m.time_admission([],r,remaining_seconds=110)
    def test_invalid_extents_and_time_refuse(self):
        m=module()
        for extent in [(32,32),(64,64),(128,256),(256,128),(True,128),(1921,1080)]:
            with self.assertRaises(ValueError):m.time_admission([],current(*extent),remaining_seconds=110)
        for value in (0,-1,121,float('nan'),float('inf'),True,'110'):
            with self.assertRaises(ValueError):m.time_admission([],current(),remaining_seconds=value)
    def test_forward_profile_and_cuda_are_explicit(self):
        m=module();r=current();r['profile'].update(backward=False,fitted_parameters=[],device='cuda')
        self.assertTrue(m.time_admission([],r,remaining_seconds=110)['startup'])
    def test_larger_frame_preserves_identity_checks(self):
        m=module();bad=anchor(256,3);bad['report']['wheel']['sha256']='other'
        with self.assertRaisesRegex(ValueError,'wheel'):m.time_admission([anchor(128,1),bad],current(512,512),remaining_seconds=110)

if __name__=='__main__':unittest.main()


def test_windowed_profile_requires_explicit_bounded_window_scope():
    import pytest
    m=module();r=current();r['ownership_profile']='public-window-pixel-replay-v3';r['profile']['window_pixels']=16
    assert m.time_admission([],r,remaining_seconds=110)['startup']
    for cap in (True,0,1,17,16.0):
        r['profile']['window_pixels']=cap
        with pytest.raises(ValueError):m.time_admission([],r,remaining_seconds=110)


def test_albedo_replay_profile_keeps_bounds_and_cannot_borrow_old_anchors():
    import pytest,copy
    m=module();r=current();r['ownership_profile']='public-window-albedo-replay-v5';r['profile'].update(window_pixels=16,request_pixels=256)
    assert m.time_admission([],r,remaining_seconds=110)['startup']
    for key,val in [('window_pixels',17),('request_pixels',257),('request_pixels',True),('backward',False)]:
        bad=copy.deepcopy(r);bad['profile'][key]=val
        if key=='backward':bad['profile']['fitted_parameters']=[]
        with pytest.raises(ValueError):m.time_admission([],bad,remaining_seconds=110)
    large=copy.deepcopy(r);large['profile'].update(width=512,height=512)
    with pytest.raises(ValueError,match='ownership'):m.time_admission([anchor(128,1),anchor(256,3)],large,remaining_seconds=110)
    rows=[anchor(128,1),anchor(256,3)]
    for row in rows:
        row['report']['ownership_profile']=r['ownership_profile'];row['report']['profile'].update(window_pixels=16,request_pixels=256)
    assert m.time_admission(rows,large,remaining_seconds=110)['forecast_seconds']==62
