"""Prospective runtime guard; forecasts never become measured timing claims."""
import pytest


def calibration(size,seconds):
    return dict(status='passed',wheel={'sha256':'a'*64},driver_sha256='b'*64,
        profile=dict(width=size,height=size,dtype='float64',device='cpu',fitted_parameters=[]),
        estimate=dict(principled_point_block_limit=4),
        forward_seconds=seconds,supervisor=dict(exit_code=0,failure=None,
        elapsed_seconds=seconds+4,sampled_peak_working_set_bytes=600000000))


def test_deferred_head_cost_uses_slower_per_pixel_rate_and_margin():
    from tools.deferred_head import projected_seconds
    assert projected_seconds(calibration(16,1),calibration(32,2),'a'*64)==2*6536/256+30


def test_deferred_head_cost_rejects_changed_artifact_or_backward_profile():
    from tools.deferred_head import projected_seconds
    with pytest.raises(ValueError,match='identity'):
        projected_seconds(calibration(16,1),calibration(32,2),'c'*64)
    value=calibration(16,1);value['profile']['fitted_parameters']=['roughness']
    with pytest.raises(ValueError,match='forward'):
        projected_seconds(value,calibration(32,2),'a'*64)


def test_deferred_head_cost_refuses_forecast_over_cooperative_deadline():
    from tools.deferred_head import projected_seconds
    with pytest.raises(ValueError,match='time'):
        projected_seconds(calibration(16,3),calibration(32,12),'a'*64)


def test_deferred_head_cost_requires_matching_explicit_point_partition():
    from tools.deferred_head import projected_seconds
    small,large=calibration(16,1),calibration(32,2)
    with pytest.raises(ValueError,match='partition'):
        projected_seconds(small,large,'a'*64,point_block_limit=1)
    small['estimate']['principled_point_block_limit']=1
    large['estimate']['principled_point_block_limit']=1
    assert projected_seconds(small,large,'a'*64,point_block_limit=1)==2*6536/256+30
    del small['estimate']
    with pytest.raises(ValueError,match='partition'):
        projected_seconds(small,large,'a'*64,point_block_limit=1)


def test_deferred_cuda_crop_forecast_keeps_original_margin_and_caps():
    from tools.deferred_head import projected_seconds
    rows=[calibration(16,1),calibration(32,2)]
    for row in rows:
        row['profile']['device']='cuda';row['peak_allocated_bytes']=1024
        row['supervisor']['sampled_peak_working_set_bytes']=1200*1024**2
    assert projected_seconds(*rows,'a'*64,device='cuda',covered_pixels=3268)==2*3268/256+30
    rows[0]['peak_allocated_bytes']=67108865
    with pytest.raises(ValueError,match='GPU'):projected_seconds(*rows,'a'*64,device='cuda',covered_pixels=3268)
