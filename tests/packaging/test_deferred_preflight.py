"""Allocation-free arithmetic controls; not numerical rendering acceptance."""
import pytest
from tools.deferred_preflight import buffer_charge,target_profiles


def test_deferred_preflight_refuses_256_buffer_without_allocating():
    row=buffer_charge(256,256)
    assert row['conservative_buffer_charge_bytes']==136314880
    assert row['disposition']=='refused_by_buffer_charge'
    assert row['measured'] is False and row['admitted'] is False


def test_deferred_preflight_small_bound_never_claims_admission():
    row=buffer_charge(128,128)
    assert row['conservative_buffer_charge_bytes']==34078720
    assert row['disposition']=='full_preflight_required' and row['admitted'] is False
    assert len(target_profiles())==8
    large=buffer_charge(1080,1920)
    assert large['rgb_mask_output_bytes']==66355200
    assert large['workspace_limit_bytes']-large['rgb_mask_output_bytes']==753664


def test_deferred_preflight_counts_only_separate_geometry_buffers():
    hard=buffer_charge(2,3)['conservative_buffer_charge_bytes']
    assert buffer_charge(2,3,soft=True)['conservative_buffer_charge_bytes']==2*hard
    assert buffer_charge(2,3,factor=2)['conservative_buffer_charge_bytes']==5*hard
    assert buffer_charge(2,3,factor=2,geometric_outputs=False)['conservative_buffer_charge_bytes']==4*hard
    assert buffer_charge(2,3,soft=True,geometric_outputs=False)['conservative_buffer_charge_bytes']==hard
    with pytest.raises(ValueError):buffer_charge(True,3)
