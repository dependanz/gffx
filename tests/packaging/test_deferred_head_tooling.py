"""DF04 profile admission metadata; no tensors or corpus generation."""
import copy
import pytest


def profile():
    return dict(id='synthetic_face_m1',recipe_version=1,vertices=[None]*1986,triangles=[None]*3968,
        scene=dict(camera=dict(position=[0.,0.,3.],target=[0.,0.,0.],up=[0.,1.,0.],
            vertical_fov_degrees=40.,width=128,height=128),
            directional_light=dict(direction=[1.,1.,2.],irradiance=[1.,1.,1.])))


def test_deferred_head_profile_metadata_is_explicit():
    from tools.deferred_head import validate_profile
    assert validate_profile(profile(),'classic') is None
    assert validate_profile(profile(),'principled') is None


def test_deferred_head_profile_rejects_changed_camera_before_geometry():
    from tools.deferred_head import validate_profile
    value=profile();value['scene']['camera']['width']=256
    with pytest.raises(ValueError,match='camera'):validate_profile(value,'classic')


def test_deferred_head_profile_rejects_other_material_or_light():
    from tools.deferred_head import validate_profile
    with pytest.raises(ValueError,match='material'):validate_profile(profile(),'custom')
    value=copy.deepcopy(profile());value['scene']['directional_light']['irradiance']=[2.,1.,1.]
    with pytest.raises(ValueError,match='light'):validate_profile(value,'classic')


def test_deferred_head_declared_albedo_and_device_profiles():
    from tools.deferred_head import validate_profile
    for albedo in ('constant','vertex','gradient','checker'):
        for device in ('cpu','cuda'):
            assert validate_profile(profile(),'principled',albedo=albedo,device=device) is None
    with pytest.raises(ValueError,match='albedo'):validate_profile(profile(),'classic',albedo='custom')
    with pytest.raises(ValueError,match='device'):validate_profile(profile(),'classic',device='meta')
