"""Fixed DF04 head bindings and metadata checks; no stored-source execution."""
import math


def projected_seconds(small,large,wheel_sha256,*,point_block_limit=4):
    """Forecast only: slower measured rate, double margin,30s other overhead.

6536 is the pinned reference's covered count, not an arbitrary-scene bound.
The process watchdog and unchanged memory preflight remain mandatory.
"""
    if type(point_block_limit)!=int or point_block_limit not in (1,2,4):
        raise ValueError('invalid candidate point partition')
    rates=[];driver=None
    for row,size in ((small,16),(large,32)):
        if row.get('status')!='passed' or row.get('wheel',{}).get('sha256')!=wheel_sha256:
            raise ValueError('calibration identity/status differs')
        if not row.get('driver_sha256') or driver not in (None,row['driver_sha256']):
            raise ValueError('calibration driver identity differs')
        driver=row['driver_sha256'];profile=row.get('profile',{})
        expected=dict(width=size,height=size,dtype='float64',device='cpu',fitted_parameters=[])
        if any(profile.get(key)!=value for key,value in expected.items()):
            raise ValueError('calibration must be matching forward-only CPU profiles')
        if row.get('estimate',{}).get('principled_point_block_limit')!=point_block_limit:
            raise ValueError('calibration point partition differs or is not recorded')
        supervision=row.get('supervisor',{})
        if (supervision.get('exit_code')!=0 or supervision.get('failure') or
            not 0<supervision.get('sampled_peak_working_set_bytes',0)<=1024**3 or
            not 0<supervision.get('elapsed_seconds',0)<=180):
            raise ValueError('calibration supervision failed')
        seconds=row.get('forward_seconds')
        if type(seconds) not in (int,float) or not math.isfinite(seconds) or seconds<=0:
            raise ValueError('invalid calibration time')
        rates.append(seconds/(size*size))
    forecast=max(rates)*6536*2+30
    if forecast>120:raise ValueError('prospective head time exceeds cooperative deadline')
    return forecast


def validate_profile(data,kind):
    if kind not in ('classic','principled'):raise ValueError('unsupported DF04 material')
    if data.get('id')!='synthetic_face_m1' or data.get('recipe_version')!=1:
        raise ValueError('fixed head identity differs')
    if len(data.get('vertices',[]))!=1986 or len(data.get('triangles',[]))!=3968:
        raise ValueError('fixed head topology differs')
    expected=dict(position=[0.,0.,3.],target=[0.,0.,0.],up=[0.,1.,0.],
                  vertical_fov_degrees=40.,width=128,height=128)
    if data.get('scene',{}).get('camera')!=expected:raise ValueError('fixed head camera differs')
    if data['scene'].get('directional_light')!=dict(direction=[1.,1.,2.],irradiance=[1.,1.,1.]):
        raise ValueError('fixed head light differs')


def create_scene(data,kind):
    """Constant albedo first; no hidden UV/texture assets or renderer invocation."""
    validate_profile(data,kind)
    import torch
    from gffx.torch import scene,mesh,material,light
    from gffx.torch.camera import Camera
    def t(value):return torch.tensor(value,dtype=torch.float64)
    vertices=t(data['vertices']);faces=torch.tensor(data['triangles'],dtype=torch.int32)
    normals=mesh.vertex_normals(vertices,faces,weighting=mesh.WEIGHTING_AREA)
    parameters=dict(albedo=t([.2,.4,.6]))
    if kind=='classic':parameters.update(specular=t(.1),exponent=t(2.))
    else:
        parameters.update(roughness=t(1.),f0=t(.04),
            budget=material.PrincipledBudget.bundled('ggx-smith-loss-cpu-f64-v1',device='cpu',dtype=torch.float64))
    mat=scene.Material('skin',getattr(material,kind),parameters)
    lamp=scene.Light('key',light.directional,dict(directions=t([[1.,1.,2.]]),irradiance=t([[1.,1.,1.]])))
    model=scene.Scene([scene.Mesh('fixed-head',vertices,faces,material='skin',
        attributes={'normals':scene.Attribute(normals,'vertex')}),mat,lamp])
    focal=64/math.tan(math.radians(20));pose=torch.eye(4,dtype=torch.float64)[None];pose[0,2,3]=3.
    camera=Camera.from_intrinsics(t(focal),t(focal),t(64.),t(64.),128,128,.1,10.,pose=pose)
    return model,camera
