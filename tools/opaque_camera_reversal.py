"""Independent camera-X certificate from exact relative-position symmetry.

For the pinned aligned camera, V(theta)-C(0) = V(0)-C(-theta).
Depth, perspective weights, normals, directional light and constant material
are unchanged. Thus the complete opaque image I_camera(theta)=I_world(-theta),
including visibility clipping, angular knots and moving shading horizons.
This transforms a source-bound complete certificate, not an execution report.
"""
from copy import deepcopy
from decimal import Decimal
from tools.opaque_projected_motion import aligned_world_motion
from tools.opaque_face_certificate import interval


def camera_from_world(data,profile,result,*,input_sha256,world_profile_sha256):
    if result.get('status')!='qualified_pixel' or result.get('motion')!='world_x' or result.get('pieces')!=list(range(13)):
        raise ValueError('complete qualified physical world-X pixel required')
    if result.get('input_sha256')!=input_sha256 or result.get('world_profile_sha256')!=world_profile_sha256:
        raise ValueError('matching physical source identities required')
    if data.get('material')!=dict(roughness=1.,f0=[.04]*3,albedo=[.2,.4,.6],direction=[1.,1.,2.]):
        raise ValueError('pinned constant material and directional light required')
    world=aligned_world_motion(data,profile,kind='world_x',input_sha256=input_sha256)
    camera=aligned_world_motion(data,profile,kind='camera_x',input_sha256=input_sha256)
    def opposite(a,b):
        if isinstance(a,list):return isinstance(b,list) and len(a)==len(b) and all(opposite(x,y) for x,y in zip(a,b))
        return a==-b
    if not opposite(world['screen_motion'],camera['screen_motion']):raise ValueError('screen reversal identity failed')
    for a,b in zip(world['faces'],camera['faces']):
        if any(not opposite(a[k],b[k]) for k in ('triangle_motion','normal_motion','view_motion','inverse_w_motion')):
            raise ValueError('complete field reversal identity failed')
    for name in ('value','derivative'):
        if not isinstance(result.get(name),list) or len(result[name])!=3:raise ValueError('complete RGB endpoints required')
        for row in result[name]:
            if not isinstance(row,list) or len(row)!=2:raise ValueError('two endpoints required')
            interval('['+','.join(row)+']')
    out=deepcopy(result);out['motion']='camera_x';out['proof']='exact_aligned_relative_translation_reversal'
    out['derivative']=[[str(Decimal(hi).copy_negate()),str(Decimal(lo).copy_negate())] for lo,hi in result['derivative']]
    out['scope']='Analytical camera-X parameter reversal of the complete source-bound world-X pixel. No new integration or arbitrary-camera admission.'
    return out
