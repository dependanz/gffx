"""Outward composition of complete independently certified visible-face pieces.

Artifact/provenance identities are checked by the caller before this pure
numerical operation. This cannot promote partial, missing or mismatched pieces.
"""
from decimal import Decimal
from tools.opaque_shading_intervals import I


def interval(text):
    if not isinstance(text,str) or not text.startswith('[') or not text.endswith(']'):raise ValueError('interval endpoints required')
    try:lo,hi=map(Decimal,text[1:-1].split(','))
    except Exception as error:raise ValueError('two decimal endpoints required') from error
    if not lo.is_finite() or not hi.is_finite() or lo>hi:raise ValueError('finite ordered endpoints required')
    q=I(lo,hi)
    if q.width>I('2e-12').lo:raise ValueError('piece reference radius exceeds acceptance')
    return q



def _motion_identity(motion,certificates,input_sha256,world_profile_sha256):
    if motion in ('normal_tilt','view_x','screen_x'):return False
    if motion not in ('world_x','camera_x','world_shear_xz'):raise ValueError('declared motion required')
    def digest(v):return isinstance(v,str) and len(v)==64 and all(c in '0123456789abcdef' for c in v)
    if not digest(input_sha256) or not digest(world_profile_sha256):raise ValueError('expected input and physical source SHA256 required')
    if not isinstance(certificates,list) or not certificates:raise ValueError('complete physical certificate set required')
    for row in certificates:
        if not isinstance(row,dict) or row.get('input_sha256')!=input_sha256 or row.get('world_profile_sha256')!=world_profile_sha256:raise ValueError('physical certificate input/source mismatch')
    return True

def compose(data,*,face,motion,certificates,input_sha256=None,world_profile_sha256=None):
    physical=_motion_identity(motion,certificates,input_sha256,world_profile_sha256)
    if type(face)!=int:raise ValueError('declared face/motion required')
    expected=[i for i,row in enumerate(data['selected']['visible_parts']) if data['faces'][row['face']]['face']==face]
    if not expected or not isinstance(certificates,list) or len(certificates)!=len(expected):raise ValueError('complete face certificate set required')
    by_piece={}
    for row in certificates:
        if row.get('status')!='qualified':raise ValueError('passing complete-piece certificate required')
        r=row.get('result',{});case=row.get('case',{});piece=r.get('piece')
        if type(piece)!=int or piece not in expected or piece in by_piece:raise ValueError('unique matching visible pieces required')
        if r.get('motion')!=motion or r.get('face')!=face or case.get('piece')!=piece or case.get('motion')!=motion:raise ValueError('piece/motion/face identity mismatch')
        by_piece[piece]=r
    result=dict(status='qualified_face',face=face,motion=motion,pieces=expected,scope='Outward sum of every saved visible piece for this face and packed-input direction; no installed or whole-pixel admission.')
    if physical:result.update(input_sha256=input_sha256,world_profile_sha256=world_profile_sha256,scope='Outward sum of all independently admitted face pieces for the pinned physical motion; no installed or whole-frame admission.')
    for quantity in ('value','derivative'):
        total=[I(0),I(0),I(0)]
        for piece in expected:
            channels=by_piece[piece].get(quantity)
            if not isinstance(channels,list) or len(channels)!=3:raise ValueError('three complete RGB channels required')
            total=[a+interval(b) for a,b in zip(total,channels)]
        if any(v.width>I('2e-12').lo for v in total):raise ValueError('combined face reference radius exceeds acceptance')
        result[quantity]=[[str(v.lo),str(v.hi)] for v in total]
    return result


def load_face_bundle(path,*,sha256,data,input_sha256,world_profile_sha256=None):
    """Verify artifact bytes and recompute all outward sums before comparison.

    Frozen producer identities and worker admission are audited by the bundle
    publisher separately. This never manufactures or relabels a certificate.
    """
    import json,hashlib
    from pathlib import Path
    def read_checked(p,digest):
        blob=Path(p).read_bytes()
        if hashlib.sha256(blob).hexdigest()!=digest:raise ValueError('face artifact digest mismatch')
        return json.loads(blob)
    try:
        bundle=read_checked(path,sha256)
        if bundle.get('world_profile_sha256')!=world_profile_sha256:raise ValueError('expected physical source identity required')
        if bundle['status']!='qualified_face' or bundle['input_sha256']!=input_sha256:raise ValueError('face input or status mismatch')
        if not isinstance(bundle['certificates'],list):raise ValueError('complete certificate list required')
        rows=[read_checked(r['path'],r['sha256']) for r in bundle['certificates']]
        r=bundle['result'];expected=compose(data,face=r['face'],motion=r['motion'],certificates=rows,input_sha256=input_sha256,world_profile_sha256=world_profile_sha256)
        if expected!=r:raise ValueError('stored face sum differs from recomputation')
        return expected
    except (KeyError,TypeError,OSError) as error:
        raise ValueError('complete checked face bundle required') from error


def compose_pixel(data,*,motion,certificates,input_sha256=None,world_profile_sha256=None):
    """Compose every visible original-pixel piece for one unchanged motion.

    Face certificates are recomputed first. Passing each face cannot bypass the
    final whole-pixel uncertainty gate. No value or derivative is borrowed from
    a different motion; producer/input/source admission remains caller-owned.
    """
    try:
        physical=_motion_identity(motion,certificates,input_sha256,world_profile_sha256)
        parts=data['selected']['visible_parts']
        if not isinstance(parts,list) or len(parts)!=13 or not isinstance(certificates,list) or len(certificates)!=13:
            raise ValueError('complete thirteen-piece pixel and declared motion required')
        faces=sorted({data['faces'][part['face']]['face'] for part in parts})
        if any(type(face)!=int for face in faces) or any(row.get('result',{}).get('face') not in faces for row in certificates):
            raise ValueError('matching original pixel face identities required')
        results=[compose(data,face=face,motion=motion,certificates=[row for row in certificates if row['result']['face']==face],input_sha256=input_sha256,world_profile_sha256=world_profile_sha256) for face in faces]
        result=dict(status='qualified_pixel',motion=motion,pieces=list(range(13)),faces=faces,scope='Outward sum of every independently admitted original visible piece for one packed-input direction. No installed, whole-frame, world-vertex, camera-pose or recomputed-normal admission.')
        if physical:result.update(input_sha256=input_sha256,world_profile_sha256=world_profile_sha256,scope='Outward sum of every independently admitted original visible piece for the pinned physical motion. No installed or whole-frame admission.')
        for quantity in ('value','derivative'):
            total=[I(0)]*3
            for face in results:
                total=[a+I(Decimal(lo),Decimal(hi)) for a,(lo,hi) in zip(total,face[quantity])]
            if any(v.width>I('2e-12').lo for v in total):raise ValueError('combined pixel reference radius exceeds acceptance')
            result[quantity]=[[str(v.lo),str(v.hi)] for v in total]
        return result
    except (KeyError,TypeError,IndexError) as error:
        raise ValueError('complete matching pixel certificate set required') from error


def load_pixel_bundle(path,*,sha256,data,input_sha256,world_profile_sha256=None):
    """Check all input/artifact bytes and recompute complete pixel endpoints."""
    import json,hashlib
    from pathlib import Path
    def read_checked(p,digest):
        blob=Path(p).read_bytes()
        if hashlib.sha256(blob).hexdigest()!=digest:raise ValueError('pixel artifact digest mismatch')
        return json.loads(blob)
    try:
        bundle=read_checked(path,sha256)
        if bundle.get('world_profile_sha256')!=world_profile_sha256:raise ValueError('expected physical source identity required')
        if bundle['status']!='qualified_pixel' or bundle['input_sha256']!=input_sha256:raise ValueError('pixel input or status mismatch')
        if not isinstance(bundle['certificates'],list):raise ValueError('complete certificate list required')
        rows=[read_checked(r['path'],r['sha256']) for r in bundle['certificates']]
        result=bundle['result'];expected=compose_pixel(data,motion=result['motion'],certificates=rows,input_sha256=input_sha256,world_profile_sha256=world_profile_sha256)
        if expected!=result:raise ValueError('stored pixel sum differs from recomputation')
        return expected
    except (KeyError,TypeError,OSError) as error:
        raise ValueError('complete checked pixel bundle required') from error
