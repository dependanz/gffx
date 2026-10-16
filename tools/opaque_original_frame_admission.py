"""Time-admission dimension for the approved original opaque frame profile.

No rendering and no accuracy/storage admission. Callers must independently verify
those prerequisites and enforce the shared whole-worker deadline before launch.
D504 changes ONLY the original mesh startup extents, not the progressive rule.
"""
import math
from tools.deferred_progressive_admission import forecast

SCENE_SHA='293d0f876b91be535fa09b78423b2f5bde56c36d2462241cec3d4acc20967b39'
STARTUPS={(128,128),(256,256)}
EXTENTS=STARTUPS|{(512,512),(1920,1080)}

def time_admission(anchors,current,*,remaining_seconds):
    if type(remaining_seconds) not in (int,float) or not math.isfinite(remaining_seconds) or not 0<remaining_seconds<=120:
        raise ValueError('finite remaining whole-worker time in (0,120] required')
    p=current.get('profile',{})
    if current.get('scene_sha256')!=SCENE_SHA or current.get('ownership_profile') not in ('public-window-pixel-replay-v2','public-window-pixel-replay-v3','public-window-pixel-replay-v4','public-window-albedo-replay-v5'):
        raise ValueError('approved original scene/ownership identity required')
    if current['ownership_profile'].endswith('v3') and (type(p.get('window_pixels')) is not int or not 2<=p['window_pixels']<=16):
        raise ValueError('explicit bounded window profile required')
    if current['ownership_profile'] in ('public-window-pixel-replay-v4','public-window-albedo-replay-v5'):
        for key,cap in (('window_pixels',16),('request_pixels',256)):
            value=p.get(key)
            if type(value) is not int or not 1<=value<=cap:raise ValueError('explicit bounded host request profile required')
    if current['ownership_profile']=='public-window-albedo-replay-v5' and p.get('backward') is not True:raise ValueError('explicit albedo replay requires backward scope')
    extent=(p.get('width'),p.get('height'))
    if any(type(v) is not int for v in extent) or extent not in EXTENTS:
        raise ValueError('approved original frame extent required')
    expected=dict(coverage='opaque',material='principled',dtype='float64',vertices=5002,faces=10000,outputs=['colour','mask'],cull_mode=2,workspace_limit_bytes=67108864,threads=1)
    if any(p.get(k)!=v for k,v in expected.items()) or any(type(p.get(k)) is not int for k in ('vertices','faces','cull_mode','workspace_limit_bytes','threads')):
        raise ValueError('approved original opaque profile required')
    if p.get('device') not in ('cpu','cuda') or type(p.get('backward')) is not bool:
        raise ValueError('explicit device and forward/backward profile required')
    if p.get('fitted_parameters')!=(['constant_RGB_albedo'] if p['backward'] else []):
        raise ValueError('approved gradient scope required')
    result=dict(startup=extent in STARTUPS,accuracy_admitted=False,storage_admitted=False,remaining_seconds=remaining_seconds)
    if result['startup']:
        if anchors:raise ValueError('startup exception does not consume timing anchors')
        return dict(result,policy='D504-original-opaque-startup-v1',forecast_seconds=None,scope='Time-policy startup exception only; accuracy/storage/resource admission still required')
    measured=forecast(anchors,current)
    if measured['seconds']>=remaining_seconds:
        raise ValueError('prospective complete frame exceeds remaining whole-worker time')
    return dict(result,policy=measured['policy'],forecast_seconds=measured['seconds'],forecast=measured,scope='Conservative time-policy admission only; accuracy/storage/resource admission still required')


def load_prerequisites(path, expected_sha256, *, wheel_sha256):
    """Validate a hash-pinned, reviewed prerequisite receipt; not a new oracle.

    The caller must freeze the receipt hash after independently reviewing the
    producers. Hashes preserve that review, and do not prove producer correctness.
    """
    import json,hashlib
    from decimal import Decimal
    from pathlib import Path
    def read(p,sha):
        raw=Path(p).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=sha:raise ValueError('prerequisite bytes changed')
        return raw
    r=json.loads(read(path,expected_sha256))
    if r.get('version')!=1 or r.get('scene_sha256')!=SCENE_SHA or r.get('wheel_sha256')!=wheel_sha256:
        raise ValueError('prerequisite scene/wheel identity differs')
    expected=[([59,10],22),([48,15],33),([67,117],28)]
    rows=r.get('pixels',[])
    if [(x.get('origin'),x.get('pieces')) for x in rows]!=expected:
        raise ValueError('three exact complete pixel prerequisites required')
    for x in rows:
        v=json.loads(read(x['path'],x['sha256']))
        if v.get('status')!='passed' or v.get('pieces')!=x['pieces'] or len(v.get('checks',[]))!=6:
            raise ValueError('incomplete pixel prerequisite')
        for c in v['checks']:
            e,a=Decimal(c['error_upper']),Decimal(c['allowance_lower'])
            if c.get('passed') is not True or not e.is_finite() or not a.is_finite() or not 0<=e<=a or a<=0:
                raise ValueError('failed pixel accuracy prerequisite')
    if not r.get('storage_sources'):raise ValueError('qualified storage source identities required')
    for x in r['storage_sources']:read(x['path'],x['sha256'])
    for x in r.get('supporting_evidence',[]):read(x['path'],x['sha256'])
    return dict(receipt_sha256=expected_sha256,verified_pixels=3,verified_pieces=83,
                scope='Reviewed representative-pixel and storage prerequisites; not per-pixel full-frame accuracy')
