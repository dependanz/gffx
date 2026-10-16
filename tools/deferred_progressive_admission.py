"""Approved progressive complete-frame forecast; no rendering or budget mutation."""
import hashlib
import json
import math
from pathlib import Path

POLICY='two_largest_complete_frames_v1'


def load_anchor(path, sha256):
    path=Path(path);raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=sha256:raise ValueError('anchor report hash differs')
    row=json.loads(raw)
    if set(row.get('arrays',{}))!={'colour','mask'}:raise ValueError('complete colour/mask evidence required')
    w,h=row['profile']['width'],row['profile']['height']
    if type(w) is not int or type(h) is not int or min(w,h)<=0:raise ValueError('invalid array profile extent')
    for name,value in row['arrays'].items():
        expected=[1,h,w,3] if name=='colour' else [1,h,w]
        if value.get('shape')!=expected or value.get('dtype')!='<f8' or value.get('bytes')!=math.prod(expected)*8:
            raise ValueError('anchor array extent/dtype/bytes differ from profile')
        blob=Path(value['path']).read_bytes()
        if len(blob)!=value['bytes'] or hashlib.sha256(blob).hexdigest()!=value['sha256']:
            raise ValueError('anchor array hash/length differs')
    return dict(path=str(path),sha256=sha256,report=row)


def forecast(anchors,current):
    profile=current['profile'];target=profile['width']*profile['height'];valid=[];seen=set()
    for anchor in anchors:
        r=anchor['report'];p=r.get('profile',{})
        if r.get('status')!='passed' or r.get('render_executed') is not True:
            raise ValueError('anchor was not a passed complete render')
        for key in ('driver_sha256','scene_sha256','source_sha256','ownership_profile'):
            if r.get(key)!=current.get(key):raise ValueError('anchor execution/ownership identity differs: '+key)
        if r.get('wheel',{}).get('sha256')!=current['wheel']['sha256']:
            raise ValueError('anchor wheel differs')
        expected={k:v for k,v in profile.items() if k not in ('width','height')}
        actual={k:v for k,v in p.items() if k not in ('width','height')}
        if actual!=expected:raise ValueError('anchor profile differs')
        for key in ('principled_point_block_limit','principled_group_point_limit'):
            if r.get('estimate',{}).get(key)!=current['estimate'].get(key):
                raise ValueError('anchor group limits differ')
        w,h=p.get('width'),p.get('height')
        if type(w) is not int or type(h) is not int or min(w,h)<=0:
            raise ValueError('invalid anchor extent')
        pixels=w*h
        if pixels>=target or w>profile['width'] or h>profile['height'] or pixels in seen:
            raise ValueError('anchors must have distinct prior extents')
        seen.add(pixels)
        supervision=r.get('supervisor',{})
        if supervision.get('exit_code')!=0 or supervision.get('failure'):
            raise ValueError('anchor supervision failed')
        seconds=r.get('measured_seconds')
        if type(seconds) not in (int,float) or not math.isfinite(seconds) or seconds<=0:
            raise ValueError('invalid anchor duration')
        valid.append((pixels,seconds/pixels,anchor))
    if len(valid)<2:raise ValueError('two complete anchors required')
    chosen=sorted(valid,key=lambda x:x[0])[-2:]
    seconds=max(x[1] for x in chosen)*target*2+30
    if not math.isfinite(seconds):raise ValueError('nonfinite forecast')
    return dict(policy=POLICY,seconds=seconds,margin=2,reserve_seconds=30,
        selected=[dict(path=x[2]['path'],sha256=x[2]['sha256'],pixels=x[0],seconds_per_pixel=x[1]) for x in chosen])
