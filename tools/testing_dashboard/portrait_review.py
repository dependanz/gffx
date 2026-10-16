"""Read-only numeric portrait-review data; no arbitrary file or HTML serving."""
import hashlib,json,math,re,struct
from pathlib import Path
from .presentation import read_asset

def load_review(root):
    root=Path(root);p=root/'reviews'/'portrait1.json'
    if p.is_symlink() or not p.is_file() or p.stat().st_size>65536:raise ValueError('Portrait review unavailable')
    raw=p.read_bytes();d=json.loads(raw)
    for key in ('image_sha256','annotation_sha256'):
        if not isinstance(d.get(key),str) or not re.fullmatch('[0-9a-f]{64}',d[key]):raise ValueError('Invalid review identity')
    w,h=d.get('width'),d.get('height')
    if type(w)!=int or type(h)!=int or not 1<=w<=1920 or not 1<=h<=1080:raise ValueError('Invalid portrait dimensions')
    blob=read_asset(root/'presentation',d['image_sha256']+'.png')
    if len(blob)<24 or struct.unpack('!II',blob[16:24])!=(w,h):raise ValueError('Portrait dimensions do not match image')
    def point(v):return isinstance(v,list) and len(v)==2 and all(type(x) in (float,int) and math.isfinite(x) for x in v) and 0<=v[0]<w and 0<=v[1]<h
    points=d.get('landmarks');polys=d.get('skin_polygons');crop=d.get('crop_xyxy')
    if not isinstance(points,list) or len(points)!=68 or not all(map(point,points)):raise ValueError('Invalid68landmark coordinates')
    if not isinstance(polys,list) or not 1<=len(polys)<=16 or any(not isinstance(v,list) or not 3<=len(v)<=128 or not all(map(point,v)) for v in polys):raise ValueError('Invalid skin polygons')
    if not isinstance(crop,list) or len(crop)!=4 or not all(type(x)==int for x in crop) or not 0<=crop[0]<crop[2]<=w or not 0<=crop[1]<crop[3]<=h:raise ValueError('Invalid crop')
    return dict(status='draft',revision_sha256=hashlib.sha256(raw).hexdigest(),annotation_sha256=d['annotation_sha256'],image_sha256=d['image_sha256'],image_url='/testing/media/'+d['image_sha256']+'.png',width=w,height=h,landmarks=points,skin_polygons=polys,crop_xyxy=crop)
