"""Exact original-piece classification, not numerical gradient acceptance."""
import os,json,hashlib
from pathlib import Path
from fractions import Fraction as Q
from tools.opaque_curved_reference import perspective_numerator,CurvedTriangle
from tools.opaque_moving_reference import P,BranchCertificate

def test_original_motion_piece_inventory(record_property):
    inp=Path(os.environ['GFFX_FULL_INPUT']);assert hashlib.sha256(inp.read_bytes()).hexdigest()=='a15bea40cb5cdead7994d828b57f5dcfac9f44fab0292a06e7d0a7b08719f73a';data=json.loads(inp.read_bytes());parts=data['selected']['visible_parts'];assert len(parts)==13;ox,oy=data['selected']['origin'];rows=[];area=Q(0)
    for i,part in enumerate(parts):
        source=data['faces'][part['face']];tri=[(Q(p[0])-ox,Q(p[1])-oy) for p in source['corners']];visible=[[Q(x) for x in p] for p in part['triangle']];iw=[1/Q(w) for w in source['clip_w']]
        motion=[[Q(.01)*Q(z),0,-Q(.01)*Q(x)] for x,y,z in source['normal']]
        co=perspective_numerator(tri,inverse_w=iw,normals=source['normal'],views=source['view'],normal_motion=motion);cert=BranchCertificate()
        def at(p):
            x,y=p;return sum((a*b for a,b in zip(co,(x*x,x*y,y*y,x,y,1))),P(0))
        v=[at(p) for p in visible];controls=v+[2*at([(visible[a][k]+visible[b][k])/2 for k in range(2)])-(v[a]+v[b])*Q(1,2) for a,b in ((0,1),(0,2),(1,2))];signs=[cert.classify(q) for q in controls]
        kind='front' if all(s>0 for s in signs) else 'back' if all(s<0 for s in signs) else 'curved';patch=CurvedTriangle(visible,co) if kind=='curved' else None
        a,b,c=visible;size=abs((b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]))/2;area+=size
        rows.append(dict(piece=i,face=source['face'],classification=kind,area=str(size),certificate=cert.report(),curved_certificate=patch.certificate if patch else None))
    assert 0<area<=1
    result=dict(status='classified',rows=rows,area=str(area),scope='Exact local topology/normal-motion sign certificates; no new integral or gradient accuracy admission.')
    out=Path(os.environ['GFFX_FULL_REFERENCE']);assert not out.exists();out.write_text(json.dumps(result,indent=2),encoding='utf8');record_property('motion_inventory',out.read_text())
