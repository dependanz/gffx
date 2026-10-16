"""Saved original visible patch,finite nonlinear GGX horizon reference."""
import json,hashlib,os
from pathlib import Path
from fractions import Fraction as Q
from decimal import Decimal as D
from types import SimpleNamespace

def test_original_visible_conic_patch_value_and_first_normal_motion(record_property):
    from tools.opaque_curved_reference import CurvedTriangle,perspective_numerator,integrate_curved,radial_homogeneous_fields
    from tools.opaque_principled_reference import perspective_field
    from tools.opaque_shading_intervals import I
    path=Path(os.environ['GFFX_CURVED_INPUT']);blob=path.read_bytes();assert hashlib.sha256(blob).hexdigest()==os.environ['GFFX_CURVED_INPUT_SHA']
    data=json.loads(blob);chosen=data['selected'];row=next(r for r in data['faces'] if r['face']==chosen['face']);ox,oy=chosen['origin']
    tri=[(Q(p[0])-ox,Q(p[1])-oy) for p in row['corners']];iw=[1/Q(w) for w in row['clip_w']];motion=[[0,0,1]]*3
    coeff=perspective_numerator(tri,inverse_w=iw,normals=row['normal'],views=row['view'],normal_motion=motion)
    assert len(chosen['selected_face_parts'])==1
    visible=[[Q(z) for z in v] for v in chosen['selected_face_parts'][0]['triangle']]
    patch=CurvedTriangle(visible,coeff)
    def radiance(n,v,l,albedo,f0):
        dot=lambda a,b:sum((x*y for x,y in zip(a,b)),n[0].lift(0))
        nv=dot(n,v)/(dot(n,n)*dot(v,v)).sqrt();nl=dot(n,l)/(dot(n,n)*dot(l,l)).sqrt()
        if nl[0,0].lo<=0:raise ValueError('light horizon in reference patch')
        # Smooth front extension of exact roughness=F0=1,albedo=0 Principled.
        # The conic certificate establishes which physical points use it.
        return [nl/((nv+nl)*(2*I(float.fromhex('0x1.921fb54442d18p+1'))))]
    normal_paths=[[[value,velocity] for value,velocity in zip(row,dm)] for row,dm in zip(row['normal'],motion)]
    fields=radial_homogeneous_fields(patch,tri,inverse_w=iw,fields={'n':normal_paths,'v':row['view']})
    def radial(t,s):
        f=fields(t,s);return radiance(f['n'],f['v'],[t.lift(z) for z in (1,1,2)],None,None)+[t.lift(1)]
    result=integrate_curved(patch,radial,radial=True)
    result['field_certificate']=fields.certificate
    assert all(q.width<=D('2e-12') for q in result['value']+result['derivative'])
    assert result['value'][0].lo>0 and result['value'][1].hi<D(1)
    assert result['derivative'][1].lo>0
    evidence=dict(status='passed',input_sha256=hashlib.sha256(blob).hexdigest(),face=row['face'],origin=chosen['origin'],visible_triangle=visible,quadratic=[[str(z) for z in p.c] for p in coeff],result=result,material=dict(roughness=1,f0=[1,1,1],albedo=[0,0,0],direction=[1,1,2]),tangent='unit positive z in all three original face normal samples; fixed projected geometry and view',channels=['GGX radiance contribution','front-facing area'],scope='Actual original visible geometry,finite analytical material control; originalF0=.04/albedoRGB full material,production quadrature and whole-frame execution not admitted.')
    dest=Path(os.environ['GFFX_CURVED_RESULT']);assert not dest.exists();dest.write_text(json.dumps(evidence,indent=2,default=str),encoding='utf-8');record_property('curved_original',json.dumps({k:v for k,v in evidence.items() if k not in ('quadratic','visible_triangle')},default=str))
