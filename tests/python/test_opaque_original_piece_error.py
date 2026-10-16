"""Installed native contributions against the completed independent reference."""
def test_original_native_face_contributions_against_complete_reference(monkeypatch,record_property):
    import json,os,hashlib,math,time
    from pathlib import Path
    from fractions import Fraction as F
    import torch
    from gffx.torch import render,_deferred
    from gffx.torch._curve_math import sum_last
    from tests.python.test_opaque_full_original import original
    from tools.material_loss_cost import check_deadline
    b=Path('D:/devspace/gffx/qualification-20260913/m1-continuation-20260913');c=b/'opaque120-20260924/original-campaign1';dest=Path(os.environ['GFFX_FULL_NATIVE']);assert not dest.exists();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    cert=json.loads((c/'complete-certificate.json').read_bytes());spec=json.loads((c/'spec.json').read_bytes());refs={}
    assert cert['piece_count']==13 and json.loads((c/'terminal.json').read_bytes())['status']=='passed'
    for i in range(13):
        v=json.loads((c/f'piece-{i:02d}-receipt.json').read_bytes());assert v['result']['status']=='passed' and v['result']['spec_sha256']==cert['spec_sha256'];face=spec['pieces'][i]['face'];row=refs.setdefault(face,dict(pieces=[],basis=[[F(0),F(0)],[F(0),F(0)]]));row['pieces'].append(i)
        for k in range(2):
            for e in range(2):row['basis'][k][e]+=F(v['result']['basis'][k][e])
    captured={};target=[None];quad=render.opaque_quadrature;shade=_deferred._shade
    def save_quad(*a,**kw):
        ids,q=quad(*a,**kw);captured['q']=q;return ids,q
    def save_shade(packed,camera,buffer,**kw):
        out=shade(packed,camera,buffer,**kw);captured['face']=buffer['_face'].reshape(-1)
        return out if target[0] is None else out*(buffer['_face'][...,0]==target[0])[...,None]
    monkeypatch.setattr(render,'opaque_quadrature',save_quad);monkeypatch.setattr(_deferred,'_shade',save_shade)
    s,camera,scene_sha=original();a=s.prims[1].parameters['albedo'];a.requires_grad_();started=time.perf_counter();out=render.opaque_window(s,camera,(78,101,1,1),cull_mode=render.CULL_BACK)
    whole=out['colour'][0,0,0];whole_jac=torch.stack([torch.autograd.grad(whole[k],a,retain_graph=True)[0] for k in range(3)]);saved=json.loads((b/'opaque63-20260920/pixel-cpu3-values.json').read_bytes());assert whole.tolist()==saved['image']['colour'][0][0][0] and whole_jac.tolist()==saved['albedo_jacobian']
    faces=captured['face'];actual_ids=set(faces[faces>=0].tolist());assert actual_ids==set(refs);rows=[]
    for face in sorted(refs):
        check_deadline(started+90);target[0]=face;keep=faces==face;part=render.opaque_window(s,camera,(78,101,1,1),cull_mode=render.CULL_BACK);rgb=part['colour'][0,0,0];grad=torch.autograd.grad(rgb[0],a)[0];assert grad[1:].tolist()==[0.,0.];matte=grad[0].item();gloss=[rgb[k].item()-a[k].item()*matte for k in range(3)];reference=refs[face]['basis'];error=[F.from_float(matte)-q for q in reference[1]]
        rows.append(dict(face=face,pieces=refs[face]['pieces'],samples=int(keep.sum()),rgb=rgb.tolist(),matte=matte,gloss=gloss,reference=[[str(q) for q in pair] for pair in reference],matte_error=[float(q) for q in error],maximum_absolute_matte_error=float(max(map(abs,error)))))
    assert all(abs(math.fsum(row['rgb'][k] for row in rows)-whole[k].item())<1e-15 for k in range(3));assert abs(math.fsum(row['matte'] for row in rows)-whole_jac[0,0].item())<1e-15
    lo,hi=map(F,cert['albedo_derivative']);v=F.from_float(whole_jac[0,0].item());passed=all(abs(v-q)<=F('1e-10')+F('1e-9')*abs(q) for q in (lo,hi))
    result=dict(status='passed' if passed else 'native_accuracy_failed',scope='Diagnostic re-evaluation with a face mask applied to installed production shading before its unchanged ordered accumulation; repeated visible pieces are combined',scene_sha256=scene_sha,certificate_sha256=sha(b/'opaque120-20260924/original-campaign1/complete-certificate.json'),seconds=time.perf_counter()-started,whole_rgb=whole.tolist(),whole_albedo_jacobian=whole_jac.tolist(),rows=rows)
    dest.write_text(json.dumps(result,indent=2));record_property('original_face_error',json.dumps(result));assert passed,'installed whole-pixel matte exceeds the independently certified original tolerance'
