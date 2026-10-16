"""Bounded radial-order and point-material diagnosis; no production promotion."""
def test_original_endpoint_mapped_radial_rows(monkeypatch,record_property):
    import json,os,hashlib,time,math
    from pathlib import Path
    from fractions import Fraction as F
    from decimal import Decimal as D,localcontext
    import torch,numpy as np
    from gffx.torch import render,_deferred
    from gffx.torch.material import principled
    from gffx.torch._reference_channels import sum3
    from tests.python.test_opaque_full_original import original
    from tools.opaque_radial_resample import resample_rows,mapped_rows
    from tools.principled_material_oracle import material_pair
    from tools.material_loss_cost import check_deadline
    b=Path('D:/devspace/gffx/qualification-20260913/m1-continuation-20260913');p=b/'opaque63-20260920/full-input1.json';assert hashlib.sha256(p.read_bytes()).hexdigest()=='a15bea40cb5cdead7994d828b57f5dcfac9f44fab0292a06e7d0a7b08719f73a';data=json.loads(p.read_bytes());cert=json.loads((b/'opaque120-20260924/original-campaign1/complete-certificate.json').read_bytes());dest=Path(os.environ['GFFX_FULL_NATIVE']);assert not dest.exists();captured={};quad=render.opaque_quadrature;shade=_deferred._shade
    def save_quad(*a,**kw):
        ids,q=quad(*a,**kw);captured['q']=q.detach().cpu().numpy().reshape(-1,4).copy();return ids,q
    def save_shade(packed,camera,buffer,**kw):
        captured['face']=buffer['_face'].reshape(-1).detach().cpu().numpy().copy();return shade(packed,camera,buffer,**kw)
    monkeypatch.setattr(render,'opaque_quadrature',save_quad);monkeypatch.setattr(_deferred,'_shade',save_shade)
    s,c,scene_sha=original();a=s.prims[1].parameters['albedo'];a.requires_grad_();budget=s.prims[1].parameters['budget'];started=time.perf_counter();out=render.opaque_window(s,c,(78,101,1,1),cull_mode=render.CULL_BACK);actual=out['colour'][0,0,0].tolist();gj=torch.autograd.grad(out['colour'][0,0,0,0],a)[0][0].item();del out
    faces={row['face']:row for row in data['faces']};ids=captured['face'];samples=captured['q'];rows=[];point_checks=[];report=dict(status='incomplete',scene_sha256=scene_sha,scope='Diagnostic reconstructed radial rows with installed material; Five-node endpoint mappings are diagnostic only, not a production implementation',orders=rows,point_checks=point_checks)
    tensor=lambda x:torch.tensor(x,dtype=torch.float64);lamp=tensor([[1.,1.,2.]]);dot=lambda a,b:sum3(a*b);passed_orders=[]
    try:
        for order in (5,'square_left','square_right','smooth3','smooth5'):
            start=time.perf_counter();values=[];matte=[];by_face=[];total_nodes=0
            for face in sorted(set(ids[ids>=0].tolist())):
                check_deadline(started+90);q=resample_rows(samples[ids==face],5) if order==5 else mapped_rows(samples[ids==face],order);row=faces[face];normal=tensor(row['normal']);view=tensor(row['view']);iw=1/tensor(row['clip_w']);fv=[];fm=[]
                for k in range(0,len(q),4):
                    check_deadline(started+90);part=tensor(q[k:k+4]);beta=part[:,:3]*iw;nn=torch.stack([sum3(beta*normal[:,j]) for j in range(3)],-1);vv=torch.stack([sum3(beta*view[:,j]) for j in range(3)],-1)
                    rgb=principled(nn.contiguous(),vv.contiguous(),lamp,albedo=a,roughness=tensor(1.),f0=tensor([.04]*3),budget=budget)[:,0,:];nl=dot(nn,lamp[0])/(dot(nn,nn).sqrt()*dot(lamp[0],lamp[0]).sqrt());radiance=rgb*nl[:,None];value=(radiance*part[:,3,None]).sum(0);grad=torch.autograd.grad(value[0],a)[0];assert grad[1:].tolist()==[0.,0.];fv.append(value.tolist());fm.append(grad[0].item())
                    if order==5 and face==5743 and len(point_checks)<3:
                        for j in range(len(part)):
                            if len(point_checks)>=3:break
                            if dot(nn[j],vv[j]).item()<=0:continue
                            params=nn[j].tolist()+vv[j].tolist()+[1.,1.,2.]+[.2,.4,.6]+[1.]+[.04]*3
                            with localcontext() as ctx:
                                ctx.prec=90;pp=list(map(D.from_float,params));cos=sum(x*y for x,y in zip(pp[:3],pp[6:9]))/(sum(x*x for x in pp[:3])*sum(x*x for x in pp[6:9])).sqrt();ref=[v*cos for v in material_pair(data['budget'],params)];errors=[abs(D.from_float(radiance[j,t].item())-ref[t]) for t in range(3)]
                            point_checks.append(dict(face=face,node=k+j,maximum_error=str(max(errors))));assert max(errors)<D('1e-12')
                values.extend(fv);matte.extend(fm);total_nodes+=len(q);by_face.append(dict(face=face,nodes=len(q),matte=math.fsum(fm),rgb=[math.fsum(v[j] for v in fv) for j in range(3)]))
            value=[math.fsum(v[j] for v in values) for j in range(3)];m=math.fsum(matte)
            if order==5:assert max(abs(v-z) for v,z in zip(value,actual))<1e-15 and abs(m-gj)<1e-15
            checks=[]
            for label,v,pair in [(f'rgb{j}',value[j],cert['value'][j]) for j in range(3)]+[('matte',m,cert['albedo_derivative'])]:
                ends=[dict(error=float(abs(F.from_float(v)-F(q))),limit=float(F('1e-10')+F('1e-9')*abs(F(q))),passed=abs(F.from_float(v)-F(q))<=F('1e-10')+F('1e-9')*abs(F(q))) for q in pair];checks.append(dict(channel=label,value=v,endpoints=ends))
            passed=all(e['passed'] for z in checks for e in z['endpoints']);rows.append(dict(order=order,nodes=total_nodes,seconds=time.perf_counter()-start,checks=checks,by_face=by_face,passed=passed))
            if passed:passed_orders.append(order)
        assert len(point_checks)==3;report.update(status='completed_diagnostic',passing_orders=passed_orders);assert passed_orders,'five-node endpoint mappings do not meet the original tolerance'
    finally:
        report['seconds']=time.perf_counter()-started;dest.write_text(json.dumps(report,indent=2));record_property('radial_diagnostic',json.dumps(report))
