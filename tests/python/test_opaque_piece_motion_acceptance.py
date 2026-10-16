"""Installed packed-input motion versus independently bounded original pieces."""
import os,json,hashlib,struct
from pathlib import Path
from fractions import Fraction as Q
import torch


def _check_original_motion(record_property, *, complete_face=False,complete_pixel=False,node_transform=None,diagnostic_label=None,require_accuracy=True):
    from gffx.torch.render import opaque_quadrature
    from gffx.torch.material import PrincipledBudget,principled
    from gffx.torch._reference_channels import sum3,expand,multiply
    from gffx.torch.light import _unit
    from tools.opaque_streamed_motion import evaluate
    device=os.environ.get('GFFX_KNOT_DEVICE','cpu');t=lambda v:torch.tensor(v,dtype=torch.float64,device=device)
    p=Path(os.environ['GFFX_FULL_INPUT']);assert hashlib.sha256(p.read_bytes()).hexdigest()=='a15bea40cb5cdead7994d828b57f5dcfac9f44fab0292a06e7d0a7b08719f73a';data=json.loads(p.read_bytes());chosen=data['selected']
    path=Path(os.environ['GFFX_MOTION_REFERENCE']);digest=hashlib.sha256(path.read_bytes()).hexdigest();assert digest==os.environ['GFFX_MOTION_REFERENCE_SHA'];report=json.loads(path.read_bytes())
    if complete_pixel:
        assert not complete_face and (node_transform is None or not require_accuracy)
        from tools.opaque_face_certificate import load_pixel_bundle
        physical=json.loads(path.read_bytes()).get('world_profile_sha256')
        expected=load_pixel_bundle(path,sha256=digest,data=data,input_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),world_profile_sha256=physical)
        ref=dict(expected);ref.update({key:['['+','.join(v)+']' for v in expected[key]] for key in ('value','derivative')})
        piece=None;face=None;motion=ref['motion'];fis=[i for i,row in enumerate(data['faces']) if row['face'] in ref['faces']]
        assert len(fis)==len(ref['faces'])
    elif complete_face:
        from tools.opaque_face_certificate import load_face_bundle
        expected=load_face_bundle(path,sha256=digest,data=data,input_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        ref=dict(expected);ref.update({key:['['+','.join(v)+']' for v in expected[key]] for key in ('value','derivative')})
        piece=None;motion=ref['motion'];face=ref['face'];indices=[i for i,row in enumerate(data['faces']) if row['face']==face];assert len(indices)==1;fi=indices[0]
    else:
        assert report['status']=='qualified';ref=report['result'];piece=ref['piece'];motion=ref['motion'];face=ref['face'];part=chosen['visible_parts'][piece];fi=part['face']
        assert sum(row['face']==fi for row in chosen['visible_parts'])==1,'need complete-face certificate sum for a face with multiple pieces'
    if not complete_pixel:fis=[fi]
    assert motion in ('normal_tilt','view_x','screen_x','world_x')
    physical_motion=None
    if motion=='world_x':
        from tools.opaque_projected_motion import aligned_world_motion
        profile_path=Path(os.environ['GFFX_WORLD_PROFILE']);assert hashlib.sha256(profile_path.read_bytes()).hexdigest()==physical
        physical_motion=aligned_world_motion(data,json.loads(profile_path.read_bytes()),kind=motion,input_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    theta=t(0).requires_grad_();vertices=t(chosen['screen_vertices']);faces=torch.tensor(chosen['faces'],dtype=torch.int32,device=device)
    if motion=='screen_x':vertices=vertices+expand(theta,vertices.shape)*t([.01,0,0])
    if physical_motion:vertices=vertices+expand(theta,vertices.shape)*t(physical_motion['screen_motion'])
    normals=[];views=[];curves=[]
    for fi,row in enumerate(data['faces']):
        n=t(row['normal']);v=t(row['view']);iw=1/t(row['clip_w'])
        if motion=='normal_tilt':n=n+expand(theta,n.shape)*.01*torch.stack((n[:,2],torch.zeros_like(n[:,1]),-n[:,0]),-1)
        if motion=='view_x':v=v+expand(theta,v.shape)*t([.01,0,0])
        if physical_motion:
            n=n+expand(theta,n.shape)*t(physical_motion['faces'][fi]['normal_motion']);v=v+expand(theta,v.shape)*t(physical_motion['faces'][fi]['view_motion'])
        normals.append(n);views.append(v);dot=lambda a,b:sum3(a*b);co=[dot(n[i],v[i])*iw[i]*iw[i] for i in range(3)];co.extend((dot(n[i],v[j])+dot(n[j],v[i]))*iw[i]*iw[j] for i,j in ((0,1),(0,2),(1,2)));curves.append(torch.stack(co))
    owners,q=opaque_quadrature(vertices,faces,vertices.new_empty((len(faces),3,0)),1,1,curves=torch.stack(curves));global_samples=int((owners>=0).sum());owner_counts={str(i):int((owners==i).sum()) for i in range(len(faces))}
    if complete_pixel:assert set(owners[0][owners[0]>=0].cpu().tolist())==set(fis),'all active owners must match independently certified visible faces'
    per_face=[];sample_count=0
    for fi in fis:
        nodes=q[0][owners[0]==fi];row=data['faces'][fi];iw=1/t(row['clip_w']);lamp=t([[1,1,2]]);budget=PrincipledBudget.bundled('ggx-smith-loss-'+device+'-f64-v1',device=device,dtype=torch.float64)
        if node_transform is not None:nodes=node_transform(nodes,fi,theta,data,physical_motion)
        def field(samples):
            beta=samples[:,:3]*iw;n=torch.stack([sum3(beta*expand(normals[fi][:,k][None,:],beta.shape)) for k in range(3)],-1);v=torch.stack([sum3(beta*expand(views[fi][:,k][None,:],beta.shape)) for k in range(3)],-1)
            rgb=principled(n.contiguous(),v.contiguous(),lamp,albedo=t([.2,.4,.6]),roughness=t(1),f0=t([.04]*3),budget=budget)[:,0,:];nl=sum3(_unit(n,'probe normal')*_unit(lamp[0],'probe lamp'));return multiply(multiply(rgb,nl[:,None]),samples[:,3,None])
        face_actual=evaluate(nodes,theta,field);per_face.append(dict(face=data['faces'][fi]['face'],**face_actual))
        sample_count+=len(nodes)
    if complete_pixel:
        from tools.opaque_streamed_motion import sum_faces
        actual=sum_faces(per_face,expected_faces=ref['faces'])
    else:actual={k:v for k,v in per_face[0].items() if k!='face'}
    checks=[]
    for name in ['value','derivative']:
        for c,(a,pair) in enumerate(zip(actual[name],ref[name])):
            lo,hi=[Q(z.strip()) for z in pair.strip('[]').split(',')];assert hi-lo<=Q('2e-12')
            for endpoint in [lo,hi]:
                error=abs(Q(a)-endpoint);limit=Q('1e-10')+Q('1e-9')*abs(endpoint);checks.append(dict(quantity=name,channel=c,error_over_limit=float(error/limit),passed=error<=limit))
    result=dict(device=device,piece=piece,face=face,motion=motion,samples=sample_count,global_samples=global_samples,owner_counts=owner_counts,actual=actual,checks=checks,bytes=struct.pack('<6d',*actual['value'],*actual['derivative']).hex(),reference_sha256=digest,scope=('Complete selected original pixel and declared packed-input direction only; no whole-frame, world-vertex, camera-pose or recomputed-normal admission.' if complete_pixel else 'This complete visible face and packed-input direction only; no world-vertex, camera-pose, whole-pixel or recomputed-normal admission.'),complete_face=complete_face,complete_pixel=complete_pixel,per_face=per_face,diagnostic_label=diagnostic_label,comparison_passed=all(v['passed'] for v in checks))
    if device=='cuda':result['gpu_peak_bytes']=torch.cuda.max_memory_allocated();assert result['gpu_peak_bytes']<=67108864
    dest=Path(os.environ['GFFX_FULL_NATIVE']);assert not dest.exists();dest.write_text(json.dumps(result,indent=2),encoding='utf8');record_property('original_motion_accuracy',json.dumps(result));assert not require_accuracy or all(v['passed'] for v in checks),checks


def test_original_motion_meets_independent_piece_endpoints(record_property):
    _check_original_motion(record_property)


def test_original_motion_meets_independent_complete_face(record_property):
    _check_original_motion(record_property,complete_face=True)


def test_complete_face_flat_tensor_diagnostic(record_property):
    from tools.opaque_flat_resample import refine
    original=Path(os.environ['GFFX_FULL_NATIVE']);rows=[]
    orders=[tuple(map(int,z.split('x'))) for z in os.environ.get('GFFX_FLAT_DIAGNOSTIC_ORDERS','5x6,5x7,6x5,7x5').split(',')];assert len(orders)<=4
    for outer,inner in orders:
        dest=original.with_name(original.stem+f'-o{outer}i{inner}.json');os.environ['GFFX_FULL_NATIVE']=str(dest)
        _check_original_motion(record_property,complete_face=True,node_transform=lambda q,*unused:refine(q,outer,inner),diagnostic_label=f'Research-only flat tensor {outer}x{inner}; not native renderer admission.',require_accuracy=False)
        r=json.loads(dest.read_bytes());rows.append(dict(outer=outer,inner=inner,samples=r['samples'],global_samples=r['global_samples'],owner_counts=r['owner_counts'],comparison_passed=r['comparison_passed'],worst_error_over_limit=max(c['error_over_limit'] for c in r['checks']),path=str(dest),sha256=hashlib.sha256(dest.read_bytes()).hexdigest()));print(rows[-1],flush=True)
    original.write_text(json.dumps(dict(status='diagnostic_complete',rows=rows,scope='Explicit bounded CPU flat-fan tensor probes against the frozen independent complete-face reference; no production, CUDA, whole-pixel or M1 admission.'),indent=2),encoding='utf8')


def test_original_motion_meets_independent_complete_pixel(record_property):
    _check_original_motion(record_property,complete_pixel=True)


def test_complete_physical_pixel_diagnostic(record_property):
    _check_original_motion(record_property,complete_pixel=True,require_accuracy=False,diagnostic_label='Packed physical world-X decomposition; not renderer acceptance.')


def test_physical_flat_reallocation_diagnostic(record_property):
    from tools.opaque_flat_resample import refine
    order=tuple(map(int,os.environ['GFFX_FLAT_REALLOCATION'].split('x')));assert order in ((6,7),(7,6),(6,6))
    # Only the three independently certified all-flat faces; curved/mixed faces stay unchanged.
    def transform(q,fi,*unused):return refine(q,*order,source=(5,8)) if fi in (2,3,4) else q
    _check_original_motion(record_property,complete_pixel=True,require_accuracy=False,node_transform=transform,diagnostic_label='Research-only all-flat face node reallocation; native rule unchanged.')


def test_physical_symmetric_triangle_diagnostic(record_property):
    from tools.opaque_symmetric_triangle import refine
    def transform(q,fi,*unused):return refine(q) if fi in (2,3,4) else q
    _check_original_motion(record_property,complete_pixel=True,require_accuracy=False,node_transform=transform,diagnostic_label='Research-only symmetric degree14 rule on all-flat faces; native rule unchanged.')


def test_physical_flat_transport_diagnostic(record_property):
    from tools.opaque_flat_transport import nodes
    from tools.opaque_moving_reference import moving_fragments
    cache={}
    def transform(q,fi,theta,data,motion):
        if fi not in (2,3,4):return q
        chosen=data['selected']
        if not cache:cache['parts'],_=moving_fragments(chosen['screen_vertices'],chosen['faces'],motion['screen_motion'])
        row=data['faces'][fi];ox,oy=chosen['origin'];tri=[[Q(x)-ox,Q(y)-oy] for x,y in row['corners']]
        return nodes(theta,tri,motion['faces'][fi]['triangle_motion'],[r for r in cache['parts'] if r['face']==fi])
    _check_original_motion(record_property,complete_pixel=True,require_accuracy=False,node_transform=transform,diagnostic_label='Research-only42slot material/relative-boundary transport on all-flat faces; exact source geometry, not native admission.')



def test_transport_rebalanced_pixel_diagnostic(record_property):
    from tools.opaque_flat_transport import nodes
    from tools.opaque_flat_resample import refine
    from tools.opaque_moving_reference import moving_fragments
    profile=tuple(map(int,os.environ.get('GFFX_TRANSPORT_PROFILE','6,6,3').split(',')));assert profile in ((6,6,3),(5,7,3));cache={}
    def transform(q,fi,theta,data,motion):
        if fi not in (2,3,4):return q
        if motion is None:return refine(q,*profile[:2],source=(5,8))
        chosen=data['selected']
        if not cache:cache['parts'],_=moving_fragments(chosen['screen_vertices'],chosen['faces'],motion['screen_motion'])
        row=data['faces'][fi];ox,oy=chosen['origin'];tri=[[Q(x)-ox,Q(y)-oy] for x,y in row['corners']]
        return nodes(theta,tri,motion['faces'][fi]['triangle_motion'],[r for r in cache['parts'] if r['face']==fi],profile=profile)
    _check_original_motion(record_property,complete_pixel=True,require_accuracy=False,node_transform=transform,diagnostic_label='Research-only rebalanced volume/edge sample budget; not production admission.')
