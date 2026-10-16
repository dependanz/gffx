"""Certify zero-overlap clipping skips and retain tiny positive contributions."""
import ctypes,json,os,hashlib
from pathlib import Path
import pytest,torch
from fractions import Fraction as Q
from tests.python.test_opaque_full_original import original
from tests.python.test_opaque_original_motion import moved

@pytest.mark.parametrize('kind,theta',[('normal_scale',0.),('radial_scale',.125),('vertex_x',-2**-10),('camera_x',-2**-10)])
def test_original_motion_separation_candidate(kind,theta,record_property):
    from gffx.torch import render
    from gffx.torch._opaque_scene import _project,_cuts,_normals,_horizon_cuts
    root=Path(os.environ['GFFX_FULL_NATIVE']).parent;manifest=json.loads((root/'native-diagnostic2.json').read_bytes());dll=root/'native-diagnostic2/diagnostic.dll';assert hashlib.sha256(dll.read_bytes()).hexdigest()==manifest['sha256']
    lib=ctypes.CDLL(str(dll));fn=lib.diagnose;P=ctypes.c_void_p;I=ctypes.c_int;fn.argtypes=[P,I,P,I,P,I,ctypes.c_double,ctypes.c_double,P,P,P];fn.restype=I
    s,c,_=original();s,c=moved(s,c,c.fx.new_tensor(theta),kind)
    with torch.no_grad():
        packed,c,v,keep,clip=_project(s,c,render.CULL_BACK);f=packed.faces;cuts=_cuts(packed,clip);normals=_normals(packed);corners=v[f.long(),:2];selected=keep&(corners[...,0].amax(-1)>=101)&(corners[...,0].amin(-1)<=102)&(corners[...,1].amax(-1)>=78)&(corners[...,1].amin(-1)<=79)
        horizon,curves=_horizon_cuts(packed,c,clip,normals,selected,remaining=16-cuts.shape[-1]);cuts=torch.cat((cuts,horizon),-1);idx=torch.arange(len(f),dtype=torch.int32)[keep];faces=f[idx.long()].contiguous();cuts=cuts[idx.long()].contiguous();curves=curves[idx.long()].contiguous();cc=torch.cat((cuts.reshape(len(faces),-1),curves),-1)[:,None,:].contiguous()
    ids=torch.empty((1,800),dtype=torch.int32);samples=torch.empty((1,800,4),dtype=torch.float64);diag=(I*7)();status=fn(v.data_ptr(),len(v),faces.data_ptr(),len(faces),cc.data_ptr(),cc.shape[-1],101.,78.,diag,ids.data_ptr(),samples.data_ptr())
    result=dict(kind=kind,theta=theta,status=status,failure_line=list(diag),candidate_faces=int(selected.sum()),original_face=int(idx[diag[3]]) if diag[3] else None)
    aa=(ctypes.c_double*128)();ii=(I*128)();lib.getfans.argtypes=[P,P];lib.getfans.restype=I;count=lib.getfans(aa,ii)
    result['native_count_pass_fans']=[dict(face=int(idx[ii[k]]),area=aa[k]) for k in range(count)]
    from tools.opaque_moving_reference import moving_fragments
    chosen=packed.faces[selected].tolist();used=sorted(set(j for face in chosen for j in face));lookup={x:i for i,x in enumerate(used)}
    vv=[[Q(p[0])-101,Q(p[1])-78,Q(p[2])] for p in v[used].tolist()];ff=[[lookup[j] for j in face] for face in chosen]
    parts,cert=moving_fragments(vv,ff,[[0,0,0]]*len(vv),origin=(0,0))
    areas=[]
    for part in parts:
        a,b,d=part['triangle'];areas.append(abs((b[0]-a[0])*(d[1]-a[1])-(b[1]-a[1])*(d[0]-a[0]))/2)
    result['exact_visibility']=dict(parts=len(parts),area=str(sum(areas)),minimum_area=str(min(areas)),areas=[str(a) for a in areas],scope='Exact rational represented screen geometry; excludes horizon subdivision and material integration')
    if status==0:assert diag[0]==0

    expected={}
    original_ids=torch.arange(len(packed.faces))[selected].tolist()
    for part,ar in zip(parts,areas):expected[original_ids[part['face']]]=expected.get(original_ids[part['face']],Q(0))+ar
    per_face={}
    for local_id in torch.unique(ids[ids>=0]).tolist():
        face=int(idx[local_id]);actual=float((samples[...,3]*(ids==local_id)).sum());target=float(expected.get(face,Q(0)));per_face[face]=dict(native=actual,exact=target,error=abs(actual-target))
    result['per_face_area']=per_face
    assert set(expected)<=set(per_face) and all(x['error']<=1e-12 for x in per_face.values())
    result['occupied_slots']=int((ids>=0).sum())
    result['weight_sum']=float(samples[...,3].sum()) if status==0 else None
    result['input_path']=str(root/(Path(os.environ['GFFX_FULL_NATIVE']).stem+'-'+kind+'-input.json'))
    input_path=Path(result['input_path']);assert not input_path.exists();input_path.write_text(json.dumps(dict(vertices=v.tolist(),faces=faces.tolist(),cuts=cc.tolist(),origin=[101,78])),encoding='utf8')
    # A separate no-horizon diagnostic isolates pure visibility fragmentation;
    # never substitute these samples for shading or claim a renderer repair.
    cc[:,:,-6:]=0;diag2=(I*7)();result['no_curve_status']=fn(v.data_ptr(),len(v),faces.data_ptr(),len(faces),cc.data_ptr(),cc.shape[-1],101.,78.,diag2,ids.data_ptr(),samples.data_ptr());result['no_curve_failure_line']=list(diag2)
    p=Path(os.environ['GFFX_FULL_NATIVE']);rows=json.loads(p.read_bytes()) if p.exists() else [];rows.append(result);p.write_text(json.dumps(rows,indent=2),encoding='utf8');record_property('separation_candidate',json.dumps(result));assert status==0,result;assert abs(result['weight_sum']-float(sum(areas)))<=1e-12

def test_separation_proof_never_discards_positive_overlap(record_property):
    import random
    from tools import opaque_pixel_oracle as oracle
    root=Path(os.environ['GFFX_FULL_NATIVE']).parent;dll=root/'native-diagnostic2/diagnostic.dll';manifest=json.loads((root/'native-diagnostic2.json').read_bytes());assert hashlib.sha256(dll.read_bytes()).hexdigest()==manifest['sha256']
    fn=ctypes.CDLL(str(dll)).separated;P=ctypes.c_void_p;fn.argtypes=[P,ctypes.c_int,P,P];fn.restype=ctypes.c_int;rng=random.Random(127);checked=admitted=overlaps=0
    fixtures=[([[0,0,0],[1,0,0],[0,1,0],[1,1,0]],[0,1,2],[2,1,3]),([[0,0,0],[1,0,0],[0,1,0],[.5,2**-40,0]],[0,1,2],[0,1,3]),([[0,0,0],[1,0,0],[0,1,0],[.5,-2**-40,0]],[0,1,2],[1,0,3])]
    for i in range(256):
        v=[[rng.uniform(-2,2),rng.uniform(-2,2),0] for _ in range(6)];fixtures.append((v,[0,1,2],[2,1,3] if i%2 else [3,4,5]))
    for v,a,b in fixtures:
        vv=torch.tensor(v,dtype=torch.float64);aa=torch.tensor(a,dtype=torch.int32);bb=torch.tensor(b,dtype=torch.int32);got=fn(vv.data_ptr(),len(v),aa.data_ptr(),bb.data_ptr());points=[[Q(x) for x in p] for p in v];poly=tuple(tuple(points[i][:2]) for i in b);tri=oracle._triangle([points[i] for i in a]);assert tri is not None;budget=oracle._Budget(lambda:None)
        for line in tri[0]:poly=oracle._clip(poly,line,budget)
        area=oracle._area(poly);assert not got or area==0,(v,a,b,area)
        if checked==0:assert got==1
        if checked==1:assert got==0 and area>0
        if checked==2:assert got==1
        checked+=1;admitted+=bool(got);overlaps+=area>0
    record_property('separation_proof',json.dumps(dict(cases=checked,certified_separations=admitted,positive_overlap_cases=overlaps,arbitrary_epsilon=False)))
@pytest.mark.parametrize('exponent',[20,40,60])
def test_installed_separation_retains_tiny_positive_overlap_and_gradient(exponent,record_property):
    from gffx.torch import render
    device=os.environ.get('GFFX_CURVED_DEVICE','cpu');e=2.**-exponent
    v=torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,-1.],[1.,0.,-1.],[.5,e,-1.]],dtype=torch.float64,device=device,requires_grad=True);f=torch.tensor([[0,1,2],[3,4,5]],dtype=torch.int32,device=device)
    ids,q=render.opaque_quadrature(v,f,v.new_empty((2,3,0)),1,1);area=(q[...,3]*(ids==1)).sum();g=torch.autograd.grad(area,v)[0]
    assert area.item()>0 and abs(area.item()/(e/2)-1)<2e-14
    assert abs(g[5,1].item()-.5)<1e-13
    record_property('tiny_overlap',json.dumps(dict(device=device,height=e,area=area.item(),height_derivative=g[5,1].item(),sample_slots=int((ids>=0).sum()))))

def test_adapter_preserves_all_registered_dispatch_entries(record_property):
    import re
    from gffx.torch import render
    # Trigger the installed extension once; inspect registrations without running
    # unrelated numerical work or claiming those operations requalified.
    v=torch.empty((0,3),dtype=torch.float64);render.opaque_quadrature(v,torch.empty((0,3),dtype=torch.int32),v.new_empty((0,3,0)),1,1)
    old=Path(os.environ['GFFX_FULL_NATIVE']).parent/'source1/adapters/pytorch/register.cpp';text=old.read_text();rows=[]
    for device,body in re.findall(r'STABLE_TORCH_LIBRARY_IMPL\(gffx, (CPU|CUDA|CompositeExplicitAutograd), m\) \{(.*?)\n\}',text,re.S):
        for name in re.findall(r'm.impl\("([^"]+)"',body):
            assert torch._C._dispatch_has_kernel_for_dispatch_key('gffx::'+name,device),(name,device);rows.append([name,device])
    assert len(rows)==135;record_property('adapter_dispatch',json.dumps(dict(registered_entries=len(rows),scope='Registration reachability only; no numerical claims for unexecuted operations')))
