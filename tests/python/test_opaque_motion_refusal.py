"""Diagnose bounded native refusals without changing production arithmetic."""
import ctypes,json,os,hashlib
from pathlib import Path
import pytest,torch
from fractions import Fraction as Q
from tests.python.test_opaque_full_original import original
from tests.python.test_opaque_original_motion import moved

@pytest.mark.parametrize('kind,theta',[('normal_scale',0.),('radial_scale',.125),('vertex_x',-2**-10),('camera_x',-2**-10)])
def test_original_motion_native_refusal(kind,theta,record_property):
    from gffx.torch import render
    from gffx.torch._opaque_scene import _project,_cuts,_normals,_horizon_cuts
    root=Path(os.environ['GFFX_FULL_NATIVE']).parent;manifest=json.loads((root/'native-diagnostic3.json').read_bytes());dll=root/'native-diagnostic3/diagnostic.dll';assert hashlib.sha256(dll.read_bytes()).hexdigest()==manifest['sha256']
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

    try:
        actual=render.opaque_quadrature(v,faces,cuts,1,1,origin=(101,78),curves=curves)
        result['installed_status']='passed';assert status==0
        assert torch.equal(actual[0],ids) and torch.equal(actual[1].contiguous().view(torch.uint8),samples.contiguous().view(torch.uint8))
        result['sample_bytes_match_installed']=True
    except OverflowError as error:
        result['installed_status']='refused';result['installed_error']=str(error);assert status==2
    # A separate no-horizon diagnostic isolates pure visibility fragmentation;
    # never substitute these samples for shading or claim a renderer repair.
    cc[:,:,-6:]=0;diag2=(I*7)();result['no_curve_status']=fn(v.data_ptr(),len(v),faces.data_ptr(),len(faces),cc.data_ptr(),cc.shape[-1],101.,78.,diag2,ids.data_ptr(),samples.data_ptr());result['no_curve_failure_line']=list(diag2)
    p=Path(os.environ['GFFX_FULL_NATIVE']);rows=json.loads(p.read_bytes()) if p.exists() else [];rows.append(result);p.write_text(json.dumps(rows,indent=2),encoding='utf8');record_property('refusal_diagnostic',json.dumps(result))
