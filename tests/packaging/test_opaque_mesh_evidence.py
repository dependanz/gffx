"""Recheck saved independently certified original-pixel evidence; no rerender."""
import hashlib,json,os
from pathlib import Path
from decimal import Decimal as D,localcontext
from fractions import Fraction as Q
import pytest

def rows(value,interval):
    output=[]
    with localcontext() as ctx:
        ctx.prec=90
        for got,bounds in zip(value,interval,strict=True):
            lo,hi=map(D,bounds.strip('[]').split(','));mid=(lo+hi)/2;radius=(hi-lo)/2
            assert 0<=radius<=D('1e-12')
            number=D.from_float(got);distance=max(lo-number,number-hi,D(0))
            assert distance<=D('1e-10')+D('1e-9')*abs(mid)
            output.append(dict(candidate=got,lower=str(lo),upper=str(hi),radius=str(radius),distance=str(distance),midpoint_error=str(abs(mid-number))))
    return output

def test_saved_original_pixel_matches_reference_and_device_bytes(record_property):
    manifest=Path(os.environ['GFFX_MESH_EVIDENCE']);assert hashlib.sha256(manifest.read_bytes()).hexdigest()==os.environ['GFFX_MESH_EVIDENCE_SHA']
    files=json.loads(manifest.read_bytes());loaded={}
    for name,entry in files.items():
        blob=Path(entry['path']).read_bytes();assert hashlib.sha256(blob).hexdigest()==entry['sha256'];r=json.loads(blob)
        assert r['status']=='passed' and r['supervisor']['failure'] is None and r['supervisor']['exit_code']==0
        assert r['supervisor']['elapsed_seconds']<120 and r['supervisor']['sampled_peak_working_set_bytes']<=r['profile']['host_bytes']
        assert r['profile']['origin']==[160,128] and r['profile']['width']==r['profile']['height']==256
        assert r['wheel']['matched_members']==74;loaded[name]=r
    ref=loaded['reference'];assert ref['accuracy_admitted'] and ref['reference_seconds']<90
    q=ref['reference'];assert q['tolerance']=='1e-12' and q['cells']<=256 and len(q['pieces'])<=32
    area=Q(0)
    for piece in q['pieces']:
        assert piece['max_depth']<=10
        a,b,c=[[Q(x) for x in p] for p in piece['triangle']]
        area+=abs((b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]))/2
    assert area==1
    result={}
    for name in ('cpu','cuda'):
        r=loaded[name];assert r['scene_sha256']==ref['scene_sha256'] and r['wheel']==ref['wheel']
        assert r['profile']['roughness']==1 and r['profile']['f0']==[.04]*3 and r['profile']['albedo']==[.2,.4,.6]
        result[name]=dict(rgb=rows(r['image']['colour'][0][0][0],q['value'][0][0]),albedo_tangent=rows([v[0] for v in r['albedo_jacobian']],q['derivative'][0][0]))
        assert abs(r['image']['mask'][0][0][0]-1)<=1e-12
    assert loaded['cpu']['bytes']==loaded['cuda']['bytes']
    assert loaded['cuda']['cuda_incremental_bytes']<=67108864
    record_property('original_pixel_comparison',json.dumps(result))

def test_saved_pixel_comparison_rejects_wrong_values_and_wide_reference():
    with pytest.raises(AssertionError):rows([.01],['[0,0]'])
    with pytest.raises(AssertionError):rows([0.],['[-1e-6,1e-6]'])
