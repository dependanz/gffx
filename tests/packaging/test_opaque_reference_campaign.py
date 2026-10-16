"""Independent certificate composition and hostile-record controls for the offline campaign."""
from copy import deepcopy
from fractions import Fraction as Q
import hashlib,json
import pytest
from tools.opaque_shading_intervals import I

def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def sha(value):return hashlib.sha256(canonical(value)).hexdigest()
def fixture():
    from tools.opaque_reference_campaign import make_spec
    parts=[dict(face=0,triangle=[[0,0],[str(Q(i+1,13)),0],[0,1]],motion=[[0,0]]*3) for i in range(13)]
    data=dict(material=dict(roughness=1.,f0=[.04]*3,albedo=[.2,.4,.6],direction=[1.,1.,2.]),budget={'fixture':'closed constants'},faces=[dict(face=42,corners=[[0,0],[1,0],[0,1]],clip_w=[1,1,1],normal=[[0,0,1]]*3,view=[[0,0,1]]*3)],selected=dict(origin=[0,0],visible_parts=parts),scene_sha256='a'*64)
    sources={'tools/reference.py':'b'*64};spec=make_spec(canonical(data),sources)
    receipts=[]
    for i,part in enumerate(parts):
        # Independent exact constant integrals: area*(2,3),not renderer output.
        area=Q(i+1,26);basis=[I(2*area),I(3*area)]
        result=dict(status='passed',protocol='original-piece-v1',spec_sha256=sha(spec),piece=i,piece_sha256=spec['pieces'][i]['sha256'],sources=sources,tolerance=spec['pieces'][i]['tolerance'],basis=[[str(x.lo),str(x.hi)] for x in basis],cells=1,max_depth=0,reference_seconds=.01)
        worker=dict(status='passed',mode='campaign-piece',pytest_exit_code=0,profile=dict(threads=1,host_bytes=1073741824,cooperative_seconds=120,watchdog_seconds=180),campaign=dict(spec_sha256=sha(spec),piece=i,result_sha256=sha(result)),supervisor=dict(exit_code=0,failure=None,elapsed_seconds=1.,sampled_peak_working_set_bytes=67108864))
        receipts.append(dict(result=result,worker=worker,result_sha256=sha(result),worker_sha256=sha(worker)))
    return data,spec,receipts

def test_complete_campaign_composes_exact_integrals_and_unchanged_radius():
    from tools.opaque_reference_campaign import compose
    data,spec,receipts=fixture();got=compose(spec,receipts)
    total_area=sum((Q(i+1,26) for i in range(13)),Q(0))
    for pair,a in zip(got['value'],data['material']['albedo']):
        want=total_area*(2+3*Q(a));assert Q(pair[0])<=want<=Q(pair[1])
        assert Q(pair[1])-Q(pair[0])<=Q(2,10**12)
    lo,hi=got['albedo_derivative'];assert Q(lo)<=3*total_area<=Q(hi)
    assert got['cells']==13 and got['piece_count']==13 and got['status']=='passed'
    assert got['scope']=='fixed_geometry_rgb_and_constant_albedo_only'

@pytest.mark.parametrize('change',['missing','duplicate','order','input','material','table','geometry','source','tolerance','nan','reversed_interval','wide_interval','result_bytes','worker_bytes'])
def test_campaign_rejects_missing_mismatched_or_modified_evidence(change):
    from tools.opaque_reference_campaign import compose
    _,spec,rows=fixture();assert compose(spec,rows)['status']=='passed';s=deepcopy(spec);r=deepcopy(rows)
    if change=='missing':r.pop()
    elif change=='duplicate':r[1]=deepcopy(r[0])
    elif change=='order':r[0],r[1]=r[1],r[0]
    elif change in ('input','material','table','geometry'):s[change+'_sha256']='c'*64
    elif change=='source':r[0]['result']['sources']['tools/reference.py']='c'*64
    elif change=='tolerance':r[0]['result']['tolerance']='1e-6'
    elif change=='nan':r[0]['result']['basis'][0]=['NaN','NaN']
    elif change=='reversed_interval':r[0]['result']['basis'][0]=['2','1']
    elif change=='wide_interval':r[0]['result']['basis'][0]=['0','1']
    elif change=='result_bytes':r[0]['result_sha256']='c'*64
    elif change=='worker_bytes':r[0]['worker_sha256']='c'*64
    # Rehash semantically malformed payloads to test validation beyond checksums.
    if change not in ('result_bytes','worker_bytes'):
        for row in r:
            row['result_sha256']=sha(row['result']);row['worker']['campaign']['result_sha256']=row['result_sha256'];row['worker_sha256']=sha(row['worker'])
    with pytest.raises(ValueError):compose(s,r)

@pytest.mark.parametrize('field,value',[('cells',257),('cells',True),('cells',-1),('max_depth',11),('reference_seconds',90.01),('reference_seconds',-1),('elapsed_seconds',120.01),('sampled_peak_working_set_bytes',1073741825),('exit_code',1),('failure','deadline'),('status','stopped')])
def test_campaign_rejects_failed_or_overbudget_jobs(field,value):
    from tools.opaque_reference_campaign import compose
    _,s,r=fixture();assert compose(s,r)['status']=='passed';row=r[0]
    target=row['result'] if field in ('cells','max_depth','reference_seconds') else row['worker'] if field=='status' else row['worker']['supervisor'];target[field]=value
    row['result_sha256']=sha(row['result']);row['worker']['campaign']['result_sha256']=row['result_sha256'];row['worker_sha256']=sha(row['worker'])
    with pytest.raises(ValueError):compose(s,r)


def test_campaign_prefix_stops_at_first_failure_without_accepting_partial_pixel():
    from tools.opaque_reference_campaign import check_prefix,compose
    _,s,r=fixture();prefix=r[:2]
    assert check_prefix(s,prefix)['complete'] is False
    with pytest.raises(ValueError,match='complete'):compose(s,prefix)
    bad=deepcopy(r[2]);bad['result']['status']='not_admitted';bad['result_sha256']=sha(bad['result']);bad['worker']['campaign']['result_sha256']=bad['result_sha256'];bad['worker_sha256']=sha(bad['worker'])
    with pytest.raises(ValueError):check_prefix(s,prefix+[bad,r[3]])


def test_campaign_prospectively_reserves_remaining_aggregate_work():
    from tools.opaque_reference_campaign import check_prefix,remaining_allowance
    _,s,r=fixture();usage=check_prefix(s,r[:12]);q=remaining_allowance(s,usage)
    assert q==dict(jobs=1,cells=3316,reference_seconds=1169.88,worker_seconds=1548.)
    usage=dict(jobs=13,cells=3328,reference_seconds=1170.,worker_seconds=1560.)
    with pytest.raises(ValueError):remaining_allowance(s,usage)


def test_campaign_loader_checks_original_file_bytes_before_parsing(tmp_path):
    from tools.opaque_reference_campaign import load_json,write_new
    p=tmp_path/'piece.json';value={'a':1};write_new(p,value);raw=p.read_bytes();h=hashlib.sha256(raw).hexdigest()
    assert load_json(p,h)==value
    with pytest.raises(FileExistsError):write_new(p,{'a':2})
    p.write_bytes(raw+b' ')
    with pytest.raises(ValueError,match='SHA'):load_json(p,h)


def test_piece_selector_retains_original_all_piece_error_allocation(monkeypatch):
    from tools import opaque_full_material_reference as ref
    # Early cancellation proves invalid selections refuse before expensive resource setup.
    for bad in (-1,13,True,1.5):
        with pytest.raises(ValueError,match='piece'):
            ref.full_pixel({},selected_piece=bad)


def test_selected_piece_uses_its_original_share_and_no_other_piece(monkeypatch):
    import os
    from pathlib import Path
    from tools import opaque_full_material_reference as ref
    data=json.loads(Path(os.environ['GFFX_FULL_INPUT']).read_bytes());calls=[]
    def stub(field,**kwargs):
        calls.append(kwargs)
        return dict(value=[I(0),I(0)],cells=1,max_depth=0)
    monkeypatch.setattr(ref,'integrate_square',stub)
    tolerances=ref.original_piece_tolerances(data['selected']['visible_parts'])
    for index in (0,12):
        before=len(calls);got=ref.full_pixel(data,selected_piece=index)
        assert len(calls)==before+1 and len(got['pieces'])==1
        assert got['pieces'][0]['piece']==index and got['selected_piece']==index
        assert calls[-1]['tolerance']==str(tolerances[index])
        assert calls[-1]['max_cells']==256
    # Mock outputs establish routing only,never numerical qualification.
