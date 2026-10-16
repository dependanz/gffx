"""Fail-closed contracts for one approved22-piece offline campaign."""
import copy,json,os,hashlib
from pathlib import Path
from fractions import Fraction as Q
import pytest
from tools.opaque_shading_intervals import I

@pytest.fixture(scope="module")
def data():
    p=os.environ.get('GFFX_STREAMED_INPUT')
    if p is None:pytest.skip('explicit frozen streamed-pixel input required')
    return json.loads(Path(p).read_bytes())

def test_streamed_input_and_allocation_bind_all22_pieces(data):
    from tools.opaque_streamed_reference_contract import validate_input,allocation
    validate_input(data);shares=allocation(data)
    assert len(shares)==22 and all(v>0 for v in shares)
    assert sum((I(v) for v in shares),I(0)).hi<=I('5e-13').lo
    changed=copy.deepcopy(data);changed['selected']['origin'][0]+=1
    with pytest.raises(ValueError):validate_input(changed)
    changed=copy.deepcopy(data);changed['selected']['visible_parts'].reverse()
    with pytest.raises(ValueError):validate_input(changed)

def reports(data):
    from tools.opaque_streamed_reference_contract import certificate_headers
    rows=[];token='a'*64
    for i,row in enumerate(certificate_headers(data,token)):
        tiny=I(Q(i+1,2**60))
        row.update(status='passed',basis=[[str(tiny.lo),str(tiny.hi)]]*2,cells=1,reference_seconds=.25,seconds=1.,supervisor=dict(exit_code=0,failure=None,elapsed_seconds=2.,sampled_peak_working_set_bytes=4096))
        rows.append(row)
    return rows,token

def test_aggregation_keeps_every_positive_tiny_piece_and_exact_order(data):
    from tools.opaque_streamed_reference_contract import aggregate
    rows,token=reports(data);r=aggregate(data,rows,token);expected=I(Q(sum(range(1,23)),2**60))
    for value in r['basis']:
        got=I(*value);assert got.lo<=expected.lo<=expected.hi<=got.hi and got.lo>0
    assert r['pieces']==22 and r['cells']==22

@pytest.mark.parametrize('change',['missing','duplicate','reordered','failed','candidate','input','geometry','allocation','radius','nan','cells','reference_time','worker_time','memory','exit'])
def test_aggregation_rejects_incomplete_or_unbound_evidence(data,change):
    from tools.opaque_streamed_reference_contract import aggregate
    rows,token=reports(data)
    if change=='missing':rows.pop()
    elif change=='duplicate':rows[-1]=rows[0]
    elif change=='reordered':rows[0],rows[1]=rows[1],rows[0]
    elif change=='radius':rows[0]['basis'][0]=['0','1e-6']
    elif change=='nan':rows[0]['basis'][0]=['NaN','NaN']
    elif change=='failed':rows[0]['status']='stopped'
    elif change=='cells':rows[0]['cells']=257
    elif change=='reference_time':rows[0]['reference_seconds']=91
    elif change=='worker_time':rows[0]['seconds']=121
    elif change=='memory':rows[0]['supervisor']['sampled_peak_working_set_bytes']=1073741825
    elif change=='exit':rows[0]['supervisor']['exit_code']=1
    else:rows[0][{'candidate':'manifest_sha256','input':'input_sha256','geometry':'piece_sha256','allocation':'allocation_sha256'}[change]]='b'*64
    with pytest.raises(ValueError):aggregate(data,rows,token)

def test_frozen_sources_are_checked_as_bytes(tmp_path):
    from tools.opaque_streamed_reference_contract import verify_files
    p=tmp_path/'proof.py';p.write_bytes(b'original');rows=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())]
    verify_files(rows);p.write_bytes(b'changed')
    with pytest.raises(ValueError):verify_files(rows)

def test_streamed_profile_keeps_limits_and_refuses_mixing(data,monkeypatch):
    from tools.opaque_reference_profile import profile,worker_profile
    import tools.opaque_full_material_reference as m
    p=profile('streamed22-offline-v1');assert (p.cells,p.depth,p.reference_seconds,p.worker_seconds,p.watchdog_seconds,p.host_bytes,p.threads)==(256,10,90,120,180,1073741824,1)
    assert worker_profile('streamed-piece',p.name)==p
    for mode,name in [('streamed-piece','standard'),('cpu',p.name),('extended-campaign-piece',p.name)]:
        with pytest.raises(ValueError):worker_profile(mode,name)
    for kwargs in (dict(streamed_piece=0),dict(reference_profile=p.name),dict(reference_profile=p.name,streamed_piece=True),dict(reference_profile=p.name,streamed_piece=22),dict(reference_profile=p.name,streamed_piece=0,selected_piece=0),dict(reference_profile=p.name,streamed_piece=0,shared_piece_budget=True)):
        with pytest.raises(ValueError):m.full_pixel(data,**kwargs)
    seen=[]
    def square(field,**kw):
        seen.append(kw);return dict(value=[I(0),I(0)],cells=1)
    monkeypatch.setattr(m,'integrate_square',square)
    result=m.full_pixel(data,reference_profile=p.name,streamed_piece=0,smooth_partition=(4,4),stable_view_base=True,aggregate_residual=True,tight_cosine_slopes=True)
    assert len(seen)==1 and seen[0]['max_cells']==256 and seen[0]['max_depth']==10
    assert len(result['pieces'])==1 and result['pieces'][0]['piece']==0 and result['streamed_piece']==0
    assert seen[0]['tolerance']==result['allocated_basis_radius']
