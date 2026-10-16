"""Two additional frozen pixels: conserved accuracy and bounded offline workers."""
import copy,json,os
from pathlib import Path
import pytest
from tools.opaque_shading_intervals import I,Unresolved
ROOT=Path(os.environ.get('GFFX_REMAINING_INPUT_ROOT','D:/devspace/gffx/qualification-20260913/m1-continuation-20260913/opaque169-20261002'))
PROFILES=['remaining33-offline-two-hour-v1','remaining28-offline-two-hour-v1']
def data(i):return json.loads((ROOT/f'pixel{i+1}-input2.json').read_bytes())

@pytest.mark.parametrize('i,n',[(0,33),(1,28)])
def test_remaining_profile_is_explicit_and_preserves_per_piece_limits(i,n):
    from tools.opaque_reference_profile import profile,worker_profile
    from tools.opaque_remaining_reference_contract import scope
    p=profile(PROFILES[i]);old=profile('streamed22-offline-two-hour-v1')
    assert {k:v for k,v in p.record().items() if k!='name'}=={k:v for k,v in old.record().items() if k!='name'}
    assert scope(p.name)['pieces']==n and worker_profile('remaining-piece',p.name)==p
    for mode,name in [('cpu',p.name),('remaining-piece',old.name),('streamed-two-hour-piece',p.name)]:
        with pytest.raises(ValueError):worker_profile(mode,name)

@pytest.mark.parametrize('i,n',[(0,33),(1,28)])
def test_remaining_allocation_and_complete_certificates_bind_each_input(i,n):
    from tools.opaque_remaining_reference_contract import certificate_headers,aggregate,validate_input,allocation
    d=data(i);name=PROFILES[i];headers=certificate_headers(d,'a'*64,reference_profile=name)
    assert len(headers)==n and all(I(h['allocated_basis_radius']).lo>0 for h in headers)
    assert sum((I(v) for v in allocation(d,reference_profile=name)),I(0)).hi<=I('5e-13').lo
    rows=[dict(h,status='passed',basis=[['0','0']]*2,cells=8192,reference_seconds=7200,seconds=7230,supervisor=dict(exit_code=0,failure=None,elapsed_seconds=7260,sampled_peak_working_set_bytes=1073741824)) for h in headers]
    result=aggregate(d,rows,'a'*64,reference_profile=name);assert result['pieces']==n and result['cells']==n*8192
    for bad in (rows[:-1],rows[::-1],[rows[0]]*n):
        with pytest.raises(ValueError):aggregate(d,bad,'a'*64,reference_profile=name)
    for key,value in [('cells',8193),('basis',[['0','1e-12']]*2),('manifest_sha256','b'*64),('profile',PROFILES[1-i])]:
        bad=copy.deepcopy(rows);bad[0][key]=value
        with pytest.raises(ValueError):aggregate(d,bad,'a'*64,reference_profile=name)
    changed=copy.deepcopy(d);changed['selected']['visible_parts'][0]['triangle'][0][0]='0.123'
    for bad in (changed,data(1-i)):
        with pytest.raises(ValueError):validate_input(bad,reference_profile=name)

@pytest.mark.parametrize('i,n',[(0,33),(1,28)])
def test_remaining_full_piece_selector_cap_and_deadline_are_enforced(i,n,monkeypatch):
    from tools import opaque_full_material_reference as m
    d=data(i);name=PROFILES[i]
    for index in (True,-1,n):
        with pytest.raises(ValueError):m.full_pixel(d,reference_profile=name,streamed_piece=index)
    with pytest.raises(ValueError):m.full_pixel(data(1-i),reference_profile=name,streamed_piece=0)
    if i==0:
        with pytest.raises(ValueError,match='cap32'):m.full_pixel(d)
    now=[0.];seen=[];monkeypatch.setattr(m.time,'monotonic',lambda:now[0])
    def expired(field,**kw):seen.append(kw);now[0]=7201;return dict(value=[I(0),I(0)],cells=1)
    monkeypatch.setattr(m,'integrate_square',expired)
    with pytest.raises(Unresolved,match='deadline'):m.full_pixel(d,reference_profile=name,streamed_piece=0)
    assert seen[0]['max_cells']==8192 and seen[0]['max_depth']==14

@pytest.mark.parametrize('i,n',[(0,33),(1,28)])
def test_remaining_controller_serial_complete_and_first_failure(i,n,tmp_path,monkeypatch):
    from tools import opaque_remaining_reference_campaign as m
    from tools import material_loss_cost as cost
    manifest=dict(profile=PROFILES[i],checker='horizon-correction-v1');d=data(i);calls=[];saved=dict(cost.PROFILE)
    monkeypatch.setattr(m,'load_manifest',lambda *a:(manifest,d));monkeypatch.setattr(m,'certificate_header',lambda *a,**kw:{});monkeypatch.setattr(m,'validate_certificate',lambda *a:None)
    def launch(fixtures,out,**kw):
        index=int(kw['worker_args'][-1]);calls.append(index);assert cost.PROFILE['cooperative_seconds']==(7230 if index<n else 120)
        out.write_text(json.dumps(dict(status='passed',accuracy_admitted=index==n)));return 0
    assert m.campaign(tmp_path/'m','a'*64,tmp_path/'ok',launch=launch)==0
    assert calls==[*m.campaign_order(manifest),n] and len(set(calls))==n+1 and cost.PROFILE==saved
    with pytest.raises(FileExistsError):m.campaign(tmp_path/'m','a'*64,tmp_path/'ok',launch=launch)
    calls.clear()
    def fail(fixtures,out,**kw):calls.append(int(kw['worker_args'][-1]));out.write_text(json.dumps(dict(status='stopped')));return 1
    assert m.campaign(tmp_path/'m','a'*64,tmp_path/'fail',launch=fail)==1 and len(calls)==1
    assert not (tmp_path/'fail/aggregate.json').exists() and cost.PROFILE==saved

@pytest.mark.parametrize('i,n',[(0,33),(1,28)])
def test_remaining_quadrature_keeps_independent_integral_and_refusals(i,n):
    from tools.opaque_square_reference import integrate_square
    from fractions import Fraction as Q
    name=PROFILES[i];r=integrate_square(lambda x,y:[x*y],reference_profile=name,max_cells=8192,max_depth=14)
    assert Q(r['value'][0].lo)<=Q(1,4)<=Q(r['value'][0].hi)
    for kw in [dict(max_cells=8193),dict(max_depth=15),dict(tolerance='1e-5')]:
        with pytest.raises(ValueError):integrate_square(lambda x,y:[x],reference_profile=name,**kw)


def test_remaining_pair_campaign_is_bounded_single_use_and_stops(tmp_path,monkeypatch):
    from tools import opaque_remaining_reference_campaign as m
    pair=dict(campaigns=[dict(path='case2',sha256='a'*64),dict(path='case1',sha256='b'*64)])
    monkeypatch.setattr(m,'load_pair',lambda *a:pair);calls=[]
    def campaign(path,token,out):
        calls.append(path);out.mkdir();(out/'campaign.json').write_text(json.dumps(dict(status='passed',accuracy_admitted=True)));return 0
    monkeypatch.setattr(m,'campaign',campaign)
    out=tmp_path/'all';assert m.pair_campaign(tmp_path/'pair','c'*64,out)==0 and calls==['case2','case1']
    with pytest.raises(FileExistsError):m.pair_campaign(tmp_path/'pair','c'*64,out)
    calls.clear()
    def fail(path,token,out):calls.append(path);out.mkdir();return 1
    monkeypatch.setattr(m,'campaign',fail)
    assert m.pair_campaign(tmp_path/'pair','c'*64,tmp_path/'fail')==1 and calls==['case2']


@pytest.mark.parametrize('i,n',[(0,33),(1,28)])
def test_remaining_manifest_rejects_changed_inputs_scope_and_limits(i,n,tmp_path):
    from tools import opaque_remaining_reference_campaign as m
    from tools.opaque_remaining_reference_contract import allocation,limits_for
    resources={key:tmp_path/key for key in ('input','wheel','cpu','cuda')}
    for key,p in resources.items():p.write_bytes((ROOT/f'pixel{i+1}-input2.json').read_bytes() if key=='input' else b'fixture')
    row=dict(schema='gffx-remaining-campaign-v1',profile=PROFILES[i],checker='horizon-correction-v1',pieces=list(range(n)),allocation=[str(v) for v in allocation(data(i),reference_profile=PROFILES[i])],limits=limits_for(PROFILES[i]).record(),files=[dict(path=str(p),sha256=m.sha(p)) for p in resources.values()],wheel_sha256=m.sha(resources['wheel']),**{k:str(p) for k,p in resources.items()})
    path=tmp_path/'manifest'
    def load(row):path.write_text(json.dumps(row));return m.load_manifest(path,m.sha(path))
    load(row)
    for key,value in [('pieces',list(range(n-1))),('profile',PROFILES[1-i]),('checker','unqualified'),('allocation',['1e-3']*n),('limits',{}),('files',row['files'][:-1])]:
        bad=copy.deepcopy(row);bad[key]=value
        with pytest.raises(ValueError):load(bad)
    load(row)
    with pytest.raises(ValueError):m.load_manifest(path,'f'*64)
    resources['input'].write_bytes(b'changed')
    with pytest.raises(ValueError):m.load_manifest(path,m.sha(path))


@pytest.mark.parametrize('case',[1,2])
def test_remaining_comparison_selects_correct_preserved_device_case(case):
    from tools.opaque_remaining_reference_campaign import compare_preserved
    cpu=dict(bytes={'colour':'yes'},values=dict(colour=[1.,2.,3.],albedo_sum_gradient=[4.,4.,4.]))
    cuda=dict(cases=[dict(case=j,bytes=cpu['bytes'] if j==case else {'colour':'wrong'}) for j in range(3)])
    ref=dict(colour=[['1','1'],['2','2'],['3','3']],albedo_sum_gradient=[['4','4']]*3)
    assert all(r['passed'] for r in compare_preserved(cpu,cuda,ref,case=case))
    with pytest.raises(ValueError):compare_preserved(cpu,cuda,ref,case=0)
    ref['colour'][0]=['9','9'];assert not all(r['passed'] for r in compare_preserved(cpu,cuda,ref,case=case))


def test_historical_contract_cannot_mint_remaining_profile_for_old_input():
    from tools.opaque_streamed_reference_contract import certificate_headers
    old=json.loads(Path(os.environ['GFFX_STREAMED_INPUT']).read_bytes())
    for name in PROFILES:
        with pytest.raises(ValueError):certificate_headers(old,'a'*64,reference_profile=name)


def test_mapped_candidate_is_explicit_and_runs_failed_piece_first():
    from tools import opaque_remaining_reference_campaign as m
    old=dict(profile=PROFILES[1],checker='horizon-correction-v1');new=dict(old,checker='light-map-weighted-v1')
    previous=dict(centered_correction=True,directional_correction=True,horizon_correction=True)
    assert m.checker_options(old)==previous
    assert m.checker_options(new)==dict(previous,light_map_power=4,weighted_light_residual=True)
    assert m.campaign_order(new)==[9,*[i for i in m.campaign_order(old) if i!=9]]
    assert sorted(m.campaign_order(new))==list(range(28))
    assert m.campaign_order(dict(profile=PROFILES[0],checker='light-map-weighted-v1'))==list(range(33))
    with pytest.raises(ValueError):m.checker_options(dict(new,checker='unqualified'))


def test_aligned_checker_is_explicit_and_prioritizes_unqualified_boundary_placements():
 from tools.opaque_remaining_reference_campaign import checker_options,campaign_order
 m=dict(profile='remaining28-offline-two-hour-v1',checker='light-map-aligned-v1')
 assert checker_options(m)==dict(centered_correction=True,directional_correction=True,horizon_correction=True,light_map_power=4,weighted_light_residual=True,align_light_boundary=True)
 order=campaign_order(m);assert order[:3]==[10,11,16] and sorted(order)==list(range(28))
 m['profile']='remaining33-offline-two-hour-v1';assert campaign_order(m)==list(range(33))
