"""Controller controls with no original-scene numerical workload."""
import json
from pathlib import Path
import pytest

def test_campaign_is_serial_single_use_and_stops_first_failure(tmp_path,monkeypatch):
    from tools import opaque_streamed_reference_campaign as m
    monkeypatch.setattr(m,'certificate_header',lambda *a:{})
    monkeypatch.setattr(m,'validate_certificate',lambda *a:None)
    monkeypatch.setattr(m,'load_manifest',lambda *a:({},{}))
    calls=[]
    def launch(fixtures,out,**kw):
        index=int(kw['worker_args'][-1]);calls.append(index)
        assert index==len(calls)-1
        out.write_text(json.dumps(dict(status='stopped' if index==2 else 'passed')))
        return int(index==2)
    out=tmp_path/'campaign'
    assert m.campaign(tmp_path/'manifest','a'*64,out,launch=launch)==1
    assert calls==[0,1,2] and not (out/'aggregate.json').exists()
    state=json.loads((out/'campaign.json').read_text());assert state['completed_pieces']==2 and not state['accuracy_admitted']
    with pytest.raises(FileExistsError):m.campaign(tmp_path/'manifest','a'*64,out,launch=launch)
    assert calls==[0,1,2]

def test_campaign_aggregates_only_after22_successes_and_rechecks_identity(tmp_path,monkeypatch):
    from tools import opaque_streamed_reference_campaign as m
    monkeypatch.setattr(m,'certificate_header',lambda *a:{})
    monkeypatch.setattr(m,'validate_certificate',lambda *a:None)
    checks=[];calls=[]
    monkeypatch.setattr(m,'load_manifest',lambda *a:checks.append(a) or ({},{}))
    def launch(fixtures,out,**kw):
        index=int(kw['worker_args'][-1]);calls.append(index)
        out.write_text(json.dumps(dict(status='passed',accuracy_admitted=index==22)))
        return 0
    out=tmp_path/'campaign'
    assert m.campaign(tmp_path/'manifest','a'*64,out,launch=launch)==0
    assert calls==list(range(23)) and len(checks)==47
    assert json.loads((out/'campaign.json').read_text())['accuracy_admitted']

def test_changed_manifest_stops_before_next_worker(tmp_path,monkeypatch):
    from tools import opaque_streamed_reference_campaign as m
    monkeypatch.setattr(m,'certificate_header',lambda *a:{})
    monkeypatch.setattr(m,'validate_certificate',lambda *a:None)
    checks=[];calls=[]
    def identity(*a):
        checks.append(a)
        if len(checks)==4:raise ValueError('changed candidate')
        return {},{}
    monkeypatch.setattr(m,'load_manifest',identity)
    def launch(fixtures,out,**kw):
        calls.append(kw);out.write_text(json.dumps(dict(status='passed')));return 0
    out=tmp_path/'campaign'
    assert m.campaign(tmp_path/'manifest','a'*64,out,launch=launch)==1 and len(calls)==1
    assert 'changed candidate' in json.loads((out/'campaign.json').read_text())['reason']

def test_preserved_device_comparison_checks_complete_values_and_derivatives():
    from tools.opaque_streamed_reference_campaign import compare_preserved
    cpu=dict(bytes={'colour':'00'},values={'colour':[1.,2.,3.],'albedo_sum_gradient':[4.,4.,4.]})
    cuda=dict(cases=[dict(case=0,bytes=cpu['bytes'])])
    ref=dict(colour=[['1','1'],['2','2'],['3','3']],albedo_sum_gradient=[['4','4']]*3)
    checks=compare_preserved(cpu,cuda,ref);assert len(checks)==6 and all(c['passed'] for c in checks)
    ref['albedo_sum_gradient']=[['5','5']]*3
    assert not all(c['passed'] for c in compare_preserved(cpu,cuda,ref))
    cuda['cases'][0]['bytes']={'colour':'ff'}
    with pytest.raises(ValueError):compare_preserved(cpu,cuda,ref)
