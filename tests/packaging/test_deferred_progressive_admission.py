"""Forecast admission must reject invalid evidence before selecting slower anchors."""
import copy
import hashlib
import json
from pathlib import Path
import pytest


def row(size=128, seconds=1.75):
    return dict(status='passed',render_executed=True,measured_seconds=seconds,
        driver_sha256='driver',wheel={'sha256':'wheel'},scene_sha256='scene',
        source_sha256={'execution.py':'source'},
        profile=dict(width=size,height=size,material='principled',soft=0.,backward=True,
            dtype='float64',device='cuda',vertices=5002,faces=10000,light_samples=1,
            outputs=['colour','mask'],fitted_parameters=['constant_RGB_albedo']),
        estimate={'principled_point_block_limit':4,'principled_group_point_limit':128},
        supervisor={'exit_code':0,'failure':None})


def anchor(size, seconds):
    return dict(path=str(size),sha256=str(size),report=row(size,seconds))


def current():
    r=row(512);r['status']='working';r.pop('supervisor');r.pop('render_executed');return r


def test_selects_two_largest_sizes_not_two_fastest_rates():
    from tools.deferred_progressive_admission import forecast
    rows=[anchor(32,10),anchor(64,.3),anchor(256,4.2),anchor(128,1.75)]
    r=forecast(rows,current())
    assert r['seconds']==86.0
    assert [x['path'] for x in r['selected']]==['128','256']
    assert forecast(list(reversed(rows)),current())==r


@pytest.mark.parametrize('where,key,value',[
    ('root','status','not_admitted'),('root','render_executed',False),
    ('root','measured_seconds',0),('root','measured_seconds',float('nan')),
    ('root','measured_seconds',float('inf')),('root','measured_seconds',True),
    ('root','driver_sha256','old'),('root','scene_sha256','other'),
    ('root','source_sha256',{'execution.py':'changed'}),
    ('root','ownership_profile','caller_cpu_outputs_native_global_windows_v1'),
    ('wheel','sha256','other'),('profile','device','cpu'),('profile','dtype','float32'),
    ('profile','soft',.2),('profile','backward',False),('profile','material','classic'),
    ('profile','outputs',['colour']),('profile','fitted_parameters',['roughness']),
    ('profile','faces',9999),('profile','light_samples',2),
    ('estimate','principled_point_block_limit',8),
    ('estimate','principled_group_point_limit',64),
    ('supervisor','exit_code',1),('supervisor','failure','watchdog'),
])
def test_rejects_bad_anchor_even_when_it_would_not_be_selected(where,key,value):
    from tools.deferred_progressive_admission import forecast
    bad=anchor(32,.1);target=bad['report'] if where=='root' else bad['report'][where]
    target[key]=value
    with pytest.raises(ValueError):forecast([bad,anchor(128,1.75),anchor(256,4.2)],current())


@pytest.mark.parametrize('sizes',[(128,), (128,128), (128,512), (128,1024)])
def test_rejects_missing_duplicate_or_nonprior_sizes(sizes):
    from tools.deferred_progressive_admission import forecast
    with pytest.raises(ValueError):forecast([anchor(s,1) for s in sizes],current())


def test_initial_pair_keeps_original_rule():
    from tools.deferred_progressive_admission import forecast
    c=row(128);r=forecast([anchor(32,.25),anchor(64,.5)],c)
    assert r['seconds']==38


def test_report_and_array_hashes_are_both_required(tmp_path):
    from tools.deferred_progressive_admission import load_anchor
    r=row(32);r['arrays']={}
    for name,channels in [('colour',3),('mask',1)]:
        p=tmp_path/(name+'.f64');blob=bytes(32*32*channels*8);p.write_bytes(blob)
        r['arrays'][name]=dict(path=str(p),bytes=len(blob),sha256=hashlib.sha256(blob).hexdigest(),dtype='<f8',shape=[1,32,32,3] if name=='colour' else [1,32,32])
    p=tmp_path/'report.json';p.write_text(json.dumps(r),encoding='utf-8');sha=hashlib.sha256(p.read_bytes()).hexdigest()
    assert load_anchor(p,sha)['report']==r
    with pytest.raises(ValueError):load_anchor(p,'0'*64)
    (tmp_path/'colour.f64').write_bytes(b'changed')
    with pytest.raises(ValueError):load_anchor(p,sha)


def test_preserves_stream_ownership_identity():
    from tools.deferred_progressive_admission import forecast
    rows=[anchor(32,.25),anchor(64,.5)];c=row(128)
    for r in [c,*[a['report'] for a in rows]]:r['ownership_profile']='caller_cpu_outputs_native_global_windows_v1'
    assert forecast(rows,c)['seconds']==38


def test_parent_preserves_child_admission_receipt_and_restores_hooks(monkeypatch,tmp_path):
    import sys
    from tools import deferred_flame_probe as driver
    from tools import deferred_progressive_probe as wrapper
    saved=[];original_forecast=driver.forecast
    def write(path,row):saved.append(copy.deepcopy(row))
    def supervise(fixtures,output,**kwargs):
        assert kwargs['worker_module']=='tools.deferred_progressive_probe'
        assert kwargs['worker_args']==['--execution','resident']
        driver.cost.write_report(output,{'admission_policy':{'selected':['child evidence']}})
        return 0
    monkeypatch.setattr(driver.cost,'write_report',write)
    monkeypatch.setattr(driver.cost,'supervise',supervise)
    monkeypatch.setattr(driver,'main',lambda:driver.cost.supervise(tmp_path,tmp_path/'out',worker_module='tools.deferred_flame_probe',worker_args=[]))
    monkeypatch.setattr(sys,'argv',['probe','--execution','resident'])
    assert wrapper.main()==0
    assert saved[0]['admission_policy']['selected']==['child evidence']
    assert driver.forecast is original_forecast
    assert driver.cost.supervise is supervise
    assert driver.cost.write_report is write
    assert sys.argv==['probe','--execution','resident']

@pytest.mark.parametrize('problem',['shape','dtype','bytes'])
def test_anchor_arrays_must_match_declared_profile(tmp_path,problem):
 from tools.deferred_progressive_admission import load_anchor
 r=row(32);r['arrays']={}
 for name,channels in [('colour',3),('mask',1)]:
  p=tmp_path/(name+'.f64');blob=bytes(32*32*channels*8)
  if name=='colour' and problem=='bytes':blob+=bytes(8)
  p.write_bytes(blob)
  r['arrays'][name]=dict(path=str(p),bytes=len(blob),sha256=hashlib.sha256(blob).hexdigest(),dtype='<f8',shape=[1,32,32,3] if name=='colour' else [1,32,32])
 if problem=='shape':r['arrays']['colour']['shape']=[1,16,64,3]
 if problem=='dtype':r['arrays']['colour']['dtype']='<f4'
 p=tmp_path/'report.json';p.write_text(json.dumps(r));sha=hashlib.sha256(p.read_bytes()).hexdigest()
 with pytest.raises(ValueError):load_anchor(p,sha)
