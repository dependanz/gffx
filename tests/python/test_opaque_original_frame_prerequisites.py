"""Fail closed on changed or incomplete prerequisites before a frame launch."""
import json,hashlib
from pathlib import Path
import pytest

def fixture(tmp_path):
    from tools.opaque_original_frame_admission import SCENE_SHA
    checks=[dict(passed=True,error_upper="1e-12",allowance_lower="1e-10") for _ in range(6)]
    rows=[]
    for origin,count in [([59,10],22),([48,15],33),([67,117],28)]:
        p=tmp_path/(str(count)+'.json');p.write_text(json.dumps(dict(status='passed',pieces=count,checks=checks)))
        rows.append(dict(origin=origin,pieces=count,path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    source=tmp_path/'execution.py';source.write_text('qualified execution')
    r=dict(version=1,scene_sha256=SCENE_SHA,wheel_sha256='a'*64,pixels=rows,storage_sources=[dict(path=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest())])
    return r

def save(tmp_path,row):
    p=tmp_path/'gate.json';p.write_text(json.dumps(row));return p,hashlib.sha256(p.read_bytes()).hexdigest()

def load(tmp_path,row):
    from tools.opaque_original_frame_admission import load_prerequisites
    return load_prerequisites(*save(tmp_path,row),wheel_sha256='a'*64)

def test_all_exact_pixels_and_sources_are_required(tmp_path):
    r=fixture(tmp_path);assert load(tmp_path,r)['verified_pieces']==83
    r['pixels'].pop()
    with pytest.raises(ValueError):load(tmp_path,r)

def test_changed_pixel_bytes_refuse_even_with_green_summary(tmp_path):
    r=fixture(tmp_path);Path(r['pixels'][0]['path']).write_text('{}')
    with pytest.raises(ValueError):load(tmp_path,r)

def test_failed_or_inconsistent_comparison_refuses(tmp_path):
    for change in [dict(passed=False),dict(error_upper='NaN'),dict(error_upper='2e-10'),dict(allowance_lower='0')]:
        r=fixture(tmp_path);p=Path(r['pixels'][0]['path']);v=json.loads(p.read_bytes());v['checks'][0].update(change);p.write_text(json.dumps(v));r['pixels'][0]['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
        with pytest.raises(ValueError):load(tmp_path,r)

def test_changed_storage_source_refuses(tmp_path):
    r=fixture(tmp_path);Path(r['storage_sources'][0]['path']).write_text('changed')
    with pytest.raises(ValueError):load(tmp_path,r)

def test_changed_gate_or_wheel_refuses(tmp_path):
    from tools.opaque_original_frame_admission import load_prerequisites
    r=fixture(tmp_path);p,sha=save(tmp_path,r)
    with pytest.raises(ValueError):load_prerequisites(p,'f'*64,wheel_sha256='a'*64)
    with pytest.raises(ValueError):load_prerequisites(p,sha,wheel_sha256='b'*64)

def test_duplicate_pixel_scope_or_piece_count_refuses(tmp_path):
    for field,value in [('origin',[59,10]),('pieces',22)]:
        r=fixture(tmp_path);r['pixels'][1][field]=value
        with pytest.raises(ValueError):load(tmp_path,r)
