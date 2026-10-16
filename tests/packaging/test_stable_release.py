"""Deterministic negative controls for artifact, source and real-time release gates.

The synthetic dates here test the gate. They are not release soak evidence.
"""
import copy
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile
import pytest

spec = importlib.util.spec_from_file_location('stable_release', Path(__file__).resolve().parents[2]/'tools/stable_release.py')
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


@pytest.fixture
def packet(tmp_path):
    root, dist = tmp_path/'root', tmp_path/'dist'
    root.mkdir(); dist.mkdir()
    source = []
    for name in release.REQUIRED_SOURCE:
        path = root/name;path.parent.mkdir(parents=True, exist_ok=True)
        text = '[project]\nversion = "0.2.0"\n' if name == 'pyproject.toml' else 'fixed release content\n'
        path.write_text(text, encoding='utf-8')
        source.append(dict(path=name, sha256=hashlib.sha256(text.encode()).hexdigest()))
    wheel = dist/'gffx-0.2.0-cp310-abi3-win_amd64.whl'
    with zipfile.ZipFile(wheel, 'w') as archive:
        archive.writestr('gffx/gffx_cuda12.dll', b'test provider placeholder')
        archive.writestr('gffx-0.2.0.dist-info/METADATA', 'Name: gffx\nVersion: 0.2.0\nRequires-Python: >=3.10\n\n')
        archive.writestr('gffx-0.2.0.dist-info/WHEEL', 'Tag: cp310-abi3-win_amd64\n\n')
    data = wheel.read_bytes();sha = hashlib.sha256(data).hexdigest()
    now = datetime(2030, 1, 20, tzinfo=timezone.utc)
    profiles = [dict(profile=name, python='3.'+str(int(name[-2:]))+'.0',
                     torch='2.10.0+cpu' if name.startswith('cpu') else '2.10.0+cu126',
                     wheel_sha256=sha, passed=20, failed=0, skipped=0, exit_code=0,
                     selected_nodeids_sha256='a'*64) for name in sorted(release.PROFILES)]
    manifest = dict(version='0.2.0', tag='v0.2.0', artifact=dict(filename=wheel.name, bytes=len(data),sha256=sha),
                    source_files=source, initial_validation=profiles,
                    soak=dict(started_at=(now-timedelta(days=15)).isoformat(),last_blocking_change_at=(now-timedelta(days=15)).isoformat(),open_blockers=[],
                              midpoint_validation=dict(completed_at=(now-timedelta(days=7)).isoformat(),profiles=copy.deepcopy(profiles)),
                              final_validation=dict(completed_at=now.isoformat(),profiles=copy.deepcopy(profiles))))
    (root/'releases').mkdir()
    return root,dist,manifest,now


def check(packet):
    root,dist,manifest,now=packet
    (root/'releases/0.2.0.json').write_text(json.dumps(manifest),encoding='utf-8')
    return release.verify(root,dist,now=now)


def test_complete_synthetic_control_passes(packet):
    assert check(packet)['mode']=='stable_publication_gates_passed'


@pytest.mark.parametrize('damage', ['early','recent-change','missing-profile','failed','skip','wrong-wheel','missing-midpoint','pending-midpoint','future-final','changed-selection','open-blocker','source-drift','extra-artifact','stale-final','wrong-torch','wrong-python'])
def test_refuse_incomplete_or_changed_release(packet,damage):
    root,dist,m,now=packet
    if damage=='early':m['soak']['started_at']=(now-timedelta(days=1)).isoformat()
    elif damage=='recent-change':m['soak']['last_blocking_change_at']=(now-timedelta(days=1)).isoformat()
    elif damage=='missing-profile':m['initial_validation'].pop()
    elif damage=='failed':m['initial_validation'][0]['failed']=1
    elif damage=='skip':m['initial_validation'][0]['skipped']=1
    elif damage=='wrong-wheel':m['initial_validation'][0]['wheel_sha256']='0'*64
    elif damage=='missing-midpoint':m['soak'].pop('midpoint_validation')
    elif damage=='pending-midpoint':m['soak']['midpoint_validation']=None
    elif damage=='future-final':m['soak']['final_validation']['completed_at']=(now+timedelta(days=1)).isoformat()
    elif damage=='changed-selection':m['soak']['final_validation']['profiles'][0]['selected_nodeids_sha256']='b'*64
    elif damage=='open-blocker':m['soak']['open_blockers']=['unresolved defect']
    elif damage=='source-drift':(root/'README.md').write_text('changed',encoding='utf-8')
    elif damage=='extra-artifact':(dist/'extra.whl').write_bytes(b'extra')
    elif damage=='stale-final':m['soak']['final_validation']['completed_at']=(now-timedelta(days=2)).isoformat()
    elif damage=='wrong-torch':m['initial_validation'][0]['torch']='2.13.0'
    elif damage=='wrong-python':m['initial_validation'][0]['python']='3.9.0'
    with pytest.raises(ValueError):check(packet)
