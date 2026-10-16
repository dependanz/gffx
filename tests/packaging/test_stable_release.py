"""Synthetic release-control tests; no fixture is actual owner publication approval."""
from datetime import datetime, timedelta, timezone
import copy
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
    for name in sorted(release.REQUIRED_SOURCE):
        path = root/name; path.parent.mkdir(parents=True, exist_ok=True)
        text = '[project]\nversion = "0.2.0"\n' if name == 'pyproject.toml' else 'fixed release content\n'
        path.write_text(text, encoding='utf-8')
        source.append(dict(path=name, sha256=hashlib.sha256(text.encode()).hexdigest()))
    wheel = dist/'gffx-0.2.0-cp310-abi3-win_amd64.whl'
    with zipfile.ZipFile(wheel, 'w') as archive:
        archive.writestr('gffx/gffx_cuda12.dll', b'synthetic provider')
        archive.writestr('gffx-0.2.0.dist-info/METADATA', 'Name: gffx\nVersion: 0.2.0\nRequires-Python: >=3.10\n\n')
        archive.writestr('gffx-0.2.0.dist-info/WHEEL', 'Tag: cp310-abi3-win_amd64\n\n')
    sha = hashlib.sha256(wheel.read_bytes()).hexdigest()
    now = datetime(2030, 1, 20, 12, tzinfo=timezone.utc)
    profiles = [dict(profile=name, python='3.'+str(int(name[-2:]))+'.0',
        torch='2.10.0+cpu' if name.startswith('cpu') else '2.10.0+cu126',
        wheel_sha256=sha, passed=20, failed=0, skipped=0, exit_code=0,
        selected_nodeids_sha256='a'*64, completed_at=(now-timedelta(hours=1)).isoformat())
        for name in sorted(release.PROFILES)]
    controls = dict(artifact_audit=dict(status='passed', wheel_sha256=sha, record_hashes_verified=True, strict_twine=True),
        native=dict(status='passed', wheel_sha256=sha, passed=12, failed=0),
        installation=dict(status='passed', wheel_sha256=sha, cases=4, commands=30),
        release_controls=dict(status='passed', passed=30, failed=0, skipped=0, source_inventory_sha256=release.inventory_digest(source)),
        hosted_cpu_ci=dict(status='passed', jobs_passed=6, runtime_inputs_unchanged=True))
    manifest = dict(version='0.2.0', tag='v0.2.0', artifact=dict(filename=wheel.name, bytes=wheel.stat().st_size, sha256=sha),
        source_files=source, initial_validation=copy.deepcopy(profiles),
        admission=dict(mode=release.MODE, owner='dependanz', fixed_calendar_wait=False, open_blockers=[],
            concrete_checks=controls, final_validation=dict(completed_at=(now-timedelta(minutes=30)).isoformat(),
                profiles=profiles, source_inventory_sha256=release.inventory_digest(source))))
    commit = 'c'*40
    event = dict(action='published', repository=dict(full_name='dependanz/gffx'), sender=dict(login='dependanz'),
        release=dict(id=123, author=dict(login='dependanz'), tag_name='v0.2.0', draft=False, prerelease=False,
            target_commitish=commit, published_at=(now-timedelta(minutes=10)).isoformat(),
            assets=[dict(name=wheel.name,size=wheel.stat().st_size,digest='sha256:'+sha)]))
    (root/'releases').mkdir()
    return root, dist, manifest, now, event, commit


def check(packet, *, technical=False, event=True):
    root, dist, manifest, now, payload, commit = packet
    (root/'releases/0.2.0.json').write_text(json.dumps(manifest), encoding='utf-8')
    return release.verify(root, dist, check=technical, release_event=payload if event else None, now=now, head_commit=commit)


def test_same_day_checks_and_owner_event_pass_without_calendar_wait(packet):
    assert check(packet)['mode'] == 'stable_publication_gates_passed'


def test_technical_readiness_does_not_record_publication_approval(packet):
    result = check(packet, technical=True, event=False)
    assert result['mode'] == 'technical_checks_passed_awaiting_owner_release_signoff'
    assert 'signed_off_by' not in result


def test_missing_owner_signoff_refuses_publication(packet):
    with pytest.raises(ValueError, match='sign-off is still required'):
        check(packet, event=False)


@pytest.mark.parametrize('damage', ['missing-profile','failed','skip','empty-tests','wrong-wheel','wrong-python','wrong-torch',
    'source-drift','duplicate-source','extra-artifact','missing-final','stale-final','future-final','stale-profile','future-profile',
    'summary-before-profile','changed-selection','changed-runtime','wrong-source-binding','open-blocker','wrong-policy',
    'native-failed','installation-incomplete','missing-audit','stale-controls','hosted-failed',
    'wrong-owner','wrong-author','wrong-repository','wrong-action','draft','prerelease','wrong-tag','wrong-commit',
    'future-signoff','signoff-before-checks','wrong-asset','wrong-asset-size','wrong-asset-digest','extra-asset'])
def test_refuse_changed_incomplete_or_unsigned_release(packet, damage):
    root, dist, m, now, event, commit = packet
    final=m['admission']['final_validation']; initial=m['initial_validation']; controls=m['admission']['concrete_checks']; signed=event['release']
    if damage=='missing-profile':initial.pop()
    elif damage=='failed':initial[0]['failed']=1
    elif damage=='skip':final['profiles'][0]['skipped']=1
    elif damage=='empty-tests':initial[0]['passed']=0
    elif damage=='wrong-wheel':initial[0]['wheel_sha256']='0'*64
    elif damage=='wrong-python':initial[0]['python']='3.9.0'
    elif damage=='wrong-torch':initial[0]['torch']='2.13.0'
    elif damage=='source-drift':(root/'README.md').write_text('changed',encoding='utf-8')
    elif damage=='duplicate-source':m['source_files'].append(m['source_files'][0])
    elif damage=='extra-artifact':(dist/'extra.whl').write_bytes(b'extra')
    elif damage=='missing-final':final['profiles'].pop()
    elif damage=='stale-final':final['completed_at']=(now-timedelta(days=2)).isoformat()
    elif damage=='future-final':final['completed_at']=(now+timedelta(days=1)).isoformat()
    elif damage=='stale-profile':final['profiles'][0]['completed_at']=(now-timedelta(days=2)).isoformat()
    elif damage=='future-profile':final['profiles'][0]['completed_at']=(now+timedelta(hours=1)).isoformat()
    elif damage=='summary-before-profile':final['completed_at']=(now-timedelta(hours=2)).isoformat()
    elif damage=='changed-selection':final['profiles'][0]['selected_nodeids_sha256']='b'*64
    elif damage=='changed-runtime':final['profiles'][0]['python']=initial[0]['python'].rsplit('.',1)[0]+'.1'
    elif damage=='wrong-source-binding':final['source_inventory_sha256']='b'*64
    elif damage=='open-blocker':m['admission']['open_blockers']=['unresolved defect']
    elif damage=='wrong-policy':m['admission']['mode']='calendar_soak'
    elif damage=='native-failed':controls['native']['failed']=1
    elif damage=='installation-incomplete':controls['installation']['cases']=3
    elif damage=='missing-audit':controls['artifact_audit']['record_hashes_verified']=False
    elif damage=='stale-controls':controls['release_controls']['source_inventory_sha256']='b'*64
    elif damage=='hosted-failed':controls['hosted_cpu_ci']['status']='failed'
    elif damage=='wrong-owner':event['sender']['login']='another-account'
    elif damage=='wrong-author':signed['author']['login']='another-account'
    elif damage=='wrong-repository':event['repository']['full_name']='another/gffx'
    elif damage=='wrong-action':event['action']='created'
    elif damage=='draft':signed['draft']=True
    elif damage=='prerelease':signed['prerelease']=True
    elif damage=='wrong-tag':signed['tag_name']='v0.2.1'
    elif damage=='wrong-commit':signed['target_commitish']='d'*40
    elif damage=='future-signoff':signed['published_at']=(now+timedelta(hours=1)).isoformat()
    elif damage=='signoff-before-checks':signed['published_at']=(now-timedelta(days=1)).isoformat()
    elif damage=='wrong-asset':signed['assets'][0]['name']='other.whl'
    elif damage=='wrong-asset-size':signed['assets'][0]['size']+=1
    elif damage=='wrong-asset-digest':signed['assets'][0]['digest']='sha256:'+'0'*64
    elif damage=='extra-asset':signed['assets'].append(dict(name='extra.whl',size=1))
    with pytest.raises(ValueError):check(packet)


def test_publisher_requires_actual_release_event_without_check_only_bypass():
    text=(Path(__file__).resolve().parents[2]/'.github/workflows/publish-alpha.yml').read_text()
    for name, following in [('stable-gates','stable-cpu'),('publish-stable',None)]:
        commands=text.split('  '+name+':\n',1)[1]
        if following:commands=commands.split('  '+following+':\n',1)[0]
        assert 'tools/stable_release.py --dist dist --release-event "$GITHUB_EVENT_PATH"' in commands
        assert '--candidate' not in commands and '--check' not in commands
