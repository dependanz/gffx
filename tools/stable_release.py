"""Verify the Windows 0.2.0 artifact, concrete evidence and owner release sign-off."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from email.parser import BytesParser
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tomllib
import zipfile

PROFILES = {'cpu310', 'cpu311', 'cpu312', 'cpu313', 'cpu314', 'gpu314', 'cuda-torch-cpu314'}
REPOSITORY = 'dependanz/gffx'
OWNER = 'dependanz'
MODE = 'explicit_owner_release_signoff'
REQUIRED_SOURCE = {'pyproject.toml', 'README.md', 'SECURITY.md', 'tools/stable_release.py',
    'tools/stable_validation.py', 'tests/packaging/test_stable_release.py',
    '.github/workflows/publish-alpha.yml', '.github/workflows/windows-functional.yml',
    'docs/RELEASE_POLICY_0_2_0.md', 'docs/support-0.2.0.json'}


def timestamp(value):
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if result.tzinfo is None:
        raise ValueError('Evidence timestamps must include a timezone')
    return result.astimezone(timezone.utc)


def inventory_digest(files):
    ordered = sorted(files, key=lambda row: row['path'])
    return hashlib.sha256(json.dumps(ordered, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def profile_matrix(rows, artifact, current, baseline=None):
    if {row['profile'] for row in rows} != PROFILES or len(rows) != len(PROFILES):
        raise ValueError('The complete seven-profile exact-artifact matrix is required')
    for row in rows:
        if (row['wheel_sha256'] != artifact['sha256'] or row['exit_code'] != 0
                or row['failed'] or row['skipped'] or row['passed'] <= 0):
            raise ValueError('Failed/skipped/empty validation or different artifact')
        minor = row['profile'][-3:]
        if '.'.join(row['python'].split('.')[:2]) != '3.'+str(int(minor[1:])):
            raise ValueError('Validation Python does not match its advertised profile')
        expected_torch = '2.10.0+cpu' if row['profile'].startswith('cpu') else '2.10.0+cu126'
        if row['torch'] != expected_torch:
            raise ValueError('Validation Torch does not match its advertised profile')
        digest = row['selected_nodeids_sha256']
        if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Missing exact selected-node identity')
        if timestamp(row['completed_at']) > current:
            raise ValueError('Future validation timestamp')
        if baseline:
            old = next(value for value in baseline if value['profile'] == row['profile'])
            for key in ['python', 'torch', 'passed', 'selected_nodeids_sha256']:
                if row[key] != old[key]:
                    raise ValueError('Final validation selection/runtime changed: '+key)
            if current-timestamp(row['completed_at']) > timedelta(days=1):
                raise ValueError('Every final profile must be fresh within 24 hours')


def verify(root: Path, dist: Path, *, candidate=False, check=False, release_event=None,
           now=None, head_commit=None):
    manifest = json.loads((root/'releases/0.2.0.json').read_text(encoding='utf-8'))
    if manifest['version'] != '0.2.0' or manifest['tag'] != 'v0.2.0':
        raise ValueError('This release is restricted to gffx 0.2.0')
    if tomllib.loads((root/'pyproject.toml').read_text(encoding='utf-8'))['project']['version'] != '0.2.0':
        raise ValueError('Source version differs from the release')
    artifact = manifest['artifact']
    wheels = list(dist.iterdir())
    if len(wheels) != 1 or not wheels[0].is_file() or wheels[0].name != artifact['filename']:
        raise ValueError('Expected exactly the reviewed Windows wheel')
    data = wheels[0].read_bytes()
    if len(data) != artifact['bytes'] or hashlib.sha256(data).hexdigest() != artifact['sha256']:
        raise ValueError('Wheel bytes differ from the release manifest')
    with zipfile.ZipFile(wheels[0]) as archive:
        metadata = BytesParser().parsebytes(archive.read('gffx-0.2.0.dist-info/METADATA'))
        tags = BytesParser().parsebytes(archive.read('gffx-0.2.0.dist-info/WHEEL'))
        if 'gffx/gffx_cuda12.dll' not in archive.namelist():
            raise ValueError('The reviewed optional provider is missing')
    if metadata['Name'] != 'gffx' or metadata['Version'] != '0.2.0' or metadata['Requires-Python'] != '>=3.10':
        raise ValueError('Unexpected package/version/Python identity')
    if metadata.get_all('Requires-Dist', []) or tags.get_all('Tag') != ['cp310-abi3-win_amd64']:
        raise ValueError('Unexpected dependencies or wheel tag')
    files = manifest.get('source_files', [])
    paths = {row['path'] for row in files}
    if not files or not REQUIRED_SOURCE.issubset(paths) or len(paths) != len(files):
        raise ValueError('Release-critical source inventory is missing or duplicated')
    for row in files:
        path = (root/row['path']).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError('Invalid release source path')
        normalized = path.read_bytes().replace(b'\r\n', b'\n')
        if hashlib.sha256(normalized).hexdigest() != row['sha256']:
            raise ValueError('Release-critical content changed: '+row['path'])
    if candidate:
        return dict(mode='candidate_only_not_publication_ready', **artifact)
    current = now if now is not None else datetime.now(timezone.utc)
    admission = manifest['admission']
    if admission['mode'] != MODE or admission['owner'] != OWNER or admission['fixed_calendar_wait'] is not False:
        raise ValueError('The explicit 0.2.0 owner sign-off policy is required')
    if admission.get('open_blockers'):
        raise ValueError('Release blockers remain open')
    initial = manifest.get('initial_validation', [])
    profile_matrix(initial, artifact, current)
    final = admission['final_validation']
    if final['source_inventory_sha256'] != inventory_digest(files):
        raise ValueError('Final validation belongs to different release-critical source')
    completed = timestamp(final['completed_at'])
    if completed > current or current-completed > timedelta(days=1):
        raise ValueError('Final validation must be real and fresh within 24 hours')
    profile_matrix(final['profiles'], artifact, current, initial)
    if any(timestamp(row['completed_at']) > completed for row in final['profiles']):
        raise ValueError('Final summary predates a profile result')
    controls = admission['concrete_checks']
    for name in ['artifact_audit', 'native', 'installation']:
        control = controls[name]
        if control['status'] != 'passed' or control['wheel_sha256'] != artifact['sha256']:
            raise ValueError('Missing/different-artifact concrete check: '+name)
    if (controls['native']['passed'] != 12 or controls['native']['failed']
            or controls['installation']['cases'] != 4 or controls['installation']['commands'] != 30):
        raise ValueError('Incomplete native/installation evidence')
    if not controls['artifact_audit']['record_hashes_verified'] or not controls['artifact_audit']['strict_twine']:
        raise ValueError('Artifact integrity/metadata audit is incomplete')
    gate_tests = controls['release_controls']
    if (gate_tests['status'] != 'passed' or gate_tests['failed'] or gate_tests['skipped']
            or gate_tests['passed'] <= 0 or gate_tests['source_inventory_sha256'] != inventory_digest(files)):
        raise ValueError('Current release-control tests are incomplete')
    hosted = controls['hosted_cpu_ci']
    if (hosted['status'] != 'passed' or hosted['jobs_passed'] != 6
            or hosted['runtime_inputs_unchanged'] is not True):
        raise ValueError('Hosted functional CI evidence is incomplete')
    if check:
        return dict(mode='technical_checks_passed_awaiting_owner_release_signoff', **artifact)
    if release_event is None:
        raise ValueError('Owner publication sign-off is still required; use --check for technical readiness')
    event = release_event
    release = event['release']
    if event.get('action') != 'published' or event['repository']['full_name'] != REPOSITORY:
        raise ValueError('Sign-off must be the reviewed repository published-release event')
    if event['sender']['login'] != OWNER or release['author']['login'] != OWNER:
        raise ValueError('The reviewed owner account must publish the release')
    if release['tag_name'] != 'v0.2.0' or release['draft'] is not False or release['prerelease'] is not False:
        raise ValueError('Sign-off requires a normal published v0.2.0 release')
    if head_commit is None:
        head_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, encoding='utf-8').strip()
    if release['target_commitish'] != head_commit:
        raise ValueError('Owner release target differs from the checked-out reviewed commit')
    published = timestamp(release['published_at'])
    if published > current or published < completed:
        raise ValueError('Release sign-off must follow actual final checks')
    assets = release['assets']
    if (len(assets) != 1 or assets[0]['name'] != artifact['filename'] or assets[0]['size'] != artifact['bytes']
            or assets[0].get('digest') not in [None, 'sha256:'+artifact['sha256']]):
        raise ValueError('The signed-off release asset differs from the reviewed wheel')
    return dict(mode='stable_publication_gates_passed', signed_off_by=OWNER,
                source_commit=head_commit, release_id=release['id'], **artifact)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--dist', required=True, type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--candidate', action='store_true', help='Audit bytes/source only; never a publisher bypass')
    mode.add_argument('--check', action='store_true', help='Check concrete readiness; does not grant publication approval')
    mode.add_argument('--release-event', type=Path, help='Actual GitHub Actions published-release event, following user approval')
    args = parser.parse_args()
    if args.release_event and (os.environ.get('GITHUB_ACTIONS') != 'true'
            or os.environ.get('GITHUB_EVENT_NAME') != 'release' or os.environ.get('GITHUB_REPOSITORY') != REPOSITORY):
        raise ValueError('Publication sign-off is accepted only in this repository GitHub release workflow')
    event = json.loads(args.release_event.read_text(encoding='utf-8')) if args.release_event else None
    print(json.dumps(verify(args.root, args.dist, candidate=args.candidate, check=args.check, release_event=event), indent=2))


if __name__ == '__main__':
    main()
