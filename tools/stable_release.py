"""Validate exact stable-candidate bytes and enforce the real RC hold. Never uploads."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from email.parser import BytesParser
import hashlib
import json
from pathlib import Path
import sys
import tomllib
import zipfile

PROFILES = {'cpu310', 'cpu311', 'cpu312', 'cpu313', 'cpu314', 'gpu314', 'cuda-torch-cpu314'}
REQUIRED_SOURCE = {'pyproject.toml', 'README.md', 'tools/stable_release.py',
    'tools/stable_validation.py', '.github/workflows/publish-alpha.yml',
    '.github/workflows/windows-functional.yml', 'docs/RELEASE_POLICY_0_2_0.md'}


def timestamp(value):
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if result.tzinfo is None:
        raise ValueError('Evidence timestamps must include a timezone')
    return result.astimezone(timezone.utc)


def verify(root: Path, dist: Path, *, candidate: bool = False, now=None) -> dict:
    manifest = json.loads((root/'releases/0.2.0.json').read_text(encoding='utf-8'))
    if manifest['version'] != '0.2.0' or manifest['tag'] != 'v0.2.0':
        raise ValueError('This release is restricted to gffx0.2.0')
    if tomllib.loads((root/'pyproject.toml').read_text(encoding='utf-8'))['project']['version'] != '0.2.0':
        raise ValueError('Source version differs from the release')
    artifact = manifest['artifact']
    files = list(dist.iterdir())
    if len(files) != 1 or not files[0].is_file() or files[0].name != artifact['filename']:
        raise ValueError('Expected exactly the approved Windows wheel')
    wheel = files[0]
    data = wheel.read_bytes()
    if len(data) != artifact['bytes'] or hashlib.sha256(data).hexdigest() != artifact['sha256']:
        raise ValueError('Wheel bytes differ from the release manifest')
    with zipfile.ZipFile(wheel) as archive:
        metadata = BytesParser().parsebytes(archive.read('gffx-0.2.0.dist-info/METADATA'))
        tags = BytesParser().parsebytes(archive.read('gffx-0.2.0.dist-info/WHEEL'))
        if 'gffx/gffx_cuda12.dll' not in archive.namelist():
            raise ValueError('The approved optional provider is missing')
    if metadata['Name'] != 'gffx' or metadata['Version'] != '0.2.0' or metadata['Requires-Python'] != '>=3.10':
        raise ValueError('Unexpected package/version/Python identity')
    if metadata.get_all('Requires-Dist', []) or tags.get_all('Tag') != ['cp310-abi3-win_amd64']:
        raise ValueError('Unexpected dependencies or wheel tag')
    files = manifest.get('source_files', [])
    if not files or not REQUIRED_SOURCE.issubset({row['path'] for row in files}) or len({row['path'] for row in files}) != len(files):
        raise ValueError('Release-critical source inventory is missing')
    for row in files:
        path = (root/row['path']).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError('Invalid release source path')
        # Git normalizes tracked text line endings on checkout across Windows/Linux.
        normalized = path.read_bytes().replace(b'\r\n', b'\n')
        if hashlib.sha256(normalized).hexdigest() != row['sha256']:
            raise ValueError('Release-critical content changed: '+row['path'])
    if candidate:
        return dict(mode='candidate_only_not_publication_ready', **artifact)
    initial = manifest.get('initial_validation', [])
    if {row['profile'] for row in initial} != PROFILES or len(initial) != len(PROFILES):
        raise ValueError('The complete exact-artifact initial profile matrix is required')
    for row in initial:
        if row['wheel_sha256'] != artifact['sha256'] or row['exit_code'] != 0 or row['failed'] or row['skipped'] or row['passed'] <= 0:
            raise ValueError('Initial validation failed or belongs to a different artifact')
        minor = row['profile'][-3:]
        if '.'.join(row['python'].split('.')[:2]) != '3.'+str(int(minor[1:])):
            raise ValueError('Validation Python does not match the advertised profile')
        expected_torch = '2.10.0+cpu' if row['profile'].startswith('cpu') else '2.10.0+cu126'
        if row['torch'] != expected_torch:
            raise ValueError('Validation Torch does not match the advertised profile')
    soak = manifest['soak']
    if soak.get('open_blockers'):
        raise ValueError('Release blockers remain open')
    start = timestamp(soak['started_at'])
    changed = timestamp(soak['last_blocking_change_at'])
    current = now if now is not None else datetime.now(timezone.utc)
    if current < start+timedelta(days=14) or current < changed+timedelta(days=7):
        raise ValueError('The real14-day RC soak/seven-clean-day hold has not elapsed')
    for key, earliest in [('midpoint_validation', start+timedelta(days=7)), ('final_validation', start+timedelta(days=14))]:
        checkpoint = soak.get(key) or {}
        rows = checkpoint.get('profiles', [])
        if {row['profile'] for row in rows} != PROFILES or len(rows) != len(PROFILES):
            raise ValueError('Missing complete '+key)
        recorded = timestamp(checkpoint['completed_at'])
        if recorded < earliest or recorded > current:
            raise ValueError('Invalid/future checkpoint timestamp: '+key)
        if key == 'final_validation' and current-recorded > timedelta(days=1):
            raise ValueError('Final validation must be fresh within24hours of publication')
        for row in rows:
            if row['wheel_sha256'] != artifact['sha256'] or row['exit_code'] != 0 or row['failed'] or row['skipped'] or row['passed'] <= 0:
                raise ValueError('Checkpoint failed or artifact changed: '+key)
            baseline = next(value for value in initial if value['profile'] == row['profile'])
            if row['passed'] != baseline['passed'] or row['selected_nodeids_sha256'] != baseline['selected_nodeids_sha256'] or row['torch'] != baseline['torch']:
                raise ValueError('Checkpoint selection/runtime changed: '+key)
    return dict(mode='stable_publication_gates_passed', **artifact)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--dist', required=True, type=Path)
    parser.add_argument('--candidate', action='store_true', help='Audit candidate bytes only; never use in a publish job')
    args = parser.parse_args()
    print(json.dumps(verify(args.root, args.dist, candidate=args.candidate), indent=2))


if __name__ == '__main__':
    main()
