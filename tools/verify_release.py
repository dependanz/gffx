"""Verify the exact tested wheel before a release upload. Never publishes."""
from __future__ import annotations

import argparse
from email.parser import BytesParser
import hashlib
import json
from pathlib import Path
import tomllib
import zipfile


def verify(root: Path, dist: Path, tag: str) -> dict:
    if tag != 'v0.2.0a1':
        raise ValueError('This publisher is restricted to v0.2.0a1')
    manifest = json.loads((root/'releases/0.2.0a1.json').read_text(encoding='utf-8'))
    project = tomllib.loads((root/'pyproject.toml').read_text(encoding='utf-8'))['project']
    if manifest['version'] != project['version'] or tag != manifest['tag'] or tag != 'v'+project['version']:
        raise ValueError('Release tag, source version and artifact manifest must agree')
    artifact = manifest['artifact']
    files = list(dist.iterdir())
    if len(files) != 1 or not files[0].is_file() or files[0].name != artifact['filename']:
        raise ValueError('Upload directory must contain exactly the approved wheel')
    wheel = files[0]
    data = wheel.read_bytes()
    if len(data) != artifact['bytes'] or hashlib.sha256(data).hexdigest() != artifact['sha256']:
        raise ValueError('Artifact bytes differ from the validated wheel')
    with zipfile.ZipFile(wheel) as archive:
        metadata = BytesParser().parsebytes(archive.read('gffx-0.2.0a1.dist-info/METADATA'))
        wheel_info = BytesParser().parsebytes(archive.read('gffx-0.2.0a1.dist-info/WHEEL'))
    if metadata['Name'] != 'gffx' or metadata['Version'] != manifest['version']:
        raise ValueError('Wheel distribution identity differs')
    if metadata['Requires-Python'] != '>=3.10' or metadata.get_all('Requires-Dist', []):
        raise ValueError('Unexpected Python requirement or mandatory dependency')
    if wheel_info.get_all('Tag') != ['cp310-abi3-win_amd64']:
        raise ValueError('Unexpected runtime/platform tag')
    return dict(tag=tag, **artifact)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--dist', type=Path, required=True)
    parser.add_argument('--tag', required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.root, args.dist, args.tag), indent=2))


if __name__ == '__main__':
    main()
