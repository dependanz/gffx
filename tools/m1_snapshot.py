"""Create an immutable, source-hashed integration build input."""
import argparse
import json
from pathlib import Path

from tools.verify_foundation import collect_source_paths, create_source_snapshot, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    if args.destination.exists() or args.manifest.exists():
        raise ValueError('preserve existing snapshot and manifest')
    paths = collect_source_paths(args.source)
    before = {str(p): sha256(args.source / p) for p in paths}
    copied = create_source_snapshot(args.source, args.destination)
    after = {str(p): sha256(args.source / p) for p in collect_source_paths(args.source)}
    saved = {str(p): sha256(args.destination / p) for p in copied}
    if before != after or before != saved:
        raise RuntimeError('source changed during snapshot; do not build this destination')
    with args.manifest.open('x', encoding='utf-8') as stream:
        json.dump({'status': 'verified', 'files': saved}, stream, indent=2)
    print(json.dumps({'status': 'verified', 'files': len(saved)}))


if __name__ == '__main__':
    main()
