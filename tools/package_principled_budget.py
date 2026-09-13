"""Create-only mechanical packaging of the already measured table; no generation.

Raw experimental reports stay in build/. Their pinned digests bind this export
to the reviewed finite profile. No runtime import of this development tool.
"""
import argparse
import hashlib
import json
from pathlib import Path
from tools import material_loss_cost as cost
from tools import public_budget_probe as probe

REPORTS = {
    'guard': '76f78076db5024c7aee3e41215876a4b7700d3d745114b0f727f90c8dcad1714',
    'directional': '178c501f2e9053215acd35a33cdd069e8d8739a13dd321668da0f2e3a8c0f957',
    'mean': '8621ce530af1e067156e56020ffeb16ee67454be7145060cc225ab566e0cb6b4',
    'furnace': '2310c62ac3dc89dfe3aacf509125c6eed56637feeb587781aa36d0e277221b62',
}
PAYLOAD_SHA = 'eb9393c7afd6e1a4964019643ee827dff72696896bd05786e8a2ec908ca538e5'


def validate_reports(root):
    previous = None
    for mode, expected in REPORTS.items():
        with (root/(mode+'.json')).open('rb') as stream:
            blob = stream.read(2*1024**2+1)
        if hashlib.sha256(blob).hexdigest() != expected:
            raise ValueError('report SHA mismatch: '+mode)
        report = json.loads(blob)
        # Exact-byte binding above protects the reviewed observations. These
        # additional structural checks make the intended chain explicit.
        if mode != 'guard':
            probe.verify_predecessor(previous, mode)
            if report['prerequisite_sha256'] != REPORTS[previous['mode']]:
                raise ValueError('report predecessor digest mismatch')
            if report['payload_hashes']['fine'] != PAYLOAD_SHA:
                raise ValueError('report candidate payload mismatch')
        if (report['status'] != mode+' passed; not public admission'
                or report['passed'] is not True
                or len(report['rows']) != dict(guard=80, directional=80, mean=16, furnace=192)[mode]):
            raise ValueError('incomplete physical report')
        previous = report
    return REPORTS.copy()


def export(fixtures, reports, output):
    if output.exists():
        raise FileExistsError('refusing to overwrite packaged data')
    verified = validate_reports(reports)
    probe.verify_references(Path(__file__).parent)
    data = cost.load_json(fixtures/'fine.json', cost.HASHES['fine'])
    value = probe.envelope(data, 'fine')
    value['evidence_id'] = 'public-budget-v0.1'
    if value['payload_sha256'] != PAYLOAD_SHA:
        raise ValueError('candidate canonical payload mismatch')
    blob = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')+b'\n'
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as stream:
        stream.write(blob)
    return dict(payload_sha256=PAYLOAD_SHA, envelope_sha256=hashlib.sha256(blob).hexdigest(),
                bytes=len(blob), reports=verified, output=str(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--reports', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.fixtures, args.reports, args.output)))


if __name__ == '__main__':
    main()
