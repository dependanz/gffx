"""The static operation inventory describes linked primitives, not GPU qualification."""


def test_support_status_separates_m1_local_evidence_from_release_support():
    from pathlib import Path
    text = (Path(__file__).resolve().parents[2] / 'docs' / 'SUPPORT_STATUS.md').read_text(encoding='utf-8')
    for required in ('Nineteen native families', 'native7', '51', '2.10.0+cu128',
                     'not requalified', 'Compositron', 'gffit', 'blur'):
        assert required in text
    assert 'no self-hosted runner registered' not in text
    assert 'operation-export behaviour are not\nmeasured at all' not in text


def test_native_operation_count_matches_public_primitive_inventory():
    from gffx import _core
    # Runtime capability transport includes the native static records. A provider
    # probe does not turn this implementation count into CUDA acceptance.
    report = _core.runtime_capabilities(include_sensitive=False)
    counts = [record['value'] for record in report['records'] if record['key'] == 12]
    assert counts == [19]
