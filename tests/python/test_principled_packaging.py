"""Development exporter checks; no physical calculations or new table generation."""
import pytest
from tools import package_principled_budget as packaging


def test_packaging_refuses_changed_reports_before_parsing(tmp_path):
    (tmp_path/'guard.json').write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='report SHA'):
        packaging.validate_reports(tmp_path)


def test_packaging_never_overwrites_existing_output(tmp_path):
    output = tmp_path/'existing.json'
    output.write_text('preserve', encoding='utf-8')
    with pytest.raises(FileExistsError):
        packaging.export(tmp_path/'absent-fixtures', tmp_path/'absent-reports', output)
    assert output.read_text(encoding='utf-8') == 'preserve'
