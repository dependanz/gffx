"""Finite cached rendering transport; fixtures are bytes, not rendered evidence."""
import hashlib
import json
from pathlib import Path
import pytest


def fixture(tmp_path):
    labels=('orbit-left','orbit-centre','orbit-right','normals','depth','coverage')
    root=tmp_path/'source';root.mkdir();(root/'demo.py').write_text('source identity',encoding='utf-8')
    wheel=tmp_path/'demo.whl';wheel.write_bytes(b'artifact identity')
    digest=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    images=tmp_path/'images';images.mkdir();entries=[]
    for number,label in enumerate(labels):
        data=bytes([number,20,30])*32*32
        sha=hashlib.sha256(data).hexdigest();raw=images/(sha+'.rgb');raw.write_bytes(data)
        entries.append(dict(label=label,shape=[32,32,3],raw_path=str(raw),raw_sha256=sha))
    row=dict(status='passed',mode='render',size=32,
             supervisor=dict(exit_code=0,failure=None),wheel=dict(sha256=digest(wheel)),
             sources={'demo.py':digest(root/'demo.py')},images=entries,logical_output_bytes=6*32*32*3)
    report=tmp_path/'render.json';report.write_text(json.dumps(row),encoding='utf-8')
    return report,wheel,digest(wheel),root,digest(report),row


def test_deferred_cached_verifies_finite_frames_without_torch(tmp_path):
    from tools.deferred_showcase_cached import load_cached
    args=fixture(tmp_path)
    row,images=load_cached(*args[:5])
    assert len(images)==6 and row['logical_output_bytes']==18432
    assert all(image.shape==(32,32,3) for image in images)


def test_deferred_cached_rejects_changed_pixels_before_publication(tmp_path):
    from tools.deferred_showcase_cached import load_cached
    args=fixture(tmp_path)
    Path(args[-1]['images'][0]['raw_path']).write_bytes(b'wrong')
    with pytest.raises(ValueError,match='image identity'):
        load_cached(*args[:5])


def test_deferred_cached_rejects_unpinned_report(tmp_path):
    from tools.deferred_showcase_cached import load_cached
    args=fixture(tmp_path)
    args[0].write_text('{}',encoding='utf-8')
    with pytest.raises(ValueError,match='report identity'):
        load_cached(*args[:5])


def test_deferred_cached_uses_only_allowed_artifact_provenance():
    from tools.deferred_showcase_cached import publication_provenance
    from gffx.dashboard._server import check_provenance
    expected='a'*64
    assert check_provenance(publication_provenance(expected))=={'artifact_sha256':expected}
