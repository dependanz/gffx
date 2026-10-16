"""Frozen, bounded opt-in ninth-piece experiment; no production selection."""
def test_original_ninth_piece_late_boundary_preflight(monkeypatch,record_property):
    import os,json,hashlib
    from pathlib import Path
    from tools import opaque_full_material_reference as m
    from tools.opaque_shading_intervals import Unresolved
    source=Path(os.environ['GFFX_FULL_INPUT']);dest=Path(os.environ['GFFX_FULL_REFERENCE']);assert not dest.exists()
    manifest=Path(os.environ['GFFX_EXPERIMENT_MANIFEST']);raw=manifest.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==os.environ['GFFX_EXPERIMENT_MANIFEST_SHA']
    frozen=json.loads(raw);root=Path('C:/dev/gffx')
    assert all(hashlib.sha256((root/r['path']).read_bytes()).hexdigest()==r['sha256'] for r in frozen['source'])
    assert hashlib.sha256(source.read_bytes()).hexdigest()=='a15bea40cb5cdead7994d828b57f5dcfac9f44fab0292a06e7d0a7b08719f73a'
    from tools.opaque_reference_profile import EXTENDED
    assert frozen['reference_profile']==EXTENDED.record()
    integrate=m.integrate_square;stats={}
    def selected(field,**kw):
        correction=kw['correction_integrator']
        assert correction.curved and correction.envelope_fallback and correction.aggregate_fallback
        from tools.opaque_late_boundary import LateBoundary
        proxy=LateBoundary(correction,tolerance=kw['tolerance']);stats['proxy']=proxy
        kw['correction_integrator']=proxy
        last=[-1]
        def progress(row):
            if row['cells']//64!=last[0]:
                last[0]=row['cells']//64
                live=dest.with_suffix('.progress.json');live.write_text(json.dumps(dict(status='diagnostic_progress_only',**row)),encoding='utf8')
        return integrate(field,**kw,seed_shape=(32,4),progress=progress)
    monkeypatch.setattr(m,'integrate_square',selected)
    try:
        from tools.opaque_decimal_convolution import decimal_convolution
        with decimal_convolution() as arithmetic:
            result=m.full_pixel(json.loads(source.read_bytes()),selected_piece=8,material_only_inputs=True,reference_profile=EXTENDED.name)
        row=dict(status='passed',result=result,scope='Bounded experimental selected-piece preflight only; not a campaign certificate or whole-pixel admission.')
    except Unresolved as e:
        row=dict(status='not_admitted',details=e.details);raise
    finally:
        row['boundary_events']=stats['proxy'].events if 'proxy' in stats else []
        row.update(reference_profile=EXTENDED.record(),arithmetic=arithmetic,source_manifest_sha256=hashlib.sha256(raw).hexdigest(),quadrature='Gauss5',work_charge=1,seed_shape=[32,4])
        dest.write_text(json.dumps(row,default=str,indent=2),encoding='utf8');record_property('ninth_preflight',json.dumps(row,default=str))
