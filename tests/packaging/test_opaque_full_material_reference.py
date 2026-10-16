"""Independent original material over all saved visible pieces."""
import json,hashlib,os
from pathlib import Path

def test_full_original_material_reference(record_property):
    from tools.opaque_full_material_reference import full_pixel
    from tools.opaque_shading_intervals import Unresolved
    p=Path(os.environ['GFFX_FULL_INPUT']);data=json.loads(p.read_bytes());dest=Path(os.environ['GFFX_FULL_REFERENCE']);assert not dest.exists();rows=[]
    def progress(row):
        rows.append(row);dest.with_suffix('.progress.json').write_text(json.dumps(rows,indent=2,default=str),encoding='utf-8')
    try:result=full_pixel(data,progress=progress)
    except Unresolved as e:
        dest.write_text(json.dumps(dict(status='not_admitted',input_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),details=e.details),indent=2,default=str),encoding='utf-8');raise
    result=dict(status='passed',input_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),**result);dest.write_text(json.dumps(result,indent=2,default=str),encoding='utf-8');record_property('full_original_reference',json.dumps({k:v for k,v in result.items() if k!='pieces'},default=str))


def test_original_residual_error_components(monkeypatch,record_property):
    from tools import opaque_full_material_reference as module
    from tools.opaque_shading_intervals import I
    from tools.opaque_peano import kernel_suprema
    rows=[]
    class Captured(Exception):pass
    def capture(field,**kwargs):
        correction=kwargs['correction_integrator']
        for box in [(I(0,1),I(0,1)),(I('.75','1'),I('.5','.75')),(I('.75','.875'),I('.5','.625'))]:
            kwargs['prepare'](0,0,box)
            rows.append(dict(box=box,axes=[correction.axis_bounds(box,axis) for axis in (0,1)],kernels=kernel_suprema()))
        raise Captured
    monkeypatch.setattr(module,'integrate_square',capture)
    data=json.loads(Path(os.environ['GFFX_FULL_INPUT']).read_bytes())
    import pytest
    with pytest.raises(Captured):module.full_pixel(data)
    result=dict(status='diagnostic_only',rows=rows)
    Path(os.environ['GFFX_FULL_REFERENCE']).write_text(json.dumps(result,indent=2,default=str),encoding='utf-8')
    record_property('error_components',json.dumps(result,default=str))


def test_original_angular_partition_diagnostic(record_property):
    from tools.opaque_partitioned_material_reference import material_partitions
    from tools.opaque_principled_reference import PreparedPrincipled
    from tools.opaque_shading_intervals import Unresolved
    data=json.loads(Path(os.environ['GFFX_FULL_INPUT']).read_bytes());curve=PreparedPrincipled(data['budget'],1.);rows=[]
    for i,part in enumerate(data['selected']['visible_parts']):
        try:
            r=material_partitions(data,part,curve);rows.append(dict(piece=i,face=data['faces'][part['face']]['face'],**r))
        except Unresolved as e:rows.append(dict(piece=i,unresolved=e.details))
    dest=Path(os.environ['GFFX_FULL_REFERENCE']);assert not dest.exists();dest.write_text(json.dumps(dict(status='diagnostic_only',rows=rows),default=str,indent=2),encoding='utf-8')
    record_property('angular_partition_diagnostic',json.dumps([dict(piece=r['piece'],error=str(r.get('error')),parts=len(r.get('pieces',[])),unresolved=r.get('unresolved')) for r in rows]))


def test_partitioned_original_material_reference(record_property):
    from tools.opaque_partitioned_material_reference import full_pixel_partitioned
    from tools.opaque_shading_intervals import Unresolved
    p=Path(os.environ['GFFX_FULL_INPUT']);data=json.loads(p.read_bytes());dest=Path(os.environ['GFFX_FULL_REFERENCE']);assert not dest.exists();rows=[]
    def progress(row):
        rows.append(row);dest.with_suffix('.progress.json').write_text(json.dumps(rows,indent=2,default=str),encoding='utf-8')
    try:result=full_pixel_partitioned(data,progress=progress)
    except Unresolved as e:
        dest.write_text(json.dumps(dict(status='not_admitted',input_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),details=e.details),indent=2,default=str),encoding='utf-8');raise
    result=dict(status='passed',input_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),**result);dest.write_text(json.dumps(result,indent=2,default=str),encoding='utf-8');record_property('full_original_reference',json.dumps({k:v for k,v in result.items() if k!='pieces'},default=str))


def test_partition_minimum_work_diagnostic(monkeypatch,record_property):
    """Count classifier cells and unavoidable subregion roots before expensive quadrature."""
    from tools import opaque_partitioned_material_reference as module
    from tools import opaque_square_reference as quadrature
    from tools.opaque_shading_intervals import I,Unresolved
    data=json.loads(Path(os.environ['GFFX_FULL_INPUT']).read_bytes());rows=[];roots=[]
    def root_only(field,**kwargs):
        roots.append(dict(tolerance=kwargs['tolerance'],max_depth=kwargs['max_depth']))
        return dict(value=[I(0),I(0)],cells=1)
    # No field is evaluated. The fake values have no numerical meaning and
    # cannot leave this diagnostic as an accepted image/reference result.
    monkeypatch.setattr(quadrature,'integrate_square',root_only)
    try:
        result=module.full_pixel_partitioned(data,progress=lambda row:rows.append({k:v for k,v in row.items() if k!='value'}))
        terminal={'reason':'classification completed; quadrature values deliberately absent','cells':result['cells']}
    except Unresolved as e:terminal=e.details
    dest=Path(os.environ['GFFX_FULL_REFERENCE']);assert not dest.exists()
    result=dict(status='diagnostic_only',completed_partitions=rows,quadrature_roots=len(roots),root_allocations=roots,terminal=terminal)
    dest.write_text(json.dumps(result,default=str,indent=2),encoding='utf-8')
    record_property('partition_minimum_work_diagnostic',json.dumps({k:v for k,v in result.items() if k!='root_allocations'},default=str))
