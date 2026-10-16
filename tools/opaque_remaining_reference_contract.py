"""Two exact original128 pixels; no generic enlargement of the historical contract."""
from decimal import localcontext,ROUND_FLOOR
from tools.opaque_shading_intervals import I
from tools.opaque_streamed_reference_contract import digest,verify_files,validate_certificate
PROFILE33='remaining33-offline-two-hour-v1'
PROFILE28='remaining28-offline-two-hour-v1'
PROFILES=(PROFILE33,PROFILE28)
_SCOPES={
 PROFILE33:dict(case=1,pieces=33,input_sha256='a148bf721ba478e1f52821bf8f4d48f9fcbd67e6aecf51390caea42076c22e86',content_sha256='6b6f96ab16913c69216277c43ce600c824f8be52b12a0c148844e91a742dab55'),
 PROFILE28:dict(case=2,pieces=28,input_sha256='de96eae0e545dc3633587049ce82e31540b81006e0b7e2a60b9fe7eab0e671af',content_sha256='4135fda3245db8d7a757b976e9f49b906d990c48d275975f8a91750fa087d78f')}
def scope(name):
    if type(name)!=str or name not in _SCOPES:raise ValueError('explicit remaining-pixel profile required')
    return dict(_SCOPES[name])
def limits_for(name):
    scope(name)
    from tools.opaque_reference_profile import worker_profile
    return worker_profile('remaining-piece',name)
def validate_input(data,*,reference_profile):
    s=scope(reference_profile)
    if digest(data)!=s['content_sha256']:raise ValueError('input differs from the frozen remaining pixel')
def allocation(data,*,reference_profile):
    validate_input(data,reference_profile=reference_profile)
    from tools.opaque_full_material_reference import piece_tolerances
    with localcontext() as ctx:
        ctx.prec=50;ctx.rounding=ROUND_FLOOR;shares=[+v for v in piece_tolerances(data['selected']['visible_parts'])]
    if len(shares)!=scope(reference_profile)['pieces'] or any(v<=0 for v in shares) or sum((I(v) for v in shares),I(0)).hi>I('5e-13').lo:raise ValueError('invalid conserved area allocation')
    return shares

def certificate_headers(data,manifest_sha,*,reference_profile):
    s=scope(reference_profile);limits_for(reference_profile)
    if type(manifest_sha)!=str or len(manifest_sha)!=64 or any(c not in '0123456789abcdef' for c in manifest_sha):raise ValueError('manifest SHA256 required')
    shares=allocation(data,reference_profile=reference_profile);allocation_sha=digest([str(v) for v in shares])
    return [dict(schema='gffx-remaining-piece-v1',profile=reference_profile,piece=i,manifest_sha256=manifest_sha,input_sha256=s['input_sha256'],piece_sha256=digest(part),allocation_sha256=allocation_sha,allocated_basis_radius=str(shares[i])) for i,part in enumerate(data['selected']['visible_parts'])]
def certificate_header(data,index,manifest_sha,*,reference_profile):
    if type(index)!=int or not 0<=index<scope(reference_profile)['pieces']:raise ValueError('remaining piece index required')
    return certificate_headers(data,manifest_sha,reference_profile=reference_profile)[index]
def aggregate(data,reports,manifest_sha,*,reference_profile):
    headers=certificate_headers(data,manifest_sha,reference_profile=reference_profile);n=len(headers)
    if not isinstance(reports,list) or len(reports)!=n:raise ValueError('all fresh remaining-pixel certificates required')
    totals=[I(0),I(0)];cells=0;seconds=0.
    for i,row in enumerate(reports):
        values=validate_certificate(row,headers[i]);totals=[a+b for a,b in zip(totals,values)];cells+=row['cells'];seconds+=row['reference_seconds']
    if any(q.width>I('1e-12').lo for q in totals):raise ValueError('combined basis radius exceeded')
    rgb=[totals[0]+I(a)*totals[1] for a in data['material']['albedo']]
    if any(q.width>I('2e-12').lo for q in rgb+[totals[1]]):raise ValueError('combined RGB/Jacobian radius exceeded')
    pair=lambda v:[str(v.lo),str(v.hi)]
    return dict(pieces=n,cells=cells,reference_seconds=seconds,basis=[pair(v) for v in totals],colour=[pair(v) for v in rgb],albedo_sum_gradient=[pair(totals[1])]*3,scope='One complete fixed packed-geometry pixel and constant-albedo first Jacobian; no frame or moving-geometry admission')
