"""Fail-closed certificates for the explicitly approved original22-piece pixel.

Hashes bind identity, not truth: the independently qualified interval checker
produces enclosures; monitored worker outcomes and full aggregation are required.
"""
import hashlib,json,math
from pathlib import Path
from decimal import Decimal,localcontext,ROUND_FLOOR
from tools.opaque_shading_intervals import I

PROFILE='streamed22-offline-v1'
EXTENDED_PROFILE='streamed22-offline-extended-v1'
HOUR_PROFILE='streamed22-offline-hour-v1'
TWO_HOUR_PROFILE='streamed22-offline-two-hour-v1'
def limits_for(name):
    if name in ('remaining33-offline-two-hour-v1','remaining28-offline-two-hour-v1'):
        from tools.opaque_remaining_reference_contract import limits_for as remaining_limits
        return remaining_limits(name)
    from tools.opaque_reference_profile import worker_profile
    if name not in (PROFILE,EXTENDED_PROFILE,HOUR_PROFILE,TWO_HOUR_PROFILE):raise ValueError('explicit streamed profile required')
    return worker_profile({PROFILE:'streamed-piece',EXTENDED_PROFILE:'streamed-extended-piece',HOUR_PROFILE:'streamed-hour-piece',TWO_HOUR_PROFILE:'streamed-two-hour-piece'}[name],name)
INPUT_SHA='058e854ed69ba529c1e42f20b213cd9f9e693a043a71a24a25ad8897e7c9326c'
CONTENT_SHA='db864f3e29026cb0f6e451aa4b3a206efa249b172910fe09f7fd7aa5c741cba2'

def digest(data):
    return hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def validate_input(data):
    if digest(data)!=CONTENT_SHA:raise ValueError('input differs from the approved frozen22-piece pixel')

def allocation(data):
    validate_input(data)
    from tools.opaque_full_material_reference import piece_tolerances
    # Reserve ten decimal guard digits for outward summation of all22 shares.
    with localcontext() as ctx:
        ctx.prec=50;ctx.rounding=ROUND_FLOOR
        shares=[+v for v in piece_tolerances(data['selected']['visible_parts'])]
    if len(shares)!=22 or any(v<=0 for v in shares) or sum((I(v) for v in shares),I(0)).hi>I('5e-13').lo:raise ValueError('invalid conserved area allocation')
    return shares

def certificate_headers(data,manifest_sha,*,reference_profile=PROFILE):
    if reference_profile not in (PROFILE,EXTENDED_PROFILE,HOUR_PROFILE,TWO_HOUR_PROFILE):raise ValueError('historical22-piece profile required')
    limits_for(reference_profile)
    if type(manifest_sha)!=str or len(manifest_sha)!=64 or any(c not in '0123456789abcdef' for c in manifest_sha):raise ValueError('manifest SHA256 required')
    shares=allocation(data);allocation_sha=digest([str(v) for v in shares])
    return [dict(schema='gffx-streamed-piece-v1',profile=reference_profile,piece=index,manifest_sha256=manifest_sha,input_sha256=INPUT_SHA,piece_sha256=digest(part),allocation_sha256=allocation_sha,allocated_basis_radius=str(shares[index])) for index,part in enumerate(data['selected']['visible_parts'])]

def certificate_header(data,index,manifest_sha,*,reference_profile=PROFILE):
    if type(index)!=int or not 0<=index<22:raise ValueError('piece index0..21 required')
    return certificate_headers(data,manifest_sha,reference_profile=reference_profile)[index]

def verify_files(rows):
    if not isinstance(rows,list) or not rows:raise ValueError('nonempty frozen file list required')
    paths=set()
    for row in rows:
        path=Path(row['path']).resolve()
        if path in paths or hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('duplicate or changed frozen file: '+str(path))
        paths.add(path)

def bounded(value,maximum,integer=False):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<=value<=maximum or integer and type(value)!=int:raise ValueError('invalid or exceeded worker accounting')

def validate_certificate(row,header):
    values=[]
    limits=limits_for(header['profile'])
    if any(row.get(k)!=v for k,v in header.items()) or row.get('status')!='passed':raise ValueError('failed, reordered or mismatched certificate')
    bounded(row.get('cells'),limits.cells,True);bounded(row.get('reference_seconds'),limits.reference_seconds);bounded(row.get('seconds'),limits.worker_seconds)
    sup=row.get('supervisor',{})
    if type(sup.get('exit_code'))!=int or sup['exit_code']!=0 or sup.get('failure') is not None:raise ValueError('worker did not succeed under supervision')
    bounded(sup.get('elapsed_seconds'),limits.watchdog_seconds);bounded(sup.get('sampled_peak_working_set_bytes'),1073741824,True)
    basis=row.get('basis')
    if not isinstance(basis,list) or len(basis)!=2:raise ValueError('two complete material basis bounds required')
    for j,endpoints in enumerate(basis):
        if not isinstance(endpoints,list) or len(endpoints)!=2 or any(type(v)!=str for v in endpoints):raise ValueError('explicit decimal endpoints required')
        try:lo,hi=map(Decimal,endpoints)
        except Exception as e:raise ValueError('malformed endpoints') from e
        if not lo.is_finite() or not hi.is_finite() or lo>hi:raise ValueError('finite ordered enclosure required')
        value=I(lo,hi)
        if value.width>(I(header['allocated_basis_radius'])*2).lo:raise ValueError('piece exceeds its conserved radius')
        values.append(value)
    return values

def aggregate(data,reports,manifest_sha,*,reference_profile=PROFILE):
    headers=certificate_headers(data,manifest_sha,reference_profile=reference_profile)
    if not isinstance(reports,list) or len(reports)!=22:raise ValueError('all22 fresh certificates required')
    totals=[I(0),I(0)];cells=0;seconds=0.
    for index,row in enumerate(reports):
        values=validate_certificate(row,headers[index])
        totals=[a+b for a,b in zip(totals,values)]
        cells+=row['cells'];seconds+=row['reference_seconds']
    if any(q.width>I('1e-12').lo for q in totals):raise ValueError('combined basis radius exceeded')
    rgb=[totals[0]+I(a)*totals[1] for a in data['material']['albedo']]
    if any(q.width>I('2e-12').lo for q in rgb+[totals[1]]):raise ValueError('combined RGB/Jacobian radius exceeded')
    pair=lambda v:[str(v.lo),str(v.hi)]
    return dict(pieces=22,cells=cells,reference_seconds=seconds,basis=[pair(v) for v in totals],colour=[pair(v) for v in rgb],albedo_sum_gradient=[pair(totals[1])]*3,scope='One complete fixed packed-geometry pixel and constant-albedo first Jacobian; no frame or moving-geometry admission')
