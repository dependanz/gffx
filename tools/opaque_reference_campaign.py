"""Offline fixed-geometry certificate composition; no renderer or external proof engine.

Trust the pinned independently tested numerical producer,Python Decimal and local
supervision. Hashes detect mismatched/modified evidence;they are not a proof of
an arbitrary producer's mathematics. Each campaign and result is create-only.
"""
from pathlib import Path
from decimal import Decimal
from fractions import Fraction
import hashlib,json,math,re
from tools.opaque_shading_intervals import I
from tools.opaque_full_material_reference import original_piece_tolerances

PROFILE='original13-offline-v1'
LIMITS=dict(jobs=13,cells_per_job=256,depth_per_job=10,reference_seconds_per_job=90,worker_seconds_per_job=120,watchdog_seconds=180,host_bytes=1073741824,threads=1,cells=3328,reference_seconds=1170,worker_seconds=1560)
MATERIAL=dict(roughness=1.,f0=[.04]*3,albedo=[.2,.4,.6],direction=[1.,1.,2.])

def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')
def digest(value):return hashlib.sha256(canonical(value)).hexdigest()
def _require(ok,message):
    if not ok:raise ValueError(message)
def _hash(value):return isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) is not None

def load_json(path,expected):
    _require(_hash(expected),'expected SHA256 required')
    with Path(path).open('rb') as stream:raw=stream.read(8*1024*1024+1)
    _require(len(raw)<=8*1024*1024,'file byte limit')
    _require(hashlib.sha256(raw).hexdigest()==expected,'file SHA256 mismatch')
    return json.loads(raw,parse_constant=lambda _:(_ for _ in ()).throw(ValueError('nonfinite JSON')))

def write_new(path,value):
    raw=canonical(value)
    _require(len(raw)<=8*1024*1024,'file byte limit')
    with Path(path).open('xb') as stream:stream.write(raw)
    return hashlib.sha256(raw).hexdigest()

def make_spec(raw,sources):
    _require(isinstance(raw,bytes) and len(raw)<=17201244,'bounded frozen input bytes required')
    data=json.loads(raw,parse_constant=lambda _:(_ for _ in ()).throw(ValueError('nonfinite JSON')))
    _require(data['material']==MATERIAL,'exact approved material required')
    parts=data['selected']['visible_parts'];_require(len(parts)==13,'exact thirteen-piece profile')
    _require(isinstance(sources,dict) and sources and all(isinstance(k,str) and _hash(v) for k,v in sources.items()),'pinned source hashes required')
    tolerances=original_piece_tolerances(parts)
    pieces=[dict(index=i,face=data['faces'][p['face']]['face'],sha256=digest(dict(part=p,face=data['faces'][p['face']],origin=data['selected']['origin'])),tolerance=str(tolerances[i])) for i,p in enumerate(parts)]
    return dict(profile=PROFILE,limits=dict(LIMITS),input_sha256=hashlib.sha256(raw).hexdigest(),material_sha256=digest(data['material']),table_sha256=digest(data['budget']),geometry_sha256=digest(dict(faces=data['faces'],selected=data['selected'])),sources=dict(sources),albedo=data['material']['albedo'],pieces=pieces)

def _spec(spec):
    _require(spec.get('profile')==PROFILE and spec.get('limits')==LIMITS,'campaign profile/limits mismatch')
    _require(all(_hash(spec.get(k)) for k in ('input_sha256','material_sha256','table_sha256','geometry_sha256')),'campaign identity hashes')
    _require(spec.get('albedo')==MATERIAL['albedo'],'campaign albedo mismatch')
    _require(isinstance(spec.get('sources'),dict) and spec['sources'] and all(_hash(v) for v in spec['sources'].values()),'campaign sources')
    parts=spec.get('pieces');_require(isinstance(parts,list) and len(parts)==13,'campaign piece count')
    total=Fraction(0)
    for i,p in enumerate(parts):
        _require(type(p.get('index')) is int and p['index']==i and _hash(p.get('sha256')),'campaign piece order/identity')
        t=I(p['tolerance']);_require(t.lo>=0 and t.width==0,'finite nonnegative tolerance');total+=Fraction(t.lo)
    _require(total<=Fraction(1,2*10**12),'aggregate basis radius exceeded')

def _interval(pair):
    _require(isinstance(pair,list) and len(pair)==2 and all(isinstance(x,str) for x in pair),'interval endpoint strings required')
    return I(*pair)

def _number(value,limit,label,*,integer=False):
    ok=type(value) is int if integer else type(value) in (int,float)
    _require(ok and math.isfinite(value) and 0<=value<=limit,label+' limit or invalid value')
    return value

def verify_piece(spec,index,receipt):
    _spec(spec);_require(type(index) is int and 0<=index<13,'piece index')
    result=receipt['result'];worker=receipt['worker'];p=spec['pieces'][index]
    _require(digest(result)==receipt['result_sha256'] and digest(worker)==receipt['worker_sha256'],'modified result/worker payload')
    _require(result.get('protocol')=='original-piece-v1' and result.get('status')=='passed','piece did not pass')
    _require(type(result.get('piece')) is int and result['piece']==index,'missing/duplicate/out-of-order piece')
    _require(result.get('spec_sha256')==digest(spec) and result.get('piece_sha256')==p['sha256'],'input/material/table/geometry identity mismatch')
    _require(result.get('sources')==spec['sources'],'numerical source mismatch')
    _require(result.get('tolerance')==p['tolerance'],'allocated tolerance mismatch')
    basis=result.get('basis');_require(isinstance(basis,list) and len(basis)==2,'two basis integrals required')
    intervals=list(map(_interval,basis));target=(2*I(p['tolerance'])).lo
    _require(all(q.width<=target for q in intervals),'piece enclosure too wide')
    cells=_number(result.get('cells'),256,'piece cells',integer=True)
    _number(result.get('max_depth'),10,'piece depth',integer=True)
    seconds=_number(result.get('reference_seconds'),90,'piece reference seconds')
    _require(worker.get('status')=='passed' and worker.get('mode')=='campaign-piece' and worker.get('pytest_exit_code')==0,'worker did not pass')
    _require(worker.get('profile')==dict(threads=1,host_bytes=1073741824,cooperative_seconds=120,watchdog_seconds=180),'worker profile changed')
    _require(worker.get('campaign')==dict(spec_sha256=digest(spec),piece=index,result_sha256=receipt['result_sha256']),'worker/result binding mismatch')
    supervisor=worker['supervisor'];_require(supervisor.get('failure') is None and supervisor.get('exit_code')==0,'supervisor failed')
    elapsed=_number(supervisor.get('elapsed_seconds'),120,'worker seconds')
    _number(supervisor.get('sampled_peak_working_set_bytes'),1073741824,'worker memory',integer=True)
    _require(elapsed>=seconds,'reference time exceeds supervised time')
    return dict(cells=cells,reference_seconds=seconds,worker_seconds=elapsed,basis=intervals)

def check_prefix(spec,receipts):
    _spec(spec);_require(isinstance(receipts,list) and len(receipts)<=13,'campaign job cap')
    rows=[verify_piece(spec,i,r) for i,r in enumerate(receipts)]
    result=dict(jobs=len(rows),cells=sum(r['cells'] for r in rows),reference_seconds=math.fsum(r['reference_seconds'] for r in rows),worker_seconds=math.fsum(r['worker_seconds'] for r in rows),complete=len(rows)==13)
    for key in ('cells','reference_seconds','worker_seconds'):_number(result[key],LIMITS[key],'aggregate '+key)
    return result

def remaining_allowance(spec,usage):
    _spec(spec)
    for key in ('jobs','cells','reference_seconds','worker_seconds'):_number(usage[key],LIMITS[key],'aggregate '+key,integer=key in ('jobs','cells'))
    _require(usage['jobs']<13,'campaign is complete; no additional job')
    left={key:LIMITS[key]-usage[key] for key in ('jobs','cells','reference_seconds','worker_seconds')}
    _require(left['cells']>=256 and left['reference_seconds']>=90 and left['worker_seconds']>=120,'cannot reserve the next complete job')
    return left

def compose(spec,receipts):
    usage=check_prefix(spec,receipts);_require(usage['complete'],'complete thirteen-piece certificate required')
    rows=[verify_piece(spec,i,r) for i,r in enumerate(receipts)]
    totals=[sum((row['basis'][k] for row in rows),I(0)) for k in range(2)]
    rgb=[totals[0]+I(a)*totals[1] for a in spec['albedo']]
    _require(all(q.width<=Decimal('2e-12') for q in rgb+[totals[1]]),'complete pixel radius exceeded')
    pair=lambda q:[str(q.lo),str(q.hi)]
    return dict(status='passed',profile=PROFILE,spec_sha256=digest(spec),piece_count=13,value=list(map(pair,rgb)),albedo_derivative=pair(totals[1]),basis=list(map(pair,totals)),cells=usage['cells'],reference_seconds=usage['reference_seconds'],worker_seconds=usage['worker_seconds'],radius='1e-12',scope='fixed_geometry_rgb_and_constant_albedo_only',receipt_sha256=[digest(r) for r in receipts])


def receipt_from_files(result_path,result_sha,worker_path,worker_sha):
    result=load_json(result_path,result_sha);worker=load_json(worker_path,worker_sha)
    return dict(result=result,worker=worker,result_sha256=digest(result),worker_sha256=digest(worker),files=dict(result=dict(path=str(Path(result_path).resolve()),sha256=result_sha),worker=dict(path=str(Path(worker_path).resolve()),sha256=worker_sha)))
