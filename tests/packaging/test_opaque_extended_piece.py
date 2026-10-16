"""One admitted original piece,under the approved offline campaign profile."""
import hashlib,json,os,time
from pathlib import Path
from tools.opaque_extended_campaign import load_json,make_spec,digest,write_new,_spec
from tools import opaque_full_material_reference as m
from tools.opaque_reference_profile import EXTENDED
from tools.opaque_decimal_convolution import decimal_convolution
from tools.opaque_shading_intervals import Unresolved,I

INPUT_SHA='a15bea40cb5cdead7994d828b57f5dcfac9f44fab0292a06e7d0a7b08719f73a'

def test_extended_original_piece_independent_certificate(monkeypatch,record_property):
    spec=load_json(os.environ['GFFX_CAMPAIGN_SPEC'],os.environ['GFFX_CAMPAIGN_SPEC_SHA']);_spec(spec)
    index=int(os.environ['GFFX_CAMPAIGN_PIECE']);assert 0<=index<13
    dest=Path(os.environ['GFFX_CAMPAIGN_RESULT']);assert not dest.exists()
    raw=Path(os.environ['GFFX_FULL_INPUT']).read_bytes();assert hashlib.sha256(raw).hexdigest()==INPUT_SHA==spec['input_sha256']
    def check_sources():
        assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in spec['sources'].items()),'campaign source changed'
    check_sources();assert make_spec(raw,spec['sources'])==spec
    row=dict(protocol='original-piece-extended-v1',reference_profile=EXTENDED.record(),spec_sha256=digest(spec),piece=index,piece_sha256=spec['pieces'][index]['sha256'],sources=spec['sources'],tolerance=spec['pieces'][index]['tolerance'])
    if index in (8,11,12):
        integrate=m.integrate_square
        def selected(field,**kw):
            if index==8:
                from tools.opaque_late_boundary import LateBoundary
                proxy=LateBoundary(kw['correction_integrator'],tolerance=kw['tolerance']);seed=(32,4)
            else:
                from tools.opaque_direction_precheck import DirectionBoundary
                from tools.opaque_tail_boundary import TailBoundary
                cls=DirectionBoundary if index==11 else TailBoundary
                proxy=cls(kw['correction_integrator'],tolerance=kw['tolerance'],piece=index);seed=(64,8)
            kw['correction_integrator']=proxy
            return integrate(field,**kw,seed_shape=seed)
        monkeypatch.setattr(m,'integrate_square',selected)
    started=time.monotonic()
    try:
        with decimal_convolution():
            result=m.full_pixel(json.loads(raw),selected_piece=index,reference_profile=EXTENDED.name,material_only_inputs=index in (8,11,12))
        assert len(result['pieces'])==1 and result['pieces'][0]['piece']==index
        check_sources();assert Path(os.environ['GFFX_FULL_INPUT']).read_bytes()==raw
        assert result['seconds']<=900 and result['cells']<=4096
        assert all(x.width<=(2*I(row['tolerance'])).lo for x in result['basis'])
        row.update(status='passed',basis=[[str(x.lo),str(x.hi)] for x in result['basis']],cells=result['cells'],max_depth=max(p.get('max_depth',0) for p in result['pieces']),reference_seconds=result['seconds'])
    except Exception as error:
        row.update(status='not_admitted',reference_seconds=time.monotonic()-started,error=type(error).__name__+': '+str(error),details=json.loads(json.dumps(getattr(error,'details',{}),default=str)))
        write_new(dest,row);raise
    h=write_new(dest,row);record_property('reference_cpu_seconds',result['cpu_seconds']);record_property('original_piece_certificate',json.dumps(dict(piece=index,result_sha256=h,cells=row['cells'],reference_seconds=row['reference_seconds'],scope='one fixed-geometry visible piece; no whole-pixel or moving-gradient claim')))
