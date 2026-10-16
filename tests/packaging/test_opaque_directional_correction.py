"""Independent integration and fail-closed directional proof selection."""
import json,os
from pathlib import Path
from fractions import Fraction as Q
import pytest
from tools.opaque_shading_intervals import I,Unresolved

@pytest.mark.parametrize('sx,sy',[(-1,-1),(-1,1),(1,-1),(1,1)])
def test_directional_correction_contains_closed_integral_and_restores_hooks(sx,sy):
    from tools.opaque_directional_correction import DirectionalMaterialCorrection
    from tests.packaging.test_opaque_localized_peano import material_fixture
    from tests.packaging.test_opaque_transverse_peano import _log_rectangle
    e,eps,slope=material_fixture();b=I('.001')*sx;c=I('.001')*sy
    a=I('.4988')+(I('.002') if sx<0 else I(0))+(I('.001') if sy<0 else I(0))
    def inputs(x,y):return (x+x*x)*b+y*c+a,x.lift('.75'),x.lift(1),1+2*x
    proof=DirectionalMaterialCorrection(e,inputs,curved=True,transverse=True,smooth_partition=True,segmented_conditioned=True)
    result=proof((I(0,1),I(0,1)))[1];expected=_log_rectangle(e,eps,slope,a,b,c,I(2))
    assert result.lo<=expected.lo<=expected.hi<=result.hi
    assert proof.inputs is inputs and proof.integrated_axis==1
    assert proof.last_method=='localized' and proof.last_integrated_axis in (0,1)

@pytest.mark.parametrize('simultaneous',[False,True])
@pytest.mark.parametrize('sign',[-1,1])
def test_directional_many_knots_keep_tiny_discontinuous_and_product_terms(simultaneous,sign):
    from types import SimpleNamespace
    from tools.opaque_full_material_reference import HorizonExtension
    from tools.opaque_directional_correction import DirectionalMaterialCorrection
    n=16;cn=[I(Q(1,4)+Q(3*i,4*n)) for i in range(n+1)];values=[I(Q((i*7)%13+1,10**30)) for i in range(n)]
    e=HorizonExtension(SimpleNamespace(cn=cn,x=[v.ln() for v in cn],controls=[[[v]*4]*2 for v in values],mean=[I(1)]*2,pi=I(1),horizon=[values[0]]*2,values=[[v]*2 for v in values]+[[values[-1]]*2]));e.indices=[0,0 if simultaneous else n-1];span=1-cn[0]
    def inputs(u,v):
        q=u.lift(cn[0])+(u if sign>0 else 1-u)*span
        return q,q if simultaneous else u.lift(1),u.lift(1),u.lift(1)
    proof=DirectionalMaterialCorrection(e,inputs,curved=True,transverse=True,envelope_fallback=True,aggregate_fallback=True)
    result=proof((I(0,1),I(0,1)))[1]
    expected=sum(((values[i]**2-values[0]**2)*(cn[i+1]**2-cn[i]**2)/(2*span) for i in range(n)),I(0)) if simultaneous else (sum(values,I(0))/n-values[0])*values[-1]
    assert result.lo<=expected.lo<=expected.hi<=result.hi
    assert expected.lo>0 and proof.last_method=='grouped'

@pytest.mark.parametrize('error',[ValueError('disjoint proof evidence'),Unresolved('cancel directional proof')])
def test_directional_invalid_evidence_and_cancellation_never_fall_back(monkeypatch,error):
    from tools.opaque_directional_correction import DirectionalMaterialCorrection
    from tools.opaque_centered_correction import CenteredMaterialCorrection
    from tests.packaging.test_opaque_localized_peano import material_fixture
    e,_,_=material_fixture();inputs=lambda x,y:(x*I('.001')+I('.4995'),x.lift('.75'),x.lift(1),x.lift(1))
    c=DirectionalMaterialCorrection(e,inputs,curved=True,transverse=True)
    monkeypatch.setattr(c,'integrated_bounds',lambda *a:(_ for _ in ()).throw(error))
    monkeypatch.setattr(CenteredMaterialCorrection,'__call__',lambda *a:pytest.fail('invalid fallback'))
    with pytest.raises(type(error),match=str(error)):c((I(0,1),I(0,1)))
    assert c.inputs is inputs and c.integrated_axis==1

def test_directional_domain_refusal_restores_fallback_and_tiny_bounds(monkeypatch):
    from tools.opaque_directional_correction import DirectionalMaterialCorrection
    from tools.opaque_centered_correction import CenteredMaterialCorrection
    from tests.packaging.test_opaque_localized_peano import material_fixture
    e,_,_=material_fixture();inputs=lambda x,y:(x*I('.001')+I('.4995'),x.lift('.75'),x.lift(1),x.lift(1))
    c=DirectionalMaterialCorrection(e,inputs,curved=True,transverse=True);calls=[]
    monkeypatch.setattr(c,'integrated_bounds',lambda *a:(_ for _ in ()).throw(ValueError('curved angular crossings need certified monotonicity')))
    def fallback(self,box):
        assert self.inputs is inputs and self.integrated_axis==1
        calls.append(1);return [I(0),I('-2e-60','3e-60')]
    monkeypatch.setattr(CenteredMaterialCorrection,'__call__',fallback)
    got=c((I(0,1),I(0,1)))[1]
    assert calls==[1] and got.lo==I('-2e-60').lo and got.hi==I('3e-60').hi

@pytest.mark.parametrize('enabled',[False,True])
def test_directional_mode_preserves_exact_piece_obligation(monkeypatch,enabled):
    from tools import opaque_full_material_reference as m
    from tools.opaque_directional_correction import DirectionalMaterialCorrection
    from tools.opaque_centered_correction import CenteredMaterialCorrection
    from tools.opaque_streamed_reference_contract import allocation
    data=json.loads(Path(os.environ['GFFX_STREAMED_INPUT']).read_bytes());seen=[]
    class Captured(Exception):pass
    def capture(field,**kw):seen.append(kw);raise Captured
    monkeypatch.setattr(m,'integrate_square',capture)
    with pytest.raises(Captured):m.full_pixel(data,reference_profile='streamed22-offline-hour-v1',streamed_piece=3,centered_correction=True,directional_correction=enabled)
    r=seen[0];assert type(r['correction_integrator']) is (DirectionalMaterialCorrection if enabled else CenteredMaterialCorrection)
    assert (r['max_cells'],r['max_depth'],r['tolerance'])==(4096,14,str(allocation(data)[3]))
    for kw in [dict(directional_correction=True),dict(directional_correction=1),dict(reference_profile='streamed22-offline-hour-v1',streamed_piece=3,directional_correction=True)]:
        with pytest.raises(ValueError,match='directional correction'):m.full_pixel({},**kw)

@pytest.mark.parametrize('fail',[False,True])
def test_directional_campaign_is_frozen_fresh_serial_and_stops_first_failure(tmp_path,monkeypatch,fail):
    from tools import opaque_streamed_reference_campaign as m
    frozen=dict(profile='streamed22-offline-hour-v1',checker='directional-correction-v1')
    assert m.checker_options(frozen)==dict(centered_correction=True,directional_correction=True)
    assert m.campaign_order(frozen)==[3,2,0,1,*range(4,22)]
    assert m.campaign_order(dict(profile=frozen['profile'],checker='centered-correction-v1'))==[2,0,1,*range(3,22)]
    monkeypatch.setattr(m,'load_manifest',lambda *a:(frozen,{}));monkeypatch.setattr(m,'certificate_header',lambda *a,**kw:{});monkeypatch.setattr(m,'validate_certificate',lambda *a:None)
    calls=[]
    def launch(fixtures,out,**kwargs):
        index=int(kwargs['worker_args'][-1]);calls.append(index);out.write_text(json.dumps(dict(status='stopped' if fail else 'passed',accuracy_admitted=index==22)));return int(fail)
    out=tmp_path/'fresh';assert m.campaign(tmp_path/'manifest','a'*64,out,launch=launch)==int(fail)
    assert calls==([3] if fail else [3,2,0,1,*range(4,23)])
    with pytest.raises(FileExistsError):m.campaign(tmp_path/'manifest','a'*64,out,launch=launch)
