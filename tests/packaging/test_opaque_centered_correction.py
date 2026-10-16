"""Independent integrals and fail-closed adoption of centered correction bounds."""
from fractions import Fraction as Q
import json,os
from pathlib import Path
import pytest
from tools.opaque_shading_intervals import I,Unresolved

@pytest.mark.parametrize('sx,sy',[(-1,-1),(-1,1),(1,-1),(1,1)])
def test_centered_correction_contains_independent_closed_log_integral(sx,sy):
    from tools.opaque_centered_correction import CenteredMaterialCorrection
    from tests.packaging.test_opaque_localized_peano import material_fixture
    from tests.packaging.test_opaque_transverse_peano import _log_rectangle
    e,eps,slope=material_fixture();b=I('.001')*sx;c=I('.001')*sy
    a=I('.4988')+(I('.002') if sx<0 else I(0))+(I('.001') if sy<0 else I(0))
    def inputs(x,y):return (x+x*x)*b+y*c+a,x.lift('.75'),x.lift(1),1+2*x
    proof=CenteredMaterialCorrection(e,inputs,curved=True,transverse=True,smooth_partition=True,segmented_conditioned=True)
    result=proof((I(0,1),I(0,1)))[1];expected=_log_rectangle(e,eps,slope,a,b,c,I(2))
    assert result.lo<=expected.lo<=expected.hi<=result.hi
    assert proof.inputs is inputs and proof.integrated_axis==1

@pytest.mark.parametrize('reason',['bounded angular segment count','bounded angular crossing count','curved angular crossings need certified monotonicity'])
def test_centered_correction_falls_back_only_after_restoring_original_inputs(monkeypatch,reason):
    from tools.opaque_centered_correction import CenteredMaterialCorrection
    from tools.opaque_peano_material import MaterialCorrection
    from tools.opaque_centered_gradient import CenteredInputs
    from tests.packaging.test_opaque_localized_peano import material_fixture
    e,_,_=material_fixture();inputs=lambda x,y:(x,y,None,x.lift(1))
    c=CenteredMaterialCorrection(e,inputs,curved=True,transverse=True,adaptive_order=True)
    calls=[]
    def unavailable(box):
        assert isinstance(c.inputs,CenteredInputs) and c.integrated_axis==0 and c.adaptive_order is False
        raise ValueError(reason)
    def fallback(self,box):
        assert self.inputs is inputs and self.integrated_axis==1 and self.adaptive_order is True
        calls.append(1);return [I(0),I('-2e-30','3e-30')]
    monkeypatch.setattr(c,'_certified_integral',unavailable);monkeypatch.setattr(MaterialCorrection,'__call__',fallback)
    got=c((I(0,1),I(0,1)))[1]
    assert calls==[1] and got.lo==I('-2e-30').lo and got.hi==I('3e-30').hi

@pytest.mark.parametrize('error',[ValueError('disjoint geometric derivative inclusions'),Unresolved('cancel centered correction')])
def test_centered_correction_propagates_invalid_proof_and_cancellation(monkeypatch,error):
    from tools.opaque_centered_correction import CenteredMaterialCorrection
    from tools.opaque_peano_material import MaterialCorrection
    from tests.packaging.test_opaque_localized_peano import material_fixture
    e,_,_=material_fixture();inputs=lambda x,y:(x,y,None,x.lift(1))
    c=CenteredMaterialCorrection(e,inputs,curved=True,transverse=True)
    def fail(box):raise error
    def forbidden(*a):raise AssertionError('invalid evidence must not become a fallback')
    monkeypatch.setattr(c,'_certified_integral',fail);monkeypatch.setattr(MaterialCorrection,'__call__',forbidden)
    with pytest.raises(type(error),match=str(error)):c((I(0,1),I(0,1)))
    assert c.inputs is inputs and c.integrated_axis==1

@pytest.mark.parametrize('enabled',[False,True])
def test_centered_correction_is_explicit_and_preserves_streamed_obligation(monkeypatch,enabled):
    from tools import opaque_full_material_reference as m
    from tools.opaque_centered_correction import CenteredMaterialCorrection
    from tools.opaque_peano_material import MaterialCorrection
    from tools.opaque_streamed_reference_contract import allocation
    data=json.loads(Path(os.environ['GFFX_STREAMED_INPUT']).read_bytes());seen=[]
    class Captured(Exception):pass
    def capture(field,**kw):seen.append(kw);raise Captured
    monkeypatch.setattr(m,'integrate_square',capture)
    with pytest.raises(Captured):m.full_pixel(data,reference_profile='streamed22-offline-hour-v1',streamed_piece=2,centered_correction=enabled)
    r=seen[0];assert type(r['correction_integrator']) is (CenteredMaterialCorrection if enabled else MaterialCorrection)
    assert (r['max_cells'],r['max_depth'],r['tolerance'])==(4096,14,str(allocation(data)[2]))
    for kwargs in [dict(centered_correction=1),dict(centered_correction=True)]:
        with pytest.raises(ValueError,match='centered correction'):m.full_pixel({},**kwargs)


def test_centered_campaign_requires_explicit_frozen_mode_and_keeps_old_order():
    from tools.opaque_streamed_reference_campaign import checker_options,campaign_order
    from tools.opaque_streamed_reference_contract import HOUR_PROFILE,PROFILE
    old=dict(profile=HOUR_PROFILE);new=dict(profile=HOUR_PROFILE,checker='centered-correction-v1')
    assert checker_options(old)=={} and campaign_order(old)==[1,0,*range(2,22)]
    assert checker_options(new)==dict(centered_correction=True)
    assert campaign_order(new)==[2,0,1,*range(3,22)]
    for invalid in (dict(profile=PROFILE,checker='centered-correction-v1'),dict(profile=HOUR_PROFILE,checker='unknown'),dict(profile=HOUR_PROFILE,checker=True)):
        with pytest.raises(ValueError):checker_options(invalid)


@pytest.mark.parametrize('fail',[False,True])
def test_centered_campaign_checks_failed_piece_first_without_reusing_certificates(tmp_path,monkeypatch,fail):
    from tools import opaque_streamed_reference_campaign as m
    frozen=dict(profile='streamed22-offline-hour-v1',checker='centered-correction-v1')
    monkeypatch.setattr(m,'load_manifest',lambda *a:(frozen,{}))
    monkeypatch.setattr(m,'certificate_header',lambda *a,**kw:{})
    monkeypatch.setattr(m,'validate_certificate',lambda *a:None)
    calls=[]
    def launch(fixtures,out,**kwargs):
        index=int(kwargs['worker_args'][-1]);calls.append(index)
        out.write_text(json.dumps(dict(status='stopped' if fail else 'passed',accuracy_admitted=index==22)))
        return int(fail)
    out=tmp_path/'fresh'
    assert m.campaign(tmp_path/'manifest','a'*64,out,launch=launch)==int(fail)
    assert calls==([2] if fail else [2,0,1,*range(3,23)])
    state=json.loads((out/'campaign.json').read_bytes())
    assert state['completed_pieces']==(0 if fail else 22) and state['accuracy_admitted']==(not fail)
    with pytest.raises(FileExistsError):m.campaign(tmp_path/'manifest','a'*64,out,launch=launch)
