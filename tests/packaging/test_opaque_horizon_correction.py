"""Independent near-horizon integrals; proof selection cannot discard tiny terms."""
import json,os
from pathlib import Path
from fractions import Fraction as Q
from types import SimpleNamespace
import pytest
from tools.opaque_shading_intervals import I,Unresolved

def fixture(*,simultaneous=False,sign=1,axis=0,h=Q(1,10**10),scale=Q(1,10**20)):
    from tools.opaque_full_material_reference import HorizonExtension
    cn=list(map(I,(h,2*h,4*h,Q(1))));values=list(map(I,(scale,3*scale,2*scale)));origin=scale/2
    e=HorizonExtension(SimpleNamespace(cn=cn,x=[v.ln() for v in cn],controls=[[[v]*4]*2 for v in values],mean=[I(1)]*2,pi=I(1),horizon=[I(origin)]*2,values=[[v]*2 for v in values]+[[values[-1]]*2]));e.indices=[-1,-1 if simultaneous else 2]
    low=h/2;high=3*h;span=high-low
    def inputs(u,v):
        z=(u,v)[axis];f=(z+z*z)/2;q=z.lift(low)+(f if sign>0 else 1-f)*span
        return q,q if simultaneous else z.lift(Q(3,4)),z.lift(1),(1+2*z)/2
    # Integrate piecewise rational polynomials in physical q, independently of jets/table proofs.
    A=origin;B=(scale-origin)/h
    def polyint(coeff,a,b):return sum((v*(b**(j+1)-a**(j+1))/(j+1) for j,v in enumerate(coeff)),Q(0))
    expected=Q(0)
    for a,b,k in ((h,2*h,scale),(2*h,high,3*scale)):
        coeff=[0,k*k-A*A,-2*A*B,-B*B] if simultaneous else [(k-A)*Q(3,4)*2*scale,-B*Q(3,4)*2*scale]
        expected+=polyint(coeff,a,b)/span
    return e,inputs,I(expected)

@pytest.mark.parametrize('simultaneous',[False,True])
@pytest.mark.parametrize('sign',[-1,1])
@pytest.mark.parametrize('axis',[0,1])
def test_horizon_correction_contains_exact_tiny_piecewise_integrals(simultaneous,sign,axis):
    from tools.opaque_horizon_correction import HorizonMaterialCorrection
    e,inputs,want=fixture(simultaneous=simultaneous,sign=sign,axis=axis)
    c=HorizonMaterialCorrection(e,inputs,curved=True,transverse=True,envelope_fallback=True,aggregate_fallback=True)
    got=c((I(0,1),I(0,1)))[1]
    assert got.lo<=want.lo<=want.hi<=got.hi and not want.zero
    assert c.last_method=='horizon-aggregate' and c.inputs is inputs

@pytest.mark.parametrize('sign',[-1,1])
@pytest.mark.parametrize('axis',[0,1])
def test_horizon_log_curvature_contains_independent_antiderivative(sign,axis):
    from decimal import Decimal as D,localcontext
    from tools.opaque_full_material_reference import HorizonExtension
    from tools.opaque_horizon_correction import HorizonMaterialCorrection
    h=Q(1,10**10);scale=I('1e-20');cn=list(map(I,(h,2*h,4*h,Q(1))));xs=[v.ln() for v in cn];controls=[];values=[]
    for i in range(3):
        z=xs[i]-xs[0];dx=xs[i+1]-xs[i];A=1+z*z;B=2*z*dx;C=dx*dx
        row=[scale*A,scale*(A+B/3),scale*(A+2*B/3+C/3),scale*(A+B+C)];controls.append([row,row]);values.append([row[0]]*2)
    values.append([controls[-1][0][-1]]*2)
    e=HorizonExtension(SimpleNamespace(cn=cn,x=xs,controls=controls,mean=[I(1)]*2,pi=I(1),horizon=[scale]*2,values=values));e.indices=[-1,2]
    def inputs(u,v):
        z=(u,v)[axis];f=(z+z*z)/2;q=z.lift(h/2)+(f if sign>0 else 1-f)*Q(5,2)*h
        return q,z.lift(Q(3,4)),z.lift(1),(1+2*z)/2
    c=HorizonMaterialCorrection(e,inputs,curved=True,transverse=True);got=c((I(0,1),I(0,1)))[1]
    with localcontext() as context:
        context.prec=100;q=D(3);z=q.ln();area=(q*(z*z-2*z+2)-2)/D('2.5')
        expected=D('1e-40')*D('.75')*(1+(D('.75')/D('1e-10')).ln()**2)*area
        assert got.lo<=expected<=got.hi

@pytest.mark.parametrize('error',[ValueError('disjoint proof evidence'),Unresolved('cancel horizon proof')])
def test_horizon_invalid_evidence_and_cancellation_propagate(monkeypatch,error):
    from tools.opaque_horizon_correction import HorizonMaterialCorrection
    from tools.opaque_aggregate_peano import AggregateMaterial
    from tools.opaque_directional_correction import DirectionalMaterialCorrection
    e,inputs,_=fixture();c=HorizonMaterialCorrection(e,inputs,curved=True,transverse=True)
    monkeypatch.setattr(AggregateMaterial,'__call__',lambda *a,**kw:(_ for _ in ()).throw(error))
    monkeypatch.setattr(DirectionalMaterialCorrection,'__call__',lambda *a:pytest.fail('invalid fallback'))
    with pytest.raises(type(error),match=str(error)):c((I(0,1),I(0,1)))
    assert c.inputs is inputs

def test_horizon_domain_refusal_restores_historical_fallback(monkeypatch):
    from tools.opaque_horizon_correction import HorizonMaterialCorrection
    from tools.opaque_aggregate_peano import AggregateMaterial
    from tools.opaque_directional_correction import DirectionalMaterialCorrection
    e,inputs,_=fixture();c=HorizonMaterialCorrection(e,inputs,curved=True,transverse=True);calls=[]
    def refuse(self,box,*,integrated_axis=None):calls.append(integrated_axis);raise ValueError('aggregate angular crossings need certified monotonicity')
    def fallback(self,box):
        assert self.inputs is inputs;return [I(0),I('-2e-70','3e-70')]
    monkeypatch.setattr(AggregateMaterial,'__call__',refuse);monkeypatch.setattr(DirectionalMaterialCorrection,'__call__',fallback)
    got=c((I(0,1),I(0,1)))[1];assert calls==[0,1] and got.lo==I('-2e-70').lo and got.hi==I('3e-70').hi

def test_horizon_selection_leaves_nonhorizon_domain_unchanged(monkeypatch):
    from tools.opaque_horizon_correction import HorizonMaterialCorrection
    from tools.opaque_directional_correction import DirectionalMaterialCorrection
    e,inputs,_=fixture();e.indices=[0,2];c=HorizonMaterialCorrection(e,inputs,curved=True,transverse=True);calls=[]
    monkeypatch.setattr(DirectionalMaterialCorrection,'__call__',lambda self,box:calls.append(self.inputs is inputs) or [I(0),I('1e-70')])
    assert c((I(0,1),I(0,1)))[1].lo==I('1e-70').lo and calls==[True]

@pytest.mark.parametrize('enabled',[False,True])
def test_horizon_mode_preserves_exact_piece_obligation(monkeypatch,enabled):
    from tools import opaque_full_material_reference as m
    from tools.opaque_horizon_correction import HorizonMaterialCorrection
    from tools.opaque_directional_correction import DirectionalMaterialCorrection
    from tools.opaque_streamed_reference_contract import allocation
    data=json.loads(Path(os.environ['GFFX_STREAMED_INPUT']).read_bytes());seen=[]
    class Captured(Exception):pass
    def capture(field,**kw):seen.append(kw);raise Captured
    monkeypatch.setattr(m,'integrate_square',capture)
    with pytest.raises(Captured):m.full_pixel(data,reference_profile='streamed22-offline-hour-v1',streamed_piece=3,centered_correction=True,directional_correction=True,horizon_correction=enabled)
    r=seen[0];assert type(r['correction_integrator']) is (HorizonMaterialCorrection if enabled else DirectionalMaterialCorrection)
    assert (r['max_cells'],r['max_depth'],r['tolerance'])==(4096,14,str(allocation(data)[3]))
    for kw in [dict(horizon_correction=True),dict(horizon_correction=1),dict(reference_profile='streamed22-offline-hour-v1',streamed_piece=3,horizon_correction=True)]:
        with pytest.raises(ValueError,match='horizon correction'):m.full_pixel({},**kw)

@pytest.mark.parametrize('fail',[False,True])
def test_horizon_frozen_campaign_keeps_serial_fresh_stop_policy(tmp_path,monkeypatch,fail):
    from tools import opaque_streamed_reference_campaign as m
    frozen=dict(profile='streamed22-offline-hour-v1',checker='horizon-correction-v1')
    assert m.checker_options(frozen)==dict(centered_correction=True,directional_correction=True,horizon_correction=True)
    assert m.campaign_order(frozen)==[3,2,0,1,*range(4,22)]
    monkeypatch.setattr(m,'load_manifest',lambda *a:(frozen,{}));monkeypatch.setattr(m,'certificate_header',lambda *a,**kw:{});monkeypatch.setattr(m,'validate_certificate',lambda *a:None)
    calls=[]
    def launch(fixtures,out,**kwargs):
        index=int(kwargs['worker_args'][-1]);calls.append(index);out.write_text(json.dumps(dict(status='stopped' if fail else 'passed',accuracy_admitted=index==22)));return int(fail)
    out=tmp_path/'fresh';assert m.campaign(tmp_path/'manifest','a'*64,out,launch=launch)==int(fail)
    assert calls==([3] if fail else [3,2,0,1,*range(4,23)])
    with pytest.raises(FileExistsError):m.campaign(tmp_path/'manifest','a'*64,out,launch=launch)
