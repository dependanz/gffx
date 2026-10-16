"""Conserve the original total error budget under explicit rational work shares."""
from fractions import Fraction as Q
from copy import deepcopy
import json
import pytest
from tools.opaque_shading_intervals import I
from tools import opaque_full_material_reference as ref


def parts(n):return [dict(triangle=[[0,0],[str(Q(i+1,7)),0],[0,1]]) for i in range(n)]

@pytest.mark.parametrize('n',[1,2,3,13,32])
@pytest.mark.parametrize('mode',['equal','varying','extreme'])
def test_work_weights_conserve_exact_basis_and_final_rgb_radius(n,mode):
 work=[Q(1) if mode=='equal' else Q(1+(i%5),2) if mode=='varying' else Q(1,16) if i%2 else Q(16) for i in range(n)];area=[Q(i+1,7) for i in range(n)];got=ref.piece_tolerances(parts(n),work_weights=work);total=sum((a*w for a,w in zip(area,work)),Q(0))
 for a,w,t in zip(area,work,got):
  exact=a*w/total/Q(2*10**12);assert 0<Q(t)<=exact and exact-Q(t)<Q(1,10**70)
 radius=sum(map(Q,got),Q(0));assert radius<=Q(1,2*10**12)
 # Independent worst-case error: RGB=spec+albedo*matte, 0<=albedo<=1.
 for albedo in [Q(0),Q(1,5),Q(3,5),Q(1)]:assert radius+albedo*radius<=Q(1,10**12)
 assert radius<=Q(1,10**12) # Constant-albedo derivative is the matte basis.
 if mode=='equal':assert got==ref.piece_tolerances(parts(n))

@pytest.mark.parametrize('work',[[1],[1,1,1],[0,1],[-1,1],[True,1],[1.,1],['1',1],[None,1],[Q(1,17),1],[17,1],{'a':1,'b':1}])
def test_invalid_work_weights_cannot_erase_nonzero_pieces_or_expand_domain(work):
 with pytest.raises(ValueError):ref.piece_tolerances(parts(2),work_weights=work)


def test_zero_area_stays_exact_zero_with_positive_work_weight():
 p=parts(3);p[1]['triangle']=[[0,0]]*3;got=ref.piece_tolerances(p,work_weights=[Q(1),Q(5,2),Q(1)])
 assert got[1]==0 and got[0]>0 and got[2]>0 and sum(map(Q,got))<=Q(1,2*10**12)


def test_original_profile_redistributes_without_increasing_the_pool():
 p=parts(13);old=ref.piece_tolerances(p);new=ref.original_piece_tolerances(p)
 expected=ref.piece_tolerances(p,work_weights=[1,1,1,Q(5,2)]+[1]*9)
 assert new==expected and new[3]>old[3] and all(new[i]<old[i] for i in range(13) if i!=3)
 assert sum(map(Q,new))<=Q(1,2*10**12)
 with pytest.raises(ValueError):ref.original_piece_tolerances(parts(12))


def test_campaign_rejects_extra_pool_even_if_every_receipt_is_rehashed():
 from tests.packaging.test_opaque_reference_campaign import fixture,sha
 from tools.opaque_reference_campaign import compose
 _,s,r=fixture();s=deepcopy(s);r=deepcopy(r);s['pieces'][3]['tolerance']=str(I(s['pieces'][3]['tolerance']).lo*2)
 for i,row in enumerate(r):
  row['result']['spec_sha256']=sha(s);row['result']['tolerance']=s['pieces'][i]['tolerance'];row['result_sha256']=sha(row['result']);row['worker']['campaign']['spec_sha256']=sha(s);row['worker']['campaign']['result_sha256']=row['result_sha256'];row['worker_sha256']=sha(row['worker'])
 with pytest.raises(ValueError,match='aggregate basis radius'):compose(s,r)


def test_campaign_spec_uses_the_same_explicit_original_work_profile():
 from tests.packaging.test_opaque_reference_campaign import fixture
 d,s,r=fixture();assert [x['tolerance'] for x in s['pieces']]==list(map(str,ref.original_piece_tolerances(d['selected']['visible_parts'])))

def test_nonzero_width_certificates_compose_at_unchanged_final_bound():
 from tests.packaging.test_opaque_reference_campaign import fixture,sha
 from tools.opaque_reference_campaign import compose
 data,s,rows=fixture()
 for i,row in enumerate(rows):
  radius=Q(s['pieces'][i]['tolerance'])*Q(9,10);area=Q(i+1,26);values=[I(I(multiplier*area-radius).lo,I(multiplier*area+radius).hi) for multiplier in (2,3)]
  row['result']['basis']=[[str(x.lo),str(x.hi)] for x in values];row['result_sha256']=sha(row['result']);row['worker']['campaign']['result_sha256']=row['result_sha256'];row['worker_sha256']=sha(row['worker'])
 cert=compose(s,rows);total=sum((Q(i+1,26) for i in range(13)),Q(0))
 for pair,albedo in zip(cert['value'],data['material']['albedo']):
  lo,hi=map(Q,pair);assert lo<=total*(2+3*Q(albedo))<=hi and hi-lo<=Q(2,10**12)
 lo,hi=map(Q,cert['albedo_derivative']);assert lo<=3*total<=hi and hi-lo<=Q(2,10**12)
