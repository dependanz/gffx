"""Independent topology and immutable registration controls for the M1 scale head."""
from collections import Counter
import copy,math
import pytest

def test_flame_scale_closed_nondegenerate_topology():
 from tools.synthetic_flame_fixture import generate,encode
 data=generate();v=data['vertices'];faces=data['triangles'];assert len(v)==5002 and len(faces)==10000
 assert len(set(map(tuple,v)))==5002
 edges=Counter();signed_volume=0.;minimum_area=math.inf
 for a,b,c in faces:
  assert len({a,b,c})==3 and all(0<=i<len(v) for i in (a,b,c))
  u=[v[b][j]-v[a][j] for j in range(3)];w=[v[c][j]-v[a][j] for j in range(3)]
  cross=[u[1]*w[2]-u[2]*w[1],u[2]*w[0]-u[0]*w[2],u[0]*w[1]-u[1]*w[0]]
  minimum_area=min(minimum_area,math.sqrt(sum(x*x for x in cross))*.5)
  assert sum(cross[j]*(v[a][j]+v[b][j]+v[c][j])/3 for j in range(3))>0
  signed_volume+=sum(v[a][j]*cross[j] for j in range(3))/6
  edges.update(((a,b),(b,c),(c,a)))
 assert minimum_area>0 and signed_volume>0
 assert all(n==1 and edges[(b,a)]==1 for (a,b),n in edges.items())
 assert len(v)-len(edges)//2+len(faces)==2
 assert encode(data)==encode(generate()) and len(encode(data))<8*1024**2

def test_flame_registration_preserves_prior_entry_and_rejects_overwrite(tmp_path):
 from pathlib import Path
 from tools.corpus_registry import register_synthetic,load_entry
 source=Path(__file__).resolve().parents[2];base='da331088f7068bcc75a1ae8bd6ab5a1110847d4a'
 first=register_synthetic(tmp_path,source,base_commit=base)
 new=register_synthetic(tmp_path,source,base_commit=base,recipe='synthetic_flame_m1')
 assert new['manifest']['id']=='synthetic_flame_m1'
 assert load_entry(tmp_path,'synthetic_face_m1',allow_development=True)==first
 with pytest.raises(FileExistsError):register_synthetic(tmp_path,source,base_commit=base,recipe='synthetic_flame_m1')
 assert load_entry(tmp_path,'synthetic_flame_m1',allow_development=True)==new
 with pytest.raises(ValueError):register_synthetic(tmp_path,source,base_commit=base,recipe='unknown')

def test_flame_profile_rejects_older_and_corrupt_geometry():
 from tools.synthetic_flame_fixture import generate
 from tools.deferred_flame import validate_profile
 from tools.synthetic_face_fixture import generate as old
 data=generate();validate_profile(data,'principled')
 with pytest.raises(ValueError):validate_profile(old(),'principled')
 bad=copy.deepcopy(data);bad['triangles'][0]=[0,0,1]
 with pytest.raises(ValueError):validate_profile(bad,'principled')


def test_scale_forecast_rejects_mismatched_or_invalid_evidence():
 from tools.deferred_flame_probe import forecast
 args=dict(width=256,height=256,material='principled',soft=0.,backward=False,device='cuda',wheel='wheel',driver='driver',scene='scene',partition=4)
 rows=[dict(status='passed',wheel={'sha256':'wheel'},driver_sha256='driver',scene_sha256='scene',estimate={'principled_point_block_limit':4},profile=dict(width=n,height=n,material='principled',soft=0.,backward=False,device='cuda'),supervisor=dict(exit_code=0,failure=None),measured_seconds=n*n/1024) for n in (32,64)]
 assert forecast(rows,**args)==158
 for key,value in [('scene_sha256','other'),('measured_seconds',float('nan')),('measured_seconds',0),('status','not_admitted')]:
  bad=copy.deepcopy(rows);bad[0][key]=value
  with pytest.raises(ValueError):forecast(bad,**args)
 bad=copy.deepcopy(rows);bad[1]['estimate']['principled_point_block_limit']=2
 with pytest.raises(ValueError):forecast(bad,**args)
 with pytest.raises(ValueError):forecast(rows[:1],**args)
