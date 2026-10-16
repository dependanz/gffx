"""Independent controls for conservative projected-mesh admission."""
import numpy as np
import pytest

def brute(v,f,h,w):
    return np.array([[sum(min(v[t,0])<=x+1 and max(v[t,0])>=x and min(v[t,1])<=y+1 and max(v[t,1])>=y for t in f) for x in range(w)] for y in range(h)])

def test_mesh_candidate_count_matches_inclusive_native_predicate():
    from tools.opaque_mesh_preflight import candidate_counts
    v=np.array([[-2.,-.3],[1.,1.],[2.,2.],[3.,0.],[-1.,4.],[0.,0.],[1.,0.],[0.,1.]])
    f=np.array([[0,1,2],[2,3,4],[5,6,7]])
    assert np.array_equal(candidate_counts(v,f,4,5),brute(v,f,4,5))

def test_mesh_candidate_count_preserves_exact_boundaries_and_one_ulp_offsets():
    from tools.opaque_mesh_preflight import candidate_counts
    for q in [np.nextafter(1.,0.),1.,np.nextafter(1.,2.)]:
        v=np.array([[q,0.],[q,1.],[q,2.]])
        f=np.array([[0,1,2]])
        assert np.array_equal(candidate_counts(v,f,3,4),brute(v,f,3,4))

def test_mesh_candidate_cap_is_not_silently_truncated():
    from tools.opaque_mesh_preflight import candidate_counts
    v=np.array([[.2,.2],[.8,.2],[.2,.8]])
    for n in (32,33):assert candidate_counts(v,np.array([[0,1,2]]*n),1,1).item()==n

def test_mesh_candidate_invalid_or_unbounded_profiles_refuse():
    from tools.opaque_mesh_preflight import candidate_counts
    v=np.array([[0.,0.],[1.,0.],[0.,1.]]);f=np.array([[0,1,2]])
    for h,w in [(0,2),(1081,1920),(1080,1921)]:
        with pytest.raises(ValueError):candidate_counts(v,f,h,w)
    with pytest.raises(ValueError):candidate_counts(v*np.nan,f,2,2)
    with pytest.raises(ValueError):candidate_counts(v,np.array([[0,1,3]]),2,2)
