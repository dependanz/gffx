"""Exact no-overlap proof preserves reference pieces without discarding slivers."""
from fractions import Fraction as Q
import pytest
from tools import opaque_pixel_oracle as g
from tools.opaque_principled_reference import visible_fragments

def poly(*xy):return tuple(tuple(map(Q,p)) for p in xy)
S=poly((0,0),(1,0),(1,1),(0,1))
def subtract(p,lines):return g._subtract_if_intersects(p,lines,g._Budget(lambda:None))
def moments(parts):
    a=x=y=Q(0)
    for p in parts:
        for u,v in zip(p,p[1:]+p[:1]):
            cross=u[0]*v[1]-u[1]*v[0];a+=cross/2;x+=(u[0]+v[0])*cross/6;y+=(u[1]+v[1])*cross/6
    return a,x,y

@pytest.mark.parametrize('last',[-1,0])
def test_disjoint_or_touching_occluder_preserves_original_polygon(last):
    # Early halfplanes cut the square, but the last proves no positive-area overlap.
    lines=((Q(1),Q(0),Q(-1,2)),(Q(0),Q(1),Q(-1,2)),(Q(-1),Q(0),Q(last)))
    assert subtract(S,lines)==[S]

@pytest.mark.parametrize('eps',[Q(1,4),Q(1,10**30)])
def test_positive_overlap_including_tiny_sliver_keeps_exact_moments(eps):
    lines=((Q(1),Q(0),-1+eps),(Q(0),Q(1),Q(0)),(Q(0),Q(-1),Q(1)))
    result=subtract(S,lines);old=g._subtract(S,lines,g._Budget(lambda:None))
    assert moments(result)==moments(old)
    assert moments(result)[0]==1-eps and result!=[S]

def test_compact_visible_cover_has_exact_per_face_area_and_moments():
    vertices=[[0,0,0],[1,0,0],[0,1,0],[1,1,0],[Q(3,4),Q(3,4),-1],[2,Q(3,4),-1],[Q(3,4),2,-1]]
    faces=[[0,1,2],[1,3,2],[4,5,6]]
    old=visible_fragments(vertices,faces);new=visible_fragments(vertices,faces,preserve_disjoint=True)
    for face in range(3):
        assert moments([p for i,p in old if i==face])==moments([p for i,p in new if i==face])
    assert len(new)<=len(old)

def test_disjoint_proof_charges_the_existing_work_budget():
    b=g._Budget(lambda:None);b.remaining=0
    with pytest.raises(ValueError,match='budget'):g._subtract_if_intersects(S,((Q(1),Q(0),Q(-2)),),b)
