"""Closed-form controls for the independent affine-radiance extension."""
from fractions import Fraction as Q
import pytest
from tools import opaque_pixel_oracle as oracle

@pytest.mark.parametrize('faces',[[(0,1,2),(0,2,3)],[(0,1,3),(1,2,3)]])
def test_affine_oracle_integrates_coordinate_fields_exactly(faces):
    v=[(0,0,0),(2,0,0),(2,2,0),(0,2,0)]
    c=[[(v[i][0],v[i][1],v[i][0]+v[i][1]) for i in f] for f in faces]
    out=oracle.integrate_affine(v,faces,c,height=2,width=2)
    assert out['mask']==((1,1),(1,1))
    assert out['colour']==tuple(tuple((Q(2*x+1,2),Q(2*y+1,2),Q(x+y+1)) for x in range(2)) for y in range(2))

def test_affine_oracle_triangle_basis_and_background_have_closed_forms():
    v=[(0,0,0),(1,0,0),(0,1,0)];c=[[(1,0,0),(0,1,0),(0,0,1)]]
    out=oracle.integrate_affine(v,[(0,1,2)],c,height=1,width=1,background=(Q(1,2),0,0))
    assert out=={'mask':((Q(1,2),),),'colour':(((Q(5,12),Q(1,6),Q(1,6)),),)}

def test_affine_oracle_constant_reduction_and_hidden_surface():
    v=[(-1,-1,0),(3,-1,0),(-1,3,0),(-1,-1,1),(3,-1,1),(-1,3,1)];f=[(0,1,2),(3,4,5)]
    c=[(Q(1,3),Q(1,5),Q(1,7)),(1,0,0)]
    assert oracle.integrate_affine(v,f,[[c0]*3 for c0 in c],height=1,width=1)==oracle.integrate(v,f,c,height=1,width=1)

def test_affine_oracle_refuses_bad_corner_colours():
    v=[(0,0,0),(1,0,0),(0,1,0)];f=[(0,1,2)]
    for c in ([],[(1,2,3)],[[[float('nan'),0,0]]*3],[[[20,0,0]]*3]):
        with pytest.raises((TypeError,ValueError)):oracle.integrate_affine(v,f,c,height=1,width=1)
