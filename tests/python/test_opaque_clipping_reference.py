from fractions import Fraction as F
from tools.opaque_clipping_reference import Oracle,area

def test_exact_square_either_diagonal():
 v=[(0,0,0),(1,0,0),(1,1,0),(0,1,0)]
 for faces in ([(0,1,2),(0,2,3)],[(0,1,3),(1,2,3)]):
  ideal=Oracle().visible(v,faces,(0,0));assert [sum(map(area,p)) for p in ideal]==[F(1,2)]*2

def test_depth_crossing_and_ties():
 v=[(0,0,0),(1,0,0),(0,1,0),(0,0,-.5),(1,0,.5),(0,1,-.5)];o=Oracle();p=o.visible(v,[(0,1,2),(3,4,5)],(0,0))
 assert [sum(map(area,q)) for q in p]==[F(1,8),F(3,8)]
 p=o.visible(v,[(0,1,2),(0,1,2)],(0,0));assert [sum(map(area,q)) for q in p]==[F(1,2),0]

def test_witness_holes_overlaps_and_wrong_owner_detected():
 v=[(0,0,0),(1,0,0),(0,1,0)];o=Oracle();ideal=o.visible(v,[(0,1,2)],(0,0));row=[0,0,0,1,0,0,1]
 assert o.compare(ideal,[row])[0]['l1_area_bound']==0
 assert o.compare(ideal,[])[0]['l1_area_bound']==F(1,2)
 assert o.compare(ideal,[row,row])[0]['l1_area_bound']==F(1,2)

def test_subpixel_positive_region_is_kept():
 e=2**-40;p=Oracle().visible([(0,0,0),(e,0,0),(0,e,0)],[(0,1,2)],(0,0))
 assert sum(map(area,p[0]))==F(1,2**81)
