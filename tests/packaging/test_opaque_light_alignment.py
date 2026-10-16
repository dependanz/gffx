"""Light support must lie on the endpoints actually flattened by the map."""
from fractions import Fraction as Q
import itertools
import pytest


def determinant(t):
 a,b,c=t;return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])


@pytest.mark.parametrize('zero_count',[1,2])
@pytest.mark.parametrize('order',list(itertools.permutations(range(3))))
def test_alignment_preserves_every_corner_area_orientation_and_moments(zero_count,order):
 from tools.opaque_light_map import align_triangle
 from tools.opaque_pixel_oracle import _eval,_Budget
 original=((Q(0),Q(0)),(Q(2),Q(0)),(Q(0),Q(3)))
 # Light x+y has one zero; light y has two zeros.
 line=(Q(1) if zero_count==1 else Q(0),Q(1),Q(0))
 tri=tuple(original[i] for i in order);part=dict(face=7,triangle=tri)
 out=align_triangle(part,line);t=out['triangle'];signs=[_eval(line,v,_Budget(lambda:None)) for v in t]
 assert signs[0]==0 and (signs[1]==0 if zero_count==2 else signs[1]>0) and signs[2]>0
 assert out['face']==7 and set(t)==set(tri) and determinant(t)==determinant(tri)
 # Independent affine-triangle integral of 1,x,y,x^2,xy,y^2.
 def moments(t):
  area=abs(determinant(t))/2;x,y=zip(*t)
  return [area,area*sum(x)/3,area*sum(y)/3,area*(sum(z*z for z in x)+sum(x[i]*x[j] for i in range(3) for j in range(i)))/6,area*(sum(x)*sum(y)+sum(a*b for a,b in zip(x,y)))/12,area*(sum(z*z for z in y)+sum(y[i]*y[j] for i in range(3) for j in range(i)))/6]
 assert moments(t)==moments(tri)
 assert part['triangle']==tri


def test_alignment_keeps_fully_lit_order_and_tiny_positive_support():
 from tools.opaque_light_map import align_triangle
 part=dict(face=0,triangle=((Q(0),Q(0)),(Q(1),Q(0)),(Q(0),Q(1))))
 assert align_triangle(part,(0,0,1)) is part
 tiny=Q(1,10**80);out=align_triangle(part,(0,tiny,0))
 assert out['triangle']==part['triangle']
 assert align_triangle(dict(face=0,triangle=(part['triangle'][1],part['triangle'][2],part['triangle'][0])),(0,tiny,0))['triangle']==part['triangle']


@pytest.mark.parametrize('line',[(0,0,0),(0,0,-1),(1,-1,0)])
def test_alignment_rejects_nonpositive_or_unclipped_support(line):
 from tools.opaque_light_map import align_triangle
 with pytest.raises(ValueError):align_triangle(dict(face=0,triangle=((0,0),(1,0),(0,1))),line)


@pytest.mark.parametrize('piece',[9,10,11,16])
def test_aligned_original_fields_preserve_physical_values_and_spatial_derivatives(piece,monkeypatch):
 import json
 from pathlib import Path
 from tools import opaque_full_material_reference as m
 from tools.opaque_light_partition import partition
 from tools.opaque_light_map import align_triangle,coordinates
 from tools.opaque_shading_intervals import I,Jet
 data=json.loads(Path('D:/devspace/gffx/qualification-20260913/m1-continuation-20260913/opaque169-20261002/pixel2-input2.json').read_bytes())
 part=data['selected']['visible_parts'][piece];parts,proof=partition(part,data['faces'][part['face']],data['selected']['origin'],data['material']['direction'])
 charts=[]
 def capture(field,**kw):
  charts.append(dict(field=field,**kw));return dict(value=[I(0),I(0)],cells=0,seconds=0)
 monkeypatch.setattr(m,'integrate_square',capture)
 for aligned in (False,True):
  m.full_pixel(data,reference_profile='remaining28-offline-two-hour-v1',streamed_piece=piece,light_map_power=4 if aligned else 1,align_light_boundary=aligned,material_only_inputs=True,aggregate_residual=True,tight_cosine_slopes=True)
 assert len(charts)==2*len(parts)
 for index,part in enumerate(parts):
  tri=part['triangle'];new=align_triangle(part,proof['line'])['triangle']
  oldchart,newchart=charts[index],charts[len(parts)+index]
  for a,b in [(Q(1,2),Q(1,4)),(Q(3,4),Q(1,2))]:
   u=Jet.variable(I(a),order=3);v=Jet(I(b),order=3);x,y,j=coordinates(u,v,4)
   weights=(1-x,x*(1-y),x*y);beta=[weights[new.index(vertex)] for vertex in tri]
   old_u=beta[1]+beta[2];old_v=beta[2]/old_u
   oldchart['prepare'](old_u,old_v,(old_u[0,0],old_v[0,0]));newchart['prepare'](u,v,(u[0,0],v[0,0]))
   expected=[z*x*j/old_u for z in oldchart['field'](old_u,old_v)];actual=newchart['field'](u,v)
   for aa,bb in zip(actual,expected):
    for k in range(4):assert max(aa[k,0].lo,bb[k,0].lo)<=min(aa[k,0].hi,bb[k,0].hi),(piece,index,k)


def test_alignment_requires_explicit_remaining_mapped_profile():
 from tools.opaque_full_material_reference import full_pixel
 for flag,power in [(1,4),('yes',4),(True,1)]:
  with pytest.raises(ValueError,match='alignment'):full_pixel({},reference_profile='remaining28-offline-two-hour-v1',align_light_boundary=flag,light_map_power=power)
