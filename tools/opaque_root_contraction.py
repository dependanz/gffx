"""Finite interval-Newton inclusions, retaining clipped endpoint possibilities.

Each contraction contains every root of every represented function. Strict
monotonicity is needed only to interpret clipped absence. Endpoint possibilities
remain in every hull; no point sample establishes root existence or absence.
"""
from fractions import Fraction as Q
from tools.opaque_shading_intervals import I
from tools.opaque_peano import crossing_location

def contract_root(domain,evaluate,knot,*,clipped=False,iterations=2,checkpoint=lambda:None,initial=None):
 domain,knot=I(domain),I(knot)
 if domain.lo>=domain.hi or type(clipped)!=bool or type(iterations)!=int or not 1<=iterations<=4:raise ValueError('positive domain,Boolean clipping and1..4iterations required')
 location=domain if initial is None else I(initial)
 if location.lo<domain.lo or location.hi>domain.hi:raise ValueError('initial root outside domain')
 endpoints=[]
 if clipped:
  checkpoint();left=evaluate(I(domain.lo))[0];right=evaluate(I(domain.hi))[0];slope=evaluate(domain)[1]
  if slope.lo<=0<=slope.hi:raise ValueError('strict monotonicity required')
  if (left.hi>=knot.lo if slope.lo>0 else left.lo<=knot.hi):endpoints.append(I(domain.lo))
  if (right.lo<=knot.hi if slope.lo>0 else right.hi>=knot.lo):endpoints.append(I(domain.hi))
 for _ in range(iterations):
  checkpoint()
  if location.lo==location.hi:return location
  middle=I((Q(location.lo)+Q(location.hi))/2);value=evaluate(middle)[0];slope=evaluate(location)[1]
  if slope.lo<=0<=slope.hi:return location
  root=crossing_location(location,value,slope,knot);possible=([root] if root is not None else [])+endpoints
  if not possible:
   if clipped:raise ValueError('inconsistent clipped root evidence')
   return None
  low=max(location.lo,min(x.lo for x in possible));high=min(location.hi,max(x.hi for x in possible))
  if low>high:raise ValueError('disjoint root evidence')
  location=I(low,high)
 return location
