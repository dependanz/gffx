"""Independent exact-rational opaque clipping oracle; no native geometry calls.

Binary64 input coordinates are interpreted as exact rationals. The certificate
bounds L1 owner-domain discrepancy, including overlaps in a rounded witness.
It does not certify radiance, texture interpolation, or moving-domain VJPs.
"""
from fractions import Fraction as F
import time

def rational(x):return F(float(x))
def area(p):
 if len(p)<3:return F(0)
 return abs(sum((a[0]*b[1]-a[1]*b[0] for a,b in zip(p,p[1:]+p[:1])),F(0)))/2

def box(p):return min(x for x,y in p),min(y for x,y in p),max(x for x,y in p),max(y for x,y in p)
def overlaps(a,b):return not (a[2]<=b[0] or b[2]<=a[0] or a[3]<=b[1] or b[3]<=a[1])
def edges(p):
 sign=1 if sum(a[0]*b[1]-a[1]*b[0] for a,b in zip(p,p[1:]+p[:1]))>0 else -1
 return [(sign*(a[1]-b[1]),sign*(b[0]-a[0]),sign*(a[0]*b[1]-b[0]*a[1])) for a,b in zip(p,p[1:]+p[:1])]
def value(l,p):return l[0]*p[0]+l[1]*p[1]+l[2]

class Oracle:
 def __init__(self,seconds=90,max_work=2000000):self.deadline=time.perf_counter()+seconds;self.work=0;self.max_work=max_work
 def clip(self,p,line):
  self.work+=len(p)
  if self.work>self.max_work:raise OverflowError('exact clipping work limit')
  if time.perf_counter()>self.deadline:raise TimeoutError('exact clipping reference deadline')
  if not p:return []
  out=[];a=p[-1];da=value(line,a)
  for b in p:
   db=value(line,b)
   if (da<0)!=(db<0):
    t=da/(da-db);out.append((a[0]+t*(b[0]-a[0]),a[1]+t*(b[1]-a[1])))
   if db>=0:out.append(b)
   a,da=b,db
  compact=[]
  for q in out:
   if not compact or q!=compact[-1]:compact.append(q)
  if len(compact)>1 and compact[0]==compact[-1]:compact.pop()
  return compact if area(compact)>0 else []
 def intersection(self,p,q):
  if not p or not q or not overlaps(box(p),box(q)):return []
  for line in edges(q):
   p=self.clip(p,line)
   if not p:break
  return p
 def subtract(self,p,lines):
  out=[]
  for l in lines:
   negative=tuple(-x for x in l);piece=self.clip(p,negative)
   if piece:out.append(piece)
   p=self.clip(p,l)
   if not p:break
  return out
 def visible(self,vertices,faces,origin):
  ox,oy=map(rational,origin);v=[(rational(x)-ox,rational(y)-oy,rational(z)) for x,y,z in vertices];geometry=[]
  for f in faces:
   xyz=[v[int(i)] for i in f];p=[q[:2] for q in xyz]
   if not area(p):geometry.append(None);continue
   x0,y0,z0=xyz[0];ux=xyz[1][0]-x0;uy=xyz[1][1]-y0;vx=xyz[2][0]-x0;vy=xyz[2][1]-y0;det=ux*vy-uy*vx
   dz1=xyz[1][2]-z0;dz2=xyz[2][2]-z0;a=(dz1*vy-dz2*uy)/det;b=(ux*dz2-vx*dz1)/det;depth=(a,b,z0-a*x0-b*y0)
   geometry.append((p,edges(p),depth,box(p)))
  square=[(F(0),F(0)),(F(1),F(0)),(F(1),F(1)),(F(0),F(1))];result=[]
  for i,g in enumerate(geometry):
   if g is None:result.append([]);continue
   p,lines,depth,bounds=g;subject=square
   for l in lines:subject=self.clip(subject,l)
   parts=[subject] if subject else []
   for j,other in enumerate(geometry):
    if i==j or other is None or not parts or not overlaps(bounds,other[3]):continue
    difference=tuple(a-b for a,b in zip(depth,other[2]));tie=not any(difference)
    if tie and j>i:continue
    constraints=other[1]+([] if tie else [difference]);remaining=[]
    for part in parts:
     if not overlaps(box(part),other[3]):remaining.append(part);continue
     remaining.extend(self.subtract(part,constraints))
    parts=remaining
   result.append(parts)
  return result
 def compare(self,ideal,rows):
  witness=[[] for _ in ideal]
  for row in rows:
   face=int(row[0]);assert row[0]==face and 0<=face<len(ideal)
   p=[(rational(row[1+2*k]),rational(row[2+2*k])) for k in range(3)]
   if area(p):witness[face].append(p)
  results=[]
  for face,(reference,actual) in enumerate(zip(ideal,witness)):
   exact=sum(map(area,reference),F(0));represented=sum(map(area,actual),F(0));common=F(0);overlap=F(0)
   for p in actual:
    for q in reference:common+=area(self.intersection(p,q))
   for i,p in enumerate(actual):
    for q in actual[i+1:]:overlap+=area(self.intersection(p,q))
   # Pairwise overlap bounds multiplicity excess; ideal pieces are disjoint by construction.
   bound=exact+represented-2*common+2*overlap
   assert bound>=0
   results.append(dict(face=face,ideal_area=exact,witness_area=represented,overlap_area=overlap,l1_area_bound=bound,ideal_pieces=len(reference),witness_triangles=len(actual)))
  return results
