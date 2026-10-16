"""Validate a fresh dyadic work schedule; never load numerical certificates.

Every chart still covers exactly its whole unit square. All leaves are evaluated
from scratch and every internal node is charged, including nodes whose coarse
integral is not evaluated. Prior evidence may choose work order only.
"""
from fractions import Fraction as Q


def prepare_schedule(charts,leaves,*,max_cells,max_depth,checkpoint=lambda:None):
    if not isinstance(charts,(tuple,list)) or not charts or not isinstance(leaves,(tuple,list)) or not 1<=len(leaves)<=max_cells<=256:raise ValueError('bounded complete schedule required')
    if any(type(r.get('depth'))!=int or not 0<=r['depth']<=max_depth for r in charts):raise ValueError('valid inherited chart depths required')
    if sum((Q(1,2**r['depth']) for r in charts),Q(0))!=1:raise ValueError('complete inherited chart tree required')
    grouped=[[] for _ in charts]
    for row in leaves:
        checkpoint()
        if not isinstance(row,dict) or set(row)!={'chart','box','depth'} or type(row['chart'])!=int or not 0<=row['chart']<len(charts) or type(row['depth'])!=int:raise ValueError('schedule contains geometry and depth only')
        box=row['box']
        if not isinstance(box,(tuple,list)) or len(box)!=4 or any(type(v) not in (str,int,Q) or len(str(v))>128 for v in box):raise ValueError('four bounded exact rational endpoints required')
        a,b,c,d=map(Q,box)
        if not 0<=a<b<=1 or not 0<=c<d<=1 or not 0<=row['depth']<=max_depth:raise ValueError('unit leaf and bounded depth required')
        grouped[row['chart']].append((a,b,c,d,row['depth'],row['chart']))
    pending=[]
    def visit(box,rows,depth):
        checkpoint()
        if not rows or depth>max_depth:raise ValueError('complete bounded binary leaf cover required')
        if len(rows)==1:
            if rows[0][:4]!=box or rows[0][4]!=depth:raise ValueError('leaf does not match complete dyadic tree')
            pending.append(rows[0]);return
        for axis in (0,1):
            k=2*axis;mid=(box[k]+box[k+1])/2
            if any(r[k]<mid<r[k+1] for r in rows):continue
            left=[r for r in rows if r[k+1]<=mid];right=[r for r in rows if r[k]>=mid]
            if not left or not right or len(left)+len(right)!=len(rows):continue
            lo=list(box);hi=list(box);lo[k+1]=mid;hi[k]=mid
            visit(tuple(lo),left,depth+1);visit(tuple(hi),right,depth+1);return
        raise ValueError('overlap, missing area or non-dyadic schedule')
    for chart,rows in enumerate(grouped):visit((Q(0),Q(1),Q(0),Q(1)),rows,charts[chart]['depth'])
    count=len(pending)-1
    if count+len(pending)>max_cells:raise ValueError('complete schedule exceeds unchanged work budget')
    return list(reversed(pending)),count
