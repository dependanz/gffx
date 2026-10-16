"""Original-piece tangent preflight; no sampled convergence is accepted."""
import json,os,hashlib
from pathlib import Path
from decimal import Decimal as D,localcontext
from fractions import Fraction as Q
import pytest
from tools.opaque_shading_intervals import I,Jet,Unresolved

def data():
    p=Path(os.environ['GFFX_FULL_INPUT']);assert hashlib.sha256(p.read_bytes()).hexdigest()=='a15bea40cb5cdead7994d828b57f5dcfac9f44fab0292a06e7d0a7b08719f73a';return json.loads(p.read_bytes())

@pytest.mark.parametrize("anchor", [0,1,2])
def test_radial_normal_tangent_matches_independent_decimal_vectors(anchor):
    from tools.opaque_original_tangent import prepare_piece,cosine_field
    p=prepare_piece(data(),anchor=anchor);angles=cosine_field(p)
    for u,s in [(Q(1,4),Q(1,4)),(Q(1,2),Q(1,3)),(Q(3,4),Q(2,3)),(Q(1),Q(1))]:
        got=angles(Jet(u),Jet(s));a,b,c=p['visible'];xy=[a[k]+u*((b[k]-a[k])+s*(c[k]-b[k])) for k in range(2)]
        a,b,c=p['triangle'];det=(b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]);q=((xy[0]-a[0])*(c[1]-a[1])-(xy[1]-a[1])*(c[0]-a[0]))/det;r=((b[0]-a[0])*(xy[1]-a[1])-(b[1]-a[1])*(xy[0]-a[0]))/det
        w=[a*b for a,b in zip([1-q-r,q,r],p['inverse_w'])]
        vectors=[[sum(w[i]*Q(rows[i][j]) for i in range(3)) for j in range(3)] for rows in [p['source']['normal'],p['source']['view'],p['motion']]]
        with localcontext() as ctx:
            ctx.prec=90;cast=lambda q:D(q.numerator)/D(q.denominator);n,v,dn=[[cast(x) for x in a] for a in vectors];l=list(map(D,[1,1,2]));dot=lambda a,b:sum(x*y for x,y in zip(a,b));norm=lambda a:dot(a,a).sqrt();N=norm(n)
            values=[dot(n,z)/(N*norm(z)) for z in [v,l]];derivatives=[dot(dn,z)/(N*norm(z))-dot(n,z)*dot(n,dn)/(N**3*norm(z)) for z in [v,l]]
            values.append(((1+dot(v,l)/(norm(v)*norm(l)))/2).sqrt());derivatives.append(D(0))
            for j,expected in enumerate([values,derivatives]):
                for g,x in zip(got,expected):assert g[0,j].lo<=x<=g[0,j].hi,(g[0,j],x)

def _certificate(record_property,anchor):
    from tools.opaque_original_tangent import normal_piece
    from tools.opaque_arb_jets import backend
    p=Path(os.environ['GFFX_FULL_REFERENCE']);assert not p.exists();rows=[]
    try:
        with backend(profile='ball'):result=normal_piece(data(),progress=rows.append,anchor=anchor)
    except Unresolved as error:
        p.write_text(json.dumps(dict(status='unresolved',details=error.details,progress=rows),default=str,indent=2),encoding='utf8');raise
    p.write_text(json.dumps(dict(status='qualified',result=result,progress=rows),default=str,indent=2),encoding='utf8');record_property('original_tangent_certificate',p.read_text())


def test_saved_residual_regions_use_tighter_validated_tangent_bounds(record_property):
    from tools.opaque_original_tangent import normal_piece
    from tools.opaque_arb_jets import backend
    boxes=[(Q(7,16),Q(1,2),Q(1,8),Q(1,4)),(Q(3,8),Q(7,16),Q(0),Q(1,8)),(Q(1,2),Q(9,16),Q(3,4),Q(7,8))]
    with backend(profile='ball'):result=normal_piece(data(),diagnostic_boxes=boxes)
    p=Path(os.environ['GFFX_FULL_REFERENCE']);assert not p.exists();p.write_text(json.dumps(result,default=str,indent=2),encoding='utf8');record_property('saved_tangent_regions',p.read_text())
    assert all(D(row['width_ratio'][1])<D('.1') for row in result['regions'])


def test_cyclic_maps_retain_physical_triangle_and_tangent():
    from tools.opaque_original_tangent import prepare_piece,cosine_field
    ps=[prepare_piece(data(),anchor=k) for k in range(3)]
    assert all(set(map(tuple,p['visible']))==set(map(tuple,ps[0]['visible'])) and p['det']==ps[0]['det'] for p in ps)
    for weights in [(Q(1,3),)*3,(Q(1,2),Q(1,3),Q(1,6)),(Q(1,10),Q(7,10),Q(2,10))]:
        samples=[]
        for k,p in enumerate(ps):
            w=weights[k:]+weights[:k];u=w[1]+w[2];s=w[2]/u
            samples.append(cosine_field(p)(Jet(u),Jet(s)))
        for role in range(3):
            for j in (0,1):assert max(row[role][0,j].lo for row in samples)<=min(row[role][0,j].hi for row in samples)
    for anchor in (-1,3,True):
        with pytest.raises(ValueError,match='cyclic anchor'):prepare_piece(data(),anchor=anchor)


def test_bounded_cyclic_map_comparison(record_property):
    from tools.opaque_original_tangent import normal_piece
    from tools.opaque_arb_jets import backend
    rows=[]
    for anchor in (0,1):
        with backend(profile='ball'):
            try:normal_piece(data(),anchor=anchor,max_cells=63,seconds=30)
            except Unresolved as e:
                cover=e.details['cause']['last_complete_enclosure'];assert cover['complete_cover']
                rows.append(dict(anchor=anchor,status='unresolved',details=e.details))
            else:raise AssertionError('update comparison to retain qualified result')
    intervals=[row['details']['cause']['last_complete_enclosure'] for row in rows]
    for key in ('value','derivative'):
        for a,b in zip(intervals[0][key],intervals[1][key]):assert max(D(a[0]),D(b[0]))<=min(D(a[1]),D(b[1]))
    result=dict(status='diagnostic_only',rows=rows,matte_derivative_width_ratio=str(D(intervals[1]['widths'][1][1])/D(intervals[0]['widths'][1][1])),scope='Same physical piece, alternate exact cyclic coordinates, 63cells/30seconds per map; no gradient certificate.')
    p=Path(os.environ['GFFX_FULL_REFERENCE']);assert not p.exists();p.write_text(json.dumps(result,default=str,indent=2),encoding='utf8');record_property('cyclic_map_comparison',p.read_text())


def test_first_original_normal_tangent_certificate(record_property):
    _certificate(record_property,0)

def test_cyclic_original_normal_tangent_certificate(record_property):
    _certificate(record_property,1)
