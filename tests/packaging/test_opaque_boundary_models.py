"""Independent images and directional derivatives for bounded native candidates."""
from fractions import Fraction as Q
import pytest
from tools.opaque_pixel_oracle import integrate


def cases():
    square=[(-1,-1,0),(4,-1,0),(4,4,0),(-1,4,0)]
    faces=[(0,1,2),(0,2,3)]
    white=[(1,1,1)]*2
    zero=[(0,0,0)]*4
    yield 'diagonal_a',square,faces,white,zero,3,3
    yield 'diagonal_b',square,[(0,1,3),(1,2,3)],white,zero,3,3
    yield 'subdivision',square+[(1,1,0)],[(0,1,4),(1,2,4),(2,3,4),(3,0,4)],white*2,zero+[(1,0,0)],3,3
    silhouette=[(-1,-1,0),(Q(5,4),-1,0),(Q(5,4),4,0),(-1,4,0)]
    motion=[(0,0,0),(1,0,0),(1,0,0),(0,0,0)]
    yield 'silhouette',silhouette,faces,white,motion,3,3
    # A full blue background plane with a moving foreground red rectangle.
    yield 'overlap',square+[(x,y,-1) for x,y,z in silhouette],faces+[(4,5,6),(4,6,7)],[(0,0,1)]*2+[(1,0,0)]*2,zero+motion,3,3
    v=[(-1,-1,Q(-9,4)),(6,-1,Q(19,4)),(-1,6,Q(-9,4)),(-1,-1,0),(6,-1,0),(-1,6,0)]
    yield 'intersection',v,[(0,1,2),(3,4,5)],[(1,0,0),(0,0,1)],[(0,0,-1)]*3+[(0,0,0)]*3,2,2
    v=[(Q(1,16),Q(1,16),0),(Q(3,16),Q(1,16),0),(Q(1,16),Q(3,16),0)]
    yield 'subpixel',v,[(0,1,2)],[(1,1,1)],[(0,0,0),(1,0,0),(0,0,0)],2,2


CASES=list(cases())


def exact(v,f,rgb,velocity,h,w):
    # Binary-representable fixture coordinates, independently differentiated by
    # exact rational symmetric differences. These area paths are linear/quadratic.
    def sample(t):
        moved=[tuple(Q(a)+t*Q(b) for a,b in zip(p,d)) for p,d in zip(v,velocity)]
        image=integrate(moved,f,rgb,height=h,width=w)
        return [(*image['colour'][y][x],image['mask'][y][x]) for y in range(h) for x in range(w)]
    centre=sample(Q(0));derivatives=[]
    for eps in [Q(1,1024),Q(1,4096)]:
        lo,hi=sample(-eps),sample(eps)
        derivatives.append([tuple((b-a)/(2*eps) for a,b in zip(x,y)) for x,y in zip(lo,hi)])
    assert derivatives[0]==derivatives[1],'fixture needs an independent non-polynomial derivative bound'
    return centre,derivatives[0]


@pytest.mark.parametrize('case',CASES,ids=[c[0] for c in CASES])
def test_compiled_complete_model_matches_exact_reference(case):
    from tools.opaque_boundary_models import evaluate
    name,v,f,rgb,velocity,h,w=case
    expected,derivative=exact(v,f,rgb,velocity,h,w)
    got=evaluate('complete',v,f,rgb,velocity,height=h,width=w)
    for a,b in zip(got['image'],expected):assert a==pytest.approx(list(map(float,b)),rel=0,abs=1e-12)
    for a,b in zip(got['derivative'],derivative):assert a==pytest.approx(list(map(float,b)),rel=0,abs=1e-12)


@pytest.mark.parametrize('case',CASES,ids=[c[0] for c in CASES])
def test_boundary_candidates_report_image_and_gradient_errors_separately(case):
    from tools.opaque_boundary_models import evaluate
    name,v,f,rgb,velocity,h,w=case
    expected,derivative=exact(v,f,rgb,velocity,h,w)
    rows={}
    for model in ('sampled_boundary','micro_edge'):
        got=evaluate(model,v,f,rgb,velocity,height=h,width=w)
        error=max(abs(float(x)-y) for a,b in zip(expected,got['image']) for x,y in zip(a,b))
        gradient_error=max(abs(float(x)-y) for a,b in zip(derivative,got['derivative']) for x,y in zip(a,b))
        rows[model]=(error,gradient_error)
    if name=='subpixel':
        # A diagnostic limitation, never a passing opaque-renderer acceptance.
        assert rows['sampled_boundary']==(1/128,1/16)
        assert rows['micro_edge']==(1/128,1/16)
    elif name in ('silhouette','overlap','intersection'):
        assert rows['sampled_boundary'][0]<=1e-12 and rows['sampled_boundary'][1]<=1e-12
        assert rows['micro_edge'][0]>=.25
        assert rows['micro_edge'][1]>=.5
    else:
        assert all(a<=1e-12 and b<=1e-12 for a,b in rows.values())


def test_native_model_rejects_unbounded_or_invalid_work():
    from tools.opaque_boundary_models import evaluate
    _,v,f,rgb,d,h,w=CASES[0]
    for kw in ({'height':0,'width':w},{'height':h,'width':10000}):
        with pytest.raises(ValueError):evaluate('complete',v,f,rgb,d,**kw)
    with pytest.raises(ValueError):evaluate('invented',v,f,rgb,d,height=h,width=w)


@pytest.mark.parametrize('case',CASES,ids=[c[0] for c in CASES])
@pytest.mark.parametrize('model',['complete','sampled_boundary','micro_edge'])
def test_global_geometry_halos_preserve_pixel_window_values_and_jvps(case,model):
    from tools.opaque_boundary_models import evaluate
    name,v,f,rgb,velocity,h,w=case
    whole=evaluate(model,v,f,rgb,velocity,height=h,width=w)
    for y in range(h):
        for x in range(w):
            tile=evaluate(model,v,f,rgb,velocity,height=1,width=1,origin=(x,y))
            assert tile['image'][0]==whole['image'][y*w+x]
            assert tile['derivative'][0]==whole['derivative'][y*w+x]
