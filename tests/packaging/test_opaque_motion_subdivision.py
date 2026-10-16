"""Independent controls for safe partial-monotonicity domain tightening."""
from fractions import Fraction as Q
import pytest
from tools.opaque_shading_intervals import I,Jet
from tools.opaque_loss_extension import bounded_cosines

@pytest.mark.parametrize('axis',[0,1])
@pytest.mark.parametrize('reverse',[False,True])
def test_one_monotone_axis_uses_edges_not_unproved_corners(axis,reverse):
    def field(a,b):
        x,y=(a,b) if axis==0 else (b,a)
        if reverse:x=1-x
        return [x+(x-1)*y*y]
    box=(I(0,1),I(-1,1)) if axis==0 else (I(-1,1),I(0,1))
    loose=bounded_cosines(field,*box)[0];tight=bounded_cosines(field,*box,monotone=True,partial=True)[0]
    assert loose.hi>1
    assert tight.lo==-1 and tight.hi==1
    # Interior y=0 attains +1. Using only corner values on the unproved axis
    # would miss that maximum at x=0 after reversing the monotone axis.
    for x in (Q(0),Q(1,2),Q(1)):
        for y in (Q(-1),Q(0),Q(1)):
            z=field(Jet(x),Jet(y))[0][0,0] if axis==0 else field(Jet(y),Jet(x))[0][0,0]
            assert tight.lo<=z.lo<=z.hi<=tight.hi

def test_unproved_axes_preserve_interior_extremum_and_parameter_tangent():
    def field(x,y):return [(1-x*x)*(1-y*y)+x.lift(0,tangent=Q(1,10**70))]
    got=bounded_cosines(field,I(-1,1),I(-1,1),monotone=True,partial=True)[0]
    assert got.lo<=0 and got.hi>=1
    assert field(Jet(0),Jet(0))[0][0,1].lo>0


def test_saved_segment_subdivision_diagnostic(record_property):
    import json,os,time
    from pathlib import Path
    from tools.opaque_original_motion_reference import prepare,mapped_inputs
    from tools.opaque_principled_reference import PreparedPrincipled
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    start=time.monotonic();data=frozen();rows=[]
    with backend(profile='ball'):
        p=prepare(data,piece=8,motion='normal_tilt',anchor=0,horizon_power=8);curve=PreparedPrincipled(data['budget'],1.)
        def evaluate(coords):
            assert time.monotonic()-start<90
            a,b,c,d=coords;directions=[]
            def field(x,y):
                vs=list(mapped_inputs(p,x,y)[:2])
                for v in vs:v.c[0,0]=I(max(0,v[0,0].lo),min(1,v[0,0].hi))
                return vs
            bs=bounded_cosines(field,I(I(a).lo,I(b).hi),I(I(c).lo,I(d).hi),monotone=True,partial=True,directional=directions.append)
            counts=[int(v.lo<=curve.cn[0].hi)+sum(v.lo<=curve.cn[k+1].hi and v.hi>=curve.cn[k].lo for k in range(len(curve.cn)-1)) for v in bs]
            return dict(box=list(map(str,coords)),bounds=list(map(str,bs)),counts=counts,directions=[[str(v) for v in row] for row in directions[0]])
        for box in [(Q(19,64),Q(39,128),Q(0),Q(1,4)),(Q(19,64),Q(39,128),Q(0),Q(1,2)),(Q(1,4),Q(1,2),Q(0),Q(1))]:
            a,b,c,d=box;x=(a+b)/2;y=(c+d)/2
            rows.append(dict(parent=evaluate(box),children=[[evaluate(z) for z in [(a,x,c,d),(x,b,c,d)]],[evaluate(z) for z in [(a,b,c,y),(a,b,y,d)]]]))
    path=Path(os.environ['GFFX_FULL_REFERENCE']);assert not path.exists();path.write_text(json.dumps(dict(status='diagnostic_only',rows=rows,seconds=time.monotonic()-start),indent=2),encoding='utf8');record_property('domain_diagnostic',path.read_text())


@pytest.mark.parametrize('inside',[False,True])
def test_horizon_split_retains_exact_partition_and_ordinary_midpoints(inside):
    from tools.opaque_original_motion_reference import horizon_split
    for a,b in [(Q(0),Q(1)),(Q(0),Q(1,8)),(Q(1,4),Q(3,4)),(Q(3,4),Q(1))]:
        m=horizon_split(a,b,inside=inside)
        assert type(m)==Q and a<m<b and (m-a)+(b-m)==b-a
        lo,hi=(1-b,1-a) if inside else (a,b)
        t=hi/4 if lo==0 else 2*lo*hi/(lo+hi)
        expected=1-t if inside else t
        assert m==expected
    with pytest.raises(ValueError):horizon_split(Q(0),Q(1),inside=None)


def test_unqualified_graded_integrator_is_refused_before_input_access():
    from tools.opaque_original_motion_reference import reference
    with pytest.raises(ValueError,match='domain-preflight only'):reference({},piece=8,motion='normal_tilt',graded=True)
    with pytest.raises(ValueError,match='explicit graded'):reference({},piece=8,motion='normal_tilt',graded=1)


def test_partial_monotonicity_requires_explicit_flag():
    for kwargs in [dict(partial=True),dict(monotone=True,partial=1)]:
        with pytest.raises(ValueError,match='partial monotonicity'):bounded_cosines(lambda x,y:[],I(0,1),I(0,1),**kwargs)
