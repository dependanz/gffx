"""Closed controls for certified approximate angular partitions."""
from fractions import Fraction as Q
import pytest
from tools.opaque_shading_intervals import I,Jet

def test_affine_surrogate_exact_polynomial_and_curvature_bound():
    from tools.opaque_branch_partition import affine_enclosure
    tri=[(Q(0),Q(0)),(Q(1),Q(0)),(Q(0),Q(1))]
    field=lambda x,y:[2+3*x-4*y+x*x/100+2*x*y/100+y*y/100]
    model=affine_enclosure(field,tri)[0]
    for x,y in [(Q(0),Q(0)),(Q(1),Q(0)),(Q(0),Q(1)),(Q(1,4),Q(1,3))]:
        exact=2+3*x-4*y+(x*x+2*x*y+y*y)/100
        approx=sum(a*b for a,b in zip(model['plane'],[x,y,Q(1)]))
        assert abs(exact-approx)<=Q(model['error'].hi)
    assert model['error'].hi<I('.02').lo

def test_exact_affine_partition_preserves_area_and_tiny_slivers():
    from tools.opaque_branch_partition import partition,area,band_area
    tri=[(Q(0),Q(0)),(Q(1),Q(0)),(Q(0),Q(1))]
    planes=[(Q(1),Q(0),-Q(1,2)),(Q(0),Q(1),-Q(1,3)),(Q(1),Q(0),-Q(1,10**12))]
    cells=partition(tri,planes)
    assert sum(area(p) for p in cells)==Q(1,2)
    assert min(area(p) for p in cells)>0
    assert band_area(tri,(Q(1),Q(0),-Q(1,2)),Q(1,10))==Q(1,10)

def test_partition_zero_constant_plane_has_one_owner():
    from tools.opaque_branch_partition import partition,area
    tri=[(Q(0),Q(0)),(Q(1),Q(0)),(Q(0),Q(1))]
    assert len(partition(tri,[(Q(0),Q(0),Q(0))]))==1
    assert sum(area(p) for p in partition(tri,[(Q(0),Q(0),Q(1))]))==Q(1,2)

def test_partition_limits_and_cancellation_are_enforced():
    from tools.opaque_branch_partition import partition
    from tools.opaque_shading_intervals import Unresolved
    tri=[(Q(0),Q(0)),(Q(1),Q(0)),(Q(0),Q(1))]
    with pytest.raises(ValueError,match='plane cap'):
        partition(tri,[(Q(1),Q(0),Q(0))]*17)
    with pytest.raises(ValueError,match='rational complexity'):
        partition(tri,[(Q(2**8193),Q(0),Q(0))])
    def stop():raise Unresolved('caller stopped')
    with pytest.raises(Unresolved,match='caller stopped'):partition(tri,[],checkpoint=stop)

def test_mismatch_error_retains_tiny_jump_and_quadratic_term():
    from tools.opaque_branch_partition import join_bound
    delta=lambda q:[q.lift(I('1e-17'))+3*(q-Q(1,2))**2]
    b=join_bound(delta,Q(1,2),I('1e-5'))[0]
    assert b.lo>=0 and b.hi>=I('3.0000001e-10').hi
    assert b.hi<I('3.000001e-10').lo

def test_affine_enclosure_rejects_unqualified_parameter_tangents():
    from tools.opaque_branch_partition import affine_enclosure
    with pytest.raises(ValueError,match='fixed geometry'):
        affine_enclosure(lambda x,y:[x+x.lift(1,tangent=1)],[(Q(0),Q(0)),(Q(1),Q(0)),(Q(0),Q(1))])


def test_simplex_cosines_match_independent_normalization_formula():
    from types import SimpleNamespace
    from tools.opaque_peano_material import radial_cosines
    from tools.opaque_moving_reference import P
    z=P(0)
    fields=SimpleNamespace(coefficients={'n':[[z,P(1),z],[z,z,z],[P(1),z,z]],'v':[[z,z,z],[z,z,z],[P(2),z,z]]})
    f=radial_cosines(fields,[0,1,1],simplex=True)
    x=Jet.variable(I('.2'),order=3);y=Jet(I('.3'),order=3)
    nv,nl,vh=f(x,y);known=(1+(x+y)**2).sqrt().reciprocal()
    for i in range(4):assert nv[i,0].lo<=known[i,0].hi and nv[i,0].hi>=known[i,0].lo

def test_join_bound_uses_outward_sub_ulp_interval_endpoints():
    from tools.opaque_branch_partition import join_bound
    k=Q(1,3);radius=I('1e-40');seen=[]
    def difference(q):seen.append(q[0,0]);return [q**3]
    join_bound(difference,k,radius)
    assert Q(seen[-1].lo)<=k-Q(radius.hi) and Q(seen[-1].hi)>=k+Q(radius.hi)


def test_bounded_partition_refinement_preserves_area_error_and_depth():
    from tools.opaque_branch_partition import refine_partition,area
    part={'triangle':[(Q(0),Q(0)),(Q(1),Q(0)),(Q(0),Q(1))],'face':3};visits=[]
    def assess(p):
        mass=area(p['triangle']);return {'error':I(mass*mass), 'pieces':[]}
    rows=refine_partition(part,assess,I('.1'),tick=lambda:visits.append(1))
    assert sum(area(r['part']['triangle']) for r in rows)==Q(1,2)
    assert sum(Q(r['certificate']['error'].hi) for r in rows)<=Q(1,10)
    assert all(r['depth']<=10 and r['part']['face']==3 for r in rows)
    assert len(visits)<=256

def test_partition_refinement_propagates_deadlines_and_depth_refusals():
    from tools.opaque_branch_partition import refine_partition
    from tools.opaque_shading_intervals import Unresolved
    part={'triangle':[(Q(0),Q(0)),(Q(1),Q(0)),(Q(0),Q(1))]};calls=[]
    def stop(p):calls.append(1);raise Unresolved('external deadline')
    with pytest.raises(Unresolved,match='external deadline'):refine_partition(part,stop,I('1e-12'))
    assert len(calls)==1
    with pytest.raises(Unresolved,match='partition depth'):
        refine_partition(part,lambda p:{'error':I(1),'pieces':[]},I('1e-12'),max_depth=0)


def test_displaced_branch_integral_encloses_closed_nonlinear_boundary():
    """Integrate the omitted strip analytically, including a value and slope jump."""
    from tools.opaque_branch_partition import band_area,join_bound
    triangles=[[(Q(0),Q(0)),(Q(1),Q(0)),(Q(1),Q(1))],[(Q(0),Q(0)),(Q(1),Q(1)),(Q(0),Q(1))]]
    epsilon=Q(1,100);j0=Q(1,10**17);j1=Q(1,10**12);c=Q(3)
    # Actual q=x+epsilon*y**2 chooses the positive polynomial above q=1/2.
    # Surrogate L=x chooses it above x=1/2. Their only difference is the
    # strip 1/2-epsilon*y**2 <= x <= 1/2. Integrate in x,then y exactly.
    exact=j0*epsilon/3+j1*epsilon**2/10+c*epsilon**3/21
    bound=join_bound(lambda q:[q.lift(j0)+j1*(q-Q(1,2))+c*(q-Q(1,2))**2],Q(1,2),I(epsilon))[0]
    mass=sum(band_area(tri,(Q(1),Q(0),-Q(1,2)),epsilon) for tri in triangles)
    enclosure=I(mass)*bound
    assert 0<exact<=Q(enclosure.hi)
    assert mass==2*epsilon and exact>c*epsilon**3/21
