"""Independent closed transport and refusal controls for integrated tangent flux."""
from fractions import Fraction as Q
from decimal import Decimal as D
import pytest
from tools.opaque_shading_intervals import I,Unresolved
from tools.opaque_square_reference import integrate_square

@pytest.mark.parametrize('seeds',[(1,1),(2,2),(4,2)])
def test_moving_step_flux_sums_only_current_disjoint_leaves(seeds):
    # Integral H(x-1/3-theta) dx dy = 2/3-theta. Smooth derivative is zero.
    def flux(box):
        x,y=box
        return [-(I(y.hi)-I(y.lo)) if Q(x.lo)<=Q(1,3)<Q(x.hi) else I(0)]
    got=integrate_square(lambda x,y:[x.lift(Q(2,3))],seed_shape=seeds,tangent_correction_integrator=flux)
    assert got['value'][0].lo<=I(Q(2,3)).lo<=got['value'][0].hi
    assert got['derivative'][0].lo<=-1<=got['derivative'][0].hi
    assert got['derivative'][0].width<D('1e-40')


def test_unresolved_flux_cannot_be_hidden_by_a_zero_smooth_derivative():
    with pytest.raises(Unresolved) as caught:
        integrate_square(lambda x,y:[x.lift(0)],max_cells=1,tangent_correction_integrator=lambda box:[I(-1,1)])
    assert caught.value.details['last_complete_enclosure']['derivative'][0]==['-1','1']


def test_tiny_nonzero_integrated_flux_is_retained():
    result=integrate_square(lambda x,y:[x.lift(0)],tangent_correction_integrator=lambda box:[I('1e-70')*(I(box[0].hi)-I(box[0].lo))*(I(box[1].hi)-I(box[1].lo))])
    assert 0<result['derivative'][0].lo<=D('1e-70')<=result['derivative'][0].hi


def test_wrong_flux_channel_count_refuses():
    with pytest.raises(Unresolved) as caught:
        integrate_square(lambda x,y:[x.lift(0)],max_depth=0,tangent_correction_integrator=lambda box:[I(0),I(0)])
    assert caught.value.details['last']['domain']=='tangent correction channel/type mismatch'
