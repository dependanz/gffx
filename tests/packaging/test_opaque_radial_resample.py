"""Diagnostic resampling is affine row reconstruction, not a new certificate."""
import numpy as np
import pytest
N=np.array([.0469100770306680036,.2307653449471584545,.5,.7692346550528415455,.9530899229693319964])
W=np.array([.1184634425280945438,.2393143352496832340,.2844444444444444444,.2393143352496832340,.1184634425280945438])
def fixture():return np.column_stack((1-N,N,N*0,(1+2*N)*W))
@pytest.mark.parametrize('order',[5,8,12,20])
def test_radial_rows_keep_closed_polynomial_integrals(order):
    from tools.opaque_radial_resample import resample_rows
    q=resample_rows(fixture(),order)
    for degree in range(8):assert abs(np.sum(q[:,1]**degree*q[:,3])-(1/(degree+1)+2/(degree+2)))<2e-14
    if order==5:assert np.array_equal(q,fixture())
@pytest.mark.parametrize('kind',['shape','nonfinite','nonlinear','negative','order','boolean'])
def test_radial_rows_refuse_unqualified_inputs(kind):
    from tools.opaque_radial_resample import resample_rows
    q=fixture();order=8
    if kind=='shape':q=q[:4]
    elif kind=='nonfinite':q[0,0]=np.nan
    elif kind=='nonlinear':q[2,0]+=.001
    elif kind=='negative':q[:,3]*=-1
    elif kind=='order':order=7
    else:order=True
    with pytest.raises(ValueError):resample_rows(q,order)

@pytest.mark.parametrize('mapping',['square_left','square_right','smooth3','smooth5'])
def test_radial_mapping_preserves_analytic_measure_and_linear_moment(mapping):
    from tools.opaque_radial_resample import mapped_rows
    q=mapped_rows(fixture(),mapping)
    assert q.shape==fixture().shape and (q[:,3]>0).all()
    assert (q[:,1]>0).all() and (q[:,1]<1).all() and (np.diff(q[:,1])>0).all()
    assert abs(q[:,3].sum()-2)<2e-14
    # Integrand barycentric u with affine measure: integral u(1+2u) du.
    # smooth5 has degree14 here, so test its exact constant-density first moment.
    if mapping!='smooth5':assert abs((q[:,1]*q[:,3]).sum()-7/6)<2e-14
    flat=fixture();flat[:,3]=W
    q=mapped_rows(flat,mapping)
    assert abs((q[:,1]*q[:,3]).sum()-.5)<2e-14

def test_radial_mapping_rejects_unknown_and_nonaffine():
    from tools.opaque_radial_resample import mapped_rows
    with pytest.raises(ValueError):mapped_rows(fixture(),'unknown')
    q=fixture();q[2,1]+=.1
    with pytest.raises(ValueError):mapped_rows(q,'smooth3')


def test_horizon_mapping_keeps_ordinary_and_selects_both_curved_sides():
    from tools.opaque_radial_resample import horizon_rows,mapped_rows
    q=fixture();q[:,3]=W
    assert np.array_equal(horizon_rows(q),q)
    lower=fixture();lower[:,3]=N*W
    assert np.array_equal(horizon_rows(lower),mapped_rows(lower,'square_right'))
    upper=fixture();upper[:,3]=(1+N)*W
    assert np.array_equal(horizon_rows(upper),mapped_rows(upper,'square_left'))
