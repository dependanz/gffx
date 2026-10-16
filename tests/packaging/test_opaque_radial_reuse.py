"""Original conic mapping and numerator, every coefficient unchanged."""
import pytest
from tools.opaque_shading_intervals import I,Jet
from tools.opaque_curved_reference import flattened_radial_map
from tools.opaque_peano_material import radial_cosines
from tests.packaging.test_opaque_horizon_factor import setup

@pytest.mark.parametrize('order',[0,1,3,6,10])
@pytest.mark.parametrize('axis',[0,1])
@pytest.mark.parametrize('positive',[False,True])
@pytest.mark.parametrize('power',[4,8])
def test_shared_radial_root_preserves_all_coefficients(order,axis,positive,power):
    p,f=setup(True,positive);fn=radial_cosines(f,[1,1,2],horizon_patch=p)
    args=[Jet(I('.51','.52'),order=order),Jet(I('.31','.32'),order=order)]
    if order:args[axis]=Jet.variable(args[axis][0,0],order=order)
    args[0].c[0,1]=I('.25')
    t,j,d=flattened_radial_map(p,*args,power=power);old=(*fn(t,args[1],horizon_distance=d),j)
    t,j,d,n=flattened_radial_map(p,*args,power=power,with_numerator=True)
    new=(*fn(t,args[1],horizon_distance=d,horizon_numerator=n),j)
    for a,b in zip(old,new):
        assert set(a.c)==set(b.c)
        assert all((a[k].lo,a[k].hi)==(b[k].lo,b[k].hi) for k in a.c)

def test_shared_root_computes_discriminant_once(monkeypatch):
    p,f=setup();calls=[];sqrt=Jet.sqrt
    def count(q):calls.append(1);return sqrt(q)
    monkeypatch.setattr(Jet,'sqrt',count)
    t,j,d,n=flattened_radial_map(p,Jet('.5'),Jet('.3'),power=4,with_numerator=True)
    assert len(calls)==1
