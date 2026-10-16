"""Exact variable substitution retains curved area, derivatives and limits."""
from fractions import Fraction as Q
from decimal import Decimal as D,localcontext
import pytest

@pytest.mark.parametrize('power',[4,8])
@pytest.mark.parametrize('positive',[False,True])
def test_flattened_map_has_independent_closed_area_and_boundary_motion(power,positive):
    from tools.opaque_curved_reference import CurvedTriangle,flattened_radial_map
    from tools.opaque_square_reference import integrate_square
    p=CurvedTriangle([(0,0),(1,0),(0,1)],[0,0,0,1,1,[-Q(1,4),-1]],positive=positive)
    def field(u,s):return [flattened_radial_map(p,u,s,power=power)[1]]
    r=integrate_square(field)
    value=D(15)/32 if positive else D(1)/32;derivative=-D(1)/4 if positive else D(1)/4
    assert r['value'][0].lo<=value<=r['value'][0].hi
    assert r['derivative'][0].lo<=derivative<=r['derivative'][0].hi

@pytest.mark.parametrize('power',[4,8])
@pytest.mark.parametrize('positive',[False,True])
def test_flattened_map_preserves_point_coordinate_jacobian_and_factored_cosine(power,positive):
    from tests.packaging.test_opaque_horizon_factor import setup
    from tools.opaque_curved_reference import flattened_radial_map
    from tools.opaque_shading_intervals import I,Jet
    from tools.opaque_peano_material import radial_cosines
    p,f=setup(True,positive);fn=radial_cosines(f,[1,1,2],horizon_patch=p)
    for x in [Q(0),Q(1,100),Q(1,2),Q(1)]:
        u=Jet.variable(I(x),order=10);s=Jet(Q(1,3),order=10);t,jac,distance=flattened_radial_map(p,u,s,power=power)
        z=1-(1-u)**power if p.inside else u**power;dz=power*((1-u)**(power-1) if p.inside else u**(power-1));old,oldjac=p.radial_map(z,s)
        for a,b in [(t,old),(jac,oldjac*dz)]:
            for k in set(a.c)|set(b.c):assert max(a[k].lo,b[k].lo)<=min(a[k].hi,b[k].hi)
        nv=fn(t,s,horizon_distance=distance)[0]
        if x==(1 if p.inside else 0):assert nv[0,0].zero

@pytest.mark.parametrize('power',[0,1,5,16,True,'8'])
def test_unqualified_powers_are_refused(power):
    from tests.packaging.test_opaque_horizon_factor import setup
    from tools.opaque_curved_reference import flattened_radial_map
    from tools.opaque_shading_intervals import Jet
    p,_=setup()
    with pytest.raises(ValueError):flattened_radial_map(p,Jet(0),Jet(0),power=power)

def test_original_ninth_piece_bounded_preflight(record_property):
    import os,json,hashlib
    from pathlib import Path
    from tools.opaque_full_material_reference import full_pixel
    from tools.opaque_shading_intervals import Unresolved
    source=Path(os.environ['GFFX_FULL_INPUT']);dest=Path(os.environ['GFFX_FULL_REFERENCE']);assert not dest.exists()
    assert hashlib.sha256(source.read_bytes()).hexdigest()=='a15bea40cb5cdead7994d828b57f5dcfac9f44fab0292a06e7d0a7b08719f73a'
    try:
        result=full_pixel(json.loads(source.read_bytes()),selected_piece=8)
        row=dict(status='passed',result=result,scope='Bounded selected-piece preflight only; not a campaign certificate or whole-pixel admission.')
    except Unresolved as e:
        row=dict(status='not_admitted',details=e.details);raise
    finally:
        dest.write_text(json.dumps(row,default=str,indent=2),encoding='utf8');record_property('ninth_preflight',json.dumps(row,default=str))
