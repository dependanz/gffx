"""Independent Principled pixel enclosures and installed point-kernel comparisons."""
from decimal import Decimal,localcontext
from fractions import Fraction as Q
import json,pytest,torch

def api():
    from tools.opaque_principled_reference import PreparedPrincipled,perspective_field,integrate_pixel
    return PreparedPrincipled,perspective_field,integrate_pixel

@pytest.fixture(scope='module')
def resource():
    from gffx.torch.material import PrincipledBudget
    b=PrincipledBudget.bundled('ggx-smith-loss-cpu-f64-v1',device='cpu',dtype=torch.float64)
    return b,{k:v.tolist() for k,v in b._data.items()}

def test_principled_interval_point_contains_independent_decimal_formula(resource):
    Prepared,_,_=api();from tools.opaque_shading_intervals import Jet
    from tools.principled_material_oracle import material_pair
    values=[.01,.02,1., -.07,.06,1., .1,.1,1., .2,.4,.6, .827,.04,.05,.08]
    curve=Prepared(resource[1],values[12]);p=[Jet(x) for x in values]
    got=curve.radiance(p[:3],p[3:6],p[6:9],p[9:12],p[13:16]);ref=material_pair(resource[1],values)
    with localcontext() as c:
        c.prec=90;D=Decimal.from_float;nn=list(map(D,values[:3]));ll=list(map(D,values[6:9]));cos=sum(a*b for a,b in zip(nn,ll))/(sum(a*a for a in nn)*sum(a*a for a in ll)).sqrt()
        for a,b in zip(got,ref):assert a[0,0].lo<=b*cos<=a[0,0].hi

def test_principled_pixel_bound_detects_vertex_colour_surrogate(resource,record_property):
    Prepared,field,integrate=api();from tools.opaque_shading_intervals import Jet
    tri=[(0,0,0),(1,0,0),(0,1,0)]
    attrs=dict(inverse_w=[1.,.85,1.1],normals=[[.01,.01,1.],[.05,.01,1.],[.01,.04,1.]],views=[[-.07,.06,1.],[-.03,.06,1.],[-.07,.08,1.]],albedo=[[.2,.4,.6],[.6,.2,.4],[.3,.7,.2]])
    fn=field(Prepared(resource[1],.827),tri,**attrs,f0=[.04,.05,.08],direction=[.1,.1,1.],tangent={'normals':[[.01,0,0]]*3,'albedo':[[.05,0,0]]*3})
    row=integrate(tri,[(0,1,2)],[fn],tolerance='1e-12')
    vals=row['value'][0][0];grads=row['derivative'][0][0]
    surrogate=[sum(fn(Jet(x),Jet(y))[k][0,0].lo for x,y,z in tri)/6 for k in range(3)]
    assert any(not a.lo<=b<=a.hi for a,b in zip(vals,surrogate))
    assert all(a.width<=Decimal('2e-12') for a in vals+grads)
    record_property('validated_pixel',json.dumps(row,default=str))

def test_principled_reference_refuses_knot_and_singular_profile(resource):
    Prepared,_,_=api();from tools.opaque_shading_intervals import Jet,Interval
    c=Prepared(resource[1],.827)
    with pytest.raises(ValueError):c.loss(Jet(Interval('.90','.99')))
    with pytest.raises(ValueError):c.radiance([Jet(0)]*3,[Jet(0),Jet(0),Jet(1)],[Jet(0),Jet(0),Jet(1)],[Jet('.5')]*3,[Jet('.04')]*3)
