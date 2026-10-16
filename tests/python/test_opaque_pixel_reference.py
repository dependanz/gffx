"""Independent smooth pixel enclosure for the installed crowded-scene path."""
import json,torch
from decimal import Decimal as D,localcontext
from test_opaque_pixel_batches import crowded,call
from test_opaque_nonlinear_scene import fixture

def test_crowded_scene_matches_independent_nonlinear_value_and_albedo_tangent(record_property):
    from tools.opaque_principled_reference import PreparedPrincipled,perspective_field,integrate_pixel
    _,_,v,f,b=fixture();data={k:a.tolist() for k,a in b._data.items()};curve=PreparedPrincipled(data,.827)
    pixels=[(0,0,0),(1,0,0),(1,1,0),(0,1,0)];fields=[]
    for face in f.tolist():
        fields.append(perspective_field(curve,[pixels[i] for i in face],inverse_w=[.5]*3,normals=[[.01,.01,1.]]*3,views=[(-v[i]).tolist() for i in face],albedo=[[.2,.4,.6]]*3,f0=[.04,.05,.08],direction=[.1,.1,1.],tangent={'albedo':[[1.,1.,1.]]*3}))
    reference=integrate_pixel(pixels,f.tolist(),fields)
    with localcontext() as context:
        context.prec=90
        for name in ('value','derivative'):
            assert all((a.hi-a.lo)/2<=D('1e-12') for a in reference[name][0][0])
    rows=[];byte_rows=[]
    for device in ('cpu','cuda'):
        s,c,a=crowded(device,grad=True);out=call(s,c);gradient,=torch.autograd.grad(out['colour'].sum(),a)
        for name,observed in (('value',out['colour'][0,0,0]),('derivative',gradient)):
            for interval,value in zip(reference[name][0][0],observed.tolist()):
                with localcontext() as context:
                    context.prec=90;x=D.from_float(value);assert max(interval.lo-x,x-interval.hi,D(0))<=D('1e-10')+D('1e-9')*abs((interval.lo+interval.hi)/2)
        byte_rows.append((out['colour'].detach().cpu().numpy().tobytes(),out['mask'].detach().cpu().numpy().tobytes(),gradient.detach().cpu().numpy().tobytes()))
        rows.append(dict(device=device,colour=out['colour'].tolist(),mask=out['mask'].tolist(),gradient=gradient.tolist()))
    assert byte_rows[0]==byte_rows[1]
    record_property('independent_batch_pixel',json.dumps(dict(reference=reference,observed=rows),default=str))
