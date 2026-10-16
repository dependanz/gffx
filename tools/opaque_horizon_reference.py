"""Finite Principled horizon reference: roughness=F0=1, albedo=0.

View horizon has a radiance jump; light horizon is continuous. The reference
integrates exact moving regions and checks a separate90-digit antiderivative.
"""
from fractions import Fraction as Q
from decimal import Decimal as D,localcontext
from tools.opaque_shading_intervals import I
from tools.opaque_material_reference import partition,integrate_partition

def horizon_pixel(kind,edge):
    edge=Q(edge)
    if kind not in ('view','light') or not 0<edge<1:raise ValueError('finite interior horizon profile')
    pi=float.fromhex('0x1.921fb54442d18p+1')
    regions=partition([[(1,0),(0,0),(-edge,-1)]])
    def field(sides):
        def evaluate(x,y):
            if sides[0]<0:return [x.lift(0)]*3
            s=x-x.lift(edge,tangent=1);cos=s/(1+s*s).sqrt();a=1/(2*pi*(1+cos))
            if kind=='light':a=a*cos
            return [a]*3
        return evaluate
    reference=integrate_partition(regions,field,gauss_points=5)
    with localcontext() as c:
        c.prec=90;b=D((1-edge).numerator)/D((1-edge).denominator);den=2*D.from_float(pi);r=(1+b*b).sqrt()
        value=(b+b**3/3-r**3/3+D(1)/3)/den;derivative=-(1+b*b-b*r)/den
        if kind=='light':value=b/den-value;derivative=-1/den-derivative
        for actual,wanted in ((reference['value'][0],value),(reference['derivative'][0],derivative)):
            assert actual.lo<=wanted<=actual.hi
        reference['closed_value']=str(value);reference['closed_derivative']=str(derivative)
        reference['boundary_flux']=str(-1/den if kind=='view' else D(0))
    return reference
