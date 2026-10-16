"""Independent pixel integral for a constant-normal moving scene view horizon."""
from tools.opaque_material_reference import partition,integrate_partition

def view_pixel():
    pi=float.fromhex('0x1.921fb54442d18p+1')
    pieces=partition([[(-1,0),(0,0),(.5,1)]])
    def field(sides):
        def evaluate(x,y):
            if sides[0]<0:return [x.lift(0)]*3
            theta=x.lift(0,tangent=1);vx=1-2*x;vy=(2*y-1)*.01;vz=x.lift(2)
            length=(vx*vx+vy*vy+vz*vz).sqrt();normal_length=(1+theta*theta).sqrt()
            nv=(vx+2*theta)/(normal_length*length);nl=1/normal_length
            value=nl/(2*pi*(nv+nl));return [value]*3
        return evaluate
    return integrate_partition(pieces,field,gauss_points=5)
