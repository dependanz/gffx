"""Outward first spatial derivative of a normalized affine normal field.

For n=A+t(B+sC), use (n x l).(n x dn)/(|n|^3 |l|).
Compute A x B, A x C and B x C in exact rational arithmetic first. This
cancels parallel terms algebraically, not by dropping small intervals.
Only first spatial or parameter coefficients are returned. Parameter bounds
retain both intrinsic normal motion and coordinate motion. Mixed and higher
coefficients remain governed by the full normalized Jet.
"""
from tools.opaque_shading_intervals import I,Jet


def normal_light_spatial_bound(coefficients,t,s):
    return _normal_light_bound(coefficients,t,s,(1,0))

def normal_light_tangent_bound(coefficients,t,s):
    return _normal_light_bound(coefficients,t,s,(0,1))

def _normal_light_bound(coefficients,t,s,key):
    if not isinstance(t,Jet) or not isinstance(s,Jet) or t.order!=s.order or key==(1,0) and t.order<1:raise ValueError('matching spatial jets required')
    if len(coefficients)!=3 or any(len(row)!=3 for row in coefficients):raise ValueError('three normal components with affine radial coefficients required')
    cross=lambda a,b:[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
    a,b,c=[[row[i][0] for row in coefficients] for i in range(3)]
    ab,ac,bc=[list(map(I,cross(x,y))) for x,y in [(a,b),(a,c),(b,c)]]
    tv,sv,dt,ds=t[0,0],s[0,0],t[key],s[key]
    n=[tv*(sv*I(z)+I(y))+I(x) for x,y,z in zip(a,b,c)]
    nd=[(y*sv+x)*dt+(y*tv+z*tv**2)*ds for x,y,z in zip(ab,ac,bc)]
    if key==(0,1):
        da,db,dc=[[row[i][1] for row in coefficients] for i in range(3)]
        aa=cross(a,da);bb=cross(b,db);cc=cross(c,dc)
        add=lambda x,y:[v+w for v,w in zip(x,y)]
        ab1=add(cross(a,db),cross(b,da));ac1=add(cross(a,dc),cross(c,da));bc1=add(cross(b,dc),cross(c,db))
        nd=[v+I(aa[i])+tv*(I(ab1[i])+sv*I(ac1[i]))+tv**2*(I(bb[i])+sv*I(bc1[i])+sv**2*I(cc[i])) for i,v in enumerate(nd)]
    nl=cross(n,list(map(I,(1,1,2))));nn=sum((x**2 for x in n),I(0))
    if nn.lo<=0:raise ValueError('positive normal norm required')
    return sum((x*y for x,y in zip(nl,nd)),I(0))/(6*nn**3).sqrt()
