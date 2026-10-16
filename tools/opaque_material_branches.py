"""Certified constant-side material branches; crossings explicitly unresolved.
The selected hemisphere is an interval fact, never a sampled midpoint.
"""
def radiance(curve,n,v,l,albedo,f0,*,two_sided=False):
    if type(two_sided)!=bool:raise TypeError('two_sided must be bool')
    dot=lambda a,b:sum((x*y for x,y in zip(a,b)),n[0].lift(0))
    # Signs of unnormalized dot products equal signs after normalization. Reject
    # zero vector domains even on a back-facing branch, matching the public API.
    for vector in (n,v,l):
        if dot(vector,vector)[0,0].lo<=0:raise ValueError('material branch vector norm unresolved')
    nv=dot(n,v)
    if two_sided:
        if nv[0,0].hi<0:n=[-x for x in n];nv=-nv
        elif nv[0,0].lo<=0:raise ValueError('two-sided horizon branch unresolved')
    nl=dot(n,l)
    if nv[0,0].hi<0 or nl[0,0].hi<0:return [n[0].lift(0)]*3
    if nv[0,0].lo<=0 or nl[0,0].lo<=0:raise ValueError('horizon branch unresolved')
    return curve.radiance(n,v,l,albedo,f0)
