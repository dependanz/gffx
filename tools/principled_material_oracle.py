"""Independent high-precision one-pair algebra over frozen loss-table constants.

Development-only, no native calls or torch graph. Bernstein polynomials and exact
Decimal roots/exp/log avoid double subtraction noise in two-step differences.
This checks composition, not new physical-table admission or a renderer oracle.
"""
from decimal import Decimal,localcontext


def material_pair(data,parameters,two_sided=False):
    if len(parameters)!=16:raise ValueError('exactly16 scalar inputs required')
    D=lambda x:x if isinstance(x,Decimal) else Decimal.from_float(float(x))
    with localcontext() as ctx:
        ctx.prec=90
        p=list(map(D,parameters));zero=Decimal(0);one=Decimal(1);three=Decimal(3)
        # Pi is the exact stored binary64 operand of the qualified expression.
        pi=D(float.fromhex('0x1.921fb54442d18p+1'))
        def dot(a,b):return sum((x*y for x,y in zip(a,b)),zero)
        def unit(v):
            norm=dot(v,v).sqrt()
            if not norm:raise ValueError('zero vector')
            return [x/norm for x in v]
        nn,vv,ll=(unit(p[a:a+3]) for a in (0,3,6))
        if two_sided and dot(nn,vv)<0:nn=[-x for x in nn]
        nv,nl=dot(nn,vv),dot(nn,ll)
        if nv<=0 or nl<=0:return [zero]*3
        hh=unit([x+y for x,y in zip(vv,ll)]);nh,vh=dot(nn,hh),dot(vv,hh);roughness=p[12]
        rn=list(map(D,data['roughness_nodes']));rx=list(map(D,data['_rlog']));cn=list(map(D,data['cosine_nodes'][1:]));x=list(map(D,data['_clog']));n=len(x)
        if not 1<=n<=512 or not rn[0]<=roughness<=rn[-1]:raise ValueError('resource domain')
        index=min(sum(v<=roughness for v in rn)-1,len(rn)-2);width=rx[index+1]-rx[index];u=(roughness.ln()-rx[index])/width
        def bernstein(b,t):
            s=1-t
            return s**3*b[0]+3*s*s*t*b[1]+3*s*t*t*b[2]+t**3*b[3]
        def interpolate(y0,y1,m0,m1):
            a,b,c,d=map(D,(y0,y1,m0,m1))
            return bernstein([a,a+width*c/3,b-width*d/3,b],u).exp()
        values=[[interpolate(data['_logs'][index][j][c],data['_logs'][index+1][j][c],data['_slopes'][index][j][c],data['_slopes'][index+1][j][c]) for c in range(2)] for j in range(n)]
        horizon=[interpolate(data['_hlogs'][index],data['_hlogs'][index+1],data['_hslopes'][index],data['_hslopes'][index+1]),zero]
        def slope(i,c):
            if n==1:return zero
            sec=lambda j:(values[j+1][c]-values[j][c])/(x[j+1]-x[j])
            before=sec(max(0,i-1));after=sec(min(i,n-2))
            if n==2:return before
            if i==0:
                h0,h1=x[1]-x[0],x[2]-x[1];raw=((2*h0+h1)*before-h0*sec(1))/(h0+h1)
            elif i==n-1:
                h0,h1=x[-1]-x[-2],x[-2]-x[-3];raw=((2*h0+h1)*after-h0*sec(n-3))/(h0+h1)
            else:
                hp,hn=x[i]-x[i-1],x[i+1]-x[i];raw=(hn*before+hp*after)/(hp+hn)
            sign=one if before>0 and after>0 else -one if before<0 and after<0 else zero
            return sign*min(max(sign*raw,zero),Decimal('1.5')*min(abs(before),abs(after)))
        slopes=[]
        for i in range(n):
            m=[slope(i,c) for c in range(2)];delta=m[0]-m[1];gap=values[i][0]-values[i][1]
            risk=delta*(x[i]-x[i-1] if i else zero) if delta>0 else -delta*(x[i+1]-x[i] if i<n-1 else zero)
            scale=min(one,3*gap/risk) if risk>0 else one
            if scale<1:scale*=one-Decimal(8)*Decimal(2)**-52
            slopes.append([v*scale for v in m])
        controls=[[[values[i][c],values[i][c]+(x[i+1]-x[i])*slopes[i][c]/3,values[i+1][c]-(x[i+1]-x[i])*slopes[i+1][c]/3,values[i+1][c]] for c in range(2)] for i in range(n-1)]
        def directional(c):
            if n==1 or c<cn[0]:return [horizon[j]+c/cn[0]*(values[0][j]-horizon[j]) for j in range(2)]
            i=min(sum(v<=c for v in cn)-1,n-2);t=(c.ln()-x[i])/(x[i+1]-x[i])
            return [bernstein(controls[i][j],t) for j in range(2)]
        mean=[cn[0]**2*(horizon[c]+2*values[0][c])/three+sum((2*cn[i+1]**2*(x[i+1]-x[i])*controls[i][c][k]*D(data['_weights'][i][k]) for i in range(n-1) for k in range(4)),zero) for c in range(2)]
        view,light=directional(nv),directional(nl);a4=roughness**4
        distribution=(a4/((1-nh)*(1+nh)+a4*nh*nh)**2)/pi
        visibility=Decimal('.5')/(nl*(nv*nv*(1-a4)+a4).sqrt()+nv*(nl*nl*(1-a4)+a4).sqrt())
        result=[]
        for c in range(3):
            f=p[13+c];mix=lambda z:(1-f)*z[0]+f*z[1]
            result.append(distribution*visibility*(f+(1-f)*(1-vh)**5)+p[9+c]*mix(view)*mix(light)/(pi*mix(mean)))
        return result
