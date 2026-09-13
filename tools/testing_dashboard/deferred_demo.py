"""Finite built-in studio scene. Real renderer pixels, no path-tracing claim."""
import math
import torch


def sphere():
    """224 outward triangles and explicit smooth normals, no pole degeneracy."""
    rings,segments=8,16
    points=[[0.,1.,0.]]
    for ring in range(1,rings):
        theta=math.pi*ring/rings
        for j in range(segments):
            phi=2*math.pi*j/segments
            points.append([math.sin(theta)*math.cos(phi),math.cos(theta),math.sin(theta)*math.sin(phi)])
    bottom=len(points);points.append([0.,-1.,0.]);faces=[]
    for j in range(segments):faces.append([0,1+(j+1)%segments,1+j])
    for ring in range(rings-2):
        a=1+ring*segments;b=a+segments
        for j in range(segments):
            k=(j+1)%segments;faces.extend(([a+j,a+k,b+j],[a+k,b+k,b+j]))
    for j in range(segments):faces.append([bottom-segments+j,bottom-segments+(j+1)%segments,bottom])
    normals=torch.tensor(points,dtype=torch.float64)
    return normals*.55+normals.new_tensor([-.45,.05,-3.]),torch.tensor(faces,dtype=torch.int32),normals


def ribbon():
    points=[];normals=[];colours=[];faces=[]
    for i in range(17):
        u=-1+i/8;twist=.6*math.pi*u
        for side in (-1,1):
            points.append([.5+.45*u,-.05+.15*math.sin(math.pi*u)+side*.26*math.cos(twist),
                           -3.15+.2*math.cos(math.pi*u)+side*.26*math.sin(twist)])
            du=(.45,.15*math.pi*math.cos(math.pi*u)-side*.26*math.sin(twist)*.6*math.pi,
                -.2*math.pi*math.sin(math.pi*u)+side*.26*math.cos(twist)*.6*math.pi)
            dv=(0.,.26*math.cos(twist),.26*math.sin(twist))
            n=(du[1]*dv[2]-du[2]*dv[1],-du[0]*dv[2],du[0]*dv[1]);length=math.sqrt(sum(x*x for x in n))
            normals.append([x/length for x in n]);colours.append([.04,.22+.12*(u+1)/2,.25+.25*(1-u)/2])
    for i in range(16):faces.extend(([2*i,2*i+2,2*i+1],[2*i+2,2*i+3,2*i+1]))
    return (torch.tensor(points,dtype=torch.float64),torch.tensor(faces,dtype=torch.int32),
            torch.tensor(normals,dtype=torch.float64),torch.tensor(colours,dtype=torch.float64))


def studio(size,angle,budget):
    from gffx.torch import scene,material,light,render
    from gffx.torch.camera import Camera
    def t(value):return torch.tensor(value,dtype=torch.float64)
    sv,sf,sn=sphere();rv,rf,rn,rc=ribbon()
    gold=scene.Material('gold',material.principled,dict(albedo=t([.025,.015,.004]),roughness=t(.36),
                        f0=t([.82,.56,.18]),budget=budget))
    foil=scene.Material('foil',material.classic,dict(albedo=render.AttributeSource('colour'),
                        specular=t(0.),exponent=t(4.),two_sided=True))
    backdrop=scene.Material('backdrop',material.classic,dict(albedo=t([.055,.07,.1]),specular=t(0.),exponent=t(4.)))
    plate=t([[-1.5,-1.,-3.8],[1.5,-1.,-3.8],[1.5,1.,-3.8],[-1.5,1.,-3.8]])
    lamp=scene.Light('studio',light.directional,dict(directions=t([[-.6,.8,1.],[.8,.3,-.2]]),
                    irradiance=t([[2.4,1.9,1.5],[.6,1.,1.8]])))
    model=scene.Scene([scene.Mesh('orb',sv,sf,material='gold',attributes={'normals':scene.Attribute(sn,'vertex')}),
        scene.Mesh('ribbon',rv,rf,material='foil',attributes={'normals':scene.Attribute(rn,'vertex'),'colour':scene.Attribute(rc,'vertex')}),
        scene.Mesh('plate',plate,torch.tensor([[0,1,2],[0,2,3]],dtype=torch.int32),material='backdrop'),gold,foil,backdrop,lamp])
    target=t([0.,0.,-3.]);eye=target+t([3*math.sin(angle),.3,3*math.cos(angle)])
    z=eye-target;z=z/torch.linalg.vector_norm(z)
    right=torch.linalg.cross(t([0.,1.,0.]),z);right=right/torch.linalg.vector_norm(right)
    up=torch.linalg.cross(z,right);pose=torch.eye(4,dtype=torch.float64)
    pose[:3,:3]=torch.stack((right,up,z),dim=1);pose[:3,3]=eye
    camera=Camera.from_intrinsics(t(size*1.3),t(size*1.3),t(size/2),t(size/2),size,size,.1,10.,pose=pose[None])
    return model,camera


def payload(size=32,*,preflight_only=False,deadline=None):
    from gffx.torch import material,render
    from gffx.torch._deferred import _plan_workload
    if type(size)!=int or size not in (32,128):raise ValueError('only planned32 or prospective128 profiles')
    if size==128 and not preflight_only:raise ValueError('128 rendering requires a later measured admission')
    budget=material.PrincipledBudget.bundled('ggx-smith-loss-cpu-f64-v1',device='cpu',dtype=torch.float64)
    frames=[];estimates=[]
    with torch.no_grad():
        for angle in (-.12,0.,.12):
            if deadline is not None:
                from tools.material_loss_cost import check_deadline
                check_deadline(deadline)
            model,camera=studio(size,angle,budget)
            outputs=('colour','mask','normal','depth')
            estimates.append(_plan_workload(model,camera,1,67108864,outputs))
            if not preflight_only:
                result=render(model,camera,outputs=outputs,cull_mode=render.CULL_NONE,background=camera.fx.new_tensor([.008,.012,.022]))
                frames.append({key:value.detach() for key,value in result.items()})
    return dict(frames=frames,estimates=estimates,size=size,angles=[-.12,0.,.12],faces=258,
        scope='Finite original studio geometry through public built-in deferred rendering. No shadows, indirect light, path tracing or Cycles comparison. Linear RGB remains unscaled; display conversion must be labelled separately.')
