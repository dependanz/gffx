"""Kinetic Atelier: finite, first-party scene presentation, not physics acceptance.

Geometry/animation are authored here. Public Scene.flatten owns hierarchical
transforms and normal transport. The existing native rasterizer and cinematic
display-lighting recipe make the stills; WebGPU uses its separate preview shader.
"""
import ctypes
import itertools
import math
import torch

FRAMES = 24
SIZE = (720, 1280)


def polygons(rows):
    vertices, faces, normals = [], [], []
    for points, outward in rows:
        p = torch.tensor(points, dtype=torch.float32)
        n = torch.linalg.cross(p[1]-p[0], p[2]-p[0])
        if float(n @ torch.tensor(outward, dtype=p.dtype)) < 0:
            p = p.flip(0); n = -n
        n = n/n.norm()
        offset = len(vertices)
        vertices.extend(p.tolist()); normals.extend([n.tolist()]*len(p))
        faces.extend([offset, offset+i, offset+i+1] for i in range(1, len(p)-1))
    return (torch.tensor(vertices), torch.tensor(faces, dtype=torch.int32),
            torch.tensor(normals))


def bevel_box(half, bevel):
    if len(half) != 3 or not 0 < bevel < min(half):
        raise ValueError('bevel must be smaller than every positive half extent')
    h = list(half); b = [x-bevel for x in h]; rows = []
    for axis in range(3):
        other = [i for i in range(3) if i != axis]
        for sign in (-1, 1):
            points = []
            for u, v in ((-1,-1),(1,-1),(1,1),(-1,1)):
                p = [0.]*3; p[axis] = sign*h[axis]
                p[other[0]], p[other[1]] = u*b[other[0]], v*b[other[1]]
                points.append(p)
            normal = [0.]*3; normal[axis] = sign; rows.append((points, normal))
    for a, c in itertools.combinations(range(3), 2):
        other = 3-a-c
        for sa, sc in itertools.product((-1,1), repeat=2):
            points = []
            for end, flip in ((-1,False),(1,False),(1,True),(-1,True)):
                p = [0.]*3; p[other] = end*b[other]
                p[a] = sa*(b[a] if flip else h[a]); p[c] = sc*(h[c] if flip else b[c])
                points.append(p)
            normal = [0.]*3; normal[a], normal[c] = sa, sc; rows.append((points, normal))
    for signs in itertools.product((-1,1), repeat=3):
        rows.append(([[signs[i]*(h[i] if i == axis else b[i]) for i in range(3)]
                     for axis in range(3)], signs))
    return polygons(rows)


def cylinder(radius, height, *, segments=32):
    rows = []
    for i in range(segments):
        a, b = i*math.tau/segments, (i+1)*math.tau/segments
        x, z, u, w = radius*math.cos(a), radius*math.sin(a), radius*math.cos(b), radius*math.sin(b)
        rows.append(([[x,0,z],[u,0,w],[u,height,w],[x,height,z]], [x+u,0,z+w]))
        rows.append(([[0,0,0],[u,0,w],[x,0,z]], [0,-1,0]))
        rows.append(([[0,height,0],[x,height,z],[u,height,w]], [0,1,0]))
    return polygons(rows)


def ring(radius, tube=.025, *, segments=40):
    rows = []
    def p(a, b):
        return [(radius+tube*math.cos(b))*math.cos(a), tube*math.sin(b),
                (radius+tube*math.cos(b))*math.sin(a)]
    for i in range(segments):
        for j in range(6):
            a, aa = math.tau*i/segments, math.tau*(i+1)/segments
            b, bb = math.tau*j/6, math.tau*(j+1)/6
            mid, cross = (a+aa)/2, (b+bb)/2
            rows.append(([p(a,b),p(aa,b),p(aa,bb),p(a,bb)],
                         [math.cos(mid)*math.cos(cross),math.sin(cross),math.sin(mid)*math.cos(cross)]))
    return polygons(rows)


def transform(x=0, y=0, z=0, *, yaw=0):
    c, s = math.cos(yaw), math.sin(yaw)
    return torch.tensor([[c,0,s,x],[0,1,0,y],[-s,0,c,z],[0,0,0,1]], dtype=torch.float32)


def frame(root_angle, branch_angle):
    from gffx import scene as s
    gold, dark = (.8,.46,.13), (.055,.075,.10)
    prims = [s.Rig('turntable', transform=transform(y=.30,yaw=root_angle)),
             s.Rig('upper_arm',parent='turntable',transform=transform(y=1.8,yaw=branch_angle))]
    def add(name, geometry, color, parent=None, matrix=None):
        v, f, n = geometry
        prims.append(s.Mesh(name,v,f,parent=parent,transform=matrix,
                           attributes={'normals':s.Attribute(n,'vertex'),
                                       'colour':s.Attribute(v.new_tensor(color),'constant')}))
    add('plinth',cylinder(1.60,.20),dark,matrix=transform(y=.025))
    add('plinth_lip',cylinder(1.54,.045),gold,matrix=transform(y=.225))
    add('inlay',cylinder(1.49,.028),(.15,.20,.24),matrix=transform(y=.27))
    add('mast',cylinder(.055,2.65),gold,'turntable')
    add('lower_link',bevel_box((.53,.035,.035),.012),gold,'turntable',transform(x=-.52,y=.65))
    add('turquoise',bevel_box((.43,.43,.43),.065),(.045,.64,.62),'turntable',transform(x=-1.05,y=.65,yaw=.20))
    add('lower_joint',cylinder(.12,.14),dark,'turntable',transform(y=.58))
    add('lower_orbit',ring(1.08),gold,'turntable',transform(y=.65))
    add('branch_joint',cylinder(.145,.18),(.94,.80,.49),'upper_arm',transform(y=-.09))
    add('upper_link',bevel_box((.54,.035,.035),.012),gold,'upper_arm',transform(x=.53))
    add('coral',bevel_box((.34,.34,.34),.05),(1.,.21,.085),'upper_arm',transform(x=1.06,yaw=-.25))
    add('counter_link',bevel_box((.39,.025,.025),.008),gold,'upper_arm',transform(x=-.38,y=.48))
    add('counter_riser',cylinder(.035,.48),gold,'upper_arm')
    add('ivory',bevel_box((.25,.25,.25),.04),(.85,.88,.79),'upper_arm',transform(x=-.76,y=.48,yaw=.4))
    add('upper_orbit',ring(.78,.017,segments=32),(.34,.42,.47),'upper_arm',transform(y=.48))
    packed = s.Scene(prims).flatten()
    # Triangle-corner display vertices retain exact crease normals. No smoothing
    # across faces is delegated to the WebGPU viewer, including on motion frames.
    vertices = packed.vertices[packed.faces.long()].reshape(-1,3)
    return dict(vertices=vertices,
                faces=torch.arange(len(vertices),dtype=torch.int32).reshape(-1,3),
                normals=packed.attributes['normals'].values.reshape(-1,3),
                colors=packed.attributes['colour'].values.reshape(-1,3),
                world={name:matrix.clone() for name,matrix in packed.world_matrices.items()})


def tiled_samples(ndc, faces, attributes, height, width, *, tile=64):
    """Demo-only conservative screen bins; native rasterization in every tile.

    No clipping/visibility/gradient support is added to the public operation.
    This helper consumes fixed in-front geometry and supplies local face attrs.
    """
    from gffx.torch import render
    pixels = torch.stack(((ndc[:,0]+1)*width/2,(1-ndc[:,1])*height/2),-1)
    triangle = pixels[faces.long()]
    lo, hi = triangle.amin(1), triangle.amax(1)
    image = ndc.new_zeros((height,width,attributes.shape[-1]))
    valid = torch.zeros((height,width),dtype=torch.bool)
    for y in range(0,height,tile):
        for x in range(0,width,tile):
            h,w = min(tile,height-y),min(tile,width-x)
            select = (hi[:,0]>=x-1e-3)&(lo[:,0]<=x+w+1e-3)&(hi[:,1]>=y-1e-3)&(lo[:,1]<=y+h+1e-3)
            if not bool(select.any()):continue
            local = ndc.clone()
            local[:,0] = (pixels[:,0]-x)*2/w-1
            local[:,1] = 1-(pixels[:,1]-y)*2/h
            indices,bary,_,_ = render.rasterize(local.contiguous(),faces[select].contiguous(),h,w,cull_mode=render.CULL_NONE)
            image[y:y+h,x:x+w] = render.interpolate(indices,bary,attributes[select].contiguous())[0,:,:,0]
            valid[y:y+h,x:x+w] = indices[0,:,:,0]>=0
    return image, valid


@torch.no_grad()
def beauty(data, *, eye=(4.,3.6,6.), size=SIZE):
    from gffx.camera import Camera
    from .showcase import look_at
    from .cinematic import highlight_lobe
    height, width = size
    if size not in (SIZE,(180,320)):
        raise ValueError('use reviewed 720p or diagnostic size')
    v, f = data['vertices'], data['faces']
    floor = v.new_tensor([[-12,0,5],[5,0,5],[5,0,-12],[-12,0,-12]])
    render_data = dict(vertices=torch.cat((v,floor)),
        faces=torch.cat((f,torch.tensor([[0,1,2],[0,2,3]],dtype=torch.int32)+len(v))),
        normals=torch.cat((data['normals'],v.new_tensor([[0,1,0]]).expand(4,3))),
        colors=torch.cat((data['colors'],v.new_tensor([[.065,.08,.11]]).expand(4,3))))
    camera = Camera.from_intrinsics(width*.96,width*.96,width/2,height/2,height,width,.1,30.,
                                    pose=look_at(eye,[0.,1.45,0.]))
    v,f = render_data['vertices'],render_data['faces']
    projected = camera.project(v,in_front_only=True)
    if not bool((projected.view_depth > 0).all()):
        raise ValueError('presentation tiling requires geometry in front of the camera')
    ndc = torch.stack((projected.pixels[:,0]*2/width-1,
                      1-projected.pixels[:,1]*2/height,projected.ndc_depth),-1)
    attrs = torch.cat((v,render_data['normals'],render_data['colors'],torch.ones_like(v[:,:1])),-1)
    attrs = attrs/projected.view_depth[:,None]
    sample,valid = tiled_samples(ndc,f,attrs[f.long()].contiguous(),height,width)
    sample = sample/sample[:,:,-1:].clamp_min(1e-10)
    point = sample[:,:,:3]; normal = torch.nn.functional.normalize(sample[:,:,3:6],dim=-1)
    albedo = sample[:,:,6:9]; viewdir = torch.nn.functional.normalize(camera.pose[0,:3,3]-point,dim=-1)
    diffuse,specular = torch.zeros_like(point),torch.zeros_like(point)
    # Same bounded Phong display recipe as cinematic.py, not public Principled.
    angle = .23*math.pi*1.6
    for position,color,power in (([3*math.cos(angle),3.7,2.5*math.sin(angle)],[1.,.60,.32],20.),
                                 ([-2.,2.,-2.],[.12,.55,1.],26.),([0.,4.,3.],[.8,.9,1.],5.)):
        delta = v.new_tensor(position)-point; d2 = delta.square().sum(-1).clamp_min(.04)
        direction = delta/d2.sqrt()[...,None]
        cosine = (normal*direction).sum(-1).clamp_min(0)
        irradiance = v.new_tensor(color)*power/d2[...,None]
        diffuse += irradiance*cosine[...,None]
        specular += irradiance*(highlight_lobe(normal,direction,viewdir)*cosine)[...,None]*.12
    linear = .88*albedo*(diffuse+.045)/math.pi+specular
    rgb = (linear/(1+linear)).clamp(0,1).pow(1/2.2)
    yy = torch.linspace(0,1,height)[:,None,None]
    background = v.new_tensor([.045,.065,.095])[None,None,:]*(.5+yy*.9)
    rgb = torch.where(valid[...,None],rgb,background)
    # Display-only edge smoothing, excluded from numerical scene evidence.
    x = rgb.permute(2,0,1)[None]
    avg = torch.nn.functional.avg_pool2d(x,3,stride=1,padding=1,count_include_pad=False)
    rgb = torch.where((x-avg).abs().amax(1,keepdim=True)>.055,x*.35+avg*.65,x)[0].permute(1,2,0)
    return (rgb.clamp(0,1)*255).round().byte()


def payload(check=lambda: None):
    # No unbounded producer. Each clip is a closed authored loop, at most24 frames.
    clips = []
    for mode in ('whole','branch'):
        clip = []
        for i in range(FRAMES):
            check(); t = math.tau*i/FRAMES
            clip.append(frame(t if mode=='whole' else 0, t if mode=='branch' else -.45))
        clips.append(clip)
    hero = frame(0,0)
    images = []
    for data, eye in ((hero,(4.,3.6,6.)), (clips[0][4],(4.,3.6,6.)),
                      (clips[1][5],(4.,3.6,6.))):
        check(); images.append(beauty(data)); check()
    return dict(hero=hero,clips=clips,images=images)


def view(value):
    from gffx.dashboard import ArrayView
    kind = 'uint8' if value.dtype == torch.uint8 else 'int32' if value.dtype == torch.int32 else 'float32'
    value = value.detach().to(device='cpu',dtype=getattr(torch,kind)).contiguous()
    return ArrayView(kind,tuple(value.shape),ctypes.string_at(value.data_ptr(),value.numel()*value.element_size()))


def publish_data(dash, data):
    artifacts = []
    def add(kind,path,label,*args,**kwargs):
        getattr(dash,kind)(path,*args,step=0,**kwargs)
        artifacts.append(dict(kind=kind,path=path,label=label))
    add('image','atelier/hero','KINETIC ATELIER · a scene built from parent transforms',view(data['images'][0]))
    add('text','atelier/guide','What to try',
        'A turquoise anchor, a coral satellite and an ivory counterweight share a turntable. '
        'Gold arms and visible joints show their parent connections. First play WHOLE ASSEMBLY: '
        'all three objects move together, while the plinth stays still. Then play UPPER BRANCH: '
        'only coral and ivory orbit; turquoise stays fixed. Drag either mesh view to inspect it. '
        'These are two finite 24-frame clips; playback speed is a viewer setting, not a measured simulation rate. '
        'Public gffx.scene.Scene.flatten computes transforms and normals; the existing native rasterizer '
        'makes the 1280×720 studio stills. Lighting is the existing cinematic display recipe, '
        'not a Principled/Cycles comparison, ray-traced shadows, physics or native animation. '
        'The interactive WebGPU preview uses different lighting. Appearance is not test acceptance.')
    h = data['hero']
    add('mesh','atelier/explore','Explore the sculpture · crisp creases and explicit colours',
        view(h['vertices']),view(h['faces']),colors=view(h['colors']),normals=view(h['normals']))
    for j, (mode, label) in enumerate((('whole','WHOLE ASSEMBLY · one parent moves all three'),
                                      ('branch','UPPER BRANCH · turquoise stays fixed'))):
        path = 'atelier/'+mode
        queue = dash.queue(path,'mesh',recent_window=FRAMES,snapshot_every=1)
        for i, row in enumerate(data['clips'][j]):
            queue.push(view(row['vertices']),view(row['faces']),colors=view(row['colors']),normals=view(row['normals']),step=i)
        artifacts.append(dict(kind='mesh',path=path,label=label))
        add('image',path+'-still',label+' · studio still',view(data['images'][j+1]))
    return artifacts


def publish(address, run, data):
    from gffx.dashboard import connect
    from .reporter import Reporter
    from tools.shading_cuda_closeout import digest
    import gffx.torch.scene as scene
    reporter = Reporter(address,run)
    def send(event,**kwargs):
        if not reporter.send(event,**kwargs):raise RuntimeError(reporter.error)
    send('start',kind='demo',provenance={'artifact_sha256':digest(scene.__file__)})
    with connect(address,run=run,wait_acks=True) as dash:
        artifacts = publish_data(dash,data)
    send('demo',page='scene/hierarchy-demo',title='Kinetic Atelier · scenes in motion',artifacts=artifacts)
    send('finish',exit_code=0)
