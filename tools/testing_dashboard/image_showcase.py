"""CHROMA: finite authored two-dimensional art made with public gffx image operations.

The procedural rasters and tiny bitmap lettering are original demo assets, not new
font/geometry/rendering functionality. Blur shadows are an authored graphic-design recipe.
"""
import ctypes
import math

import torch
import gffx.image as image


_FONT = {
    'A':'01110 10001 10001 11111 10001 10001 10001', 'B':'11110 10001 10001 11110 10001 10001 11110',
    'C':'01111 10000 10000 10000 10000 10000 01111', 'D':'11110 10001 10001 10001 10001 10001 11110',
    'E':'11111 10000 10000 11110 10000 10000 11111', 'F':'11111 10000 10000 11110 10000 10000 10000',
    'G':'01111 10000 10000 10111 10001 10001 01111', 'H':'10001 10001 10001 11111 10001 10001 10001',
    'I':'11111 00100 00100 00100 00100 00100 11111', 'J':'00111 00010 00010 00010 10010 10010 01100',
    'K':'10001 10010 10100 11000 10100 10010 10001', 'L':'10000 10000 10000 10000 10000 10000 11111',
    'M':'10001 11011 10101 10101 10001 10001 10001', 'N':'10001 11001 10101 10011 10001 10001 10001',
    'O':'01110 10001 10001 10001 10001 10001 01110', 'P':'11110 10001 10001 11110 10000 10000 10000',
    'Q':'01110 10001 10001 10001 10101 10010 01101', 'R':'11110 10001 10001 11110 10100 10010 10001',
    'S':'01111 10000 10000 01110 00001 00001 11110', 'T':'11111 00100 00100 00100 00100 00100 00100',
    'U':'10001 10001 10001 10001 10001 10001 01110', 'V':'10001 10001 10001 10001 10001 01010 00100',
    'W':'10001 10001 10001 10101 10101 10101 01010', 'X':'10001 10001 01010 00100 01010 10001 10001',
    'Y':'10001 10001 01010 00100 00100 00100 00100', 'Z':'11111 00001 00010 00100 01000 10000 11111',
    '0':'01110 10001 10011 10101 11001 10001 01110', '1':'00100 01100 00100 00100 00100 00100 01110',
    '2':'01110 10001 00001 00010 00100 01000 11111', '3':'11110 00001 00001 01110 00001 00001 11110',
    '4':'00010 00110 01010 10010 11111 00010 00010',
    '/':'00001 00010 00010 00100 01000 01000 10000', '-':'00000 00000 00000 11111 00000 00000 00000',
}


def _text(canvas, text, x, y, scale, colour):
    """Authored five-column fixture lettering; not a font API."""
    scale=max(1,scale)
    for index,char in enumerate(text):
        if char not in _FONT:
            continue
        for row,line in enumerate(_FONT[char].split()):
            for col,on in enumerate(line):
                if on=='1':
                    xx=x+(index*6+col)*scale; yy=y+row*scale
                    canvas[max(0,yy):min(len(canvas),yy+scale),max(0,xx):min(canvas.shape[1],xx+scale)]=canvas.new_tensor(colour)


def _display(rgba):
    return (image.linear_to_srgb(rgba[...,:3]).clamp(0,1)*255).round().to(torch.uint8)


def _asset(n, kind):
    yy,xx=torch.meshgrid(torch.linspace(-1,1,n),torch.linspace(-1,1,n),indexing='ij')
    radius=torch.sqrt(xx.square()+yy.square())
    angle=torch.atan2(yy,xx)
    outer=((.89-radius)*n*.5+.5).clamp(0,1)
    if kind==0:
        coverage=outer*((radius-.48)*n*.5+.5).clamp(0,1)
        t=(angle/math.tau+.5).unsqueeze(-1)
        rgb=(1-t)*torch.tensor([.008,.32,.44])+t*torch.tensor([.04,.72,.49])
    elif kind==1:
        coverage=outer
        t=((xx+yy+2)/4).unsqueeze(-1)
        rgb=(1-t)*torch.tensor([.96,.055,.025])+t*torch.tensor([.44,.008,.16])
        rgb=rgb*(.84+.16*torch.cos(angle*13+radius*6)).unsqueeze(-1)
    else:
        coverage=outer*((radius-.63)*n*.5+.5).clamp(0,1)
        t=((yy+1)/2).unsqueeze(-1)
        rgb=(1-t)*torch.tensor([.99,.67,.06])+t*torch.tensor([.88,.22,.018])
    rgba=torch.cat((rgb,coverage.unsqueeze(-1)*.94),-1)
    return image.convert_alpha(rgba)


@torch.no_grad()
def hero(size=(720,1280)):
    """Designed layered poster: three transformed colour studies with soft graphic shadows."""
    h,w=size
    yy,xx=torch.meshgrid(torch.linspace(0,1,h),torch.linspace(0,1,w),indexing='ij')
    vignette=1-.12*((xx-.55).square()+(yy-.4).square())
    rgb=vignette.unsqueeze(-1)*torch.tensor([.88,.89,.84])
    canvas=torch.cat((rgb,torch.ones(h,w,1)),-1)
    n=max(32,round(h*.76))
    for kind,cx,cy,rotation,sx,sy in ((0,.71,.43,-.32,1.12,.92),
                                      (1,.76,.56,.28,.70,.70),
                                      (2,.61,.66,.45,.82,.51)):
        layer=_asset(n,kind)
        # A source-space fade is deliberately separate from layer opacity.
        mask=torch.linspace(.84,1,n).reshape(n,1,1).expand(n,n,1)
        layer=image.apply_mask(image.adjust(layer,exposure=.12),mask,opacity=.96)
        pivot=torch.tensor([(n-1)/2,(n-1)/2])
        translation=torch.tensor([cx*w-pivot[0],cy*h-pivot[1]])
        transform=image.affine_matrix(translation,torch.tensor(rotation),torch.tensor([sx,sy]),pivot)
        shadow=torch.cat((torch.zeros_like(layer[...,:3]),layer[...,3:4]*.22),-1)
        shadow=image.gaussian_blur(shadow,max(.7,n/120),radius=min(12,max(1,n//40)))
        shadow_transform=transform.clone(); shadow_transform[:,2]+=torch.tensor([h*.015,h*.025])
        canvas=image.composite([image.warp_affine(shadow,shadow_transform,(h,w)),
                                image.warp_affine(layer,transform,(h,w))],canvas)
    result=_display(canvas)
    ink=(18,40,49); muted=(75,91,93); scale=max(1,w//640)
    _text(result,'GFFX / IMAGE LAB',round(w*.055),round(h*.075),scale,ink)
    _text(result,'CHROMA',round(w*.055),round(h*.32),max(1,w//160),ink)
    _text(result,'LINEAR LIGHT',round(w*.057),round(h*.47),scale,muted)
    _text(result,'LAYERS / MASKS / WARP',round(w*.057),round(h*.53),scale,muted)
    _text(result,'01 / PUBLIC IMAGE OPERATIONS',round(w*.055),round(h*.90),scale,ink)
    y=round(h*.85); result[y:y+max(1,scale//2),round(w*.055):round(w*.945)]=result.new_tensor((141,154,151))
    return result


@torch.no_grad()
def transparency():
    """Contrast an explicitly labelled wrong straight-RGB mix with correct premultiplied filtering."""
    h,w=240,512
    source=torch.tensor([[[1.,0,0,0],[0,0,1.,1.]]]).expand(h,2,4).clone()
    transform=torch.tensor([[float(w-1),0.,0.],[0.,1.,0.]])
    correct=image.warp_affine(source,transform,(h,w),alpha='straight')
    a=torch.linspace(0,1,w).reshape(1,w,1).expand(h,w,1)
    naive=torch.cat((torch.cat((1-a,torch.zeros_like(a),a),-1),a),-1)
    yy,xx=torch.meshgrid(torch.arange(h),torch.arange(w),indexing='ij')
    check=((xx//16+yy//16)%2).float()*.10+.07
    background=torch.cat((check[...,None].expand(h,w,3),torch.ones(h,w,1)),-1)
    left=_display(image.composite([naive],background,alpha='straight'))
    right=_display(image.composite([correct],background,alpha='straight'))
    result=torch.full((h+96,w*2+24,3),235,dtype=torch.uint8)
    result[64:64+h,:w]=left; result[64:64+h,w+24:]=right
    _text(result,'WRONG / STRAIGHT RGB MIX',16,22,2,(92,25,36))
    _text(result,'RIGHT / PREMULTIPLIED',w+40,22,2,(12,85,75))
    return result


@torch.no_grad()
def sampling():
    """Display magnification makes aliasing visible; it is not an extra library resize operation."""
    n=48
    yy,xx=torch.meshgrid(torch.arange(n),torch.arange(n),indexing='ij')
    stripe=((xx+2*yy)%4<2).float()
    rgb=stripe[...,None]*torch.tensor([.01,.55,.49])+(1-stripe[...,None])*torch.tensor([.018,.03,.06])
    source=torch.cat((rgb,torch.ones(n,n,1)),-1)
    base=image.affine_matrix(torch.tensor([1.3,-.7]),torch.tensor(.24),torch.tensor([.75,.75]),torch.tensor([24.,24.]))
    panels=[]
    for mode,factor in (('nearest',1),('bilinear',1),('bilinear',4)):
        matrix=base.clone()
        # High-resolution output centre k corresponds to (k+.5)/factor-.5 in the base grid.
        matrix[:,:2]*=factor
        matrix[:,2]=factor*(matrix[:,2]+.5)-.5
        rendered=image.warp_affine(source,matrix,(48*factor,48*factor),filter=mode,border='clamp')
        if factor>1: rendered=image.reconstruct(rendered,factor)
        panels.append(_display(rendered).repeat_interleave(6,0).repeat_interleave(6,1))
    result=torch.full((380,912,3),235,dtype=torch.uint8)
    for i,(panel,label) in enumerate(zip(panels,('NEAREST','BILINEAR','4X BOX RESOLVE'))):
        x=12+i*300; result[70:358,x:x+288]=panel
        _text(result,label,x+6,28,2,(18,40,49))
    return result


def payload(check=lambda:None):
    images=[]
    for create in (hero,transparency,sampling):
        check(); images.append(create()); check()
    return images


def _view(value):
    from gffx.dashboard import ArrayView
    value=value.contiguous()
    return ArrayView('uint8',tuple(value.shape),ctypes.string_at(value.data_ptr(),value.numel()))


def publish_data(dash,images):
    artifacts=[]
    for path,label,pixels in zip(('chroma/hero','chroma/transparency','chroma/sampling'),
                                ('CHROMA / public image operations','Transparent edges / wrong versus right',
                                 'Sampling / inspect the aliasing'),images):
        dash.image(path,_view(pixels),step=0)
        artifacts.append(dict(kind='image',path=path,label=label))
    text=('CHROMA is authored 2D graphic design using public gffx mask, opacity, exposure, affine sampling, '
          'Gaussian blur and compositing. The shadow recipe, gradient rasters and bitmap lettering are '
          'authored demonstration assets, not physical lighting, a font engine or new geometry functionality. '
          'These three finite images are not test acceptance, a Compositron integration proof or a realtime claim. '
          'The transparency comparison intentionally labels the straight-RGB counterexample WRONG. '
          'Sampling panels are display-magnified actual outputs; bilinear downscaling can alias. '
          'Linear-to-sRGB conversion and display clipping occur explicitly at the final presentation boundary.')
    dash.text('chroma/guide',text,step=0)
    artifacts.append(dict(kind='text',path='chroma/guide',label='What this demonstration establishes'))
    return artifacts


def publish(address,run,images):
    from gffx.dashboard import connect
    from .reporter import Reporter
    from tools.shading_cuda_closeout import digest
    import gffx.torch.image as backend
    reporter=Reporter(address,run)
    def send(event,**kwargs):
        if not reporter.send(event,**kwargs): raise RuntimeError(reporter.error)
    send('start',kind='demo',provenance={'artifact_sha256':digest(backend.__file__)})
    with connect(address,run=run,wait_acks=True) as dash:
        artifacts=publish_data(dash,images)
    send('demo',page='image/chroma-demo',title='CHROMA / linear light in layers',artifacts=artifacts)
    send('finish',exit_code=0)
