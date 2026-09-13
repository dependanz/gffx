"""gffx-side synthetic integration readiness; consumer implementation is outside this fixture.

Prospective recipe:32x32, one asymmetric fixed foreground over a locked background;
target translation(.4,-.3)/opacity.8; initial(1.1,-.9)/opacity.6; Adam100 steps lr.08.
Target is independent scalar bilinear+over. No random seed, private asset or cached graph.
"""
import math

import torch


def _source(size=32):
    values=[]
    for y in range(size):
        row=[]
        for x in range(size):
            ellipse=((x-13)/8)**2+((y-16)/10)**2
            a=max(0.,min(1.,(1-ellipse)*4))
            notch=0.3 if x>13 and 12<y<17 else 1.
            a*=notch
            rgb=(.15+.65*x/(size-1),.2+.3*y/(size-1),.7-.4*x/(size-1))
            row.append([*(c*a for c in rgb),a])
        values.append(row)
    return torch.tensor(values,dtype=torch.float64)


def _target(source, dx, dy, opacity):
    pixels=source.tolist(); h,w=source.shape[:2]; out=[]
    background=(.03,.05,.08,1.)
    for y in range(h):
        row=[]
        for x in range(w):
            sx,sy=x-dx,y-dy; x0,y0=math.floor(sx),math.floor(sy)
            fx,fy=sx-x0,sy-y0; sample=[0.]*4
            for iy,wy in ((y0,1-fy),(y0+1,fy)):
                for ix,wx in ((x0,1-fx),(x0+1,fx)):
                    if 0<=iy<h and 0<=ix<w:
                        for c in range(4): sample[c]+=pixels[iy][ix][c]*wx*wy*opacity
            row.append([sample[c]+(1-sample[3])*background[c] for c in range(4)])
        out.append(row)
    return torch.tensor(out,dtype=source.dtype)


def test_known_translation_opacity_recovery_against_independent_target(record_property):
    """A fixed optimizer must improve a fresh graph while leaving source and locked background unchanged."""
    from gffx.torch import image
    source=_source(); original=source.clone()
    background=source.new_tensor([.03,.05,.08,1.]).expand_as(source).clone()
    locked=background.clone(); target=_target(source,.4,-.3,.8)
    params=torch.tensor([1.1,-.9,math.log(.6/.4)],dtype=source.dtype,requires_grad=True)
    def evaluate():
        zero=params.new_zeros(()); one=params.new_ones(())
        matrix=torch.stack((torch.stack((one,zero,params[0])),torch.stack((zero,one,params[1]))))
        layer=image.apply_mask(source,opacity=params[2].sigmoid())
        layer=image.warp_affine(layer,matrix,(32,32))
        return image.composite([layer],background)
    initial=float(((evaluate()[...,:3]-target[...,:3])**2).mean().detach())
    optimizer=torch.optim.Adam([params],lr=.08)
    for _ in range(100):
        optimizer.zero_grad(set_to_none=True)
        loss=((evaluate()[...,:3]-target[...,:3])**2).mean()
        loss.backward()
        assert params.grad is not None and bool(torch.isfinite(params.grad).all())
        optimizer.step()
    final=float(((evaluate()[...,:3]-target[...,:3])**2).mean().detach())
    actual=[float(params[0].detach()),float(params[1].detach()),float(params[2].detach().sigmoid())]
    record_property('initial_mse',initial); record_property('final_mse',final)
    record_property('recovered_translation_opacity',str(actual))
    record_property('independent_target','scalar-bilinear-over-v1')
    assert initial>0 and final<=initial*.01
    assert torch.equal(source,original) and torch.equal(background,locked)
