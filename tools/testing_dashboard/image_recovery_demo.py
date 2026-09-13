"""A finite, inspectable presentation of the existing library-only recovery fixture.

Metadata imports no tensor framework. Capture is CPU-only and must run inside the
existing supervised worker, not as a dashboard request or continuous sender.
"""
import importlib.util
import math
from pathlib import Path


PROFILE={'size':32,'steps':100,'learning_rate':.08,'residual_full_scale':.1,'display_scale':8}
GUIDE=(
    'A gffx-only synthetic recovery for CI-07 integration readiness, not a consumer '
    'implementation or acceptance claim by this visual alone. A 32 by 32 independent scalar '
    'bilinear target is matched by fitting translation and opacity with public gffx '
    'operations. Pixels are enlarged eightfold for inspection, not recomputed at a '
    'higher resolution. The source and background remain locked. The error panel '
    'shows maximum absolute linear-RGB error per pixel on a fixed 0 to 0.1 scale; '
    'orange means more error. The loss chart has fixed log10 limits from 1 at the '
    'top to 1e-10 at the bottom; values below that display floor remain unchanged '
    'in the saved measurements. Display conversion is separate from linear-light MSE. '
    'This single fixed initialization does not establish a unique solution, arbitrary '
    'optimizer convergence, Compositron integration, cancellation, undo or real-time speed.'
)
BOARD_LABELS=('GFFX / IMAGE RECOVERY','INDEPENDENT TARGET','BEFORE FITTING',
              'RECOVERED','ABSOLUTE ERROR','FIXED ERROR SCALE / BLACK TO ORANGE',
              '32 X 32 / ENLARGED FOR INSPECTION')


def fixture_path():
    """Resolve only the checked-in, already qualified fixture; never a caller-selected module."""
    return Path(__file__).resolve().parents[2]/'tests/python/test_image_recovery.py'


def _fixture():
    spec=importlib.util.spec_from_file_location('_gffx_image_recovery_fixture',fixture_path())
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def capture(*,checkpoint):
    """Save target/before/after/residuals and all 101 measured states of the fixed recipe."""
    import torch
    from gffx.torch import image
    fixture=_fixture()
    source=fixture._source(PROFILE['size']);original=source.clone()
    background=source.new_tensor([.03,.05,.08,1.]).expand_as(source).clone()
    locked=background.clone();target=fixture._target(source,.4,-.3,.8)
    params=torch.tensor([1.1,-.9,math.log(.6/.4)],dtype=source.dtype,requires_grad=True)
    def evaluate():
        zero=params.new_zeros(());one=params.new_ones(())
        matrix=torch.stack((torch.stack((one,zero,params[0])),torch.stack((zero,one,params[1]))))
        layer=image.apply_mask(source,opacity=params[2].sigmoid())
        return image.composite([image.warp_affine(layer,matrix,(32,32))],background)
    def values():
        return [float(params[0].detach()),float(params[1].detach()),float(params[2].detach().sigmoid())]
    optimizer=torch.optim.Adam([params],lr=PROFILE['learning_rate'])
    losses=[];parameters=[];before=None
    for step in range(PROFILE['steps']):
        checkpoint();optimizer.zero_grad(set_to_none=True)
        output=evaluate()
        if step==0: before=output.detach().clone()
        loss=((output[...,:3]-target[...,:3])**2).mean()
        losses.append(float(loss.detach()));parameters.append(values())
        loss.backward()
        if params.grad is None or not bool(torch.isfinite(params.grad).all()):
            raise RuntimeError('recovery produced missing/nonfinite parameter derivatives')
        optimizer.step()
        del output,loss
    checkpoint()
    after=evaluate().detach()
    losses.append(float(((after[...,:3]-target[...,:3])**2).mean()))
    parameters.append(values())
    if not losses[0]>0 or not losses[-1]<=losses[0]*.01:
        raise RuntimeError('fixed recovery did not meet its existing 99 percent improvement gate')
    if not torch.equal(source,original) or not torch.equal(background,locked):
        raise RuntimeError('recovery changed a locked source or background')
    result={'target':target.detach(),'before':before,'after':after,
            'residual_before':(before[...,:3]-target[...,:3]).abs(),
            'residual_after':(after[...,:3]-target[...,:3]).abs(),
            'loss_history':losses,'parameter_history':parameters,
            'target_parameters':[.4,-.3,.8],'profile':dict(PROFILE),
            'source_unchanged':True,'background_unchanged':True,
            'independent_target':'scalar-bilinear-over-v1'}
    checkpoint()
    return result


def board(captured):
    """Arrange saved pixels; never run fitting again or renormalize the error for appearance."""
    import torch
    from gffx.torch import image
    from .image_showcase import _text
    scale=PROFILE['display_scale'];size=PROFILE['size']*scale
    canvas=torch.full((800,608,3),247,dtype=torch.uint8)
    ink=(20,43,51);muted=(87,101,104)
    _text(canvas,BOARD_LABELS[0],24,26,2,ink)
    _text(canvas,BOARD_LABELS[-1],24,65,1,muted)
    def display(value):
        rgb=(image.linear_to_srgb(value[...,:3]).clamp(0,1)*255).round().to(torch.uint8)
        return rgb.repeat_interleave(scale,0).repeat_interleave(scale,1)
    magnitude=(captured['residual_after'].amax(-1)/PROFILE['residual_full_scale']).clamp(0,1)
    error=(magnitude.unsqueeze(-1)*torch.tensor([255.,104.,34.],dtype=magnitude.dtype)).round().to(torch.uint8)
    error=error.repeat_interleave(scale,0).repeat_interleave(scale,1)
    panels=(display(captured['target']),display(captured['before']),display(captured['after']),error)
    for index,pixels in enumerate(panels):
        x=32+(index%2)*288;y=130+(index//2)*320
        _text(canvas,BOARD_LABELS[index+1],x,y-25,1,ink)
        canvas[y:y+size,x:x+size]=pixels
    _text(canvas,BOARD_LABELS[-2],32,737,1,muted)
    for x in range(256):
        canvas[761:773,32+x]=torch.tensor([round(255*x/255),round(104*x/255),round(34*x/255)],dtype=torch.uint8)
    return canvas


def history_chart(captured):
    """Plot the actual 101 MSE values with fixed log10 limits [-10,0], no rescaled drama."""
    import torch
    from .image_showcase import _text
    canvas=torch.full((280,608,3),247,dtype=torch.uint8)
    _text(canvas,'ACTUAL LOSS HISTORY',24,20,2,(20,43,51))
    _text(canvas,'LOG MSE / TOP 1 / BOTTOM 1E-10',24,51,1,(87,101,104))
    x0,x1,y0,y1=32,576,81,229
    for level in range(6):
        y=round(y0+(y1-y0)*level/5)
        canvas[y:y+1,x0:x1+1]=torch.tensor([205,215,214],dtype=torch.uint8)
    points=[]
    for step,loss in enumerate(captured['loss_history']):
        log_loss=max(-10.,min(0.,math.log10(max(loss,1e-10))))
        points.append((round(x0+(x1-x0)*step/100),round(y0+(y1-y0)*(-log_loss)/10)))
    for (xa,ya),(xb,yb) in zip(points,points[1:]):
        count=max(abs(xb-xa),abs(yb-ya),1)
        for part in range(count+1):
            x=round(xa+(xb-xa)*part/count);y=round(ya+(yb-ya)*part/count)
            canvas[y-1:y+2,x-1:x+2]=torch.tensor([0,122,119],dtype=torch.uint8)
    _text(canvas,'0 TO 100 OPTIMIZER STEPS',32,252,1,(87,101,104))
    return canvas
