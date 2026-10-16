"""Explicit PNG export using the standard library; importing this module is inert."""
from pathlib import Path
import math
import struct
import zlib

__all__=['write_png']


def write_png(path, image, *, tone_map='aces', exposure=0., overwrite=False):
    """Export linear RGB (HWC or a one-image BHWC tensor) as an sRGB PNG.

    Export explicitly detaches and transfers to CPU. It does not change the
    supplied tensor or its gradient graph. No NumPy or Pillow is required.
    """
    import torch
    from .torch.image import linear_to_srgb
    if not isinstance(image,torch.Tensor):raise TypeError('image must be a tensor')
    if image.ndim==4 and image.shape[0]==1:image=image[0]
    if image.ndim!=3 or image.shape[-1]!=3 or min(image.shape[:2])<1:
        raise ValueError('PNG export requires HWC RGB or one-image BHWC RGB')
    if image.dtype not in (torch.float32,torch.float64) or not bool(torch.isfinite(image).all()):
        raise ValueError('image must have finite float32/float64 channels')
    if not bool((image>=0).all()):raise ValueError('linear RGB must be nonnegative')
    if isinstance(exposure,bool) or not isinstance(exposure,(int,float)) or not math.isfinite(exposure):
        raise ValueError('exposure must be finite stops')
    if tone_map not in ('aces','clip'):raise ValueError('tone_map must be aces or clip')
    if type(overwrite) is not bool:raise TypeError('overwrite must be boolean')
    target=Path(path)
    if target.exists() and not overwrite:raise FileExistsError(target)
    with torch.no_grad():
        rgb=image.detach()*2.**exposure
        if not bool(torch.isfinite(rgb).all()):raise ValueError('exposure overflows image precision')
        if tone_map=='aces':
            # Algebraically equivalent rational curve without large-value squaring.
            x=rgb.clamp_max(1e6)
            rgb=(x*(2.51*x+.03)/(x*(2.43*x+.59)+.14)).clamp(0,1)
        else:rgb=rgb.clamp(0,1)
        data=(linear_to_srgb(rgb.contiguous()).clamp(0,1)*255).round().to(torch.uint8).cpu()
    h,w,_=data.shape
    def chunk(kind,payload):
        return struct.pack('>I',len(payload))+kind+payload+struct.pack('>I',zlib.crc32(kind+payload)&0xffffffff)
    raw=b''.join(b'\0'+bytes(row.reshape(-1).tolist()) for row in data)
    encoded=(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',w,h,8,2,0,0,0))
             +chunk(b'sRGB',b'\0')+chunk(b'IDAT',zlib.compress(raw,6))+chunk(b'IEND',b''))
    with target.open('wb' if overwrite else 'xb') as output:output.write(encoded)
    return target
