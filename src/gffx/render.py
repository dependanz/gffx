"""Lazy rendering facade. Arithmetic lives exclusively in the bound adapter."""
import sys
from types import ModuleType

__all__=['opaque_albedo_replay','opaque_material_quadrature_batch','opaque_material_quadrature','opaque_boundary_quadrature','opaque_window','opaque_quadrature','opaque_integrate','opaque_coverage','render','mesh','visibility','AttributeSource','TextureSource','rasterize','interpolate','texture','texture_pyramid']
_CONSTANTS={'CULL_NONE','CULL_BACK','CULL_FRONT','FILTER_NEAREST','FILTER_BILINEAR',
            'MIP_NEAREST','MIP_LINEAR','WRAP_REPEAT','WRAP_CLAMP','WRAP_MIRROR','WRAP_BORDER'}


def render(*args,**kwargs):
    from .torch import render as bound
    return bound(*args,**kwargs)


def __getattr__(name):
    if name not in __all__ and name not in _CONSTANTS:
        raise AttributeError("module 'gffx.render' has no attribute %r" % name)
    from .torch import render as bound
    return getattr(bound,name)


def __dir__():return sorted(set(__all__)|_CONSTANTS)


class _CallableRender(ModuleType):
    def __call__(self,*args,**kwargs):return render(*args,**kwargs)


sys.modules[__name__].__class__=_CallableRender
