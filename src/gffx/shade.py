"""Lazy shading facade; PyTorch is the first and currently sole adapter.

Importing this module does not import a framework or native binary. Calling an
operation opts into the PyTorch adapter. All arithmetic lives in that adapter.
"""
__all__ = ['lambert', 'blinn_phong', 'ggx_distribution', 'smith_visibility', 'fresnel_schlick']


def lambert(albedo):
    from .torch.shade import lambert as operation
    return operation(albedo)


def blinn_phong(n_dot_h, exponent):
    from .torch.shade import blinn_phong as operation
    return operation(n_dot_h, exponent)


def ggx_distribution(n_dot_h, roughness):
    from .torch.shade import ggx_distribution as operation
    return operation(n_dot_h, roughness)


def smith_visibility(n_dot_v, n_dot_l, roughness):
    from .torch.shade import smith_visibility as operation
    return operation(n_dot_v, n_dot_l, roughness)


def fresnel_schlick(v_dot_h, f0):
    from .torch.shade import fresnel_schlick as operation
    return operation(v_dot_h, f0)
